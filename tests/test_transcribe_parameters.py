"""Uebergabe der Auftragsparameter an whisper_timestamped.transcribe.

Frueher fing der Worker jeden TypeError ab und rief die Bibliothek erneut auf, erst ohne vad,
dann ohne weitere Decoding-Parameter. Ein Auftrag mit VAD konnte so still ohne VAD enden.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.services import gpu_diagnostics, transcription, transcription_worker
from app.services.transcription import TranscriptionError, WhisperTimestampedService


def _fake_whisper(monkeypatch, *, error: Exception | None = None, without: tuple[str, ...] = ()):
    """Ersetzt whisper_timestamped; transcribe hat die Signatur von 1.15.9 (ohne `without`)."""
    calls: list[dict] = []

    def transcribe(model, audio, language=None, task="transcribe", vad=False, detect_disfluencies=False,
                   temperature=0.0, best_of=None, beam_size=None, no_speech_threshold=0.6, fp16=None,
                   condition_on_previous_text=True, initial_prompt=None):
        calls.append(dict(language=language, task=task, vad=vad, detect_disfluencies=detect_disfluencies,
                          temperature=temperature, best_of=best_of, beam_size=beam_size,
                          no_speech_threshold=no_speech_threshold, fp16=fp16,
                          condition_on_previous_text=condition_on_previous_text, initial_prompt=initial_prompt))
        if error is not None:
            raise error
        return {"text": "hallo", "language": "de", "segments": []}

    if without:
        # Aeltere Bibliothek: dieselbe Funktion ohne die genannten Parameter.
        import inspect
        signature = inspect.signature(transcribe)
        reduced = signature.replace(parameters=[p for p in signature.parameters.values() if p.name not in without])
        transcribe.__signature__ = reduced  # type: ignore[attr-defined]

    module = types.ModuleType("whisper_timestamped")
    module.transcribe = transcribe
    monkeypatch.setitem(sys.modules, "whisper_timestamped", module)
    monkeypatch.setattr(WhisperTimestampedService, "_resolve_runtime_device", lambda self: "cpu")
    monkeypatch.setattr(WhisperTimestampedService, "_load_model", lambda self, model, device, whisper: object())
    monkeypatch.setattr(gpu_diagnostics, "observe_model", lambda model, name: None)
    monkeypatch.setattr(gpu_diagnostics, "inference_succeeded", lambda observation: None)
    return calls


def _run(service: WhisperTimestampedService, *, vad: bool, vad_mode: str | None = None) -> dict:
    return service._transcribe_wav(
        wav_path="/tmp/egal.wav", model="large-v3", language="de", prompt="Kontext", beam_size=5, best_of=5,
        temperature=None, condition_on_previous_text=False, vad=vad, vad_mode=vad_mode, task="transcribe",
        no_speech_threshold=0.5, detect_disfluencies=True, accurate=False,
    )


def test_base_passes_vad_false_explicitly(monkeypatch) -> None:
    calls = _fake_whisper(monkeypatch)
    _run(WhisperTimestampedService(), vad=False)
    assert len(calls) == 1
    assert "vad" in calls[0] and calls[0]["vad"] is False


@pytest.mark.parametrize("vad_mode", [None, "auditok"])
def test_vad_passes_auditok(monkeypatch, vad_mode) -> None:
    calls = _fake_whisper(monkeypatch)
    _run(WhisperTimestampedService(), vad=True, vad_mode=vad_mode)
    assert [c["vad"] for c in calls] == ["auditok"]


def test_successful_call_keeps_all_parameters(monkeypatch) -> None:
    calls = _fake_whisper(monkeypatch)
    _run(WhisperTimestampedService(), vad=True)
    assert calls == [dict(language="de", task="transcribe", vad="auditok", detect_disfluencies=True, temperature=0.0,
                          best_of=5, beam_size=5, no_speech_threshold=0.5, fp16=False,
                          condition_on_previous_text=False, initial_prompt="Kontext")]


def test_internal_type_error_is_a_failure_without_second_call(monkeypatch) -> None:
    calls = _fake_whisper(monkeypatch, error=TypeError("unsupported operand type(s)"))
    with pytest.raises(TranscriptionError, match="TypeError: unsupported operand"):
        _run(WhisperTimestampedService(), vad=True)
    assert len(calls) == 1 and calls[0]["vad"] == "auditok"


def test_missing_library_parameter_fails_before_inference(monkeypatch) -> None:
    calls = _fake_whisper(monkeypatch, without=("vad",))
    with pytest.raises(TranscriptionError, match="does not support required parameters: vad"):
        _run(WhisperTimestampedService(), vad=True)
    assert calls == []


def test_failed_job_is_not_reported_as_success(monkeypatch, tmp_path: Path) -> None:
    _fake_whisper(monkeypatch, error=TypeError("internal"))
    request = tmp_path / "request.json"
    response = tmp_path / "response.json"
    request.write_text(json.dumps({"wav_path": str(tmp_path / "a.wav"), "model": "large-v3", "vad": True,
                                   "vad_mode": "auditok"}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["worker", "--request", str(request), "--response", str(response)])
    assert transcription_worker.main() == 1
    body = json.loads(response.read_text(encoding="utf-8"))
    assert "text" not in body and "TypeError: internal" in body["error"]


def test_vad_request_and_passed_value_are_logged(monkeypatch, caplog) -> None:
    _fake_whisper(monkeypatch)
    with caplog.at_level("INFO", logger=transcription.__name__):
        _run(WhisperTimestampedService(), vad=True)
    line = next(r.getMessage() for r in caplog.records if "whisper transcribe:" in r.getMessage())
    assert "vad_requested=True" in line and "vad_passed='auditok'" in line
    assert "Kontext" not in line and "prompt_chars=7" in line
