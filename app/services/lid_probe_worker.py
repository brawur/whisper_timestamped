from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any

from app.models import LidCandidate, LidProbeResponse
from app.services.transcription import TranscriptionError, WhisperTimestampedService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def _write_error(response_path: Path, message: str) -> None:
    response_path.write_text(json.dumps({"error": message}), encoding="utf-8")


def _read_pcm_window(wav_path: str, offset_seconds: float, duration_seconds: float) -> tuple[list[float], int, float]:
    with wave.open(wav_path, "rb") as wav_file:
        frame_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if frame_rate <= 0 or channels != 1 or sample_width != 2:
            raise TranscriptionError("LID probe requires a mono 16-bit WAV input.", status_code=400)
        total_duration = total_frames / frame_rate
        if offset_seconds >= total_duration:
            raise TranscriptionError(
                f"offset_seconds ({offset_seconds}) is past the end of the audio ({total_duration:.3f}s).",
                status_code=400,
            )
        start_frame = max(0, int(offset_seconds * frame_rate))
        end_frame = min(total_frames, int((offset_seconds + duration_seconds) * frame_rate))
        wav_file.setpos(start_frame)
        raw = wav_file.readframes(end_frame - start_frame)

    import numpy as np  # type: ignore[import-not-found]

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    actual_duration = round(len(samples) / frame_rate, 3)
    return samples.tolist(), frame_rate, actual_duration


def main() -> int:
    args = _parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _write_error(response_path, f"Failed to load worker request: {exc}")
        return 1

    started = time.monotonic()
    try:
        wav_path = str(payload["wav_path"])
        model_name = str(payload["model"])
        offset_seconds = float(payload.get("offset_seconds", 0.0))
        duration_seconds = float(payload.get("duration_seconds", 30.0))
        top_k = max(1, int(payload.get("top_k", 3)))

        samples, sample_rate, actual_duration = _read_pcm_window(wav_path, offset_seconds, duration_seconds)
        if not samples:
            raise TranscriptionError("Selected audio window is empty.", status_code=400)

        try:
            import numpy as np  # type: ignore[import-not-found]
            import torch  # type: ignore[import-not-found]
            import whisper_timestamped as whisper  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TranscriptionError(
                "Whisper service requires optional dependencies. Install with: pip install -e .[local]"
            ) from exc

        service = WhisperTimestampedService()
        runtime_device = service._resolve_runtime_device()
        model_instance = service._load_model(model_name, runtime_device, whisper)

        audio = np.asarray(samples, dtype=np.float32)
        audio = whisper.pad_or_trim(audio)
        use_fp16 = runtime_device.startswith("cuda")
        mel = whisper.log_mel_spectrogram(audio, n_mels=getattr(model_instance, "dims").n_mels).to(runtime_device)
        if use_fp16:
            mel = mel.half()

        with torch.no_grad():
            _tokens, probs = model_instance.detect_language(mel)
        if isinstance(probs, list):
            probs_dict: dict[str, Any] = probs[0] if probs else {}
        else:
            probs_dict = probs or {}
        if not probs_dict:
            raise TranscriptionError("Whisper produced no language probabilities.")

        ranked = sorted(probs_dict.items(), key=lambda kv: float(kv[1]), reverse=True)
        top = ranked[:top_k]
        candidates = [LidCandidate(language=str(lang), confidence=float(prob)) for lang, prob in top]

        response = LidProbeResponse(
            language=candidates[0].language,
            confidence=candidates[0].confidence,
            candidates=candidates,
            model=model_name,
            sample_offset_seconds=round(float(offset_seconds), 3),
            sample_duration_seconds=actual_duration,
            processing_ms=int((time.monotonic() - started) * 1000),
        )
        response_path.write_text(response.model_dump_json(), encoding="utf-8")
        return 0
    except (KeyError, TypeError, ValueError, TranscriptionError) as exc:
        _write_error(response_path, str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        _write_error(response_path, f"Unexpected LID worker failure: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
