from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from app.models import FileTranscriptionResponse, WhisperSegment, WhisperWord


class TranscriptionError(RuntimeError):
    pass


class WhisperTimestampedService:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}

    def transcribe_file(
        self,
        *,
        media_bytes: bytes,
        source_name: str | None,
        model: str,
        language: str | None,
        prompt: str,
        beam_size: int,
        best_of: int,
        temperature: float,
        condition_on_previous_text: bool,
        vad: bool,
    ) -> FileTranscriptionResponse:
        started = time.monotonic()
        wav_path = self._normalize_media_to_wav(media_bytes, source_name=source_name)
        try:
            duration_seconds = self._duration_seconds(wav_path)
            result = self._transcribe_wav(
                wav_path=wav_path,
                model=model,
                language=language,
                prompt=prompt,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                vad=vad,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        segments = self._extract_segments(result)
        transcript = str(result.get("text") or "").strip()
        detected_language = result.get("language")
        return FileTranscriptionResponse(
            text=transcript,
            language=language or detected_language,
            duration_seconds=duration_seconds,
            model=model,
            processing_ms=int((time.monotonic() - started) * 1000),
            segments=segments,
        )

    def _normalize_media_to_wav(self, media_bytes: bytes, *, source_name: str | None) -> str:
        suffix = Path(source_name or "input.bin").suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as input_file:
            input_path = Path(input_file.name)
            input_file.write(media_bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = Path(output_file.name)

        try:
            completed = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise TranscriptionError("ffmpeg is not installed.") from exc
        finally:
            input_path.unlink(missing_ok=True)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            output_path.unlink(missing_ok=True)
            raise TranscriptionError(f"ffmpeg failed to decode media input: {stderr or 'unknown ffmpeg error'}")

        return str(output_path)

    def _duration_seconds(self, wav_path: str) -> float:
        with wave.open(wav_path, "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
        if frame_rate <= 0:
            raise TranscriptionError("Invalid WAV sample rate.")
        return round(frame_count / frame_rate, 3)

    def _transcribe_wav(
        self,
        *,
        wav_path: str,
        model: str,
        language: str | None,
        prompt: str,
        beam_size: int,
        best_of: int,
        temperature: float,
        condition_on_previous_text: bool,
        vad: bool,
    ) -> dict[str, Any]:
        try:
            import whisper_timestamped as whisper  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TranscriptionError(
                "Whisper service requires optional dependencies. Install with: pip install -e .[local]"
            ) from exc

        runtime_device = self._resolve_runtime_device()
        model_instance = self._load_model(model, runtime_device, whisper)
        try:
            result = whisper.transcribe(
                model_instance,
                wav_path,
                language=language,
                initial_prompt=prompt or None,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                vad=vad,
                fp16=runtime_device.startswith("cuda"),
            )
        except TypeError:
            try:
                result = whisper.transcribe(
                    model_instance,
                    wav_path,
                    language=language,
                    initial_prompt=prompt or None,
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=temperature,
                    condition_on_previous_text=condition_on_previous_text,
                    fp16=runtime_device.startswith("cuda"),
                )
            except TypeError:
                result = whisper.transcribe(
                    model_instance,
                    wav_path,
                    language=language,
                    fp16=runtime_device.startswith("cuda"),
                )
        if not isinstance(result, dict):
            raise TranscriptionError("Whisper returned an unsupported response type.")
        return result

    def _load_model(self, model: str, runtime_device: str, whisper: Any) -> Any:
        cache_key = (model, runtime_device)
        cached = self._models.get(cache_key)
        if cached is not None:
            return cached
        model_instance = whisper.load_model(model, device=runtime_device)
        self._models[cache_key] = model_instance
        return model_instance

    def _resolve_runtime_device(self) -> str:
        requested = os.environ.get("WHISPER_DEVICE", "auto").strip().lower() or "auto"
        cuda_available = self._cuda_available()
        if requested == "auto":
            return "cuda" if cuda_available else "cpu"
        if requested.startswith("cuda"):
            return requested if cuda_available else "cpu"
        return requested

    def _cuda_available(self) -> bool:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def _extract_segments(self, result: dict[str, Any]) -> list[WhisperSegment]:
        segments: list[WhisperSegment] = []
        for index, item in enumerate(result.get("segments", [])):
            if not isinstance(item, dict):
                continue
            start = self._coerce_float(item.get("start"))
            end = self._coerce_float(item.get("end"))
            text = str(item.get("text") or "").strip()
            if start is None or end is None:
                continue
            words: list[WhisperWord] = []
            for word_item in item.get("words", []):
                if not isinstance(word_item, dict):
                    continue
                word_start = self._coerce_float(word_item.get("start"))
                word_end = self._coerce_float(word_item.get("end"))
                word_text = str(word_item.get("text") or word_item.get("word") or "").strip()
                if word_start is None or word_end is None or not word_text:
                    continue
                words.append(WhisperWord(text=word_text, start=word_start, end=word_end))
            segments.append(
                WhisperSegment(
                    id=index,
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                )
            )
        return segments

    def _coerce_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


_service: WhisperTimestampedService | None = None


def get_service() -> WhisperTimestampedService:
    global _service
    if _service is None:
        _service = WhisperTimestampedService()
    return _service
