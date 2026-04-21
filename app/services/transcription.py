from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from app.models import FileTranscriptionResponse, WhisperSegment, WhisperWord


class TranscriptionError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


class WhisperTimestampedService:
    def __init__(self) -> None:
        self._models: dict[tuple[str, str], Any] = {}
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self._active_process_lock = asyncio.Lock()

    async def transcribe_file(
        self,
        *,
        media_bytes: bytes,
        source_name: str | None,
        model: str,
        language: str | None,
        prompt: str | None,
        beam_size: int,
        best_of: int,
        temperature: float | None,
        condition_on_previous_text: bool,
        vad: bool,
        vad_mode: str | None,
        task: str,
        no_speech_threshold: float | None,
        detect_disfluencies: bool,
        accurate: bool,
        request_id: str | None = None,
    ) -> FileTranscriptionResponse:
        started = time.monotonic()
        self._resolve_vad_mode(vad=vad, vad_mode=vad_mode)
        wav_path = self._normalize_media_to_wav(media_bytes, source_name=source_name)
        try:
            worker_response = await self._run_worker_process(
                wav_path=wav_path,
                model=model,
                language=language,
                prompt=prompt,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                condition_on_previous_text=condition_on_previous_text,
                vad=vad,
                vad_mode=vad_mode,
                task=task,
                no_speech_threshold=no_speech_threshold,
                detect_disfluencies=detect_disfluencies,
                accurate=accurate,
                request_id=request_id,
            )
        finally:
            Path(wav_path).unlink(missing_ok=True)

        return worker_response.model_copy(
            update={
                "processing_ms": int((time.monotonic() - started) * 1000),
                "model": model,
            }
        )

    def _normalize_media_to_wav(self, media_bytes: bytes, *, source_name: str | None) -> str:
        if self._is_normalized_wav(media_bytes):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                output_path = Path(output_file.name)
                output_file.write(media_bytes)
            return str(output_path)

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

    def _is_normalized_wav(self, media_bytes: bytes) -> bool:
        try:
            with wave.open(io.BytesIO(media_bytes), "rb") as wav_file:
                return (
                    wav_file.getsampwidth() == 2
                    and wav_file.getnchannels() == 1
                    and wav_file.getframerate() == 16_000
                )
        except Exception:
            return False

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
        prompt: str | None,
        beam_size: int,
        best_of: int,
        temperature: float | None,
        condition_on_previous_text: bool,
        vad: bool,
        vad_mode: str | None,
        task: str,
        no_speech_threshold: float | None,
        detect_disfluencies: bool,
        accurate: bool,
    ) -> dict[str, Any]:
        try:
            import whisper_timestamped as whisper  # type: ignore[import-not-found]
        except ImportError as exc:
            raise TranscriptionError(
                "Whisper service requires optional dependencies. Install with: pip install -e .[local]"
            ) from exc

        runtime_device = self._resolve_runtime_device()
        model_instance = self._load_model(model, runtime_device, whisper)
        resolved_vad = self._resolve_vad_mode(vad=vad, vad_mode=vad_mode)
        effective_beam_size, effective_best_of, effective_temperature = self._resolve_accuracy_options(
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            accurate=accurate,
        )
        try:
            result = whisper.transcribe(
                model_instance,
                wav_path,
                task=task,
                language=language,
                initial_prompt=prompt,
                beam_size=effective_beam_size,
                best_of=effective_best_of,
                temperature=effective_temperature,
                condition_on_previous_text=condition_on_previous_text,
                vad=resolved_vad,
                no_speech_threshold=no_speech_threshold,
                detect_disfluencies=detect_disfluencies,
                fp16=runtime_device.startswith("cuda"),
            )
        except TypeError:
            try:
                result = whisper.transcribe(
                    model_instance,
                    wav_path,
                    task=task,
                    language=language,
                    initial_prompt=prompt,
                    beam_size=effective_beam_size,
                    best_of=effective_best_of,
                    temperature=effective_temperature,
                    condition_on_previous_text=condition_on_previous_text,
                    no_speech_threshold=no_speech_threshold,
                    detect_disfluencies=detect_disfluencies,
                    fp16=runtime_device.startswith("cuda"),
                )
            except TypeError:
                result = whisper.transcribe(
                    model_instance,
                    wav_path,
                    task=task,
                    language=language,
                    initial_prompt=prompt,
                    temperature=effective_temperature,
                    fp16=runtime_device.startswith("cuda"),
                )
        if not isinstance(result, dict):
            raise TranscriptionError("Whisper returned an unsupported response type.")
        return result

    async def cancel_request(self, request_id: str) -> None:
        async with self._active_process_lock:
            process = self._active_processes.get(request_id)
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    def _worker_command(self, request_path: Path, response_path: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "app.services.transcription_worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]

    async def _run_worker_process(
        self,
        *,
        wav_path: str,
        model: str,
        language: str | None,
        prompt: str | None,
        beam_size: int,
        best_of: int,
        temperature: float | None,
        condition_on_previous_text: bool,
        vad: bool,
        vad_mode: str | None,
        task: str,
        no_speech_threshold: float | None,
        detect_disfluencies: bool,
        accurate: bool,
        request_id: str | None,
    ) -> FileTranscriptionResponse:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as request_file:
            request_path = Path(request_file.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as response_file:
            response_path = Path(response_file.name)

        request_path.write_text(
            json.dumps(
                {
                    "wav_path": wav_path,
                    "model": model,
                    "language": language,
                    "prompt": prompt,
                    "beam_size": beam_size,
                    "best_of": best_of,
                    "temperature": temperature,
                    "condition_on_previous_text": condition_on_previous_text,
                    "vad": vad,
                    "vad_mode": vad_mode,
                    "task": task,
                    "no_speech_threshold": no_speech_threshold,
                    "detect_disfluencies": detect_disfluencies,
                    "accurate": accurate,
                }
            ),
            encoding="utf-8",
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *self._worker_command(request_path, response_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if request_id:
                async with self._active_process_lock:
                    self._active_processes[request_id] = process
            _stdout, stderr = await process.communicate()
            stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""
            if process.returncode != 0:
                if process.returncode < 0:
                    raise TranscriptionError("Transcription cancelled.", status_code=499)
                raise TranscriptionError(self._read_worker_error(response_path, stderr_text))

            if not response_path.exists():
                raise TranscriptionError("Whisper worker did not produce a response file.")
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise TranscriptionError("Whisper worker returned invalid JSON.") from exc
            return FileTranscriptionResponse.model_validate(payload)
        finally:
            if request_id:
                async with self._active_process_lock:
                    self._active_processes.pop(request_id, None)
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)

    def _resolve_vad_mode(self, *, vad: bool, vad_mode: str | None) -> bool | str:
        normalized = (vad_mode or "").strip().lower()
        if not vad:
            return False
        if not normalized:
            return "auditok"
        if normalized.startswith("auditok"):
            return normalized
        raise TranscriptionError(
            "Only auditok VAD is supported in offline mode. Use vad=true or vad_mode=auditok."
        )

    def _resolve_accuracy_options(
        self,
        *,
        beam_size: int,
        best_of: int,
        temperature: float | None,
        accurate: bool,
    ) -> tuple[int, int, float | tuple[float, ...]]:
        if not accurate:
            return beam_size, best_of, (0.0 if temperature is None else temperature)
        return max(beam_size, 5), max(best_of, 5), (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    def _read_worker_error(self, response_path: Path, stderr: str | None) -> str:
        if response_path.exists():
            try:
                payload = json.loads(response_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                detail = payload.get("error")
                if isinstance(detail, str) and detail.strip():
                    return detail.strip()
        stderr_text = (stderr or "").strip()
        if stderr_text:
            return f"Whisper worker failed: {stderr_text}"
        return "Whisper worker failed."

    def _load_model(self, model: str, runtime_device: str, whisper: Any) -> Any:
        model_ref, download_root = self._resolve_model_source(model)
        cache_key = (model_ref, runtime_device)
        cached = self._models.get(cache_key)
        if cached is not None:
            return cached
        model_instance = whisper.load_model(model_ref, device=runtime_device, download_root=download_root)
        self._models[cache_key] = model_instance
        return model_instance

    def _resolve_model_source(self, model: str) -> tuple[str, str | None]:
        model_name = model.strip()
        if not model_name:
            raise TranscriptionError("Model name must not be empty.")

        model_dir = self._model_dir()
        if self._looks_like_path(model_name):
            explicit_path = Path(model_name).expanduser()
            if explicit_path.exists():
                return str(explicit_path), model_dir

        if model_dir:
            candidate_names = [model_name]
            if not model_name.endswith(".pt"):
                candidate_names.append(f"{model_name}.pt")
            for candidate_name in candidate_names:
                candidate_path = Path(model_dir) / candidate_name
                if candidate_path.exists():
                    return str(candidate_path), model_dir

        return model_name, model_dir

    def _model_dir(self) -> str | None:
        raw = os.environ.get("WHISPER_MODEL_DIR", "").strip()
        if not raw:
            return None
        return str(Path(raw).expanduser())

    def _looks_like_path(self, value: str) -> bool:
        return value.startswith(("/", ".", "~")) or "/" in value

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
                words.append(
                    WhisperWord(
                        text=word_text,
                        start=word_start,
                        end=word_end,
                        confidence=self._coerce_float(word_item.get("confidence")),
                    )
                )
            segments.append(
                WhisperSegment(
                    id=index,
                    start=start,
                    end=end,
                    text=text,
                    temperature=self._coerce_float(item.get("temperature")),
                    avg_logprob=self._coerce_float(item.get("avg_logprob")),
                    compression_ratio=self._coerce_float(item.get("compression_ratio")),
                    no_speech_prob=self._coerce_float(item.get("no_speech_prob")),
                    confidence=self._coerce_float(item.get("confidence")),
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
