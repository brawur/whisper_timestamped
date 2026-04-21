import asyncio
import os
import sys
import tempfile
from pathlib import Path

import httpx

from app.main import app
import app.services.transcription as transcription


class FakeService:
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
    ):
        assert media_bytes == b"RIFFfake"
        assert source_name == "sample.wav"
        assert model == "small"
        assert language == "de"
        assert beam_size == 3
        assert best_of == 5
        assert prompt is None
        assert temperature is None
        assert condition_on_previous_text is True
        assert vad is False
        assert vad_mode is None
        assert task == "transcribe"
        assert no_speech_threshold is None
        assert detect_disfluencies is True
        assert accurate is True
        return {
            "text": "hallo welt",
            "language": "de",
            "duration_seconds": 1.25,
            "model": model,
            "processing_ms": 123,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.25,
                    "text": "hallo welt",
                    "temperature": 0.3,
                    "avg_logprob": -0.085798696,
                    "compression_ratio": 1.4421768,
                    "no_speech_prob": 1.1901208e-11,
                    "confidence": 0.979,
                    "words": [
                        {"text": "hallo", "start": 0.0, "end": 0.6, "confidence": 0.991},
                        {"text": "welt", "start": 0.7, "end": 1.25, "confidence": 0.976},
                    ],
                }
            ],
        }


class FakeWorkerService(transcription.WhisperTimestampedService):
    def _normalize_media_to_wav(self, media_bytes: bytes, *, source_name: str | None) -> str:
        assert media_bytes == b"RIFFfake"
        assert source_name == "sample.wav"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            path = Path(output_file.name)
        path.write_bytes(b"fake-wav")
        return str(path)

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
        request_id: str | None = None,
    ) -> transcription.FileTranscriptionResponse:
        assert Path(wav_path).exists()
        assert model == "small"
        assert language == "de"
        assert beam_size == 3
        assert best_of == 5
        assert prompt is None
        assert temperature is None
        assert condition_on_previous_text is True
        assert vad is False
        assert vad_mode is None
        assert task == "transcribe"
        assert no_speech_threshold is None
        assert detect_disfluencies is True
        assert accurate is True
        return transcription.FileTranscriptionResponse(
            text="hallo welt",
            language="de",
            duration_seconds=1.25,
            model=model,
            processing_ms=12,
            segments=[
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.25,
                    "text": "hallo welt",
                    "temperature": 0.3,
                    "avg_logprob": -0.085798696,
                    "compression_ratio": 1.4421768,
                    "no_speech_prob": 1.1901208e-11,
                    "confidence": 0.979,
                    "words": [
                        {"text": "hallo", "start": 0.0, "end": 0.6, "confidence": 0.991},
                        {"text": "welt", "start": 0.7, "end": 1.25, "confidence": 0.976},
                    ],
                }
            ],
        )


async def _get_client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def test_health() -> None:
    async def run() -> None:
        async with await _get_client() as client:
            response = await client.get("/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "ok",
                "service": "whisper_timestamped",
            }

    asyncio.run(run())


def test_metadata() -> None:
    async def run() -> None:
        async with await _get_client() as client:
            response = await client.get("/metadata")
            assert response.status_code == 200
            body = response.json()
            assert body["service"] == "whisper_timestamped"
            assert body["supports_word_timestamps"] is True
            assert body["supports_speaker_diarization"] is False
            assert body["mode"] == "local_model"
            assert "beam_size" in body["supported_parameters"]

    asyncio.run(run())


def test_file_transcription_rejects_empty_upload() -> None:
    async def run() -> None:
        async with await _get_client() as client:
            response = await client.post(
                "/transcribe/file",
                files={"file": ("empty.wav", b"", "audio/wav")},
            )
            assert response.status_code == 400
            assert response.json()["detail"] == "Uploaded file is empty."

    asyncio.run(run())


def test_file_transcription_response() -> None:
    async def run() -> None:
        transcription._service = FakeWorkerService()
        async with await _get_client() as client:
            response = await client.post(
                "/transcribe/file",
                files={
                    "file": ("sample.wav", b"RIFFfake", "audio/wav"),
                    "model": (None, "small"),
                    "language": (None, "de"),
                    "beam_size": (None, "3"),
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["text"] == "hallo welt"
            assert body["language"] == "de"
            assert body["duration_seconds"] == 1.25
            assert body["model"] == "small"
            assert body["processing_ms"] >= 0
            assert body["segments"][0]["temperature"] == 0.3
            assert body["segments"][0]["avg_logprob"] == -0.085798696
            assert body["segments"][0]["compression_ratio"] == 1.4421768
            assert body["segments"][0]["no_speech_prob"] == 1.1901208e-11
            assert body["segments"][0]["confidence"] == 0.979
            assert body["segments"][0]["words"][0]["text"] == "hallo"
            assert body["segments"][0]["words"][0]["confidence"] == 0.991
        transcription._service = None

    asyncio.run(run())


def test_file_transcription_uses_auditok_when_vad_enabled() -> None:
    class FakeVadWorkerService(transcription.WhisperTimestampedService):
        def _normalize_media_to_wav(self, media_bytes: bytes, *, source_name: str | None) -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                path = Path(output_file.name)
            path.write_bytes(b"fake-wav")
            return str(path)

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
        request_id: str | None = None,
    ) -> transcription.FileTranscriptionResponse:
            assert vad is True
            assert vad_mode is None
            assert task == "transcribe"
            assert prompt is None
            assert temperature is None
            assert no_speech_threshold is None
            assert detect_disfluencies is True
            assert accurate is True
            assert self._resolve_vad_mode(vad=vad, vad_mode=vad_mode) == "auditok"
            return transcription.FileTranscriptionResponse(
                text="hallo welt",
                language="de",
                duration_seconds=1.25,
                model=model,
                processing_ms=12,
                segments=[],
            )

    async def run() -> None:
        transcription._service = FakeVadWorkerService()
        async with await _get_client() as client:
            response = await client.post(
                "/transcribe/file",
                files={
                    "file": ("sample.wav", b"RIFFfake", "audio/wav"),
                    "model": (None, "small"),
                    "language": (None, "de"),
                    "vad": (None, "true"),
                },
            )
            assert response.status_code == 200
        transcription._service = None

    asyncio.run(run())


def test_cancel_endpoint_stops_running_worker_process() -> None:
    class CancelTestService(transcription.WhisperTimestampedService):
        def _normalize_media_to_wav(self, media_bytes: bytes, *, source_name: str | None) -> str:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
                path = Path(output_file.name)
            path.write_bytes(b"fake-wav")
            return str(path)

        def _worker_command(self, request_path: Path, response_path: Path) -> list[str]:
            return [sys.executable, "-c", "import time; time.sleep(60)"]

    async def run() -> None:
        service = CancelTestService()
        transcription._service = service
        async with await _get_client() as client:
            post_task = asyncio.create_task(
                client.post(
                    "/transcribe/file",
                    headers={"X-Transcription-Request-ID": "req-1"},
                    files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
                )
            )
            await asyncio.sleep(0.2)
            cancel_response = await client.post("/transcribe/cancel/req-1")
            assert cancel_response.status_code == 204
            response = await post_task
            assert response.status_code == 499
            assert response.json()["detail"] == "Transcription cancelled."
        transcription._service = None

    asyncio.run(run())


def test_file_transcription_rejects_non_auditok_vad_mode() -> None:
    async def run() -> None:
        transcription._service = transcription.WhisperTimestampedService()
        async with await _get_client() as client:
            response = await client.post(
                "/transcribe/file",
                files={
                    "file": ("sample.wav", b"RIFFfake", "audio/wav"),
                    "model": (None, "small"),
                    "language": (None, "de"),
                    "vad": (None, "true"),
                    "vad_mode": (None, "silero"),
                },
            )
            assert response.status_code == 502
            assert "Only auditok VAD is supported in offline mode" in response.json()["detail"]
        transcription._service = None

    asyncio.run(run())


def test_resolve_model_source_prefers_mounted_model_dir_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "small.pt"
        model_path.write_bytes(b"fake")
        previous = os.environ.get("WHISPER_MODEL_DIR")
        os.environ["WHISPER_MODEL_DIR"] = temp_dir
        try:
            service = transcription.WhisperTimestampedService()
            resolved_model, download_root = service._resolve_model_source("small")
            assert resolved_model == str(model_path)
            assert download_root == temp_dir
        finally:
            if previous is None:
                os.environ.pop("WHISPER_MODEL_DIR", None)
            else:
                os.environ["WHISPER_MODEL_DIR"] = previous


def test_resolve_model_source_uses_model_dir_as_download_root() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        previous = os.environ.get("WHISPER_MODEL_DIR")
        os.environ["WHISPER_MODEL_DIR"] = temp_dir
        try:
            service = transcription.WhisperTimestampedService()
            resolved_model, download_root = service._resolve_model_source("large-v1")
            assert resolved_model == "large-v1"
            assert download_root == temp_dir
        finally:
            if previous is None:
                os.environ.pop("WHISPER_MODEL_DIR", None)
            else:
                os.environ["WHISPER_MODEL_DIR"] = previous
