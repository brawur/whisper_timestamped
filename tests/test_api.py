import asyncio

import httpx

from app.main import app
import app.services.transcription as transcription


class FakeService:
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
    ):
        assert media_bytes == b"RIFFfake"
        assert source_name == "sample.wav"
        assert model == "small"
        assert language == "de"
        assert beam_size == 3
        assert best_of == 5
        assert temperature == 0.0
        assert condition_on_previous_text is True
        assert vad is False
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
                    "words": [
                        {"text": "hallo", "start": 0.0, "end": 0.6},
                        {"text": "welt", "start": 0.7, "end": 1.25},
                    ],
                }
            ],
        }


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
        transcription._service = FakeService()  # type: ignore[assignment]
        async with await _get_client() as client:
            response = await client.post(
                "/transcribe/file?model=small&language=de&beam_size=3",
                files={"file": ("sample.wav", b"RIFFfake", "audio/wav")},
            )
            assert response.status_code == 200
            body = response.json()
            assert body["text"] == "hallo welt"
            assert body["language"] == "de"
            assert body["duration_seconds"] == 1.25
            assert body["model"] == "small"
            assert body["processing_ms"] == 123
            assert body["segments"][0]["words"][0]["text"] == "hallo"
        transcription._service = None

    asyncio.run(run())
