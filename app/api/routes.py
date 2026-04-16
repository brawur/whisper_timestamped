from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.models import FileTranscriptionResponse, HealthResponse, MetadataResponse
from app.services.transcription import TranscriptionError, WhisperTimestampedService, get_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", service="whisper_timestamped")


@router.get("/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        service="whisper_timestamped",
        supports_word_timestamps=True,
        supports_speaker_diarization=False,
        mode="local_model",
        supported_parameters=[
            "model",
            "language",
            "prompt",
            "beam_size",
            "best_of",
            "temperature",
            "condition_on_previous_text",
            "vad",
        ],
    )


@router.post("/transcribe/file", response_model=FileTranscriptionResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    model: str = "small",
    language: str | None = None,
    prompt: str = "",
    beam_size: int = 5,
    best_of: int = 5,
    temperature: float = 0.0,
    condition_on_previous_text: bool = True,
    vad: bool = False,
    service: WhisperTimestampedService = Depends(get_service),
) -> FileTranscriptionResponse:
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return service.transcribe_file(
            media_bytes=payload,
            source_name=file.filename,
            model=model,
            language=language,
            prompt=prompt,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            vad=vad,
        )
    except TranscriptionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
