from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile

from app.models import FileTranscriptionResponse, HealthResponse, MetadataResponse
from app.services.transcription import TranscriptionError, WhisperTimestampedService, get_service

router = APIRouter()


def _pick_text(request: Request, form: dict, key: str) -> str | None:
    value = form.get(key)
    if value is None:
        value = request.query_params.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_bool(request: Request, form: dict, key: str) -> bool | None:
    value = form.get(key)
    if value is None:
        value = request.query_params.get(key)
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _pick_int(request: Request, form: dict, key: str) -> int | None:
    value = form.get(key)
    if value is None:
        value = request.query_params.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid integer value for '{key}'.") from None


def _pick_float(request: Request, form: dict, key: str) -> float | None:
    value = form.get(key)
    if value is None:
        value = request.query_params.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(str(value))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid float value for '{key}'.") from None


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
            "initial_prompt",
            "beam_size",
            "best_of",
            "temperature",
            "condition_on_previous_text",
            "vad",
            "vad_mode",
            "task",
            "no_speech_threshold",
            "detect_disfluencies",
            "accurate",
        ],
    )


@router.post("/transcribe/file", response_model=FileTranscriptionResponse)
async def transcribe_file(
    request: Request,
    file: UploadFile = File(...),
    service: WhisperTimestampedService = Depends(get_service),
) -> FileTranscriptionResponse:
    form = dict(await request.form())
    model = _pick_text(request, form, "model") or "small"
    language = _pick_text(request, form, "language")
    prompt = _pick_text(request, form, "initial_prompt") or _pick_text(request, form, "prompt")
    beam_size = _pick_int(request, form, "beam_size") or 5
    best_of = _pick_int(request, form, "best_of") or 5
    temperature = _pick_float(request, form, "temperature")
    condition_on_previous_text = _pick_bool(request, form, "condition_on_previous_text")
    vad = _pick_bool(request, form, "vad") or False
    vad_mode = _pick_text(request, form, "vad_mode")
    task = _pick_text(request, form, "task") or "transcribe"
    no_speech_threshold = _pick_float(request, form, "no_speech_threshold")
    detect_disfluencies = _pick_bool(request, form, "detect_disfluencies")
    accurate = _pick_bool(request, form, "accurate")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    request_id = request.headers.get("X-Transcription-Request-ID")
    try:
        return await service.transcribe_file(
            media_bytes=payload,
            source_name=file.filename,
            model=model,
            language=language,
            prompt=prompt,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            condition_on_previous_text=True if condition_on_previous_text is None else condition_on_previous_text,
            vad=vad,
            vad_mode=vad_mode,
            task=task if task in {"transcribe", "translate"} else "transcribe",
            no_speech_threshold=no_speech_threshold,
            detect_disfluencies=True if detect_disfluencies is None else detect_disfluencies,
            accurate=True if accurate is None else accurate,
            request_id=request_id,
        )
    except TranscriptionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post("/transcribe/cancel/{request_id}", status_code=204)
async def cancel_transcription(
    request_id: str,
    service: WhisperTimestampedService = Depends(get_service),
) -> Response:
    await service.cancel_request(request_id)
    return Response(status_code=204)
