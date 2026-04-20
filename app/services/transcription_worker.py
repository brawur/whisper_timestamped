from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.models import FileTranscriptionResponse
from app.services.transcription import TranscriptionError, WhisperTimestampedService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--response", required=True)
    return parser.parse_args()


def _write_error(response_path: Path, message: str) -> None:
    response_path.write_text(json.dumps({"error": message}), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    request_path = Path(args.request)
    response_path = Path(args.response)

    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _write_error(response_path, f"Failed to load worker request: {exc}")
        return 1

    service = WhisperTimestampedService()
    started = time.monotonic()
    try:
        wav_path = str(payload["wav_path"])
        result = service._transcribe_wav(
            wav_path=wav_path,
            model=str(payload["model"]),
            language=payload.get("language"),
            prompt=str(payload.get("prompt") or ""),
            beam_size=int(payload.get("beam_size", 5)),
            best_of=int(payload.get("best_of", 5)),
            temperature=float(payload.get("temperature", 0.0)),
            condition_on_previous_text=bool(payload.get("condition_on_previous_text", True)),
            vad=bool(payload.get("vad", False)),
        )
        transcript = str(result.get("text") or "").strip()
        detected_language = result.get("language")
        response = FileTranscriptionResponse(
            text=transcript,
            language=payload.get("language") or detected_language,
            duration_seconds=service._duration_seconds(wav_path),
            model=str(payload["model"]),
            processing_ms=int((time.monotonic() - started) * 1000),
            segments=service._extract_segments(result),
        )
        response_path.write_text(response.model_dump_json(), encoding="utf-8")
        return 0
    except (KeyError, TypeError, ValueError, TranscriptionError) as exc:
        _write_error(response_path, str(exc))
        return 1
    except Exception as exc:  # pragma: no cover
        _write_error(response_path, f"Unexpected worker failure: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
