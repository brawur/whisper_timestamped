# Whisper Timestamped Service

Separate transcription service for Whisper-based batch transcription.

This repository is intentionally separate from the Parakeet gateway so that:

- dependencies remain isolated
- container images remain isolated
- licensing remains easier to reason about

## API

Implemented:

- `GET /health`
- `GET /metadata`
- `POST /transcribe/file`

`POST /transcribe/file` accepts:

- `file`
- `model`
- `language`
- `prompt`
- `beam_size`
- `best_of`
- `temperature`
- `condition_on_previous_text`
- `vad`

The service is meant to be called by the Parakeet gateway over HTTP.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,local]
uvicorn app.main:app --reload
```

## Docker

```bash
docker build -t whisper-timestamped-service .
docker run --rm -p 8001:8000 whisper-timestamped-service
```
