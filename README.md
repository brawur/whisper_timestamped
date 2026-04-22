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
- `GET /metrics/runtime`
- `POST /transcribe/file`
- `POST /transcribe/cancel/{request_id}`

`POST /transcribe/file` accepts `multipart/form-data` fields:

- `file`
- `model`
- `language`
- `initial_prompt`
- `beam_size`
- `best_of`
- `temperature`
- `condition_on_previous_text`
- `vad`
- `vad_mode`
- `task`
- `no_speech_threshold`
- `detect_disfluencies`
- `accurate`

Body-first API note:

- Whisper options are expected in the multipart body, not in the query string
- existing query parameters are still accepted as a temporary compatibility fallback
- `initial_prompt`, `temperature`, and `no_speech_threshold` are optional and are only forwarded when explicitly provided

Cancellation note:

- callers can attach `X-Transcription-Request-ID` to `POST /transcribe/file`
- the worker registers the spawned subprocess under that request ID
- `POST /transcribe/cancel/{request_id}` terminates the active subprocess and returns `204`
- cancelled requests return `499` with `Transcription cancelled.`

Whisper-specific output note:

- the service preserves Whisper-only metadata when it is available in the upstream result
- per segment this includes:
  - `temperature`
  - `avg_logprob`
  - `compression_ratio`
  - `no_speech_prob`
  - `confidence`
- per word this includes:
  - `confidence`

The service is meant to be called by the Parakeet gateway over HTTP.

Runtime metrics note:

- `GET /metrics/runtime` exposes CPU/RAM/GPU metrics from the worker runtime
- the gateway uses this internal endpoint for MCC multi-job metrics while Whisper ASR steps are active

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev,local]
uvicorn app.main:app --reload
```

Offline VAD note:

- `whisper-timestamped` supports multiple VAD backends such as Silero and Auditok
- for this service, offline-safe VAD is limited to `auditok`
- `auditok` is therefore installed as part of `.[local]`
- `vad=true` in this service maps internally to `vad_mode=auditok`
- `silero` is intentionally not used here because it may trigger online model/version resolution

Model resolution:

- if `WHISPER_MODEL_DIR` is set, the service first looks for mounted files like `small.pt`, `medium.pt`, or `large-v1.pt` in that directory
- if a matching file exists, the service loads it directly and does not download the model again
- if no matching file exists, Whisper still uses `WHISPER_MODEL_DIR` as its download/cache root, so downloaded models land in the shared mounted directory

Example:

```bash
WHISPER_MODEL_DIR="$HOME/.cache/whisper" \
uvicorn app.main:app --reload
```

VAD examples:

```bash
curl -F "file=@sample.wav" \
  -F "model=small" \
  -F "vad=true" \
  http://127.0.0.1:8001/transcribe/file
```

Explicit offline-safe VAD mode:

```bash
curl -F "file=@sample.wav" \
  -F "model=small" \
  -F "vad=true" \
  -F "vad_mode=auditok" \
  http://127.0.0.1:8001/transcribe/file
```

## Docker

```bash
docker build -t whisper-timestamped-service .
docker run --rm \
  -p 8001:8000 \
  -e WHISPER_MODEL_DIR=/models/whisper \
  -v "$HOME/.cache/whisper:/models/whisper" \
  whisper-timestamped-service
```

With this mount, existing host models like `~/.cache/whisper/large-v1.pt` or `~/.cache/whisper/small.pt` are reused by the container.

### Build-Time `openai_public.py` Override

If you have a patched `tiktoken_ext/openai_public.py` for fully offline execution, you can bake it directly into the image build:

1. copy your patched file to:
   `whisper_timestamped/docker-overrides/tiktoken_ext/openai_public.py`
2. build the image again:

```bash
docker build -t whisper-timestamped-service .
```

During the build, that file is copied into:

- `/usr/local/lib/python3.11/site-packages/tiktoken_ext/openai_public.py`

This is usually cleaner than a runtime file mount because the image already contains the offline patch.

## Docker Compose

Start from the example environment file:

```bash
cp .env.example .env
```

Adjust these values as needed:

- `WHISPER_CACHE_DIR` for the host Whisper model cache directory
- `WHISPER_MODEL_DIR` for the container-side mount target
- `WHISPER_OPENAI_PUBLIC_PATCH_FILE` for the optional runtime `openai_public.py` patch source
- `WHISPER_DEVICE` for CPU/GPU selection

```bash
docker compose up --build
```

The provided Compose file mounts `${WHISPER_CACHE_DIR:-~/.cache/whisper}` into the container at `${WHISPER_MODEL_DIR:-/models/whisper}` so the host Whisper cache is reused directly.

The Compose setup also requests one NVIDIA GPU for the container, mirroring the Parakeet setup. With `WHISPER_DEVICE=auto`, the service prefers CUDA when Docker GPU support is available and falls back to CPU otherwise.

Default Compose behavior:

- reuses the host Whisper model cache via volume mount
- uses the build-time `openai_public.py` override if you placed it under `docker-overrides/tiktoken_ext/openai_public.py` before `docker compose up --build`

If you want to keep the image generic and inject the patched `openai_public.py` only at runtime, use the additional Compose overlay:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.runtime-patch.yml \
  up --build
```

That overlay mounts:

- host source default:
  `/home/media4cast/.local/share/pipx/venvs/whisper-timestamped/lib/python3.12/site-packages/tiktoken_ext/openai_public.py`
- container target:
  `/usr/local/lib/python3.11/site-packages/tiktoken_ext/openai_public.py`

Override the host source path with `WHISPER_OPENAI_PUBLIC_PATCH_FILE` if needed.

## Combined Root Compose

The repository root also provides a shared Compose setup for both services:

```bash
cd ..
cp .env.example .env
docker compose up --build
```

Optional runtime patch overlay from the repository root:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.runtime-patch.yml \
  up --build
```
