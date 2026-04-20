FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL_DIR=/models/whisper

WORKDIR /app

RUN mkdir -p /models/whisper

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY app /app/app
COPY docker-overrides /app/docker-overrides

RUN pip install --upgrade pip setuptools wheel
RUN pip install --retries 10 --default-timeout 300 ".[local]"
RUN python - <<'PY'
from pathlib import Path
import shutil

override_path = Path("/app/docker-overrides/tiktoken_ext/openai_public.py")
if not override_path.exists():
    raise SystemExit(0)

target_path = Path("/usr/local/lib/python3.11/site-packages/tiktoken_ext/openai_public.py")
target_path.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(override_path, target_path)
PY

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
