FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL_DIR=/models/whisper \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache

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

RUN pip install pip-licenses \
    && pip-licenses --format=json --with-urls --ignore-packages pip-licenses \
       --with-license-file --with-notice-file --no-license-path \
       --output-file=/app/THIRD_PARTY_LICENSES.full.json

# If somebody drops a custom tiktoken_ext/openai_public.py into docker-overrides
# (e.g. for downstream patches), copy it over the pip-installed file. No-op when
# the directory is empty - that's the supported default for offline operation,
# which relies on the pre-populated TIKTOKEN_CACHE_DIR below instead of patching.
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

# Pre-populate the tiktoken cache so the worker can run fully offline. tiktoken
# uses sha1(url) as the cache key inside TIKTOKEN_CACHE_DIR, so we mirror that
# layout. The list mirrors every URL that tiktoken_ext/openai_public.py resolves
# (gpt2 via data_gym + the four .tiktoken-format encodings; p50k_edit shares
# the p50k_base file). Each download is verified against the hash that
# openai_public.py also enforces at runtime.
RUN python - <<'PY'
import hashlib, os, sys, urllib.request

cache_dir = os.environ["TIKTOKEN_CACHE_DIR"]
os.makedirs(cache_dir, exist_ok=True)

ENCODINGS = [
    # gpt2 data_gym files
    ("https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe",
     "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"),
    ("https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json",
     "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"),
    # tiktoken-format encodings
    ("https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
     "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930"),
    ("https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
     "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"),
    ("https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
     "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"),
    ("https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
     "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"),
]

for url, expected in ENCODINGS:
    key = hashlib.sha1(url.encode()).hexdigest()
    target = os.path.join(cache_dir, key)
    print(f"fetching {url} -> {target}")
    with urllib.request.urlopen(url, timeout=180) as r:
        data = r.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        sys.exit(f"hash mismatch for {url}: expected {expected}, got {actual}")
    with open(target, "wb") as f:
        f.write(data)

print("tiktoken cache populated:", sorted(os.listdir(cache_dir)))
PY

# Fail-fast offline check: load every encoding the override exposes with the
# loopback DNS poisoned, so the build aborts immediately if the cache misses
# anything instead of surfacing as a runtime failure in production.
RUN python - <<'PY'
import os, socket, tiktoken

# Block any outbound DNS/HTTP so a cache miss surfaces as a clear error here
# rather than silently being papered over by network access during build.
def _no_network(*_args, **_kwargs):
    raise OSError("network access disabled during offline tiktoken self-test")

socket.getaddrinfo = _no_network  # type: ignore[assignment]
socket.create_connection = _no_network  # type: ignore[assignment]

for name in ("gpt2", "r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base"):
    enc = tiktoken.get_encoding(name)
    assert enc.encode("offline self-test"), name
    print(f"OK offline: {name}")
PY

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
