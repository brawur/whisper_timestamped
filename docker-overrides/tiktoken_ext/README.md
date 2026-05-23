Place a patched `openai_public.py` into this directory before building the image:

- source path expected by the Docker build:
  `whisper_timestamped_worker/docker-overrides/tiktoken_ext/openai_public.py`
- install target inside the image:
  `/usr/local/lib/python3.11/site-packages/tiktoken_ext/openai_public.py`

If the file is present, the Docker build copies it into the image after
`pip install -e .[local]`.
