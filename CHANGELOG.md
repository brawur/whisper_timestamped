# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Changed
- The image installs its runtime packages from `requirements.lock` (exact
  versions with hashes, `--no-deps --require-hashes`) instead of resolving
  `.[local]` on every build, so a rebuild reproduces torch 2.12.0, numpy,
  whisper-timestamped and auditok exactly.

### Fixed
- torch is pinned to 2.12.0 (was `>=2.3.0,<3.0.0`). Images built since early
  October pulled torch 2.14.1 with cuDNN 9.24, under which large-v3 fell into
  repetition loops at borderline passages (reference audiobook: VAD output at
  05:08, 06:54 and 10:24, base output at 09:50). Same code, file and GPU with
  torch 2.12.0 / cuDNN 9.20 transcribe these passages correctly again, as the
  2.0.7 worker did.
- Transcription no longer drops job parameters silently. A `TypeError` from
  `whisper_timestamped.transcribe` used to trigger a second call without `vad`
  and a third one without further decoding parameters (beam size, best of,
  condition on previous text, no-speech threshold, disfluency detection), so a
  job with `vad=true`/`vad_mode=auditok` could finish successfully without VAD
  (the library defaults to `vad=False`). Now the call is made once with all
  parameters; a `TypeError` fails the job with its cause. Before inference the
  worker checks that the installed library accepts every parameter and fails
  with the missing names otherwise (whisper-timestamped 1.15.9 in the image
  accepts all, auditok 0.5.2 is installed). The worker logs requested and passed
  VAD settings and the other decoding options, without audio, prompt text or
  transcript; it does not claim that VAD found segments.

### Added
- Several GPUs (MCC-305 E3): `X-MCC-GPU` (GPU UUID) from the gateway limits the
  transcription and language-probe job processes to that card
  (`CUDA_VISIBLE_DEVICES`).
- CUDA MPS (MCC-305): with `MCC_MPS_PIPE_DIRECTORY` set, the transcription and language-probe job processes become clients
  of the host's MPS daemon (`CUDA_MPS_PIPE_DIRECTORY`); without a running daemon
  they compute as before.

## [2.1.3] - 2026-09-25

### Changed
- A model missing in `WHISPER_MODEL_DIR` is an error; the worker never
  downloads models.
- Runtime runs offline: `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`,
  `HF_DATASETS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`, `DO_NOT_TRACK`.

## [2.1.2] - 2026-09-24

### Changed
- No changes; released with the 2.1.2 stack.

## [2.1.1] - 2026-09-23

### Added
- Report the interface contract in `/metadata`: `worker_api` (gateway interface
  version, currently 1) and `build_version` (installed package version).

### Fixed
- Include gcc and libc6-dev for Triton's CUDA-helper compilation during automatic
  language detection; check the compiler and Python headers at image build time.

### Changed
- Generate version-specific source references for Debian and Python packages,
  with explicit unresolved entries and release-source documentation.
- Extend the full license report with Debian package notices and shared full
  license texts; include exact source-package versions in a separate manifest.
- Include worker license documentation in the image.
- Reduce THIRD-PARTY-LICENSES.md to components with obligations beyond notices;
  the generated report is the authoritative list. Release steps moved to the README.

## [2.1.0] - 2026-09-10

### Changed

- The license report is generated with `pip-licenses --with-license-file --with-notice-file`, so the license and notice texts of the dependencies ship with the image. MIT requires the copyright and license notice "in all copies or substantial portions", Apache-2.0 section 4(a) requires the license copy explicitly and 4(d) the NOTICE file — a shipped image is such a copy.
## [2.0.7] - 2026-06-07

### Added

- cancellable worker execution for request-scoped transcription subprocesses
- `POST /transcribe/cancel/{request_id}` to terminate a running transcription subprocess
- API tests verifying that cancellation really stops the spawned worker process
- restored Whisper metadata fields in the JSON response:
  - segment `temperature`
  - segment `avg_logprob`
  - segment `compression_ratio`
  - segment `no_speech_prob`
  - segment `confidence`
  - word `confidence`

### Changed

- `POST /transcribe/file` now accepts `X-Transcription-Request-ID` and maps cancelled subprocesses to HTTP `499`
- `POST /transcribe/file` now also accepts `shared_path` for pre-normalized WAV files from the local shared jobs volume
