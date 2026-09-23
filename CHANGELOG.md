# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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
