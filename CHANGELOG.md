# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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
