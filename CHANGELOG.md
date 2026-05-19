# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

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
