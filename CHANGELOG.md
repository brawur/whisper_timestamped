# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog.

## [Unreleased]

### Added

- cancellable worker execution for request-scoped transcription subprocesses
- `POST /transcribe/cancel/{request_id}` to terminate a running transcription subprocess
- API tests verifying that cancellation really stops the spawned worker process

### Changed

- `POST /transcribe/file` now accepts `X-Transcription-Request-ID` and maps cancelled subprocesses to HTTP `499`
