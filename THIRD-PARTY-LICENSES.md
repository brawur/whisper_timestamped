# Third-Party Licenses

This repository is licensed under the **GNU Affero General Public License
v3.0 or later (AGPL-3.0-or-later)**. See [LICENSE.md](LICENSE.md) for the
full license text and the README for §13 (network-use) compliance notes.

## Authoritative component list

This document does not list every dependency. Each image build generates the
authoritative inventory in `/app`:

- `THIRD_PARTY_LICENSES.full.json`: every installed Python and Debian package
  with version, license, license text and copyright notices. Shared texts from
  `/usr/share/common-licenses` are included once as separate entries; original
  Debian notices remain in `/usr/share/doc/<package>/copyright`.
- `DEBIAN_SOURCE_PACKAGES.json`: exact Debian source-package names and versions.
- `THIRD_PARTY_SOURCES.json`: source references for these packages, including
  explicitly unresolved entries.

Permissive components (MIT, BSD, Apache-2.0 and similar) only require their
license texts and notices, which the generated report contains. The components
below carry obligations beyond that.

## Components with further obligations

### whisper-timestamped (AGPL-3.0)

- Source: https://github.com/linto-ai/whisper-timestamped

This service imports `whisper_timestamped` as a Python library into its own
process (see `pyproject.toml` `[project.optional-dependencies].local`). The
combined work is therefore subject to AGPL-3.0, including the network-use
clause (§13): anyone who interacts with this service over a network is
entitled to the complete corresponding source of the running version. The
`GET /license` endpoint exposes the source URL configured via
`WHISPER_TS_SOURCE_URL`. Modifications to this service or to
`whisper-timestamped` must be published under AGPL-3.0 when the service is
operated or distributed.

The other services in the wider transcription stack (`gateway`,
`parakeet_worker`, `diarization_worker`, `summary_worker`, `gatetop`) remain
under MIT. They only communicate with this service via HTTP between separate
processes / containers ("mere aggregation"). Do not import
`whisper-timestamped` or copy its code into any of those repositories.

### GPL/LGPL Debian packages

The image contains Debian packages under GPL or LGPL:

- **FFmpeg: GPL-2.0-or-later.** FFmpeg's source is mostly LGPL-2.1-or-later,
  but Debian builds it with `--enable-gpl` and GPL-licensed files, so the
  binaries (`ffmpeg`, `libavcodec`, `libavfilter`, `libpostproc`, ...) are
  GPL-2.0-or-later, as stated in `/usr/share/doc/ffmpeg/copyright`. The build
  also links GPL codec libraries such as `libx264`, `libx265` and `libxvidcore`,
  which Debian installs as dependencies. Verified on 2026-09-23 for Debian 13.7
  (`python:3.11-slim`), FFmpeg `7:7.1.5-0+deb13u1`.
- **GCC and binutils: GPL-3.0-or-later.** Certain GCC runtime libraries carry
  the GCC Runtime Library Exception 3.1.
- **glibc development files:** primarily LGPL-2.1-or-later.

GCC and libc6-dev are installed because Triton compiles its CUDA helper on
first use. The component-specific Debian copyright notices are authoritative;
a base-image update can change versions and dependencies.

This service runs `ffmpeg` as a separate program to decode audio; it does not
link FFmpeg into its own code. The GPL therefore does not extend to the
worker, but it applies to distributing the FFmpeg binaries in the image.

Distributing the image requires offering the corresponding source of these
packages; `DEBIAN_SOURCE_PACKAGES.json` and `THIRD_PARTY_SOURCES.json` identify
it.

### NVIDIA CUDA libraries (proprietary)

The PyTorch wheels pull in NVIDIA runtime libraries (`nvidia-*` packages such
as cuBLAS, cuDNN and NCCL). They are not open source; they are distributed
under NVIDIA's license agreements, which the generated report contains.

### Whisper model weights

The weights (MIT, OpenAI, https://github.com/openai/whisper) are not part of
this image; they are mounted at `WHISPER_MODEL_DIR`. When weights are shipped
with a product, their attribution belongs to that product's model license
documentation.

## Release documents

How to produce and publish these documents for a release is described in the
README under "Release License Documents".

This notice is informational; it is not legal advice. Consult counsel if you
intend to redistribute or commercially offer this service.
