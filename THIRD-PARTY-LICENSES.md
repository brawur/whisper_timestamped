# Third-Party Licenses

This repository is licensed under the **GNU Affero General Public License
v3.0 or later (AGPL-3.0-or-later)**. See [LICENSE.md](LICENSE.md) for the
full license text and the README for §13 (network-use) compliance notes.

The AGPL applies because this service imports the `whisper-timestamped`
Python library (itself AGPL-3.0) into its own process. The combined work is
therefore subject to the AGPL, including the network-use clause: anyone who
interacts with this service over a network is entitled to receive the
complete corresponding source code of the running version. The `GET
/license` endpoint exposes the source URL configured by the operator.

## Bundled / Linked Components

### whisper-timestamped (upstream library)

- Component: `whisper-timestamped` (Python package)
- Licensor: LINAGORA / linto-ai contributors
- **License: GNU Affero General Public License v3.0 (AGPL-3.0)**
- Source: https://github.com/linto-ai/whisper-timestamped
- License text: https://www.gnu.org/licenses/agpl-3.0.html

This service imports `whisper_timestamped` as a Python library inside its own
process (see `pyproject.toml` `[project.optional-dependencies].local`). The
combined work is therefore subject to AGPL-3.0 terms, including the
network-use clause (§13): if you offer the service over a network, recipients
have the right to obtain the complete corresponding source of the modified
version under AGPL-3.0.

### OpenAI Whisper (transitive via whisper-timestamped)

- Component: `openai-whisper`
- License: MIT License
- Source: https://github.com/openai/whisper

### Whisper Model Weights

- Components: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, ...
- Licensor: OpenAI
- License: MIT License (model weights)
- Source: https://github.com/openai/whisper
- Model weights are downloaded at runtime and are not redistributed by this
  repository. Review upstream for the authoritative terms.

## Runtime Dependencies (this repository's own code)

### FastAPI

- License: MIT License
- Source: https://github.com/tiangolo/fastapi

### Uvicorn

- License: BSD 3-Clause License
- Source: https://github.com/encode/uvicorn

### python-multipart

- License: Apache License 2.0
- Source: https://github.com/Kludex/python-multipart

### PyTorch

- Components: `torch`
- License: BSD 3-Clause License
- Source: https://github.com/pytorch/pytorch

### NumPy

- License: BSD 3-Clause License
- Source: https://github.com/numpy/numpy

### auditok

- License: MIT License
- Source: https://github.com/amsehili/auditok

### tiktoken (used by Whisper for tokenization)

- License: MIT License
- Source: https://github.com/openai/tiktoken

### FFmpeg (runtime, not bundled)

- License: LGPL/GPL depending on build configuration
- Source: https://ffmpeg.org/

## Copyleft Notice — AGPL-3.0 Implications

Because `whisper-timestamped` is linked into this service's process as a
Python library, the combined work is a derivative work under AGPL-3.0.
Practical consequences:

- Anyone who interacts with this service over a network is entitled to
  receive the complete corresponding source code of the running version
  under AGPL-3.0 (§13). The `GET /license` endpoint surfaces the configured
  `source_url` — set the `WHISPER_TS_SOURCE_URL` environment variable to a
  publicly reachable location of the deployed source.
- Any modifications made to this service or to `whisper-timestamped` must be
  published under AGPL-3.0 when the service is operated.
- The other services in the wider transcription stack (`gateway`,
  `parakeet_worker`, `diarization_worker`, `summary_worker`, `gatetop`)
  remain under MIT. They only communicate with this service via HTTP between
  separate processes / containers ("mere aggregation"). Do not import
  `whisper-timestamped` or copy its code into any of those repositories.
- This notice is informational; it is not legal advice. Consult counsel if
  you intend to redistribute or commercially offer this service.
