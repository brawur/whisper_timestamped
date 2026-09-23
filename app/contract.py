"""Interface contract this worker reports to the gateway."""
from importlib.metadata import PackageNotFoundError, version

# Raise only on an incompatible change of the gateway <-> worker interface. The
# gateway accepts a range per worker (app/services/compatibility.py there);
# extend that range in the same release.
WORKER_API = 1


def build_version() -> str:
    try:
        return version("whisper-timestamped-service")
    except PackageNotFoundError:
        return "unknown"
