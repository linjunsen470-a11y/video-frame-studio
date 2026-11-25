"""Keep processing commands on the managed Codex Cloud workspace."""

from pathlib import Path
import sys


CLOUD_WORKSPACE = Path("/workspace")


def is_codex_cloud() -> bool:
    if sys.platform != "linux":
        return False
    try:
        # These directories are provisioned by the managed workspace, not the checkout.
        return all((CLOUD_WORKSPACE / name).is_dir() for name in ("library-files", "scratch", "shared"))
    except OSError:
        return False


def cloud_startup_allowed() -> bool:
    if is_codex_cloud():
        return True
    print("Error: Codex Cloud workspace could not be verified. Processing stopped. "
          "Open the repository in Codex Cloud and rerun the command there.", file=sys.stderr)
    return False
