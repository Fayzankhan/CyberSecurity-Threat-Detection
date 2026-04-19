"""Render: deploy bundle may omit build-time untracked files; train DL once at runtime if missing."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_thread: threading.Thread | None = None


def maybe_bootstrap_dl_async(root_dir: Path, dl_binary_model: Path, dl_multiclass_model: Path) -> None:
    """If running on Render and DL weights are missing, start train_dl --quick in a daemon thread."""
    global _thread
    if _thread is not None:
        return
    if not os.environ.get("RENDER"):
        return
    if dl_binary_model.is_file() and dl_multiclass_model.is_file():
        return

    def run() -> None:
        logger.warning(
            "DL checkpoints missing at runtime (deploy often ships only git-tracked files). "
            "Running train_dl --quick in background; wait a few minutes then refresh /health."
        )
        env = {**os.environ, "PYTHONPATH": str(root_dir)}
        r = subprocess.run(
            [sys.executable, "-m", "src.train_dl", "--quick"],
            cwd=str(root_dir),
            env=env,
            timeout=3600,
        )
        if r.returncode != 0:
            logger.error("Runtime train_dl --quick failed with exit code %s", r.returncode)
        else:
            logger.info("Runtime train_dl --quick completed")

    _thread = threading.Thread(target=run, name="render-dl-bootstrap", daemon=True)
    _thread.start()


def bootstrap_thread_alive() -> bool:
    return _thread is not None and _thread.is_alive()
