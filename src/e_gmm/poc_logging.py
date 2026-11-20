"""Shared timestamped logging & stdout/stderr tee helper for all PoC scripts (stdlib only)."""

from contextlib import contextmanager
from datetime import datetime
import pathlib
import platform
import shutil
import sys
from typing import Iterator, Optional

# Resolve <repo>/poc/logs cleanly regardless of working directory
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
LOGS_ROOT = REPO_ROOT / "poc" / "logs"


class _TeeStream:
    """Mirror writes to both original console stream and a UTF-8 log file."""

    def __init__(self, primary, log_file):
        self.primary = primary
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())


def _detect_compute_env() -> str:
    parts = []
    try:
        import numpy as np

        parts.append(f"numpy={np.__version__}")
    except ImportError:
        parts.append("numpy=missing")
    try:
        import torch

        if torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            parts.append(f"torch={torch.__version__} (cuda={torch.version.cuda}, gpus={names})")
        else:
            parts.append(f"torch={torch.__version__} (CPU)")
    except ImportError:
        parts.append("torch=not installed (NumPy/CPU mode)")
    return " | ".join(parts)


@contextmanager
def tee_poc_logs(poc_name: str, run_tag: Optional[str] = None) -> Iterator[pathlib.Path]:
    """
    Create `poc/logs/<poc_name>/run_<timestamp>.log`, mirror stdout & stderr cleanly,
    record system/CLI metadata, and update `poc/logs/<poc_name>/latest.log` on exit.
    """
    log_dir = LOGS_ROOT / poc_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    log_path = log_dir / f"run_{ts}{suffix}.log"
    latest_path = log_dir / "latest.log"

    orig_out, orig_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as fh:
        fh.write("=" * 84 + "\n")
        fh.write(f"POC_RUN      : {poc_name}{suffix}\n")
        fh.write(f"TIMESTAMP    : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"COMMAND      : {' '.join(sys.argv)}\n")
        fh.write(f"CWD          : {pathlib.Path.cwd()}\n")
        fh.write(f"PLATFORM     : {platform.platform()} | Python {platform.python_version()}\n")
        fh.write(f"COMPUTE_ENV  : {_detect_compute_env()}\n")
        fh.write(f"LOG_FILE     : {log_path}\n")
        fh.write("=" * 84 + "\n\n")

        sys.stdout = _TeeStream(orig_out, fh)
        sys.stderr = _TeeStream(orig_err, fh)
        try:
            yield log_path
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = orig_out
            sys.stderr = orig_err

    try:
        shutil.copyfile(log_path, latest_path)
    except OSError:
        pass
    print(f"[Log Saved] {log_path}")
