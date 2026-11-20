"""Dedicated timestamped logging & stdout/stderr tee helper inside `experiments/logs/`."""

from contextlib import contextmanager
from datetime import datetime
import pathlib
import platform
import shutil
import sys
from typing import Generator, Optional

EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parent
LOGS_ROOT = EXPERIMENTS_DIR / "logs"
RESULTS_ROOT = EXPERIMENTS_DIR / "results"


class _TeeStream:
    """Mirror writes to both console stream and a UTF-8 log file."""

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


def _detect_env() -> str:
    parts = []
    try:
        import numpy as np

        parts.append(f"numpy={np.__version__}")
    except ImportError:
        parts.append("numpy=missing")
    try:
        import torch

        if torch.cuda.is_available():
            gpus = [
                f"{torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.1f} GiB)"
                for i in range(torch.cuda.device_count())
            ]
            parts.append(f"torch={torch.__version__} (cuda={torch.version.cuda}, gpus={gpus})")
        else:
            parts.append(f"torch={torch.__version__} (CPU)")
    except ImportError:
        parts.append("torch=not installed (NumPy/CPU mode)")
    return " | ".join(parts)


@contextmanager
def tee_experiment_logs(exp_name: str, run_tag: Optional[str] = None) -> Generator[pathlib.Path, None, None]:
    """
    Create `experiments/logs/<exp_name>/run_<YYYYMMDD_HHMMSS>[_<tag>].log`,
    mirror stdout & stderr cleanly, and update `latest.log`.
    """
    log_dir = LOGS_ROOT / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_tag}" if run_tag else ""
    log_path = log_dir / f"run_{ts}{suffix}.log"
    latest_path = log_dir / "latest.log"

    orig_out, orig_err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as fh:
        fh.write("=" * 88 + "\n")
        fh.write(f"EXPERIMENT   : {exp_name}{suffix}\n")
        fh.write(f"TIMESTAMP    : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"COMMAND      : {' '.join(sys.argv)}\n")
        fh.write(f"CWD          : {pathlib.Path.cwd()}\n")
        fh.write(f"PLATFORM     : {platform.platform()} | Python {platform.python_version()}\n")
        fh.write(f"COMPUTE_ENV  : {_detect_env()}\n")
        fh.write(f"LOG_FILE     : {log_path}\n")
        fh.write("=" * 88 + "\n\n")

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
    print(f"[Experiment Log Saved] {log_path}")
