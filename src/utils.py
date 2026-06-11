"""Shared utilities: seeding, timing, device detection, logging."""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
SUBMISSIONS = ROOT / "submissions"
CONFIGS = ROOT / "configs"
RESULTS_CSV = ROOT / "results.csv"
EXPERIMENTS_MD = ROOT / "EXPERIMENTS.md"

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_device() -> str:
    """Return 'cuda' if an NVIDIA GPU is usable, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0 and "GPU" in out.stdout:
            return "cuda"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "cpu"


@contextmanager
def timer(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Context manager that logs elapsed wall time."""
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time() - t0:.1f}s"
    (logger.info if logger else print)(msg)


def get_logger(name: str = "btk") -> logging.Logger:
    """Console logger with a compact format (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: dict, path: Path) -> None:
    """Write a dict as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def clip_preds(preds: np.ndarray) -> np.ndarray:
    """Clip predictions to the valid target range [0, 100]."""
    return np.clip(preds, 0.0, 100.0)
