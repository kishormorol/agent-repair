"""Consistent logging to console + a persistent file on Drive."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) a file.

    Idempotent: repeated calls with the same name won't duplicate handlers
    (important in notebooks where cells re-run).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")

    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                     for h in logger.handlers)
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fpath = str(Path(log_dir) / f"{name}.log")
        has_file = any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(Path(fpath).resolve())
                       for h in logger.handlers)
        if not has_file:
            fh = logging.FileHandler(fpath, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
