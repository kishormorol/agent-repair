"""Seeding utilities for reproducibility."""
from __future__ import annotations

import hashlib
import os
import random


def set_seed(seed: int) -> None:
    """Seed all RNGs we might touch (stdlib, numpy, torch if available)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def derive_seed(*parts: object) -> int:
    """Deterministically derive a 32-bit seed from arbitrary parts.

    Used so that e.g. (trajectory_id, strategy, repeat_index) always maps to the
    same seed -> repair experiments are reproducible and independently repeatable.
    """
    key = "::".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return int(digest[:8], 16)  # 32-bit
