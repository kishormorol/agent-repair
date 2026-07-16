"""Configuration loader.

Loads config/config.yaml, detects the runtime (Colab Drive vs local), resolves
all relative paths to absolute ones under the correct base directory, and
creates the output folders. Every notebook starts with:

    from src.utils.config import load_config
    cfg = load_config()
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def _on_colab() -> bool:
    """True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def _drive_mounted(drive_base: str) -> bool:
    """True if the Drive base path's parent (/content/drive) is mounted."""
    return Path("/content/drive").exists()


def _to_namespace(d: Any) -> Any:
    """Recursively convert dicts to attribute-accessible namespaces.

    Keeps dict access working too via the `_raw` copy stored on the root.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_namespace(v) for v in d]
    return d


class Config(SimpleNamespace):
    """Attribute-accessible config with a resolved absolute `base` path and a
    `path(key)` helper. Also keeps the original dict in `.raw`."""

    raw: Dict[str, Any]
    base: str

    def path(self, key: str) -> str:
        """Return the absolute path for a key under `paths:` (e.g. 'trajectories')."""
        rel = self.raw["paths"][key]
        return str(Path(self.base) / rel)

    def ensure_dirs(self) -> None:
        """Create every directory listed under `paths:` (except *_base keys)."""
        for key, rel in self.raw["paths"].items():
            if key.endswith("_base"):
                continue
            Path(self.base, rel).mkdir(parents=True, exist_ok=True)


def _find_config_file(explicit: str | None) -> Path:
    """Locate config.yaml by walking up from CWD, or use an explicit path."""
    if explicit:
        return Path(explicit)
    here = Path.cwd()
    for parent in [here, *here.parents]:
        candidate = parent / "config" / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate config/config.yaml")


def load_config(config_path: str | None = None, base_override: str | None = None) -> Config:
    """Load and resolve the experiment config.

    Args:
        config_path: explicit path to config.yaml (optional; auto-located otherwise).
        base_override: force a base directory (useful for tests).

    Returns:
        Config namespace with `.raw` (dict), `.base` (absolute str), and helpers.
    """
    cfg_file = _find_config_file(config_path)
    with open(cfg_file, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)

    # Decide base directory: explicit > Drive (Colab) > local project root.
    if base_override:
        base = base_override
    elif _on_colab() and _drive_mounted(raw["paths"]["drive_base"]):
        base = raw["paths"]["drive_base"]
    else:
        # Local: project root is the config file's parent's parent.
        base = str(cfg_file.parent.parent)

    ns = _to_namespace(raw)
    cfg = Config(**ns.__dict__)
    cfg.raw = raw
    cfg.base = base
    cfg.ensure_dirs()
    return cfg


if __name__ == "__main__":
    c = load_config()
    print(f"Base directory : {c.base}")
    print(f"On Colab       : {_on_colab()}")
    print(f"Agent model    : {c.models.agent.name}")
    print(f"Trajectory dir : {c.path('trajectories')}")
