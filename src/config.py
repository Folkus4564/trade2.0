"""
Module: config.py
Purpose: Load, validate, and cache central YAML configuration
Author: Auto-generated
Date: 2026-03-10
"""

import yaml
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).parents[1]
_DEFAULT_PATH = ROOT / "config.yaml"

_cached_config: Dict[str, Any] = None


def load_config(path: str = None) -> Dict[str, Any]:
    """Load YAML config and cache in module-level singleton."""
    global _cached_config
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    _cached_config = cfg
    return cfg


def get_config() -> Dict[str, Any]:
    """Return cached config, loading from default path if needed."""
    global _cached_config
    if _cached_config is None:
        load_config()
    return _cached_config


def validate_config(cfg: Dict[str, Any]) -> None:
    """Check required keys, type correctness, date consistency, numeric sanity."""
    required_sections = ["data", "splits", "hmm", "features", "smc", "costs", "acceptance"]
    for s in required_sections:
        if s not in cfg:
            raise ValueError(f"Missing required config section: '{s}'")

    # Date consistency: train_end < val_start < val_end < test_start
    sp = cfg["splits"]
    for key in ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]:
        if key not in sp:
            raise ValueError(f"Missing splits.{key}")

    if sp["train_end"] >= sp["val_start"]:
        raise ValueError(f"train_end ({sp['train_end']}) must be before val_start ({sp['val_start']})")
    if sp["val_end"] >= sp["test_start"]:
        raise ValueError(f"val_end ({sp['val_end']}) must be before test_start ({sp['test_start']})")

    # Numeric sanity
    hmm = cfg["hmm"]
    if hmm.get("n_states", 3) < 2:
        raise ValueError("hmm.n_states must be >= 2")
    if hmm.get("n_iter", 200) < 1:
        raise ValueError("hmm.n_iter must be >= 1")

    costs = cfg["costs"]
    if costs.get("spread_pips", 3) < 0:
        raise ValueError("costs.spread_pips must be >= 0")
    if costs.get("commission_rt", 0.0002) < 0:
        raise ValueError("costs.commission_rt must be >= 0")
