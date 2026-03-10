"""
config/loader.py - Load and validate YAML configuration with deep merge support.
No global singleton - config is passed explicitly to all modules.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import copy
import yaml


def load_config(
    base_path: str | Path = None,
    override_path: str | Path = None,
) -> Dict[str, Any]:
    """
    Load base config YAML, optionally deep-merged with an override file.

    Args:
        base_path:     Path to base.yaml (defaults to configs/base.yaml relative to project root)
        override_path: Optional path to override YAML (e.g. configs/xauusd_mtf.yaml)

    Returns:
        Merged config dict.
    """
    if base_path is None:
        base_path = Path(__file__).parents[4] / "configs" / "base.yaml"

    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path) as f:
            overrides = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, overrides)

    validate_config(cfg)
    return cfg


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override into base. Override values take precedence."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def validate_config(cfg: Dict[str, Any]) -> None:
    """Check required sections, date ordering, and numeric sanity."""
    required = ["data", "splits", "hmm", "features", "smc", "costs", "acceptance"]
    for s in required:
        if s not in cfg:
            raise ValueError(f"Missing required config section: '{s}'")

    sp = cfg["splits"]
    for key in ["train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]:
        if key not in sp:
            raise ValueError(f"Missing splits.{key}")

    if sp["train_end"] >= sp["val_start"]:
        raise ValueError(f"train_end must be before val_start")
    if sp["val_end"] >= sp["test_start"]:
        raise ValueError(f"val_end must be before test_start")

    hmm = cfg["hmm"]
    if hmm.get("n_states", 3) < 2:
        raise ValueError("hmm.n_states must be >= 2")

    costs = cfg["costs"]
    if costs.get("spread_pips", 3) < 0:
        raise ValueError("costs.spread_pips must be >= 0")
    if costs.get("commission_rt", 0.0002) < 0:
        raise ValueError("costs.commission_rt must be >= 0")
