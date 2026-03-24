"""
live/retrainer.py - Weekly HMM retrain on expanding window.

Checks every loop iteration whether a Sunday retrain is due.
Backs up the current model, retrains on full dataset (original + live bars),
saves the new model, and reloads it in all StrategyInstance objects.
"""

import logging
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from trade2.features.builder import add_1h_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.config.loader import load_config

logger = logging.getLogger(__name__)


class Retrainer:
    """
    Manages the weekly HMM retrain cycle.

    Usage:
        retrainer = Retrainer(live_cfg, data_accumulator, strategy_instances)
        # In the main loop:
        retrainer.check_and_retrain_if_due()
    """

    def __init__(
        self,
        live_cfg: Dict[str, Any],
        data_accumulator,
        strategy_instances: List,
        project_root: Path,
    ):
        self.cfg            = live_cfg["retrain"]
        self.accumulator    = data_accumulator
        self.strategies     = strategy_instances
        self.project_root   = Path(project_root)
        self._last_retrain_week: int = -1  # ISO week number of last retrain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_retrain_if_due(self) -> bool:
        """
        Call this once per loop iteration.

        Returns True if a retrain was performed.
        """
        if not self.cfg.get("enabled", True):
            return False

        schedule = self.cfg.get("schedule", "sunday").lower()
        now      = datetime.now(tz=timezone.utc)

        # Saturday = weekday 5, Sunday = weekday 6
        is_scheduled_day = (
            (schedule == "saturday" and now.weekday() == 5) or
            (schedule == "sunday"   and now.weekday() == 6)
        )
        iso_week         = now.isocalendar()[1]
        already_done     = (iso_week == self._last_retrain_week)

        if is_scheduled_day and not already_done:
            logger.info(f"[Retrainer] Scheduled retrain triggered (week {iso_week})")
            self._run_retrain()
            self._last_retrain_week = iso_week
            return True

        return False

    def force_retrain(self, full: bool = False) -> None:
        """Trigger an immediate retrain regardless of schedule.

        Args:
            full: If True, ignore mode config and do a full scratch retrain.
        """
        label = "full retrain" if full else f"{self.cfg.get('mode','full')} retrain"
        logger.info(f"[Retrainer] Force {label} requested")
        self._run_retrain(force_full=full)
        now = datetime.now(tz=timezone.utc)
        self._last_retrain_week = now.isocalendar()[1]

    # ------------------------------------------------------------------
    # Core retrain logic
    # ------------------------------------------------------------------

    def _run_retrain(self, force_full: bool = False) -> None:
        retrain_cfg = self.cfg
        mode        = "full" if force_full else retrain_cfg.get("mode", "full")
        original_1h = self.project_root / retrain_cfg["original_1h"]

        # Step 1: Optionally close all open positions before model swap
        if retrain_cfg.get("close_positions", True):
            logger.info("[Retrainer] Closing all open positions before retrain")
            for strat in self.strategies:
                strat.position_manager.close_all(reason="pre_retrain")

        # Step 2: Load 1H dataset
        try:
            df_1h = self.accumulator.load_full_1h(original_1h)
        except Exception as e:
            logger.error(f"[Retrainer] Failed to load 1H data: {e}")
            return

        # Step 3: Build HMM features
        try:
            first_cfg = self.strategies[0].config if self.strategies else None
            reg_feat  = add_1h_features(df_1h, first_cfg)
            X, _      = get_hmm_feature_matrix(reg_feat, first_cfg)
        except Exception as e:
            logger.error(f"[Retrainer] Feature build failed: {e}")
            return

        if len(X) < 200:
            logger.error(f"[Retrainer] Not enough HMM rows ({len(X)}) — aborting retrain")
            return

        # Step 4: Slice to recent window for warm mode
        if mode == "warm":
            warm_bars = int(retrain_cfg.get("warm_update_bars", 6500))
            X_train   = X[-warm_bars:] if len(X) > warm_bars else X
            logger.info(
                f"[Retrainer] Warm mode: using last {len(X_train)} of {len(X)} bars"
            )
        else:
            X_train = X
            logger.info(f"[Retrainer] Full retrain on {len(X_train)} bars")

        # Step 5: For each strategy, retrain/update and reload
        for strat in self.strategies:
            self._retrain_strategy(strat, X_train, retrain_cfg, mode=mode)

        logger.info(f"[Retrainer] {mode.capitalize()} retrain cycle complete")

    def _retrain_strategy(
        self, strat, X, retrain_cfg: Dict[str, Any], mode: str = "full"
    ) -> None:
        model_path = Path(strat.model_path)
        cfg        = strat.config

        # Backup current model
        if retrain_cfg.get("backup_model", True) and model_path.exists():
            ts          = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir  = self.project_root / "artefacts" / "models" / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stem        = model_path.stem
            backup_path = backup_dir / f"{stem}_{ts}.pkl"
            shutil.copy2(model_path, backup_path)
            logger.info(f"[Retrainer] Backed up model to {backup_path}")

        hmm_cfg = cfg["hmm"]

        if mode == "warm":
            # Warm update: load existing model and adapt to recent bars
            logger.info(f"[Retrainer] Warm-updating {strat.name} on {len(X)} bars ...")
            try:
                current_model = XAUUSDRegimeModel.load(model_path)
                n_iter        = int(retrain_cfg.get("warm_n_iter", 30))
                current_model.warm_update(X, n_iter=n_iter)
                new_model = current_model
            except Exception as e:
                logger.error(f"[Retrainer] Warm update failed for {strat.name}: {e}")
                return
        else:
            # Full retrain from scratch
            new_model = XAUUSDRegimeModel(
                n_states    = hmm_cfg["n_states"],
                n_iter      = hmm_cfg.get("n_iter", 200),
                random_seed = hmm_cfg.get("random_seed", 42),
            )
            logger.info(f"[Retrainer] Full retrain for {strat.name} on {len(X)} bars ...")
            try:
                new_model.fit(X)
            except Exception as e:
                logger.error(f"[Retrainer] HMM fit failed for {strat.name}: {e}")
                return

        # Save updated model
        try:
            new_model.save(model_path)
            logger.info(f"[Retrainer] Saved model to {model_path}")
        except Exception as e:
            logger.error(f"[Retrainer] Model save failed: {e}")
            return

        # Reload in strategy instance
        strat.reload_model(new_model)
        logger.info(f"[Retrainer] {strat.name} model hot-swapped ({mode})")

        # Log state distribution
        dist = new_model.state_distribution(X)
        logger.info(
            f"[Retrainer] {strat.name} | n_bars={len(X)} | "
            f"state_dist={dist}"
        )
