"""
live/strategy_instance.py - Binds config + model + pipeline + position manager
for one trading strategy.

One StrategyInstance per approved strategy (magic number).
"""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from trade2.config.loader import load_config
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.live.signal_pipeline import SignalPipeline
from trade2.live.position_manager import PositionManager
from trade2.live.trade_logger import TradeLogger
from trade2.live.reporter import generate_report
from trade2.live.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class StrategyInstance:
    """
    Self-contained strategy runner.

    Attributes:
        name:             Strategy identifier (used in logs and file names)
        magic:            MT5 magic number (unique per strategy)
        config:           Full resolved config dict
        model_path:       Path to current model .pkl
        pipeline:         SignalPipeline (wraps feature+HMM+signal chain)
        position_manager: PositionManager (tracks open position)
        trade_logger:     TradeLogger (CSV writer)
    """

    def __init__(
        self,
        strategy_cfg: Dict[str, Any],
        connector: MT5Connector,
        trade_log_dir: Path,
        project_root: Path,
        base_allocation_frac: float = 0.10,
    ):
        self.name           = strategy_cfg["name"]
        self.magic          = strategy_cfg["magic_number"]
        self.base_alloc     = base_allocation_frac
        self.project_root   = Path(project_root)
        self.connector      = connector

        # Resolve config path relative to project root
        cfg_path  = self.project_root / strategy_cfg["config_path"]
        self.model_path = self.project_root / strategy_cfg["model_path"]

        # Load merged config (base + strategy override)
        self.config = load_config(cfg_path)

        # Load HMM model
        hmm_model = XAUUSDRegimeModel.load(self.model_path)

        # Signal pipeline
        self.pipeline = SignalPipeline(hmm_model, self.config)

        # Position manager
        self.position_manager = PositionManager(
            connector     = connector,
            magic         = self.magic,
            strategy_name = self.name,
        )

        # Trade logger
        trade_log_dir = Path(trade_log_dir)
        self.trade_logger = TradeLogger(
            trade_log_dir / f"live_trades_{self.name}.csv"
        )

        logger.info(
            f"[StrategyInstance] Initialized | name={self.name} | magic={self.magic} | "
            f"config={cfg_path.name}"
        )

    # ------------------------------------------------------------------
    # Per-bar update
    # ------------------------------------------------------------------

    def on_bar(self, df_1h: pd.DataFrame, df_5m: pd.DataFrame) -> None:
        """Process a new 5M bar close: run pipeline + manage position."""
        try:
            state = self.pipeline.run(df_1h, df_5m)
        except Exception as e:
            logger.error(f"[{self.name}] Signal pipeline error: {e}", exc_info=True)
            return

        # Compute lot size from current account equity
        lots = self._compute_lots(state)

        # Delegate to position manager
        self.position_manager.update(
            state        = state,
            lots         = lots,
            trade_logger = self.trade_logger,
        )

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def recover(self) -> None:
        """Discover and resume any orphaned MT5 position on startup."""
        self.position_manager.recover()

    # ------------------------------------------------------------------
    # Model reload (after retrain)
    # ------------------------------------------------------------------

    def reload_model(self, new_model: XAUUSDRegimeModel) -> None:
        """Hot-swap the HMM model without restarting."""
        self.pipeline.reload_model(new_model)

    # ------------------------------------------------------------------
    # Performance reporting
    # ------------------------------------------------------------------

    def generate_report(self, report_dir: Path) -> Dict[str, Any]:
        return generate_report(
            trade_log_path = self.trade_logger.log_path,
            report_dir     = report_dir,
            strategy_name  = self.name,
            initial_equity = self._get_equity(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_lots(self, state: Dict[str, Any]) -> float:
        """
        Compute lot size matching backtest sizing logic.

        pos_val = equity * base_alloc * position_size_multiplier
        lots    = pos_val / entry_price / 100
        """
        equity   = self._get_equity()
        ps_long  = state.get("position_size_long",  0.5)
        ps_short = state.get("position_size_short", 0.5)
        ps_mult  = ps_long if state.get("signal_long") == 1 else ps_short
        ps_mult  = max(ps_mult, 0.01)  # safety floor

        price    = max(state.get("close_5m", 1.0), 1.0)
        pos_val  = equity * self.base_alloc * ps_mult
        lots     = pos_val / price / 100.0

        # Clamp and round to broker step
        lots = max(round(lots, 2), 0.01)
        return lots

    def _get_equity(self) -> float:
        try:
            info = self.connector.get_account_info()
            return info["equity"]
        except Exception:
            return 10_000.0   # fallback to avoid div-by-zero
