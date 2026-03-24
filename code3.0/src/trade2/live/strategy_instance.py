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
        self.lot_min        = strategy_cfg["lot_min"]
        self.lot_max        = strategy_cfg["lot_max"]
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

        # Dashboard cache
        self._last_regime    = "?"
        self._last_bull_prob = 0.0
        self._last_bear_prob = 0.0

        logger.info(
            f"[StrategyInstance] Initialized | name={self.name} | magic={self.magic} | "
            f"config={cfg_path.name}"
        )
        self._log_lot_range()

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

        # Cache regime info for dashboard display
        self._last_regime    = state.get("regime", "?")
        self._last_bull_prob = state.get("bull_prob", 0.0)
        self._last_bear_prob = state.get("bear_prob", 0.0)

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
        pm = self.position_manager
        if pm.ticket is not None:
            # Pre-populate TradeLogger so exit row has entry data on restart
            self.trade_logger.log_entry(
                ticket        = pm.ticket,
                strategy      = self.name,
                direction     = pm.direction or "",
                entry_price   = pm.entry_price or 0.0,
                sl            = pm.sl or 0.0,
                tp            = pm.tp or 0.0,
                lots          = pm.lots or 0.0,
                entry_time    = pm.entry_time,
                regime        = pm.entry_regime or "?",
                bull_prob     = pm.entry_bull_prob,
                bear_prob     = pm.entry_bear_prob,
                signal_source = pm.signal_source or "recovered",
            )

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
        Scale lot size by regime confidence between lot_min and lot_max (from live.yaml).

        confidence = bear_prob if short signal, bull_prob if long signal
        lots = lot_min + (lot_max - lot_min) * confidence, snapped to 0.01 grid
        """
        is_short   = state.get("signal_short") == 1
        confidence = state.get("bear_prob", 0.0) if is_short else state.get("bull_prob", 0.0)

        raw  = self.lot_min + (self.lot_max - self.lot_min) * confidence
        lots = round(round(raw, 2) / 0.01) * 0.01   # snap to 0.01 grid
        lots = max(self.lot_min, min(self.lot_max, round(lots, 2)))
        return lots

    def _log_lot_range(self) -> None:
        """Print the lot size range this strategy will use based on regime confidence."""
        try:
            equity = self._get_equity()
            print(
                f"[{self.name}] Lot size range: {self.lot_min} (confidence=0.0) "
                f"to {self.lot_max} (confidence=1.0) | equity={equity:.2f} | method=confidence_scaled"
            )
            logger.info(
                f"[{self.name}] Lot range: min={self.lot_min} max={self.lot_max} equity={equity:.2f}"
            )
        except Exception as e:
            logger.warning(f"[{self.name}] Could not compute lot range: {e}")

    def _get_equity(self) -> float:
        try:
            info = self.connector.get_account_info()
            return info["equity"]
        except Exception:
            return 10_000.0   # fallback to avoid div-by-zero
