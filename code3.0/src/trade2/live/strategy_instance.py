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
from trade2.live.multi_position_manager import MultiPositionManager
from trade2.live.partial_tp_position_manager import PartialTPPositionManager
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
        self.name             = strategy_cfg["name"]
        self.magic            = strategy_cfg["magic_number"]
        self.base_alloc       = base_allocation_frac
        self.lot_min          = strategy_cfg["lot_min"]
        self.lot_max          = strategy_cfg["lot_max"]
        self.max_concurrent   = int(strategy_cfg.get("max_concurrent_positions", 1))
        self.project_root     = Path(project_root)
        self.connector        = connector

        # Resolve config path relative to project root
        cfg_path  = self.project_root / strategy_cfg["config_path"]
        self.model_path = self.project_root / strategy_cfg["model_path"]

        # Load merged config (base + strategy override)
        self.config = load_config(override_path=cfg_path)

        # Load HMM model
        hmm_model = XAUUSDRegimeModel.load(self.model_path)

        # Load XGBoost reversal model (optional — only for strategies like Q)
        xgb_model = None
        xgb_path_str = strategy_cfg.get("xgb_model_path", None)
        if xgb_path_str:
            from trade2.models.reversal_xgb import ReversalXGBModel
            xgb_model_path = self.project_root / xgb_path_str
            xgb_model = ReversalXGBModel.load(str(xgb_model_path))
            logger.info(
                f"[StrategyInstance] Loaded XGB reversal model | path={xgb_model_path.name}"
            )

        # Signal pipeline
        self.pipeline = SignalPipeline(hmm_model, self.config, xgb_model=xgb_model)

        # Position manager — partial-TP, multi, or single
        partial_tp_cfg   = self.config.get("risk", {}).get("partial_tp", None)
        fixed_dollar_tps = partial_tp_cfg.get("fixed_dollar_tps", None) if partial_tp_cfg else None

        # A single TP level with size_frac=1.0 is NOT a partial-TP split —
        # it's a regular single-exit trade.  When max_concurrent > 1 we must
        # use MultiPositionManager so ATR-based tp_long/tp_short are honoured.
        _is_real_partial = (
            fixed_dollar_tps is not None
            and len(fixed_dollar_tps) > 1   # genuinely more than one TP level
        )
        # live.yaml can also force the multi-position manager explicitly
        _force_multi = bool(strategy_cfg.get("force_multi_position_manager", False))

        if _is_real_partial and not _force_multi:
            self.position_manager = PartialTPPositionManager(
                connector        = connector,
                magic            = self.magic,
                strategy_name    = self.name,
                max_concurrent   = self.max_concurrent,
                fixed_dollar_tps = fixed_dollar_tps,
                size_fracs       = partial_tp_cfg.get("size_fracs", None),
                be_after_tp1     = partial_tp_cfg.get("be_after_tp1", True),
            )
        elif self.max_concurrent > 1 or _force_multi:
            self.position_manager = MultiPositionManager(
                connector      = connector,
                magic          = self.magic,
                strategy_name  = self.name,
                max_concurrent = self.max_concurrent,
            )
        else:
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
        """Discover and resume any orphaned MT5 position(s) on startup."""
        self.position_manager.recover()
        pm = self.position_manager

        # MultiPositionManager or PartialTPPositionManager: log each recovered position
        if isinstance(pm, (MultiPositionManager, PartialTPPositionManager)):
            for pos in pm._positions:
                self.trade_logger.log_entry(
                    ticket        = pos["ticket"],
                    strategy      = self.name,
                    direction     = pos["direction"],
                    entry_price   = pos["entry_price"],
                    sl            = pos["sl"],
                    tp            = pos["tp"],
                    lots          = pos["lots"],
                    entry_time    = pos["entry_time"],
                    regime        = pos.get("entry_regime", "?"),
                    bull_prob     = pos.get("entry_bull_prob", 0.0),
                    bear_prob     = pos.get("entry_bear_prob", 0.0),
                    signal_source = pos.get("signal_source", "recovered"),
                )
        elif pm.ticket is not None:
            # Single PositionManager: log the one position
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

    def reload_xgb_model(self, new_xgb_model: Any) -> None:
        """Hot-swap the XGBoost reversal model without restarting."""
        self.pipeline.reload_xgb_model(new_xgb_model)

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

        For MultiPositionManager: divide by max_concurrent so each slot gets an equal share
        of the total budget (total exposure constant regardless of how many slots are filled).

        For PartialTPPositionManager: do NOT divide — lot_min/lot_max represent the per-signal
        budget. The manager splits internally by size_fracs (e.g. 0.5/0.5 for 2 TP levels).

        confidence = bear_prob if short signal, bull_prob if long signal
        """
        is_short   = state.get("signal_short") == 1
        confidence = state.get("bear_prob", 0.0) if is_short else state.get("bull_prob", 0.0)

        raw = self.lot_min + (self.lot_max - self.lot_min) * confidence

        # Only divide by max_concurrent for multi-slot engines, not for partial-TP groups
        if not isinstance(self.position_manager, PartialTPPositionManager):
            raw = raw / self.max_concurrent   # per-slot share of total budget

        lots = round(round(raw, 2) / 0.01) * 0.01   # snap to 0.01 grid
        lots = max(0.01, round(lots, 2))
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
