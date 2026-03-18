"""
live/position_manager.py - Tracks and manages a single open position per strategy.

Each StrategyInstance owns one PositionManager bound to its magic number.
Responsibilities:
- Open new positions when signal fires and no position is open
- Close positions on exit signal or regime flip
- Update trailing stop-loss on each bar
- Crash recovery: discover existing positions on startup
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from trade2.live.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages at most ONE open position per strategy (magic number).

    Thread-safety: designed for single-threaded polling loop.
    """

    def __init__(self, connector: MT5Connector, magic: int, strategy_name: str):
        self.connector     = connector
        self.magic         = magic
        self.strategy_name = strategy_name

        # Live position state
        self.ticket:      Optional[int]   = None
        self.direction:   Optional[str]   = None   # "long" or "short"
        self.entry_price: Optional[float] = None
        self.lots:        Optional[float] = None
        self.sl:          Optional[float] = None
        self.tp:          Optional[float] = None
        self.entry_time:  Optional[datetime] = None
        self.entry_regime: Optional[str] = None
        self.entry_bull_prob: float = 0.0
        self.entry_bear_prob: float = 0.0
        self.signal_source: str = ""

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------

    def recover(self) -> None:
        """On startup, discover any orphaned position for this magic number."""
        positions = self.connector.get_positions(magic=self.magic)
        if not positions:
            logger.info(f"[{self.strategy_name}] No existing position to recover")
            return
        pos = positions[0]
        self.ticket      = pos["ticket"]
        self.direction   = pos["type"]
        self.entry_price = pos["open_price"]
        self.lots        = pos["volume"]
        self.sl          = pos["sl"]
        self.tp          = pos["tp"]
        self.entry_time  = pos["open_time"]
        logger.info(
            f"[{self.strategy_name}] Recovered position | ticket={self.ticket} | "
            f"dir={self.direction} | entry={self.entry_price:.2f} | lots={self.lots:.2f}"
        )

    # ------------------------------------------------------------------
    # Per-bar update
    # ------------------------------------------------------------------

    def update(
        self,
        state: Dict[str, Any],
        lots: float,
        trade_logger,
    ) -> None:
        """
        Process the latest signal state and act accordingly.

        Args:
            state:        Output of SignalPipeline.run()
            lots:         Pre-computed lot size for this bar
            trade_logger: TradeLogger instance for recording events
        """
        if self.ticket is not None:
            self._handle_open_position(state, trade_logger)
        else:
            self._handle_no_position(state, lots, trade_logger)

    def close_all(self, reason: str = "manual") -> None:
        """Close the open position if any (used before retrain)."""
        if self.ticket is None:
            return
        logger.info(f"[{self.strategy_name}] Closing position {self.ticket} — reason: {reason}")
        ok = self.connector.close_position(self.ticket, magic=self.magic)
        if ok:
            self._clear_state()

    def has_position(self) -> bool:
        return self.ticket is not None

    # ------------------------------------------------------------------
    # Internal — position logic
    # ------------------------------------------------------------------

    def _handle_open_position(self, state: Dict[str, Any], trade_logger) -> None:
        direction = self.direction
        exit_flag = state["exit_long"] if direction == "long" else state["exit_short"]

        if exit_flag:
            logger.info(
                f"[{self.strategy_name}] Exit signal | dir={direction} | "
                f"regime={state['regime']}"
            )
            ok = self.connector.close_position(self.ticket, magic=self.magic)
            if ok:
                exit_price = state["close_5m"]
                pnl = self._compute_pnl(exit_price)
                trade_logger.log_exit(
                    ticket=self.ticket,
                    strategy=self.strategy_name,
                    exit_price=exit_price,
                    exit_time=state["bar_time"],
                    pnl=pnl,
                    exit_reason="signal",
                )
                self._clear_state()
            return

        # Update trailing stop
        trail_mult = (
            state["trailing_atr_mult_long"]  if direction == "long"
            else state["trailing_atr_mult_short"]
        )
        if trail_mult > 0 and state["atr_1h"] > 0:
            close = state["close_5m"]
            atr   = state["atr_1h"]
            if direction == "long":
                new_sl = close - trail_mult * atr
                if new_sl > self.sl:
                    ok = self.connector.modify_sl(self.ticket, new_sl)
                    if ok:
                        self.sl = new_sl
                        logger.debug(f"[{self.strategy_name}] Trail SL -> {new_sl:.2f}")
            else:
                new_sl = close + trail_mult * atr
                if new_sl < self.sl:
                    ok = self.connector.modify_sl(self.ticket, new_sl)
                    if ok:
                        self.sl = new_sl
                        logger.debug(f"[{self.strategy_name}] Trail SL -> {new_sl:.2f}")

    def _handle_no_position(
        self,
        state: Dict[str, Any],
        lots: float,
        trade_logger,
    ) -> None:
        sig_long  = state["signal_long"]
        sig_short = state["signal_short"]

        if sig_long == 1:
            self._open(state, "long", lots, trade_logger)
        elif sig_short == 1:
            self._open(state, "short", lots, trade_logger)

    def _open(
        self,
        state: Dict[str, Any],
        direction: str,
        lots: float,
        trade_logger,
    ) -> None:
        sl = state["stop_long"]  if direction == "long" else state["stop_short"]
        tp = state["tp_long"]    if direction == "long" else state["tp_short"]

        ticket = self.connector.open_order(
            direction=direction,
            lots=lots,
            sl_price=sl,
            tp_price=tp,
            magic=self.magic,
            comment=f"{self.strategy_name[:20]}_{state['signal_source'][:8]}",
        )
        if ticket is None:
            logger.error(f"[{self.strategy_name}] open_order failed — signal skipped")
            return

        entry_price = state["close_5m"]
        self.ticket       = ticket
        self.direction    = direction
        self.entry_price  = entry_price
        self.lots         = lots
        self.sl           = sl
        self.tp           = tp
        self.entry_time   = state["bar_time"]
        self.entry_regime = state["regime"]
        self.entry_bull_prob = state["bull_prob"]
        self.entry_bear_prob = state["bear_prob"]
        self.signal_source   = state["signal_source"]

        trade_logger.log_entry(
            ticket=ticket,
            strategy=self.strategy_name,
            direction=direction,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            lots=lots,
            entry_time=state["bar_time"],
            regime=state["regime"],
            bull_prob=state["bull_prob"],
            bear_prob=state["bear_prob"],
            signal_source=state["signal_source"],
        )

    def _compute_pnl(self, exit_price: float) -> float:
        if self.entry_price is None or self.lots is None:
            return 0.0
        multiplier = 1.0 if self.direction == "long" else -1.0
        # 100 oz per standard lot on Exness XAUUSD
        return multiplier * (exit_price - self.entry_price) * self.lots * 100.0

    def _clear_state(self) -> None:
        self.ticket       = None
        self.direction    = None
        self.entry_price  = None
        self.lots         = None
        self.sl           = None
        self.tp           = None
        self.entry_time   = None
        self.entry_regime = None
        self.entry_bull_prob = 0.0
        self.entry_bear_prob = 0.0
        self.signal_source   = ""
