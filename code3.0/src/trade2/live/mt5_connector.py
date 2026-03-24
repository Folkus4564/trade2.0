"""
live/mt5_connector.py - MetaTrader5 connection, bar fetching, and order execution.

All MT5 operations are wrapped here. Import MetaTrader5 lazily so the rest of
the package can be imported without MT5 installed (useful for testing/CI).
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# MT5 order type constants (mirrored here to avoid hard imports at module level)
_ORDER_TYPE_BUY  = 0
_ORDER_TYPE_SELL = 1


def _mt5():
    """Lazy import of MetaTrader5."""
    try:
        import MetaTrader5 as mt5
        return mt5
    except ImportError as e:
        raise ImportError(
            "MetaTrader5 package not installed. "
            "Install it with: pip install MetaTrader5"
        ) from e


class MT5Connector:
    """
    Thin wrapper around the MetaTrader5 Python API.

    Handles:
    - Connection / reconnection
    - Bar fetching (1H and 5M)
    - Market order placement with SL/TP
    - Position query and modification
    - Account info
    """

    def __init__(
        self,
        login: int,
        password: str,
        server: str,
        symbol: str = "XAUUSD",
        timeout_ms: int = 60_000,
    ):
        self.login      = login
        self.password   = password
        self.server     = server
        self.symbol     = symbol
        self.timeout_ms = timeout_ms
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Initialize and log in to MT5 terminal."""
        mt5 = _mt5()
        if not mt5.initialize(
            login=self.login,
            password=self.password,
            server=self.server,
            timeout=self.timeout_ms,
        ):
            err = mt5.last_error()
            logger.error(f"[MT5] initialize() failed: {err}")
            self._connected = False
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("[MT5] Could not retrieve account info after initialize()")
            self._connected = False
            return False

        self._connected = True
        logger.info(
            f"[MT5] Connected | login={info.login} | server={info.server} | "
            f"balance={info.balance:.2f} {info.currency}"
        )
        return True

    def disconnect(self) -> None:
        """Shut down MT5 connection."""
        if self._connected:
            _mt5().shutdown()
            self._connected = False
            logger.info("[MT5] Disconnected")

    def is_connected(self) -> bool:
        mt5 = _mt5()
        return self._connected and mt5.terminal_info() is not None

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account_info(self) -> Dict[str, Any]:
        mt5   = _mt5()
        info  = mt5.account_info()
        if info is None:
            raise RuntimeError(f"[MT5] account_info() failed: {mt5.last_error()}")
        return {
            "login":    info.login,
            "server":   info.server,
            "balance":  info.balance,
            "equity":   info.equity,
            "margin":   info.margin,
            "free_margin": info.margin_free,
            "currency": info.currency,
            "leverage": info.leverage,
        }

    # ------------------------------------------------------------------
    # Bar fetching
    # ------------------------------------------------------------------

    def fetch_bars(
        self,
        timeframe_str: str,
        count: int,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch the last `count` closed bars for the given timeframe.

        Args:
            timeframe_str: "1H" or "5M"
            count:         Number of bars to fetch
            symbol:        Override symbol (defaults to self.symbol)

        Returns:
            DataFrame with DatetimeIndex (UTC) and columns:
            Open, High, Low, Close, Volume
            Sorted ascending by time (oldest first).
        """
        mt5  = _mt5()
        sym  = symbol or self.symbol
        tf   = self._parse_timeframe(timeframe_str)

        # Fetch count+1 bars then drop the last (potentially open) bar
        rates = mt5.copy_rates_from_pos(sym, tf, 0, count + 1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"[MT5] copy_rates_from_pos failed for {sym} {timeframe_str}: "
                f"{mt5.last_error()}"
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time").sort_index()

        # Drop the last bar (it may still be forming)
        df = df.iloc[:-1]

        df = df.rename(columns={
            "open":       "Open",
            "high":       "High",
            "low":        "Low",
            "close":      "Close",
            "tick_volume": "Volume",
            "real_volume": "RealVolume",
        })
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        return df[cols].copy()

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def open_order(
        self,
        direction: str,        # "long" or "short"
        lots: float,
        sl_price: float,
        tp_price: float,
        magic: int,
        comment: str = "",
        symbol: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> Optional[int]:
        """
        Place a market order.

        Returns:
            MT5 ticket (position ID) on success, None on failure.
        """
        mt5  = _mt5()
        sym  = symbol or self.symbol

        order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
        tick       = mt5.symbol_info_tick(sym)
        if tick is None:
            logger.error(f"[MT5] symbol_info_tick({sym}) failed")
            return None

        price = tick.ask if direction == "long" else tick.bid

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       sym,
            "volume":       lots,
            "type":         order_type,
            "price":        price,
            "sl":           round(sl_price,  2),
            "tp":           round(tp_price,  2),
            "deviation":    20,
            "magic":        magic,
            "comment":      comment[:31],   # MT5 max 31 chars
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        for attempt in range(1, max_retries + 1):
            result = mt5.order_send(request)
            if result is None:
                logger.warning(f"[MT5] order_send returned None (attempt {attempt})")
                time.sleep(retry_delay_sec)
                continue
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    f"[MT5] Order opened | ticket={result.order} | "
                    f"dir={direction} | lots={lots:.2f} | price={result.price:.2f} | "
                    f"sl={sl_price:.2f} | tp={tp_price:.2f}"
                )
                return result.order
            logger.warning(
                f"[MT5] order_send failed (attempt {attempt}): "
                f"retcode={result.retcode} | comment={result.comment}"
            )
            time.sleep(retry_delay_sec)

        logger.error(f"[MT5] open_order failed after {max_retries} attempts")
        return None

    def close_position(
        self,
        ticket: int,
        magic: int,
        symbol: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_sec: float = 2.0,
    ) -> bool:
        """Close an open position by ticket."""
        mt5  = _mt5()
        sym  = symbol or self.symbol

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"[MT5] Position {ticket} not found for close")
            return False

        pos       = positions[0]
        lots      = pos.volume
        direction = pos.type  # 0=buy, 1=sell
        close_type = mt5.ORDER_TYPE_SELL if direction == 0 else mt5.ORDER_TYPE_BUY
        tick      = mt5.symbol_info_tick(sym)
        if tick is None:
            logger.error(f"[MT5] symbol_info_tick failed during close")
            return False

        price = tick.bid if direction == 0 else tick.ask

        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       sym,
            "volume":       lots,
            "type":         close_type,
            "position":     ticket,
            "price":        price,
            "deviation":    20,
            "magic":        magic,
            "comment":      "close",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        for attempt in range(1, max_retries + 1):
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"[MT5] Position {ticket} closed at {result.price:.2f}")
                return True
            code = result.retcode if result else "None"
            logger.warning(f"[MT5] close attempt {attempt} failed: retcode={code}")
            time.sleep(retry_delay_sec)

        logger.error(f"[MT5] close_position {ticket} failed after {max_retries} attempts")
        return False

    def modify_sl(
        self,
        ticket: int,
        new_sl: float,
        symbol: Optional[str] = None,
    ) -> bool:
        """Ratchet stop-loss for an open position (only moves in profit direction)."""
        mt5  = _mt5()
        sym  = symbol or self.symbol

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False

        pos = positions[0]

        # Only ratchet (never widen SL)
        if pos.type == 0:  # long
            if new_sl <= pos.sl:
                return True   # no move needed
        else:               # short
            if new_sl >= pos.sl:
                return True

        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   sym,
            "position": ticket,
            "sl":       round(new_sl, 2),
            "tp":       pos.tp,
        }
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.debug(f"[MT5] SL modified ticket={ticket} new_sl={new_sl:.2f}")
            return True
        code = result.retcode if result else "None"
        logger.warning(f"[MT5] modify_sl failed: retcode={code}")
        return False

    def get_positions(self, magic: Optional[int] = None, symbol: Optional[str] = None) -> List[Dict]:
        """Return list of open positions, optionally filtered by magic and/or symbol."""
        mt5  = _mt5()
        sym  = symbol or self.symbol
        raw  = mt5.positions_get(symbol=sym)
        if raw is None:
            return []

        result = []
        for p in raw:
            if magic is not None and p.magic != magic:
                continue
            result.append({
                "ticket":     p.ticket,
                "magic":      p.magic,
                "symbol":     p.symbol,
                "type":       "long" if p.type == 0 else "short",
                "volume":     p.volume,
                "open_price": p.price_open,
                "sl":         p.sl,
                "tp":         p.tp,
                "open_time":  datetime.fromtimestamp(p.time, tz=timezone.utc),
                "profit":     p.profit,
                "comment":    p.comment,
            })
        return result

    def get_deal_exit_price(self, ticket: int) -> Optional[float]:
        """
        Look up the actual exit price for a closed position from MT5 deal history.

        Searches the last 30 days of history for a deal that closes the given position ticket.
        Returns the deal price, or None if not found.
        """
        mt5 = _mt5()
        from datetime import timedelta
        date_to   = datetime.now(tz=timezone.utc)
        date_from = date_to - timedelta(days=30)

        deals = mt5.history_deals_get(date_from, date_to, position=ticket)
        if not deals:
            return None

        # The closing deal has DEAL_ENTRY_OUT (value 1)
        DEAL_ENTRY_OUT = 1
        for deal in deals:
            if deal.entry == DEAL_ENTRY_OUT:
                return deal.price

        return None

    def get_symbol_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        mt5  = _mt5()
        sym  = symbol or self.symbol
        info = mt5.symbol_info(sym)
        if info is None:
            raise RuntimeError(f"[MT5] symbol_info({sym}) failed: {mt5.last_error()}")
        return {
            "point":        info.point,
            "digits":       info.digits,
            "trade_contract_size": info.trade_contract_size,
            "volume_min":   info.volume_min,
            "volume_max":   info.volume_max,
            "volume_step":  info.volume_step,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_timeframe(tf_str: str) -> int:
        """Convert "1H" / "5M" string to MT5 TIMEFRAME_* constant."""
        mt5 = _mt5()
        mapping = {
            "1M":  mt5.TIMEFRAME_M1,
            "5M":  mt5.TIMEFRAME_M5,
            "15M": mt5.TIMEFRAME_M15,
            "30M": mt5.TIMEFRAME_M30,
            "1H":  mt5.TIMEFRAME_H1,
            "4H":  mt5.TIMEFRAME_H4,
            "1D":  mt5.TIMEFRAME_D1,
        }
        key = tf_str.upper()
        if key not in mapping:
            raise ValueError(f"Unknown timeframe '{tf_str}'. Known: {list(mapping)}")
        return mapping[key]
