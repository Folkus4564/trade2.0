"""
live/health.py - Connection monitoring, auto-reconnect, and weekend detection.

Implements:
- Weekend skip (Sat/Sun UTC)
- MT5 connection health check
- Exponential backoff reconnection (5s -> 15s -> 60s -> 300s)
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from trade2.live.mt5_connector import MT5Connector

logger = logging.getLogger(__name__)

# Backoff sequence in seconds
_BACKOFF_STEPS = [5, 15, 60, 300]


class HealthMonitor:
    """
    Monitors MT5 connection health and provides weekend-skip logic.

    Usage in main loop:
        health = HealthMonitor(connector)
        while True:
            if health.is_weekend():
                health.sleep_until_monday()
                continue
            if not health.ensure_connected():
                continue   # still not connected, wait and retry
            # ... normal trading logic
    """

    def __init__(self, connector: MT5Connector, heartbeat_interval_sec: int = 60):
        self.connector  = connector
        self.heartbeat_interval = heartbeat_interval_sec
        self._backoff_idx = 0
        self._last_heartbeat: Optional[float] = None

    # ------------------------------------------------------------------
    # Weekend detection
    # ------------------------------------------------------------------

    @staticmethod
    def is_weekend() -> bool:
        """Return True when markets are closed (Saturday or Sunday UTC)."""
        now = datetime.now(tz=timezone.utc)
        return now.weekday() >= 5   # 5=Saturday, 6=Sunday

    @staticmethod
    def sleep_until_monday(poll_sec: int = 300) -> None:
        """Sleep in chunks until Monday UTC, logging progress."""
        while True:
            now = datetime.now(tz=timezone.utc)
            if now.weekday() < 5:
                logger.info("[Health] Markets reopened — resuming")
                return
            # Compute seconds until Monday 00:00 UTC
            days_left = (7 - now.weekday()) % 7  # days until Monday
            if days_left == 0:
                days_left = 7
            target    = now.replace(hour=0, minute=0, second=0, microsecond=0)
            wait_sec  = max(poll_sec, 60)
            logger.info(f"[Health] Weekend — sleeping {wait_sec}s (checking for Monday)")
            time.sleep(wait_sec)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def ensure_connected(self) -> bool:
        """
        Check connection and attempt reconnect with exponential backoff if needed.

        Returns True if connected (either already was, or successfully reconnected).
        """
        if self.connector.is_connected():
            self._backoff_idx = 0   # reset on success
            return True

        backoff = _BACKOFF_STEPS[min(self._backoff_idx, len(_BACKOFF_STEPS) - 1)]
        logger.warning(f"[Health] MT5 disconnected — reconnecting in {backoff}s ...")
        time.sleep(backoff)
        self._backoff_idx += 1

        ok = self.connector.connect()
        if ok:
            logger.info("[Health] Reconnected to MT5")
            self._backoff_idx = 0
        else:
            logger.error("[Health] Reconnect failed")
        return ok

    def heartbeat(self) -> None:
        """Log periodic heartbeat (every heartbeat_interval seconds)."""
        now = time.monotonic()
        if self._last_heartbeat is None or (now - self._last_heartbeat) >= self.heartbeat_interval:
            self._last_heartbeat = now
            try:
                info = self.connector.get_account_info()
                logger.info(
                    f"[Health] Heartbeat | equity={info['equity']:.2f} | "
                    f"free_margin={info['free_margin']:.2f}"
                )
            except Exception as e:
                logger.warning(f"[Health] Heartbeat failed: {e}")
