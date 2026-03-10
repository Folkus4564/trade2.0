"""
Module: broker.py
Purpose: Abstract BrokerWrapper — skeleton for future broker integration
All methods raise NotImplementedError.
"""

from typing import Any, Dict, List, Optional


class BrokerWrapper:
    """
    Abstract interface for broker connectivity.
    Implement a subclass for each broker (Oanda, MT5, IBKR, etc.)
    """

    def connect(self, config: Dict[str, Any]) -> None:
        """Establish connection to broker API."""
        raise NotImplementedError("connect() not implemented")

    def disconnect(self) -> None:
        """Close connection to broker API."""
        raise NotImplementedError("disconnect() not implemented")

    def get_account_info(self) -> Dict[str, Any]:
        """Return account balance, equity, margin info."""
        raise NotImplementedError("get_account_info() not implemented")

    def place_order(
        self,
        instrument: str,
        direction:  str,      # "long" or "short"
        size:       float,    # notional in account currency
        stop_loss:  Optional[float] = None,
        take_profit: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """
        Place a market or limit order.

        Returns:
            Order confirmation dict with order_id, fill_price, etc.
        """
        raise NotImplementedError("place_order() not implemented")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID. Returns True if successful."""
        raise NotImplementedError("cancel_order() not implemented")

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return list of open positions."""
        raise NotImplementedError("get_positions() not implemented")

    def close_position(self, position_id: str) -> Dict[str, Any]:
        """Close an open position by ID."""
        raise NotImplementedError("close_position() not implemented")

    def get_current_price(self, instrument: str) -> Dict[str, float]:
        """Return current bid/ask for instrument."""
        raise NotImplementedError("get_current_price() not implemented")
