import logging

import requests

from nba_bot.config import (
    DATA_API,
    HEADERS,
    MAX_SLIPPAGE_PCT,
    SLIPPAGE_BASE,
    SLIPPAGE_IMPACT_FACTOR,
)

logger = logging.getLogger(__name__)


def _extract_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "items", "results", "trades"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _trade_asset_id(trade: dict) -> str | None:
    for key in ("asset", "asset_id", "token_id", "tokenId"):
        value = trade.get(key)
        if value:
            return str(value)
    return None


def fetch_recent_trades(asset_id: str | None = None, limit: int = 100) -> list[dict]:
    try:
        params = {"limit": limit}
        if asset_id:
            params["asset"] = asset_id
        resp = requests.get(
            f"{DATA_API}/trades",
            params=params,
            headers=HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        trades = _extract_items(resp.json())
        if asset_id:
            return [trade for trade in trades if _trade_asset_id(trade) == str(asset_id)]
        return trades
    except Exception as e:
        logger.error("Data API /trades error: %s", e)
        return []


def fetch_open_interest() -> dict:
    try:
        resp = requests.get(f"{DATA_API}/oi", headers=HEADERS, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload
        return {"data": payload}
    except Exception as e:
        logger.error("Data API /oi error: %s", e)
        return {}


def estimate_slippage_from_trades(
    asset_id: str,
    target_size: float,
    side: str,
    enter_price: float,
    recent_trades: list[dict] | None = None,
) -> tuple[float | None, float]:
    if recent_trades is None:
        recent_trades = fetch_recent_trades(asset_id=asset_id, limit=500)

    asset_trades = [trade for trade in recent_trades if _trade_asset_id(trade) == str(asset_id)]
    if not asset_trades:
        logger.warning("No trades found for asset %s, using default slippage", asset_id)
        return None, min(max(SLIPPAGE_BASE * 4, SLIPPAGE_BASE), MAX_SLIPPAGE_PCT)

    total_notional = 0.0
    total_shares = 0.0
    valid_trade_count = 0
    for trade in asset_trades:
        try:
            price = float(trade.get("price", 0) or 0)
            size = float(trade.get("size", trade.get("amount", 0)) or 0)
        except (TypeError, ValueError):
            continue
        if price <= 0 or price >= 1 or size <= 0:
            continue
        total_notional += price * size
        total_shares += size
        valid_trade_count += 1

    if total_shares <= 0 or valid_trade_count <= 0:
        return None, min(max(SLIPPAGE_BASE * 4, SLIPPAGE_BASE), MAX_SLIPPAGE_PCT)

    vwap = total_notional / total_shares
    avg_trade_size = total_shares / valid_trade_count
    size_ratio = target_size / avg_trade_size if avg_trade_size > 0 else 1.0
    size_ratio = max(size_ratio, 1.0)
    slippage = SLIPPAGE_BASE + (size_ratio * SLIPPAGE_IMPACT_FACTOR * 0.01)
    slippage = min(max(slippage, SLIPPAGE_BASE), MAX_SLIPPAGE_PCT)

    reference_price = max(enter_price, vwap) if 0 < enter_price < 1 else vwap
    estimated_price = reference_price * (1 + slippage)

    estimated_price = max(0.02, min(0.98, estimated_price))
    logger.debug(
        "Trade slippage estimate — asset_id=%s | side=%s | enter_price=%.4f | vwap=%.4f | reference=%.4f | target_size=%.4f | avg_trade_size=%.4f | slippage=%.4f | estimated=%.4f",
        asset_id,
        side,
        enter_price,
        vwap,
        reference_price,
        target_size,
        avg_trade_size,
        slippage,
        estimated_price,
    )
    return estimated_price, slippage


def check_data_api_available() -> bool:
    try:
        resp = requests.get(f"{DATA_API}/oi", headers=HEADERS, timeout=5)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error("Data API unavailable: %s", e)
        return False
