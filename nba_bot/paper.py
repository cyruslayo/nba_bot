"""
nba_bot/paper.py
================
JSON-backed paper trading engine.

Provides:
  init_bankroll(initial_amount, reset_if_exists)  — create / reset bankroll file
  load_trades()                                    — read paper_trades.json
  save_trades(trades)                              — atomic write to paper_trades.json
  has_active_position(market_id)                   — idempotency guard
  execute_paper_trade(alert)                       — log a new PENDING trade
"""

import json
import logging
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from nba_bot.config import (
    CONVERGENCE_TARGET_PCT,
    DEFAULT_BANKROLL,
    DISABLE_STOCHASTIC_FILL,
    FILL_PROB_BASE,
    FILL_PROB_EDGE_BONUS,
    FILL_PROB_LIQUIDITY_SCALE,
    LATENCY_MAX_MS,
    LATENCY_MEAN_MS,
    LATENCY_STD_MS,
    LIVE_PAPER_TRADING_ENABLED,
    MAX_CONCURRENT_POSITIONS,
    MAX_FIRST_HALF_EXPOSURE_PCT,
    MAX_GAME_EXPOSURE_PCT,
    MAX_MONEYLINE_EXPOSURE_PCT,
    MAX_OTHER_EXPOSURE_PCT,
    MAX_SLIPPAGE_PCT,
    MAX_SPREAD_EXPOSURE_PCT,
    MAX_STAKE_LIQUIDITY_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    MIN_EXIT_EDGE,
    PAPER_BANKROLL_PATH,
    PAPER_TRADES_PATH,
    PLATFORM_FEE_RATE,
    PRICE_DRIFT_STD,
    PRICE_UPDATE_INTERVAL_SEC,
    PRE_GAME_EXIT_MINUTES,
    SLIPPAGE_BASE,
    SPREAD_CLUSTER_DISTANCE,
    STOP_LOSS_PCT,
    TOTAL_CLUSTER_DISTANCE,
)

logger = logging.getLogger(__name__)
PENDING_TRADE_MAX_AGE_HOURS = 6
HARDENED_EXECUTION_STATS: dict[str, int] = {}
ACTIVE_POSITION_STATUSES = {"PENDING", "OPEN"}


def _normalize_model_key(model_key: str | None) -> str | None:
    if model_key is None:
        return None
    normalized = str(model_key).strip()
    return normalized or None


def _trade_model_key(trade: dict) -> str | None:
    return _normalize_model_key(trade.get("model_key"))


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_active_status(status: str | None) -> bool:
    return str(status or "").upper() in ACTIVE_POSITION_STATUSES


def count_active_positions(model_key: str | None = None) -> int:
    normalized_model_key = _normalize_model_key(model_key)
    count = 0
    for trade in load_trades():
        if not _is_active_status(trade.get("status")):
            continue
        if normalized_model_key is not None and _trade_model_key(trade) != normalized_model_key:
            continue
        count += 1
    return count


def _load_bankroll_payload() -> dict:
    if not os.path.exists(PAPER_BANKROLL_PATH):
        return {}
    with open(PAPER_BANKROLL_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload if isinstance(payload, dict) else {}


def load_model_bankrolls() -> dict[str, float]:
    payload = _load_bankroll_payload()
    raw_bankrolls = payload.get("bankrolls")
    if not isinstance(raw_bankrolls, dict):
        return {}

    bankrolls: dict[str, float] = {}
    for key, value in raw_bankrolls.items():
        normalized_key = _normalize_model_key(str(key))
        if normalized_key is None:
            continue
        try:
            bankrolls[normalized_key] = float(value)
        except (TypeError, ValueError):
            continue
    return bankrolls


def _total_model_bankroll(bankrolls: dict[str, float]) -> float:
    return round(sum(float(amount) for amount in bankrolls.values()), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Atomic JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write(path: str, data) -> None:
    """Write *data* as JSON to *path* atomically via a .tmp file + os.replace."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Bankroll management
# ─────────────────────────────────────────────────────────────────────────────

def init_bankroll(initial_amount: float, reset_if_exists: bool = False, model_keys: list[str] | None = None) -> None:
    """
    Initialise the bankroll JSON file.

    If *reset_if_exists* is True (or the file doesn't exist yet),
    the bankroll is written (or overwritten) to *initial_amount*.
    If the file already exists and *reset_if_exists* is False, this is a no-op
    so that successive runs without --bankroll preserve accumulated P&L.
    """
    normalized_model_keys = []
    for model_key in model_keys or []:
        normalized_key = _normalize_model_key(model_key)
        if normalized_key is not None and normalized_key not in normalized_model_keys:
            normalized_model_keys.append(normalized_key)

    if os.path.exists(PAPER_BANKROLL_PATH) and not reset_if_exists:
        if not normalized_model_keys:
            logger.debug("Bankroll file exists; skipping init (reset_if_exists=False).")
            return

        payload = _load_bankroll_payload()
        bankrolls = load_model_bankrolls()
        updated = False
        try:
            seed_amount = float(payload.get("bankroll", initial_amount))
        except (TypeError, ValueError):
            seed_amount = float(initial_amount)

        for model_key in normalized_model_keys:
            if model_key not in bankrolls:
                bankrolls[model_key] = round(seed_amount, 2)
                updated = True

        if updated:
            payload["bankrolls"] = bankrolls
            payload["bankroll"] = _total_model_bankroll(bankrolls)
            _atomic_write(PAPER_BANKROLL_PATH, payload)
            logger.info("Model bankrolls initialised at $%.2f → %s", seed_amount, PAPER_BANKROLL_PATH)
        else:
            logger.debug("Requested model bankrolls already exist; skipping init.")
        return

    payload = {"bankroll": round(initial_amount, 2)}
    if normalized_model_keys:
        bankrolls = {
            model_key: round(initial_amount, 2)
            for model_key in normalized_model_keys
        }
        payload["bankrolls"] = bankrolls
        payload["bankroll"] = _total_model_bankroll(bankrolls)
    _atomic_write(PAPER_BANKROLL_PATH, payload)
    logger.info("Bankroll initialised at $%.2f → %s", initial_amount, PAPER_BANKROLL_PATH)


def _load_bankroll(model_key: str | None = None) -> float:
    """Read current bankroll from JSON. Returns DEFAULT_BANKROLL if file missing."""
    normalized_model_key = _normalize_model_key(model_key)
    bankrolls = load_model_bankrolls()
    if normalized_model_key is not None:
        if normalized_model_key in bankrolls:
            return bankrolls[normalized_model_key]
        if bankrolls:
            return DEFAULT_BANKROLL

    payload = _load_bankroll_payload()
    if payload:
        try:
            if "bankroll" in payload:
                return float(payload.get("bankroll", DEFAULT_BANKROLL))
        except (TypeError, ValueError):
            pass

    if normalized_model_key is None and bankrolls:
        return _total_model_bankroll(bankrolls)

    return DEFAULT_BANKROLL


def _save_bankroll(amount: float, model_key: str | None = None) -> None:
    """Atomically persist updated bankroll."""
    payload = _load_bankroll_payload()
    normalized_model_key = _normalize_model_key(model_key)
    if normalized_model_key is None:
        payload["bankroll"] = round(amount, 2)
    else:
        bankrolls = load_model_bankrolls()
        bankrolls[normalized_model_key] = round(amount, 2)
        payload["bankrolls"] = bankrolls
        payload["bankroll"] = _total_model_bankroll(bankrolls)
    _atomic_write(PAPER_BANKROLL_PATH, payload)


# ─────────────────────────────────────────────────────────────────────────────
# Trade log management
# ─────────────────────────────────────────────────────────────────────────────

def load_trades() -> list[dict]:
    """Read paper_trades.json; returns an empty list if file doesn't exist yet."""
    if not os.path.exists(PAPER_TRADES_PATH):
        return []
    with open(PAPER_TRADES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def save_trades(trades: list[dict]) -> None:
    """Atomically persist updated trade list to paper_trades.json."""
    _atomic_write(PAPER_TRADES_PATH, trades)


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency guard
# ─────────────────────────────────────────────────────────────────────────────

def has_active_position(market_id: str, model_key: str | None = None) -> bool:
    """
    Returns True if there is already a PENDING paper trade for *market_id*.
    Used to prevent duplicate entries across scanner loops.
    """
    normalized_model_key = _normalize_model_key(model_key)
    for trade in load_trades():
        if trade.get("market_id") != market_id or not _is_active_status(trade.get("status")):
            continue
        if normalized_model_key is not None and _trade_model_key(trade) != normalized_model_key:
            continue
        return True
    return False


def _pending_cutoff_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=PENDING_TRADE_MAX_AGE_HOURS)


def _is_recent_pending_trade(trade: dict, cutoff_time: datetime | None = None) -> bool:
    if not _is_active_status(trade.get("status")):
        return False

    cutoff_time = cutoff_time or _pending_cutoff_time()
    timestamp = trade.get("timestamp")
    if timestamp:
        try:
            trade_time = datetime.fromisoformat(timestamp)
            if trade_time.tzinfo is None:
                trade_time = trade_time.replace(tzinfo=timezone.utc)
            if trade_time < cutoff_time:
                return False
        except (TypeError, ValueError):
            pass

    return True


def classify_market_bucket(market: str) -> str:
    market_text = (market or "").lower()
    if "1h" in market_text or "1st half" in market_text or "first half" in market_text:
        return "first_half"
    if "spread" in market_text:
        return "spread"
    if "o/u" in market_text or "total" in market_text or "over/under" in market_text:
        return "total"
    if "moneyline" in market_text or ("vs." in market_text and ":" not in market_text):
        return "moneyline"
    return "other"


def _extract_line_value(market: str) -> float | None:
    numeric_tokens = re.findall(r"[-+]?\d+(?:\.\d+)?", market or "")
    if not numeric_tokens:
        return None
    try:
        return float(numeric_tokens[-1])
    except ValueError:
        return None


def _market_family_key(market: str) -> str:
    normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", " ", (market or "").lower())
    normalized = normalized.replace("(", " ").replace(")", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip(" :")


def _bucket_exposure_pct(bucket: str) -> float:
    if bucket == "moneyline":
        return MAX_MONEYLINE_EXPOSURE_PCT
    if bucket == "spread":
        return MAX_SPREAD_EXPOSURE_PCT
    if bucket == "total":
        return MAX_TOTAL_EXPOSURE_PCT
    if bucket == "first_half":
        return MAX_FIRST_HALF_EXPOSURE_PCT
    return MAX_OTHER_EXPOSURE_PCT


def get_game_exposure(event_slug: str, bucket: str | None = None, model_key: str | None = None) -> float:
    total = 0.0
    cutoff_time = _pending_cutoff_time()
    normalized_model_key = _normalize_model_key(model_key)

    for trade in load_trades():
        if trade.get("event_slug") != event_slug:
            continue
        if normalized_model_key is not None and _trade_model_key(trade) != normalized_model_key:
            continue
        if bucket is not None:
            trade_bucket = trade.get("bucket") or classify_market_bucket(trade.get("market", ""))
            if trade_bucket != bucket:
                continue
        if not _is_recent_pending_trade(trade, cutoff_time=cutoff_time):
            continue

        total += float(trade.get("stake", 0) or 0)

    return total


def find_clustered_trade(event_slug: str, market: str, bucket: str, market_id: str, cluster_distance: float, model_key: str | None = None) -> dict | None:
    if bucket not in {"spread", "total"} or cluster_distance <= 0:
        return None

    line_value = _extract_line_value(market)
    if line_value is None:
        return None

    family_key = _market_family_key(market)
    cutoff_time = _pending_cutoff_time()
    normalized_model_key = _normalize_model_key(model_key)
    for trade in load_trades():
        if str(trade.get("market_id")) == str(market_id):
            continue
        if trade.get("event_slug") != event_slug:
            continue
        if normalized_model_key is not None and _trade_model_key(trade) != normalized_model_key:
            continue
        if not _is_recent_pending_trade(trade, cutoff_time=cutoff_time):
            continue

        trade_bucket = trade.get("bucket") or classify_market_bucket(trade.get("market", ""))
        if trade_bucket != bucket:
            continue
        if _market_family_key(trade.get("market", "")) != family_key:
            continue

        trade_line = trade.get("line_value")
        if trade_line is None:
            trade_line = _extract_line_value(trade.get("market", ""))
        try:
            trade_line_value = float(trade_line)
        except (TypeError, ValueError):
            continue

        if abs(trade_line_value - line_value) <= cluster_distance:
            return trade

    return None


def _record_hardened_decision(reason: str, alert: dict, level: str = "info", **fields) -> None:
    HARDENED_EXECUTION_STATS[reason] = HARDENED_EXECUTION_STATS.get(reason, 0) + 1

    payload = {
        "count": HARDENED_EXECUTION_STATS[reason],
        "model_key": alert.get("model_key"),
        "model_label": alert.get("model_label"),
        "market_id": alert.get("market_id"),
        "event_slug": alert.get("event_slug", ""),
        "market": alert.get("market", ""),
        "direction": alert.get("direction", ""),
    }
    payload.update(fields)
    details = " | ".join(
        f"{key}={value}"
        for key, value in payload.items()
        if value not in (None, "")
    )
    getattr(logger, level, logger.info)("Hardened decision — reason=%s | %s", reason, details)


def _clock_to_seconds(clock: str | None) -> int | None:
    if not clock or ":" not in str(clock):
        return None
    try:
        minutes, seconds = str(clock).split(":", 1)
        return max(int(minutes), 0) * 60 + max(int(seconds), 0)
    except (TypeError, ValueError):
        return None


def _game_near_completion(game: dict | None, bucket: str) -> bool:
    if not game:
        return False
    remaining = _clock_to_seconds(game.get("clock"))
    if remaining is None:
        return False
    threshold = PRE_GAME_EXIT_MINUTES * 60
    try:
        period = int(game.get("period", 0) or 0)
    except (TypeError, ValueError):
        return False
    if bucket == "first_half":
        return period >= 2 and remaining <= threshold
    return period >= 4 and remaining <= threshold


def _resolve_trade_price(
    trade: dict,
    market_lookup: dict[str, dict] | None = None,
    price_cache: dict | None = None,
) -> tuple[float | None, float, str]:
    token_id = trade.get("trade_token")
    if token_id and price_cache is not None:
        cached_price = _coerce_float(price_cache.get(token_id), default=-1.0)
        if 0 < cached_price < 1:
            return cached_price, _coerce_float(trade.get("liquidity")), "ws"

    if token_id:
        from nba_bot.polymarket import fetch_clob_midpoint

        midpoint = fetch_clob_midpoint(token_id)
        if midpoint is not None:
            return midpoint, _coerce_float(trade.get("liquidity")), "clob"

    market = None
    if market_lookup is not None:
        market = market_lookup.get(str(trade.get("market_id")))
    if not market:
        return None, _coerce_float(trade.get("liquidity")), "missing"

    yes_token = market.get("yes_token") or market.get("clob_yes_id")
    no_token = market.get("no_token") or market.get("clob_no_id")
    price = None
    if token_id and str(token_id) == str(yes_token):
        price = _coerce_float(market.get("yes_price"), default=0.0)
    elif token_id and str(token_id) == str(no_token):
        price = _coerce_float(market.get("no_price"), default=0.0)
    return (price if 0 < (price or 0.0) < 1 else None), _coerce_float(market.get("liquidity")), "gamma"


def _entry_fill_price(trade: dict) -> float:
    return _coerce_float(trade.get("final_price") or trade.get("adjusted_price") or trade.get("enter_price"))


def _compute_target_exit_price(trade: dict) -> float:
    model_prob = _coerce_float(trade.get("model_prob"))
    entry_price = _entry_fill_price(trade)
    stored_target = _coerce_float(trade.get("target_exit_price"), default=0.0)
    if stored_target > 0:
        return stored_target
    if model_prob <= 0 or entry_price <= 0:
        return entry_price
    return round(entry_price + max(model_prob - entry_price, 0.0) * CONVERGENCE_TARGET_PCT, 4)


def _should_exit_trade(trade: dict, current_price: float, current_edge: float | None, game: dict | None) -> str | None:
    entry_price = _entry_fill_price(trade)
    if entry_price <= 0 or current_price <= 0:
        return None
    if current_price <= entry_price * (1.0 - STOP_LOSS_PCT):
        return "STOP_LOSS"
    target_exit_price = _compute_target_exit_price(trade)
    if target_exit_price > 0 and current_price >= target_exit_price:
        return "CONVERGENCE_TARGET"
    if current_edge is not None and current_edge <= MIN_EXIT_EDGE:
        return "EDGE_THRESHOLD"
    if _game_near_completion(game, trade.get("bucket") or classify_market_bucket(trade.get("market", ""))):
        return "TIME_BASED"
    return None


def _close_trade(trade: dict, exit_price: float, exit_reason: str, liquidity: float, price_source: str) -> tuple[dict, float]:
    shares = _coerce_float(trade.get("shares"))
    stake = _coerce_float(trade.get("stake"))
    notional = max(shares * exit_price, 0.0)
    adjusted_exit_price, exit_slippage = apply_slippage(
        exit_price,
        "SELL",
        notional,
        liquidity,
        trade.get("trade_token"),
        trade_side="SELL",
    )
    final_exit_price, exit_latency_ms = simulate_latency_and_drift(adjusted_exit_price)
    payout = round(max(shares * final_exit_price, 0.0), 2)
    fee = 0.0
    profit = round(payout - stake - fee, 2)
    updated = dict(trade)
    updated.update({
        "status": "CLOSED",
        "current_price": round(exit_price, 4),
        "exit_price": round(final_exit_price, 4),
        "exit_timestamp": _current_timestamp(),
        "exit_reason": exit_reason,
        "exit_slippage": round(exit_slippage, 4),
        "exit_latency_ms": round(exit_latency_ms, 1),
        "price_source": price_source,
        "payout": payout,
        "profit": profit,
        "realized_pnl": profit,
        "unrealized_pnl": 0.0,
        "closed_before_resolution": True,
    })
    return updated, payout


def monitor_live_positions(
    markets: list[dict],
    games_by_event_slug: dict[str, dict] | None = None,
    price_cache: dict | None = None,
) -> dict[str, float]:
    summary = {
        "checked": 0,
        "closed": 0,
        "missing_price": 0,
        "realized_pnl": 0.0,
    }
    if not LIVE_PAPER_TRADING_ENABLED:
        return summary

    trades = load_trades()
    if not trades:
        return summary

    market_lookup = {str(market.get("market_id")): market for market in markets}
    bankroll_deltas: dict[str | None, float] = defaultdict(float)
    changed = False
    now = datetime.now(timezone.utc)

    for index, trade in enumerate(trades):
        if not bool(trade.get("hardened", False)):
            continue
        if not _is_active_status(trade.get("status")):
            continue
        if _coerce_float(trade.get("shares")) <= 0:
            continue

        last_check = _parse_iso_datetime(trade.get("last_price_check"))
        if last_check is not None and (now - last_check).total_seconds() < PRICE_UPDATE_INTERVAL_SEC:
            continue

        summary["checked"] += 1
        current_price, liquidity, price_source = _resolve_trade_price(trade, market_lookup=market_lookup, price_cache=price_cache)
        if current_price is None:
            summary["missing_price"] += 1
            continue

        model_prob = _coerce_float(trade.get("model_prob"), default=-1.0)
        current_edge = None if model_prob <= 0 else round(model_prob - current_price, 4)
        unrealized_pnl = round(_coerce_float(trade.get("shares")) * current_price - _coerce_float(trade.get("stake")), 2)
        updated_trade = dict(trade)
        updated_trade.update({
            "status": "OPEN",
            "current_price": round(current_price, 4),
            "current_edge": current_edge,
            "unrealized_pnl": unrealized_pnl,
            "last_price_check": now.isoformat(),
            "liquidity": liquidity,
            "price_source": price_source,
            "target_exit_price": _compute_target_exit_price(trade),
            "max_favorable_price": round(max(_coerce_float(trade.get("max_favorable_price"), default=current_price), current_price), 4),
            "max_adverse_price": round(min(_coerce_float(trade.get("max_adverse_price"), default=current_price), current_price), 4),
        })

        exit_reason = _should_exit_trade(updated_trade, current_price, current_edge, (games_by_event_slug or {}).get(trade.get("event_slug", "")))
        if exit_reason is not None:
            closed_trade, payout = _close_trade(updated_trade, current_price, exit_reason, liquidity, price_source)
            trades[index] = closed_trade
            bankroll_deltas[_trade_model_key(trade)] += payout
            summary["closed"] += 1
            summary["realized_pnl"] += _coerce_float(closed_trade.get("realized_pnl"))
            changed = True
            print(
                f"  LIVE PAPER EXIT: {closed_trade.get('direction')} | "
                f"{closed_trade.get('market', closed_trade.get('market_id'))[:40]} | "
                f"Reason: {exit_reason} | Exit: {closed_trade.get('exit_price'):.4f} | "
                f"P&L: ${_coerce_float(closed_trade.get('realized_pnl')):+.2f}"
            )
            continue

        trades[index] = updated_trade
        changed = True

    if changed:
        save_trades(trades)
    for model_key, payout in bankroll_deltas.items():
        if payout <= 0:
            continue
        _save_bankroll(round(_load_bankroll(model_key=model_key) + payout, 2), model_key=model_key)
    summary["realized_pnl"] = round(summary["realized_pnl"], 2)
    return summary


def compute_fill_probability(liquidity: float, edge: float) -> float:
    liquidity_signal = max(liquidity, 0.0) * max(FILL_PROB_LIQUIDITY_SCALE, 0.0) * 10.0
    liquidity_factor = liquidity_signal / (1.0 + liquidity_signal)
    probability = 0.05 + max(FILL_PROB_BASE - 0.05, 0.0) * liquidity_factor
    probability += abs(edge) * FILL_PROB_EDGE_BONUS
    return min(max(probability, 0.05), 0.95)


def apply_slippage(
    enter_price: float,
    direction: str,
    stake: float,
    liquidity: float,
    asset_id: str | None = None,
    trade_side: str = "BUY",
) -> tuple[float, float]:
    if enter_price <= 0 or enter_price >= 1:
        return enter_price, SLIPPAGE_BASE

    def _clamp_slippage(raw_slippage: float, source: str) -> float:
        clamped = min(max(raw_slippage, SLIPPAGE_BASE), MAX_SLIPPAGE_PCT)
        if clamped < raw_slippage:
            logger.warning(
                "Slippage capped — source=%s | asset_id=%s | raw=%.4f | capped=%.4f | enter_price=%.4f | stake=%.2f",
                source,
                asset_id,
                raw_slippage,
                clamped,
                enter_price,
                stake,
            )
        return clamped

    def _cap_price_move(candidate_price: float, source: str) -> tuple[float, float]:
        candidate_price = min(max(candidate_price, 0.02), 0.98)
        raw_slippage = abs(candidate_price - enter_price) / max(enter_price, 0.01)
        capped_slippage = _clamp_slippage(raw_slippage, source)

        if candidate_price >= enter_price:
            adjusted_price = enter_price * (1 + capped_slippage)
        else:
            adjusted_price = enter_price * (1 - capped_slippage)

        adjusted_price = min(max(adjusted_price, 0.02), 0.98)
        return adjusted_price, capped_slippage

    if asset_id:
        from nba_bot.polymarket import compute_vwap_from_book, fetch_order_book

        book = fetch_order_book(asset_id)
        if book:
            vwap_price, filled_shares = compute_vwap_from_book(book, trade_side, stake)
            if vwap_price is not None:
                adjusted_price, slippage = _cap_price_move(vwap_price, "order_book")
                logger.debug(
                    "Order-book fill estimate — asset_id=%s | direction=%s | trade_side=%s | enter_price=%.4f | vwap=%.4f | adjusted=%.4f | slippage=%.4f | filled_shares=%.4f",
                    asset_id,
                    direction,
                    trade_side,
                    enter_price,
                    vwap_price,
                    adjusted_price,
                    slippage,
                    filled_shares,
                )
                return adjusted_price, slippage
            logger.info(
                "Insufficient order book depth for hardened fill — asset_id=%s | direction=%s | stake=%.2f | filled_shares=%.4f",
                asset_id,
                direction,
                stake,
                filled_shares,
            )

    if asset_id and trade_side == "BUY":
        from nba_bot.market_analytics import estimate_slippage_from_trades

        shares = stake / enter_price
        adjusted_price, slippage = estimate_slippage_from_trades(
            asset_id,
            shares,
            direction,
            enter_price,
        )
        if adjusted_price is not None:
            adjusted_price, slippage = _cap_price_move(adjusted_price, "trades")
            logger.debug(
                "Trade-history fill estimate — asset_id=%s | direction=%s | trade_side=%s | enter_price=%.4f | adjusted=%.4f | slippage=%.4f | shares=%.4f",
                asset_id,
                direction,
                trade_side,
                enter_price,
                adjusted_price,
                slippage,
                shares,
            )
            return adjusted_price, slippage

    base_liquidity = max(liquidity, 1.0)
    slippage = SLIPPAGE_BASE + (stake / base_liquidity) * 0.001
    slippage = _clamp_slippage(slippage, "liquidity")
    if trade_side == "SELL":
        adjusted_price = enter_price * (1 - slippage)
    else:
        adjusted_price = enter_price * (1 + slippage)
    adjusted_price = min(max(adjusted_price, 0.02), 0.98)
    slippage = abs(adjusted_price - enter_price) / max(enter_price, 0.01)
    slippage = min(max(slippage, SLIPPAGE_BASE), MAX_SLIPPAGE_PCT)
    return adjusted_price, slippage


def simulate_latency_and_drift(enter_price: float) -> tuple[float, float]:
    latency_ms = random.gauss(LATENCY_MEAN_MS, LATENCY_STD_MS)
    latency_ms = min(max(latency_ms, 0.0), LATENCY_MAX_MS)
    time.sleep(latency_ms / 1000.0)
    drift = random.gauss(0, PRICE_DRIFT_STD)
    drifted_price = enter_price * (1 + drift)
    drifted_price = min(max(drifted_price, 0.02), 0.98)
    return drifted_price, latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# Paper trade execution
# ─────────────────────────────────────────────────────────────────────────────

def execute_paper_trade(alert: dict) -> None:
    """
    Evaluate an alert and, if eligible, record a paper trade.

    Steps:
      1. Skip if a PENDING trade already exists for this market_id.
      2. Skip if current bankroll is <= 0.
      3. Compute stake = raw_stake * current_bankroll.
      4. Append PENDING trade; atomic-write paper_trades.json.
      5. Deduct stake from bankroll; atomic-write paper_bankroll.json.
      6. Print confirmation to console.
    """
    market_id = alert.get("market_id")
    direction = alert.get("direction", "")
    enter_price = alert.get("enter_price", 0.0)
    raw_stake = alert.get("raw_stake", 0.0)
    edge = alert.get("edge", 0.0)
    market = alert.get("market", "")
    bucket = alert.get("bucket") or classify_market_bucket(market)
    line_value = _extract_line_value(market)
    model_key = _normalize_model_key(alert.get("model_key"))
    model_label = alert.get("model_label")

    # 1. Idempotency check
    if has_active_position(market_id, model_key=model_key):
        logger.debug("Skipping paper trade — active position already open for market %s", market_id)
        return

    # 2. Bankroll guard
    bankroll = _load_bankroll(model_key=model_key)
    if bankroll <= 0:
        logger.warning("Bankroll is $%.2f — skipping paper trade.", bankroll)
        return

    # 3. Stake calculation
    stake = round(raw_stake * bankroll, 2)
    if stake <= 0:
        logger.debug("Computed stake is $0.00 — skipping paper trade.")
        return

    # 4. Record trade
    trades = load_trades()
    trades.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "market_id": market_id,
        "event_slug": alert.get("event_slug", ""),
        "market": market,
        "direction": direction,
        "bucket": bucket,
        "line_value": line_value,
        "model_key": model_key,
        "model_label": model_label,
        "enter_price": enter_price,
        "stake": stake,
        "edge": round(edge, 4),
        "status": "PENDING",
    })
    save_trades(trades)

    # 5. Deduct from bankroll (only after trade write succeeds)
    new_bankroll = round(bankroll - stake, 2)
    _save_bankroll(new_bankroll, model_key=model_key)

    # 6. Console confirmation
    print(
        f"\n  📝 PAPER TRADE EXECUTED: {direction} | "
        f"Stake: ${stake:.2f} | Price: {enter_price} | "
        f"Bankroll: ${new_bankroll:.2f}"
    )
    logger.info(
        "Paper trade executed — %s | market=%s | stake=$%.2f | enter_price=%s",
        direction, market_id, stake, enter_price,
    )


def execute_paper_trade_hardened(alert: dict, data_api_checked: bool = True) -> bool:
    market_id = alert.get("market_id")
    event_slug = alert.get("event_slug", "")
    direction = alert.get("direction", "")
    market = alert.get("market", "")
    bucket = alert.get("bucket") or classify_market_bucket(market)
    line_value = _extract_line_value(market)
    enter_price = float(alert.get("enter_price", 0.0) or 0.0)
    raw_stake = float(alert.get("raw_stake", 0.0) or 0.0)
    edge = float(alert.get("calibrated_edge", alert.get("edge", 0.0)) or 0.0)
    liquidity = float(alert.get("liquidity", 0.0) or 0.0)
    model_key = _normalize_model_key(alert.get("model_key"))
    model_label = alert.get("model_label")
    asset_id = alert.get("trade_token")
    if not asset_id:
        if "NO" in direction.upper():
            asset_id = alert.get("no_token") or alert.get("clob_no_id")
        else:
            asset_id = alert.get("yes_token") or alert.get("clob_yes_id")

    if not data_api_checked:
        from nba_bot.market_analytics import check_data_api_available

        if not check_data_api_available():
            _record_hardened_decision("data_api_unavailable", alert, level="warning")
            return False

    if has_active_position(market_id, model_key=model_key):
        _record_hardened_decision("active_position", alert, level="debug")
        return False

    bankroll = _load_bankroll(model_key=model_key)
    if bankroll <= 0:
        _record_hardened_decision("bankroll_nonpositive", alert, level="warning", bankroll=f"{bankroll:.2f}")
        return False

    active_positions = count_active_positions(model_key=model_key)
    if active_positions >= MAX_CONCURRENT_POSITIONS:
        _record_hardened_decision(
            "max_concurrent_positions",
            alert,
            active_positions=str(active_positions),
            max_positions=str(MAX_CONCURRENT_POSITIONS),
        )
        return False

    stake = round(raw_stake * bankroll, 2)
    if liquidity > 0:
        stake = min(stake, round(liquidity * MAX_STAKE_LIQUIDITY_PCT, 2))
    if stake <= 0:
        _record_hardened_decision("stake_zero", alert, level="debug")
        return False

    max_game_exposure = bankroll * MAX_GAME_EXPOSURE_PCT
    current_game_exposure = get_game_exposure(event_slug, model_key=model_key)
    if event_slug and current_game_exposure + stake > max_game_exposure:
        _record_hardened_decision(
            "event_exposure_cap",
            alert,
            current_exposure=f"{current_game_exposure:.2f}",
            stake=f"{stake:.2f}",
            max_exposure=f"{max_game_exposure:.2f}",
        )
        return False

    bucket_cap_pct = _bucket_exposure_pct(bucket)
    bucket_cap = bankroll * bucket_cap_pct
    current_bucket_exposure = get_game_exposure(event_slug, bucket=bucket, model_key=model_key)
    if event_slug and bucket_cap_pct > 0 and current_bucket_exposure + stake > bucket_cap:
        _record_hardened_decision(
            "bucket_exposure_cap",
            alert,
            bucket=bucket,
            current_bucket_exposure=f"{current_bucket_exposure:.2f}",
            stake=f"{stake:.2f}",
            bucket_cap=f"{bucket_cap:.2f}",
        )
        return False

    cluster_distance = SPREAD_CLUSTER_DISTANCE if bucket == "spread" else TOTAL_CLUSTER_DISTANCE if bucket == "total" else 0.0
    clustered_trade = find_clustered_trade(event_slug, market, bucket, market_id, cluster_distance, model_key=model_key)
    if clustered_trade is not None:
        _record_hardened_decision(
            "line_cluster_block",
            alert,
            bucket=bucket,
            cluster_distance=f"{cluster_distance:.1f}",
            blocked_by=clustered_trade.get("market_id"),
        )
        return False

    fill_prob = compute_fill_probability(liquidity, edge)
    if not DISABLE_STOCHASTIC_FILL and random.random() > fill_prob:
        _record_hardened_decision("fill_probability_reject", alert, fill_prob=f"{fill_prob:.3f}")
        print(f"  HARDENED TRADE NOT FILLED: {direction} | Fill prob: {fill_prob * 100:.1f}%")
        return False

    adjusted_price, slippage = apply_slippage(
        enter_price,
        direction,
        stake,
        liquidity,
        asset_id,
    )
    final_price, latency_ms = simulate_latency_and_drift(adjusted_price)
    shares = stake / final_price if final_price > 0 else 0.0
    if shares <= 0:
        _record_hardened_decision("invalid_shares", alert, level="warning", final_price=f"{final_price:.4f}")
        return False

    model_prob = _coerce_float(alert.get("model_prob"), default=0.0)
    target_exit_price = round(final_price + max(model_prob - final_price, 0.0) * CONVERGENCE_TARGET_PCT, 4)
    timestamp = _current_timestamp()

    trades = load_trades()
    trades.append({
        "timestamp":      timestamp,
        "market_id":      market_id,
        "event_slug":     event_slug,
        "market":         market,
        "direction":      direction,
        "bucket":         bucket,
        "line_value":     line_value,
        "model_key":      model_key,
        "model_label":    model_label,
        "trade_token":    asset_id,
        "enter_price":    enter_price,
        "adjusted_price": round(adjusted_price, 4),
        "final_price":    round(final_price, 4),
        "current_price":  round(final_price, 4),
        "stake":          stake,
        "shares":         round(shares, 6),
        "model_prob":     round(model_prob, 4) if model_prob > 0 else None,
        "edge":           round(edge, 4),
        "current_edge":   round(model_prob - final_price, 4) if model_prob > 0 else None,
        "slippage":       round(slippage, 4),
        "latency_ms":     round(latency_ms, 1),
        "fill_prob":      round(fill_prob, 3),
        "liquidity":      liquidity,
        "target_exit_price": target_exit_price,
        "unrealized_pnl": 0.0,
        "realized_pnl":   0.0,
        "max_favorable_price": round(final_price, 4),
        "max_adverse_price": round(final_price, 4),
        "last_price_check": timestamp,
        "status":         "OPEN",
        "hardened":       True,
        "closed_before_resolution": False,
    })
    save_trades(trades)

    new_bankroll = round(bankroll - stake, 2)
    _save_bankroll(new_bankroll, model_key=model_key)

    print(
        f"\n  HARDENED PAPER TRADE: {direction} | "
        f"Stake: ${stake:.2f} | "
        f"Price: {enter_price:.4f} -> {final_price:.4f} | "
        f"Slippage: {slippage * 100:.2f}% | "
        f"Bankroll: ${new_bankroll:.2f}"
    )
    logger.info(
        "Hardened paper trade executed — %s | market=%s | stake=$%.2f | enter_price=%.4f | final_price=%.4f",
        direction,
        market_id,
        stake,
        enter_price,
        final_price,
    )
    _record_hardened_decision("executed", alert, level="debug", bucket=bucket, stake=f"{stake:.2f}")
    return True
