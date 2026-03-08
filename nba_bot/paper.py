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
from datetime import datetime, timedelta, timezone

from nba_bot.config import (
    DEFAULT_BANKROLL,
    DISABLE_STOCHASTIC_FILL,
    FILL_PROB_BASE,
    FILL_PROB_EDGE_BONUS,
    FILL_PROB_LIQUIDITY_SCALE,
    LATENCY_MAX_MS,
    LATENCY_MEAN_MS,
    LATENCY_STD_MS,
    MAX_FIRST_HALF_EXPOSURE_PCT,
    MAX_GAME_EXPOSURE_PCT,
    MAX_MONEYLINE_EXPOSURE_PCT,
    MAX_OTHER_EXPOSURE_PCT,
    MAX_SLIPPAGE_PCT,
    MAX_SPREAD_EXPOSURE_PCT,
    MAX_STAKE_LIQUIDITY_PCT,
    MAX_TOTAL_EXPOSURE_PCT,
    PAPER_BANKROLL_PATH,
    PAPER_TRADES_PATH,
    PRICE_DRIFT_STD,
    SLIPPAGE_BASE,
    SPREAD_CLUSTER_DISTANCE,
    TOTAL_CLUSTER_DISTANCE,
)

logger = logging.getLogger(__name__)
PENDING_TRADE_MAX_AGE_HOURS = 6
HARDENED_EXECUTION_STATS: dict[str, int] = {}


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

def init_bankroll(initial_amount: float, reset_if_exists: bool = False) -> None:
    """
    Initialise the bankroll JSON file.

    If *reset_if_exists* is True (or the file doesn't exist yet),
    the bankroll is written (or overwritten) to *initial_amount*.
    If the file already exists and *reset_if_exists* is False, this is a no-op
    so that successive runs without --bankroll preserve accumulated P&L.
    """
    if os.path.exists(PAPER_BANKROLL_PATH) and not reset_if_exists:
        logger.debug("Bankroll file exists; skipping init (reset_if_exists=False).")
        return
    _atomic_write(PAPER_BANKROLL_PATH, {"bankroll": initial_amount})
    logger.info("Bankroll initialised at $%.2f → %s", initial_amount, PAPER_BANKROLL_PATH)


def _load_bankroll() -> float:
    """Read current bankroll from JSON. Returns DEFAULT_BANKROLL if file missing."""
    if not os.path.exists(PAPER_BANKROLL_PATH):
        return DEFAULT_BANKROLL
    with open(PAPER_BANKROLL_PATH, encoding="utf-8") as fh:
        return float(json.load(fh).get("bankroll", DEFAULT_BANKROLL))


def _save_bankroll(amount: float) -> None:
    """Atomically persist updated bankroll."""
    _atomic_write(PAPER_BANKROLL_PATH, {"bankroll": amount})


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

def has_active_position(market_id: str) -> bool:
    """
    Returns True if there is already a PENDING paper trade for *market_id*.
    Used to prevent duplicate entries across scanner loops.
    """
    for trade in load_trades():
        if trade.get("market_id") == market_id and trade.get("status") == "PENDING":
            return True
    return False


def _pending_cutoff_time() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=PENDING_TRADE_MAX_AGE_HOURS)


def _is_recent_pending_trade(trade: dict, cutoff_time: datetime | None = None) -> bool:
    if trade.get("status") != "PENDING":
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


def get_game_exposure(event_slug: str, bucket: str | None = None) -> float:
    total = 0.0
    cutoff_time = _pending_cutoff_time()

    for trade in load_trades():
        if trade.get("event_slug") != event_slug:
            continue
        if bucket is not None:
            trade_bucket = trade.get("bucket") or classify_market_bucket(trade.get("market", ""))
            if trade_bucket != bucket:
                continue
        if not _is_recent_pending_trade(trade, cutoff_time=cutoff_time):
            continue

        total += float(trade.get("stake", 0) or 0)

    return total


def find_clustered_trade(event_slug: str, market: str, bucket: str, market_id: str, cluster_distance: float) -> dict | None:
    if bucket not in {"spread", "total"} or cluster_distance <= 0:
        return None

    line_value = _extract_line_value(market)
    if line_value is None:
        return None

    family_key = _market_family_key(market)
    cutoff_time = _pending_cutoff_time()
    for trade in load_trades():
        if str(trade.get("market_id")) == str(market_id):
            continue
        if trade.get("event_slug") != event_slug:
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
            vwap_price, filled_shares = compute_vwap_from_book(book, "BUY", stake)
            if vwap_price is not None:
                adjusted_price, slippage = _cap_price_move(vwap_price, "order_book")
                logger.debug(
                    "Order-book fill estimate — asset_id=%s | direction=%s | enter_price=%.4f | vwap=%.4f | adjusted=%.4f | slippage=%.4f | filled_shares=%.4f",
                    asset_id,
                    direction,
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

    if asset_id:
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
                "Trade-history fill estimate — asset_id=%s | direction=%s | enter_price=%.4f | adjusted=%.4f | slippage=%.4f | shares=%.4f",
                asset_id,
                direction,
                enter_price,
                adjusted_price,
                slippage,
                shares,
            )
            return adjusted_price, slippage

    base_liquidity = max(liquidity, 1.0)
    slippage = SLIPPAGE_BASE + (stake / base_liquidity) * 0.001
    slippage = _clamp_slippage(slippage, "liquidity")
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

    # 1. Idempotency check
    if has_active_position(market_id):
        logger.debug("Skipping paper trade — active position already open for market %s", market_id)
        return

    # 2. Bankroll guard
    bankroll = _load_bankroll()
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
        "enter_price": enter_price,
        "stake": stake,
        "edge": round(edge, 4),
        "status": "PENDING",
    })
    save_trades(trades)

    # 5. Deduct from bankroll (only after trade write succeeds)
    new_bankroll = round(bankroll - stake, 2)
    _save_bankroll(new_bankroll)

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

    if has_active_position(market_id):
        _record_hardened_decision("active_position", alert, level="debug")
        return False

    bankroll = _load_bankroll()
    if bankroll <= 0:
        _record_hardened_decision("bankroll_nonpositive", alert, level="warning", bankroll=f"{bankroll:.2f}")
        return False

    stake = round(raw_stake * bankroll, 2)
    if liquidity > 0:
        stake = min(stake, round(liquidity * MAX_STAKE_LIQUIDITY_PCT, 2))
    if stake <= 0:
        _record_hardened_decision("stake_zero", alert, level="debug")
        return False

    max_game_exposure = bankroll * MAX_GAME_EXPOSURE_PCT
    current_game_exposure = get_game_exposure(event_slug)
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
    current_bucket_exposure = get_game_exposure(event_slug, bucket=bucket)
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
    clustered_trade = find_clustered_trade(event_slug, market, bucket, market_id, cluster_distance)
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

    trades = load_trades()
    trades.append({
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "market_id":      market_id,
        "event_slug":     event_slug,
        "market":         market,
        "direction":      direction,
        "bucket":         bucket,
        "line_value":     line_value,
        "trade_token":    asset_id,
        "enter_price":    enter_price,
        "adjusted_price": round(adjusted_price, 4),
        "final_price":    round(final_price, 4),
        "stake":          stake,
        "shares":         round(shares, 6),
        "edge":           round(edge, 4),
        "slippage":       round(slippage, 4),
        "latency_ms":     round(latency_ms, 1),
        "fill_prob":      round(fill_prob, 3),
        "liquidity":      liquidity,
        "status":         "PENDING",
        "hardened":       True,
    })
    save_trades(trades)

    new_bankroll = round(bankroll - stake, 2)
    _save_bankroll(new_bankroll)

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
