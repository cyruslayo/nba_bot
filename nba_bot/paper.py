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
from datetime import datetime, timezone

from nba_bot.config import (
    DEFAULT_BANKROLL,
    PAPER_BANKROLL_PATH,
    PAPER_TRADES_PATH,
)

logger = logging.getLogger(__name__)


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
    market_id  = alert.get("market_id")
    direction  = alert.get("direction", "")
    enter_price = alert.get("enter_price", 0.0)
    raw_stake  = alert.get("raw_stake", 0.0)
    edge       = alert.get("edge", 0.0)

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
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "market_id":   market_id,
        "event_slug":  alert.get("event_slug", ""),
        "market":      alert.get("market", ""),
        "direction":   direction,
        "enter_price": enter_price,
        "stake":       stake,
        "edge":        round(edge, 4),
        "status":      "PENDING",
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
