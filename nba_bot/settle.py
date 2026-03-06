"""
nba_bot/settle.py
=================
Batch settlement CLI for paper trades — `nba-bot-settle`.

Usage:
    nba-bot-settle                  # settle all pending trades
    nba-bot-settle --status         # print current bankroll + pending trades
    nba-bot-settle --dry-run        # print settlement math; do NOT write files
"""

import argparse
import json
import logging
import sys
from collections import defaultdict

import requests

from nba_bot.config import GAMMA_API, HEADERS, PAPER_BANKROLL_PATH, PAPER_TRADES_PATH
from nba_bot.paper import _load_bankroll, _save_bankroll, load_trades, save_trades

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Status display
# ─────────────────────────────────────────────────────────────────────────────

def _run_status() -> None:
    """Print current bankroll and all PENDING trades, then exit."""
    bankroll = _load_bankroll()
    trades   = load_trades()
    pending  = [t for t in trades if t.get("status") == "PENDING"]

    print()
    print("=" * 60)
    print("  NBA BOT PAPER TRADING — STATUS")
    print("=" * 60)
    print(f"  Current bankroll : ${bankroll:,.2f}")
    print(f"  Pending trades   : {len(pending)}")

    if pending:
        print()
        for i, t in enumerate(pending, 1):
            print(f"  [{i}] {t.get('market', t.get('market_id', '?'))}")
            print(f"       Direction   : {t.get('direction')}")
            print(f"       Enter price : {t.get('enter_price')}")
            print(f"       Stake       : ${t.get('stake', 0):,.2f}")
            print(f"       Edge        : {t.get('edge', 0) * 100:.2f}%")
            print(f"       Placed at   : {t.get('timestamp', '?')}")
    else:
        print("  No pending trades.")

    print()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# Gamma API lookup
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_event_by_slug(slug: str) -> dict | None:
    """
    Query Gamma API for a single event by slug (no active/closed filter so
    resolved events are included).

    Returns the first event dict, or None on error / not found.
    """
    url = f"{GAMMA_API}/events"
    try:
        resp = requests.get(
            url,
            params  = {"slug": slug},
            headers = HEADERS,
            timeout = 10,
        )
        resp.raise_for_status()
        events = resp.json()
        if events:
            return events[0]
        logger.warning("No event found for slug=%s", slug)
        return None
    except requests.exceptions.Timeout:
        logger.error("Gamma API timed out fetching slug=%s", slug)
        return None
    except Exception as e:
        logger.error("Gamma API error for slug=%s: %s", slug, e)
        return None


def _find_market_in_event(event: dict, market_id: str) -> dict | None:
    """Return the specific market dict matching *market_id* within *event*."""
    for mkt in event.get("markets", []):
        if str(mkt.get("id")) == str(market_id):
            return mkt
    return None


def _parse_outcome_prices(mkt: dict) -> tuple[str | None, str | None]:
    """
    Parse *outcomePrices* from a Gamma market dict.
    Returns (yes_price_str, no_price_str) or (None, None) on failure.
    """
    try:
        prices = json.loads(mkt.get("outcomePrices", "[]"))
        yes = prices[0] if len(prices) > 0 else None
        no  = prices[1] if len(prices) > 1 else None
        return yes, no
    except (json.JSONDecodeError, TypeError):
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Settlement logic
# ─────────────────────────────────────────────────────────────────────────────

def _settle_trade(trade: dict, mkt: dict) -> dict | None:
    """
    Attempt to settle *trade* against resolved *mkt*.

    Returns an updated trade dict if the market is closed and resolved,
    or None if the market is still open.
    """
    if not mkt.get("closed", False):
        return None  # market not yet resolved

    yes_price, no_price = _parse_outcome_prices(mkt)
    if yes_price is None:
        logger.warning("Could not parse outcomePrices for market_id=%s", trade.get("market_id"))
        return None

    direction   = trade.get("direction", "")
    enter_price = float(trade.get("enter_price", 0))
    stake       = float(trade.get("stake", 0))

    if enter_price <= 0:
        logger.warning("Invalid enter_price for trade market_id=%s; skipping.", trade.get("market_id"))
        return None

    # Determine win condition
    won = (
        (direction == "BUY YES" and yes_price == "1") or
        (direction == "BUY NO"  and no_price  == "1")
    )

    shares = stake / enter_price
    if won:
        payout = shares          # each share pays $1.00
        profit = payout - stake
        status = "WON"
    else:
        payout = 0.0
        profit = -stake
        status = "LOST"

    updated = dict(trade)
    updated.update({
        "status":      status,
        "shares":      round(shares, 6),
        "payout":      round(payout, 2),
        "profit":      round(profit, 2),
        "settled_yes": yes_price,
        "settled_no":  no_price,
    })
    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Main settlement run
# ─────────────────────────────────────────────────────────────────────────────

def _run_settle(dry_run: bool) -> None:
    """Core settlement routine."""
    trades  = load_trades()
    pending = [(i, t) for i, t in enumerate(trades) if t.get("status") == "PENDING"]

    if not pending:
        print("\n  No pending trades to settle.\n")
        sys.exit(0)

    print()
    print("=" * 60)
    print(f"  NBA BOT PAPER TRADING — {'DRY RUN ' if dry_run else ''}SETTLEMENT")
    print("=" * 60)
    print(f"  Pending trades   : {len(pending)}")
    if dry_run:
        print("  *** DRY RUN — no files will be written ***")
    print()

    # Group pending trades by event_slug to minimise API requests
    slug_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, trade in pending:
        slug = trade.get("event_slug", "")
        slug_groups[slug].append((idx, trade))

    settled_count = 0
    skipped_count = 0
    total_profit  = 0.0
    bankroll_delta = 0.0

    for slug, group in slug_groups.items():
        if not slug:
            logger.warning("Trade(s) missing event_slug — skipping: %s", [t.get("market_id") for _, t in group])
            skipped_count += len(group)
            continue

        event = _fetch_event_by_slug(slug)
        if event is None:
            logger.warning("Could not fetch event for slug=%s; skipping %d trade(s).", slug, len(group))
            skipped_count += len(group)
            continue

        for idx, trade in group:
            market_id = trade.get("market_id")
            mkt = _find_market_in_event(event, market_id)
            if mkt is None:
                logger.warning("market_id=%s not found in event slug=%s; skipping.", market_id, slug)
                skipped_count += 1
                continue

            updated = _settle_trade(trade, mkt)
            if updated is None:
                print(f"  ⏳ PENDING  | {trade.get('market', market_id)[:50]} — market not yet closed")
                skipped_count += 1
                continue

            won_str  = "WON " if updated["status"] == "WON" else "LOST"
            profit   = updated["profit"]
            total_profit   += profit
            bankroll_delta += updated["payout"]

            print(
                f"  {'✅' if updated['status'] == 'WON' else '❌'} {won_str} "
                f"| {trade.get('market', market_id)[:50]}"
            )
            print(
                f"       Direction  : {trade.get('direction')}  |  "
                f"Stake: ${trade.get('stake', 0):,.2f}  |  "
                f"Payout: ${updated['payout']:,.2f}  |  "
                f"P&L: ${profit:+,.2f}"
            )
            print()

            if not dry_run:
                trades[idx] = updated
            settled_count += 1

    # Summary
    print("-" * 60)
    print(f"  Settled : {settled_count}   Skipped/Pending : {skipped_count}")
    print(f"  Total P&L this run : ${total_profit:+,.2f}")

    if not dry_run and settled_count > 0:
        save_trades(trades)
        new_bankroll = round(_load_bankroll() + bankroll_delta, 2)
        _save_bankroll(new_bankroll)
        print(f"  New bankroll       : ${new_bankroll:,.2f}")
        print()
        print("  Files updated: paper_trades.json, paper_bankroll.json")
    elif dry_run:
        print()
        print(f"  Projected bankroll : ${_load_bankroll() + bankroll_delta:,.2f}  (not written)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level  = logging.WARNING,
        format = "%(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="NBA bot paper trade settlement tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nba-bot-settle\n"
            "  nba-bot-settle --status\n"
            "  nba-bot-settle --dry-run\n"
        ),
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current bankroll and pending trades, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute settlement math but do not write any files",
    )
    args = parser.parse_args()

    if args.status:
        _run_status()
    else:
        _run_settle(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
