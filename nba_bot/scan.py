"""
nba_bot/scan.py
===============
Unified NBA × Polymarket edge scanner CLI.

Usage:
    nba-bot-scan [--ws] [--mode live|test|markets] [--model-path PATH] [--interval SEC]

Modes:
    live     (default) — continuous polling loop, alerts on edge >= MIN_EDGE
    test     — hardcoded Lakers @ Warriors scenario, no live data required
    markets  — list active NBA Polymarket markets and exit

Flags:
    --ws             Use WebSocket price stream instead of REST CLOB polls
    --model-path     Override model file path (default: config.MODEL_PATH)
    --interval       Scan interval in seconds for live mode (default: SCAN_INTERVAL_SEC)
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from nba_bot import config
from nba_bot.model import load_model, predict_home_win_prob
from nba_bot.nba_live import fetch_live_nba_games
from nba_bot.polymarket import (
    compute_edge,
    fetch_nba_markets,
    kelly_stake,
    print_alert,
    print_no_edge,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STARTTYPE derivation for Tier-2 live inference
# ─────────────────────────────────────────────────────────────────────────────

# Heuristic mapping: EVENTMSGTYPE int (from nba_api PlayByPlay)
# → starttype_encoded int (matching STARTTYPE_MAP in features.py)
#   5 (turnover)         → 0 (LiveBallTurnover)
#   4 (rebound)          → 1 (DefensiveRebound / OffensiveRebound)
#   1 (made field goal)  → 2 (MadeShot)
#   3 (free throw made)  → 2 (FreeThrow)
#   other                → 3 (fallback)
_EVENTMSGTYPE_TO_STARTTYPE = {
    5: 0,  # Turnover
    4: 1,  # Rebound
    1: 2,  # Made FG
    3: 2,  # Made FT
}
_STARTTYPE_FALLBACK = 3


def _get_live_advanced_ctx(game_id: str, team_stats: dict) -> dict:
    """
    Builds a Tier-2 advanced_ctx dict for a live game.

    Fetches the last play's EVENTMSGTYPE from nba_api.PlayByPlay to derive
    starttype_encoded. Reads player_quality from the team stats cache.

    Returns a dict with fallback values on any error — never raises.
    """
    starttype_encoded = _STARTTYPE_FALLBACK

    try:
        from nba_api.stats.endpoints import playbyplayv2
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        df  = pbp.get_data_frames()[0]
        if not df.empty:
            last_play     = df.iloc[-1]
            event_type    = int(last_play.get("EVENTMSGTYPE", 0))
            starttype_encoded = _EVENTMSGTYPE_TO_STARTTYPE.get(event_type, _STARTTYPE_FALLBACK)
    except Exception as e:
        logger.debug("Could not fetch live PBP for advanced_ctx (game %s): %s", game_id, e)

    return {
        "starttype_encoded":   starttype_encoded,
        "player_quality_home": 0.0,   # will be overwritten by caller from team_stats
        "player_quality_away": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Scanner modes
# ─────────────────────────────────────────────────────────────────────────────

def run_test_mode(model, feature_cols):
    """
    Tests the full pipeline using a hardcoded game state.
    Works any time — no live NBA game or network required.

    Scenario: Lakers lead Warriors 88-84 with 3:20 left in Q4.
    Simulated Polymarket price: 0.64 for Lakers YES.
    """
    print()
    print("=" * 60)
    print("  MANUAL TEST MODE")
    print("=" * 60)
    print("  Game:   Golden State Warriors @ Los Angeles Lakers")
    print("  Score:  84-88  |  Q4  3:20")
    print()

    if model is not None:
        prob = predict_home_win_prob(
            model        = model,
            feature_cols = feature_cols,
            home_score   = 88,
            away_score   = 84,
            period       = 4,
            clock        = "3:20",
        )
        print(f"  Model home win probability: {prob * 100:.1f}%")
        print(f"  Model away win probability: {(1 - prob) * 100:.1f}%")
    else:
        print("  [No model loaded — using placeholder probability 0.73]")
        prob = 0.73

    simulated_price = 0.64
    edge  = prob - simulated_price
    stake = kelly_stake(edge, simulated_price)

    print(f"\n  Simulated Polymarket price (Lakers YES): {simulated_price}")
    print(f"  Edge:         {edge * 100:.2f}%")
    print(f"  Direction:    {'BUY YES' if edge > 0 else 'BUY NO'}")
    print(f"  Kelly stake:  {stake * 100:.2f}% of bankroll")
    print()
    sys.exit(0)


def run_markets_mode():
    """Fetches and prints all active NBA Polymarket markets, then exits."""
    print()
    print("=" * 60)
    print("  ACTIVE NBA MARKETS ON POLYMARKET")
    print("=" * 60)

    from nba_bot.polymarket import fetch_clob_midpoint

    markets = fetch_nba_markets()

    if not markets:
        print("  No active NBA game markets found.")
        print("  This is normal outside of game days.")
        print()
        sys.exit(0)

    for i, m in enumerate(markets, 1):
        clob_yes = fetch_clob_midpoint(m.get("clob_yes_id"))
        live_str = f"  CLOB live: {clob_yes:.3f}" if clob_yes else ""

        print(f"\n  [{i}] {m['event_title']}")
        print(f"       Question:  {m['question']}")
        print(f"       YES price: {m['yes_price']:.3f}{live_str}")
        print(f"       NO price:  {m['no_price']:.3f}")
        print(f"       Liquidity: ${m['liquidity']:,.0f}")
        print(f"       URL:       {m['url']}")

    print()
    sys.exit(0)


def run_live_mode(model, feature_cols, use_ws: bool, interval: int):
    """
    Continuous polling loop. Fetches live NBA scores and Polymarket prices,
    computes edge, and prints alerts.

    If --ws is set, also starts the Polymarket WebSocket listener thread
    and uses WS-streamed prices instead of REST CLOB calls.
    """
    print()
    print("=" * 60)
    print(f"  NBA × POLYMARKET  —  {'WebSocket' if use_ws else 'REST'} Edge Scanner")
    print("=" * 60)
    print(f"  Min edge threshold : {config.MIN_EDGE * 100:.0f}%")
    print(f"  Min liquidity      : ${config.MIN_LIQUIDITY:,}")
    print(f"  Kelly fraction     : {config.KELLY_FRACTION * 100:.0f}%  (quarter Kelly)")
    print(f"  Scan interval      : {interval}s")
    print(f"  Price source       : {'WS real-time' if use_ws else 'CLOB REST mid-point'}")
    if model is None:
        print("  [!] Model not loaded — edge computation disabled")
    print("=" * 60)
    print()

    # Load team stats cache (for Tier-2 player quality)
    from nba_bot.team_stats_cache import load_team_stats
    team_stats = load_team_stats()

    # Determine model tier
    use_t2 = False
    if model is not None:
        try:
            use_t2 = model.n_features_in_ == len(config.FEATURES_ALL)
        except AttributeError:
            if feature_cols is not None:
                use_t2 = len(feature_cols) == len(config.FEATURES_ALL)

    # Start WebSocket listener if requested
    ws_price_cache = None
    ws_stream      = None
    if use_ws:
        markets, token_ids = fetch_nba_markets(return_token_ids=True)
        if token_ids:
            from nba_bot.ws_stream import PolymarketPriceStream, price_cache
            ws_price_cache = price_cache
            ws_stream = PolymarketPriceStream(token_ids)
            ws_stream.start()
            time.sleep(2)  # brief wait for initial book snapshots
        else:
            logger.warning("No token IDs from Polymarket — WS mode unavailable, falling back to REST")
            use_ws = False

    # How often to refresh WS markets list (every N scans = ~N * interval seconds)
    _WS_MARKET_REFRESH_EVERY = 10  # refresh every ~10 min at default 60s interval

    scan_count = 0

    try:
        while True:
            scan_count += 1
            print(f"[Scan #{scan_count}]  {datetime.now().strftime('%H:%M:%S')}")

            if use_ws and ws_stream:
                print(f"  [WS] {ws_stream.status()}")
                # H2 FIX: Periodically refresh markets to pick up new games/markets
                if scan_count % _WS_MARKET_REFRESH_EVERY == 1:
                    fresh_markets, new_token_ids = fetch_nba_markets(return_token_ids=True)
                    if fresh_markets:
                        markets = fresh_markets
                        logger.info("[WS] Markets refreshed: %d markets", len(markets))
            else:
                markets = fetch_nba_markets()

            live_games = fetch_live_nba_games()

            if not markets:
                print("  No active NBA markets on Polymarket. Waiting...")

            elif not live_games:
                print("  No NBA games currently in progress.")

            else:
                all_alerts = []

                for game in live_games:
                    # Build Tier-2 context if model needs it
                    advanced_ctx = None
                    if use_t2:
                        from nba_bot.team_stats_cache import get_team_quality
                        ctx = _get_live_advanced_ctx(game["game_id"], team_stats)
                        # H1 FIX: Inject player quality using team_id now exposed by nba_live.py
                        ctx["player_quality_home"] = get_team_quality(
                            game.get("home_team_id", 0), team_stats
                        )
                        ctx["player_quality_away"] = get_team_quality(
                            game.get("away_team_id", 0), team_stats
                        )
                        advanced_ctx = ctx


                    alerts = compute_edge(
                        model         = model,
                        game          = game,
                        markets       = markets,
                        feature_cols  = feature_cols,
                        advanced_ctx  = advanced_ctx,
                        price_cache   = ws_price_cache,
                    )
                    all_alerts.extend(alerts)

                if all_alerts:
                    all_alerts.sort(key=lambda x: abs(x["edge"]), reverse=True)
                    print(f"\n  >>> {len(all_alerts)} ALERT(S) THIS SCAN <<<")
                    for alert in all_alerts:
                        print_alert(alert)
                else:
                    print_no_edge(len(live_games), len(markets))

            print(f"  Sleeping {interval}s...  (Ctrl+C to stop)\n")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[Scanner stopped]")
        if ws_stream:
            ws_stream.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NBA × Polymarket edge scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nba-bot-scan --mode test\n"
            "  nba-bot-scan --mode markets\n"
            "  nba-bot-scan --mode live --interval 30\n"
            "  nba-bot-scan --mode live --ws\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["live", "test", "markets"],
        default="live",
        help="Scanner mode (default: live)",
    )
    parser.add_argument(
        "--ws",
        action="store_true",
        help="Use WebSocket price stream instead of REST CLOB polling",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        metavar="PATH",
        help=f"Model .pkl path (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=config.SCAN_INTERVAL_SEC,
        metavar="SEC",
        help=f"Scan interval in seconds for live mode (default: {config.SCAN_INTERVAL_SEC})",
    )

    args = parser.parse_args()

    # Load model (non-fatal if not found)
    model_path   = args.model_path or config.MODEL_PATH
    model, fcols = load_model(model_path)

    if args.mode == "test":
        run_test_mode(model, fcols)

    elif args.mode == "markets":
        run_markets_mode()

    else:  # live
        run_live_mode(
            model        = model,
            feature_cols = fcols,
            use_ws       = args.ws,
            interval     = args.interval,
        )


if __name__ == "__main__":
    main()
