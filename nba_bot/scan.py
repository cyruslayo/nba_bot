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
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from nba_bot import config
from nba_bot.model import load_model, predict_home_win_prob
from nba_bot.nba_live import fetch_live_nba_games
from nba_bot.polymarket import (
    compute_edge,
    fetch_nba_markets,
    kelly_stake,
    match_game_to_markets,
    print_alert,
    print_no_edge,
)


def _load_auxiliary_models():
    """
    Loads spread, total, and first_half models if configured.
    
    Returns dict with model and feature_cols for each market type.
    """
    from nba_bot.model import load_model
    
    models = {}
    
    # Load spread model
    if config.SPREAD_MODEL_PATH and config.ENABLE_SPREAD_TRADING:
        spread_model, spread_cols = load_model(config.SPREAD_MODEL_PATH)
        if spread_model:
            models["spread"] = {"model": spread_model, "feature_cols": spread_cols}
            logger.info("Spread model loaded from %s", config.SPREAD_MODEL_PATH)
    
    # Load total model
    if config.TOTAL_MODEL_PATH and config.ENABLE_TOTAL_TRADING:
        total_model, total_cols = load_model(config.TOTAL_MODEL_PATH)
        if total_model:
            models["total"] = {"model": total_model, "feature_cols": total_cols}
            logger.info("Total model loaded from %s", config.TOTAL_MODEL_PATH)
    
    # Load first half model
    if config.FIRST_HALF_MODEL_PATH and config.ENABLE_FIRST_HALF_TRADING:
        fh_model, fh_cols = load_model(config.FIRST_HALF_MODEL_PATH)
        if fh_model:
            models["first_half"] = {"model": fh_model, "feature_cols": fh_cols}
            logger.info("First half model loaded from %s", config.FIRST_HALF_MODEL_PATH)
    
    return models

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


def _infer_uses_t2(model, feature_cols) -> bool:
    if model is None:
        return False
    try:
        return model.n_features_in_ == len(config.FEATURES_ALL)
    except AttributeError:
        return feature_cols is not None and len(feature_cols) == len(config.FEATURES_ALL)


def _build_model_entry(model, feature_cols, model_path: str | None, fallback_key: str) -> dict:
    label = Path(model_path).stem if model_path else fallback_key
    key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or fallback_key
    return {
        "key": key,
        "label": label,
        "path": model_path,
        "model": model,
        "feature_cols": feature_cols,
        "use_t2": _infer_uses_t2(model, feature_cols),
    }


def _print_model_comparison_summary(alerts: list[dict], model_entries: list[dict]) -> None:
    if len(model_entries) < 2 or not alerts:
        return

    grouped_alerts: dict[str, list[dict]] = {}
    for alert in alerts:
        market_id = str(alert.get("market_id") or "")
        grouped_alerts.setdefault(market_id, []).append(alert)

    print("  Model Compare:")
    for _, group in sorted(
        grouped_alerts.items(),
        key=lambda item: max(
            abs(float(alert.get("calibrated_edge", alert.get("edge", 0.0)) or 0.0))
            for alert in item[1]
        ),
        reverse=True,
    ):
        market = group[0].get("market", "")
        if len(group) == 1:
            alert = group[0]
            print(
                f"    - {alert.get('model_label', 'model')} only | {alert.get('direction')} | "
                f"edge={float(alert.get('calibrated_edge', alert.get('edge', 0.0)) or 0.0):.4f} | "
                f"{market[:70]}"
            )
            continue

        directions = {alert.get("direction") for alert in group}
        status = "DISAGREE" if len(directions) > 1 else "AGREE"
        details = "; ".join(
            (
                f"{alert.get('model_label', alert.get('model_key', 'model'))}: "
                f"{alert.get('direction')} edge={float(alert.get('calibrated_edge', alert.get('edge', 0.0)) or 0.0):.4f} "
                f"stake={float(alert.get('raw_stake', 0.0) or 0.0):.4f}"
            )
            for alert in sorted(group, key=lambda item: item.get("model_label") or item.get("model_key") or "")
        )
        print(f"    - {status} | {market[:70]}")
        print(f"      {details}")


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


def run_live_mode(
    model,
    feature_cols,
    use_ws: bool,
    interval: int,
    use_paper: bool = False,
    use_paper_hardened: bool = False,
    initial_bankroll: float | None = None,
    random_seed: int | None = None,
    model_entries: list[dict] | None = None,
):
    """
    Continuous polling loop. Fetches live NBA scores and Polymarket prices,
    computes edge, and prints alerts.

    If --ws is set, also starts the Polymarket WebSocket listener thread
    and uses WS-streamed prices instead of REST CLOB calls.

    If --paper is set, auto-executes and logs paper trades to JSON files.
    """
    if random_seed is not None:
        random.seed(random_seed)
        logger.info("Random seed set to %d", random_seed)

    active_model_entries = [
        dict(entry)
        for entry in (model_entries or [_build_model_entry(model, feature_cols, config.MODEL_PATH, "primary")])
        if entry.get("model") is not None
    ]

    print()
    print("=" * 60)
    print(f"  NBA × POLYMARKET  —  {'WebSocket' if use_ws else 'REST'} Edge Scanner")
    print("=" * 60)
    print(f"  Min edge threshold : {config.MIN_EDGE * 100:.0f}%")
    print(f"  Min liquidity      : ${config.MIN_LIQUIDITY:,}")
    print(f"  Kelly fraction     : {config.KELLY_FRACTION * 100:.0f}%  (quarter Kelly)")
    print(f"  Scan interval      : {interval}s")
    print(f"  Price source       : {'WS real-time' if use_ws else 'CLOB REST mid-point'}")
    if active_model_entries:
        print("  Models loaded      : " + ", ".join(entry["label"] for entry in active_model_entries))
    else:
        print("  [!] Model not loaded — edge computation disabled")

    # Initialise paper trading if requested
    if use_paper_hardened:
        from nba_bot.market_analytics import check_data_api_available
        from nba_bot.paper import _load_bankroll, init_bankroll, load_model_bankrolls

        if not check_data_api_available():
            print("  [!] Data API unavailable — hardened paper trading requires a live Data API connection.")
            return

        init_bankroll(
            initial_amount=initial_bankroll or config.DEFAULT_BANKROLL,
            reset_if_exists=bool(initial_bankroll),
            model_keys=[entry["key"] for entry in active_model_entries],
        )
        model_bankrolls = load_model_bankrolls()
        if len(active_model_entries) > 1:
            bankroll_summary = ", ".join(
                f"{entry['label']}: ${model_bankrolls.get(entry['key'], _load_bankroll(model_key=entry['key'])):.2f}"
                for entry in active_model_entries
            )
            print(f"  Paper Trading      : HARDENED  ({bankroll_summary})")
        else:
            current_bankroll = _load_bankroll(model_key=active_model_entries[0]["key"] if active_model_entries else None)
            print(f"  Paper Trading      : HARDENED  (bankroll: ${current_bankroll:.2f})")
    elif use_paper:
        from nba_bot.paper import init_bankroll, _load_bankroll
        init_bankroll(
            initial_amount=initial_bankroll or config.DEFAULT_BANKROLL,
            reset_if_exists=bool(initial_bankroll),
            model_keys=[entry["key"] for entry in active_model_entries],
        )
        if len(active_model_entries) > 1:
            bankroll_summary = ", ".join(
                f"{entry['label']}: ${_load_bankroll(model_key=entry['key']):.2f}"
                for entry in active_model_entries
            )
            print(f"  Paper Trading      : ACTIVE  ({bankroll_summary})")
        else:
            current_bankroll = _load_bankroll(model_key=active_model_entries[0]["key"] if active_model_entries else None)
            print(f"  Paper Trading      : ACTIVE  (bankroll: ${current_bankroll:.2f})")

    from nba_bot.team_stats_cache import load_team_stats
    team_stats = load_team_stats()

    # Load auxiliary models (spread, total, first_half)
    auxiliary_models = _load_auxiliary_models()

    use_t2 = any(entry["use_t2"] for entry in active_model_entries)

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
                all_rejections: dict[str, int] = {}
                games_by_event_slug: dict[str, dict] = {}
                for game in live_games:
                    for market, _ in match_game_to_markets(game, markets):
                        event_slug = market.get("event_slug", "")
                        if event_slug:
                            games_by_event_slug[event_slug] = game

                if use_paper_hardened:
                    from nba_bot.paper import monitor_live_positions

                    monitor_summary = monitor_live_positions(
                        markets,
                        games_by_event_slug=games_by_event_slug,
                        price_cache=ws_price_cache,
                    )
                    if monitor_summary["checked"] or monitor_summary["closed"]:
                        print(
                            "  Live monitor      : "
                            f"checked={int(monitor_summary['checked'])} | "
                            f"closed={int(monitor_summary['closed'])} | "
                            f"missing_price={int(monitor_summary['missing_price'])} | "
                            f"realized_pnl=${float(monitor_summary['realized_pnl']):+.2f}"
                        )

                bankroll_by_model: dict[str, float] = {}
                if use_paper_hardened or use_paper:
                    from nba_bot.paper import _load_bankroll
                    for model_entry in active_model_entries:
                        bankroll_by_model[model_entry["key"]] = _load_bankroll(model_key=model_entry["key"])

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

                    for model_entry in active_model_entries:
                        result = compute_edge(
                            model         = model_entry["model"],
                            game          = game,
                            markets       = markets,
                            feature_cols  = model_entry["feature_cols"],
                            advanced_ctx  = advanced_ctx if model_entry["use_t2"] else None,
                            price_cache   = ws_price_cache,
                            hardened      = use_paper_hardened,
                            bankroll      = bankroll_by_model.get(model_entry["key"], 0.0),
                            model_key     = model_entry["key"],
                            model_label   = model_entry["label"],
                            return_rejections = use_paper_hardened,
                            spread_model  = auxiliary_models.get("spread", {}).get("model"),
                            spread_feature_cols = auxiliary_models.get("spread", {}).get("feature_cols"),
                            total_model   = auxiliary_models.get("total", {}).get("model"),
                            total_feature_cols = auxiliary_models.get("total", {}).get("feature_cols"),
                            first_half_model = auxiliary_models.get("first_half", {}).get("model"),
                            first_half_feature_cols = auxiliary_models.get("first_half", {}).get("feature_cols"),
                        )
                        if use_paper_hardened:
                            alerts, rejections = result
                            for key, count in rejections.items():
                                all_rejections[key] = all_rejections.get(key, 0) + count
                        else:
                            alerts = result
                        all_alerts.extend(alerts)

                if all_alerts:
                    if len(active_model_entries) > 1:
                        _print_model_comparison_summary(all_alerts, active_model_entries)
                    sort_key = "calibrated_edge" if use_paper_hardened else "edge"
                    all_alerts.sort(key=lambda x: abs(float(x.get(sort_key, x.get("edge", 0.0)) or 0.0)), reverse=True)
                    print(f"\n  >>> {len(all_alerts)} ALERT(S) THIS SCAN <<<")
                    for alert in all_alerts:
                        print_alert(alert)
                        if use_paper_hardened:
                            from nba_bot.paper import execute_paper_trade_hardened
                            executed = execute_paper_trade_hardened(alert)
                            if not executed:
                                logger.info(
                                    "Hardened alert rejected during execution | market_id=%s | direction=%s | edge=%.4f | price_source=%s",
                                    alert.get("market_id"),
                                    alert.get("direction"),
                                    float(alert.get("calibrated_edge", alert.get("edge", 0.0)) or 0.0),
                                    alert.get("price_source", "unknown"),
                                )
                        elif use_paper:
                            from nba_bot.paper import execute_paper_trade
                            execute_paper_trade(alert)
                else:
                    rejections_arg = all_rejections if use_paper_hardened else None
                    print_no_edge(len(live_games), len(markets), rejections=rejections_arg)

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
        "--compare-model-path",
        default=config.COMPARE_MODEL_PATH,
        metavar="PATH",
        help="Optional secondary .pkl path for side-by-side live comparison",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=config.SCAN_INTERVAL_SEC,
        metavar="SEC",
        help=f"Scan interval in seconds for live mode (default: {config.SCAN_INTERVAL_SEC})",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help="Enable paper trading mode — auto-log trades to paper_trades.json",
    )
    parser.add_argument(
        "--paper-hardened",
        action="store_true",
        help="Enable hardened paper trading with realistic slippage, fees, fill probability, and latency",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=None,
        metavar="AMOUNT",
        help="Initial paper trading bankroll (resets if already exists)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="Random seed for reproducible hardened paper-trading simulations",
    )

    args = parser.parse_args()

    # Warn if --paper used outside live mode
    if args.paper and args.mode != "live":
        print("  [!] --paper flag is only supported in --mode live. Ignoring.")
        args.paper = False
    if args.paper_hardened and args.mode != "live":
        print("  [!] --paper-hardened flag is only supported in --mode live. Ignoring.")
        args.paper_hardened = False
    if args.paper and args.paper_hardened:
        print("  [!] --paper and --paper-hardened are mutually exclusive. Using --paper-hardened.")
        args.paper = False

    # Load model (non-fatal if not found)
    model_path   = args.model_path or config.MODEL_PATH
    model, fcols = load_model(model_path)
    model_entries = []
    if model is not None:
        model_entries.append(_build_model_entry(model, fcols, model_path, "primary"))

    if args.compare_model_path:
        compare_model, compare_fcols = load_model(args.compare_model_path)
        if compare_model is None:
            print(f"  [!] Compare model not loaded — ignoring {args.compare_model_path}")
        else:
            compare_entry = _build_model_entry(compare_model, compare_fcols, args.compare_model_path, "compare")
            if any(existing["key"] == compare_entry["key"] for existing in model_entries):
                compare_entry["key"] = f"{compare_entry['key']}_compare"
                compare_entry["label"] = f"{compare_entry['label']} (compare)"
            model_entries.append(compare_entry)

    if args.mode == "test":
        run_test_mode(model, fcols)

    elif args.mode == "markets":
        run_markets_mode()

    else:  # live
        run_live_mode(
            model            = model,
            feature_cols     = fcols,
            use_ws           = args.ws,
            interval         = args.interval,
            use_paper        = args.paper,
            use_paper_hardened = args.paper_hardened,
            initial_bankroll = args.bankroll,
            random_seed      = args.seed,
            model_entries    = model_entries,
        )

if __name__ == "__main__":
    main()
