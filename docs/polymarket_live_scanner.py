"""
Polymarket Live NBA Edge Scanner
==================================
Automatically:
  1. Finds all active NBA game markets on Polymarket (no API key needed)
  2. Fetches live YES/NO prices for each market via REST
  3. Fetches live NBA game scores via nba_api
  4. Compares scores against your trained XGBoost win probability model
  5. Prints an alert + Kelly stake when edge >= MIN_EDGE threshold

Dependencies:
  pip install requests pandas numpy joblib xgboost nba_api

Files required in same directory:
  xgb_win_prob_model.pkl   (trained by nba_win_probability.py)

Usage:
  python polymarket_live_scanner.py          # live continuous scan
  python polymarket_live_scanner.py test     # manual test with hardcoded game
  python polymarket_live_scanner.py markets  # list all active NBA markets and exit
"""

import requests
import json
import time
import sys
import numpy as np
import joblib
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_API          = "https://gamma-api.polymarket.com"
CLOB_API           = "https://clob.polymarket.com"

# Polymarket NBA series + game-winner tag IDs
NBA_SERIES_ID      = 10345
GAME_TAG_ID        = 100639

# Edge detection thresholds
MIN_EDGE           = 0.05    # Only alert when model edge >= 5%
MIN_LIQUIDITY      = 500     # Skip markets with less than $500 total liquidity
KELLY_FRACTION     = 0.25   # Bet 25% of full Kelly (conservative variance control)

# Scanner timing
SCAN_INTERVAL_SEC  = 60     # Poll every 60 seconds

# Feature columns (must match order used during model training)
FEATURES = ["score_diff", "time_remaining", "time_pressure", "game_progress", "period"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str = "xgb_win_prob_model.pkl"):
    """
    Loads the trained XGBoost win probability model from disk.
    Run nba_win_probability.py first to generate this file.
    """
    try:
        model = joblib.load(path)
        print(f"[Model] Loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"[WARN] Model not found at '{path}'.")
        print("       Run nba_win_probability.py first to train and save the model.")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2. WIN PROBABILITY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_win_prob(
    model,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,       # Format "MM:SS" e.g. "4:32"
) -> float:
    """
    Computes the home team's win probability given the current game state.

    Features engineered:
      score_diff    = home_score - away_score
      time_remaining = total seconds left in regulation
      time_pressure  = score_diff / sqrt(time_remaining + 1)
                       → grows as lead widens and time runs out
      game_progress  = 1 - (time_remaining / 2880)
                       → 0.0 at tipoff, 1.0 at final buzzer
      period         = current quarter (1-4, 5+ for OT)

    Returns:
      float in [0, 1] — probability that the home team wins
    """
    # Parse clock string "MM:SS" → seconds remaining in this period
    try:
        mins, secs     = map(int, clock.split(":"))
        time_in_period = mins * 60 + secs
    except Exception:
        time_in_period = 720   # default to start of period if parse fails

    # Total seconds remaining in regulation (each quarter = 12 min = 720s)
    if period <= 4:
        time_remaining = (4 - period) * 720 + time_in_period
    else:
        # Overtime: 5 min periods, just use remaining time
        time_remaining = max(0, time_in_period)

    score_diff    = home_score - away_score
    time_pressure = score_diff / np.sqrt(time_remaining + 1)
    game_progress = 1 - (time_remaining / 2880)    # 2880 = 4 quarters * 720s

    X = np.array([[score_diff, time_remaining, time_pressure, game_progress, period]])
    return float(model.predict_proba(X)[0][1])


# ─────────────────────────────────────────────────────────────────────────────
# 3. POLYMARKET MARKET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nba_markets() -> list[dict]:
    """
    Queries the Polymarket Gamma API for all active, open NBA game markets.

    Filters applied:
      - series_id = 10345  (NBA series)
      - tag_id    = 100639 (game winner markets only — excludes futures/season bets)
      - active    = true
      - closed    = false
      - liquidity >= MIN_LIQUIDITY

    Returns a list of market dicts, each containing:
      event_title, question, yes_price, no_price,
      clob_yes_id, clob_no_id, liquidity, url
    """
    url    = f"{GAMMA_API}/events"
    params = {
        "series_id": NBA_SERIES_ID,
        "tag_id":    GAME_TAG_ID,
        "active":    "true",
        "closed":    "false",
        "limit":     50,
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        events = resp.json()
    except requests.exceptions.Timeout:
        print("[ERROR] Gamma API timed out. Will retry next scan.")
        return []
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR] Gamma API HTTP error: {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Gamma API unexpected error: {e}")
        return []

    markets = []

    for event in events:
        for mkt in event.get("markets", []):
            try:
                outcome_prices = json.loads(mkt.get("outcomePrices", "[]"))
                outcomes       = json.loads(mkt.get("outcomes", "[]"))
                clob_ids       = json.loads(mkt.get("clobTokenIds", "[]"))
                liquidity      = float(mkt.get("liquidity", 0))

                # Skip thin markets — hard to fill positions
                if len(outcome_prices) < 2 or liquidity < MIN_LIQUIDITY:
                    continue

                markets.append({
                    "event_title": event.get("title", "Unknown"),
                    "question":    mkt.get("question", ""),
                    "market_id":   mkt.get("id"),
                    "event_slug":  event.get("slug", ""),
                    "outcomes":    outcomes,
                    "yes_price":   float(outcome_prices[0]),
                    "no_price":    float(outcome_prices[1]),
                    "clob_yes_id": clob_ids[0] if len(clob_ids) > 0 else None,
                    "clob_no_id":  clob_ids[1] if len(clob_ids) > 1 else None,
                    "liquidity":   liquidity,
                    "url":         f"https://polymarket.com/event/{event.get('slug', '')}",
                })

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    print(f"[Polymarket] {len(markets)} active NBA game market(s) found")
    return markets


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIVE CLOB PRICE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_clob_midpoint(token_id: str) -> float | None:
    """
    Fetches the real-time order-book mid-point price for a single token
    from Polymarket's CLOB REST endpoint.

    This is more accurate than the Gamma API's cached outcomePrices,
    which can lag by several minutes during fast-moving games.

    Returns float in [0, 1] or None on failure.
    """
    if not token_id:
        return None

    try:
        resp = requests.get(
            f"{CLOB_API}/midpoint",
            params  = {"token_id": token_id},
            headers = HEADERS,
            timeout = 5,
        )
        resp.raise_for_status()
        mid = float(resp.json().get("mid", 0))
        return mid if 0 < mid < 1 else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. KELLY CRITERION
# ─────────────────────────────────────────────────────────────────────────────

def kelly_stake(edge: float, market_price: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Computes the recommended bet size as a fraction of bankroll using
    fractional Kelly criterion.

    Full Kelly formula:
      f* = edge / (1 - market_price)

    where edge = model_prob - market_price

    We then multiply by `fraction` (default 0.25) to reduce variance.
    At full Kelly, a single bad streak can be catastrophic.
    At quarter Kelly, drawdowns are much more manageable.

    Args:
      edge         : model_prob - polymarket_yes_price
      market_price : Polymarket YES token price (implied probability)
      fraction     : Kelly multiplier (0.25 = quarter Kelly)

    Returns:
      float: recommended fraction of bankroll to stake (0–1)
             Returns 0 if edge is negative or market_price is degenerate.
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if edge <= 0:
        return 0.0

    full_kelly      = edge / (1 - market_price)
    fractional_kelly = full_kelly * fraction
    return round(max(fractional_kelly, 0.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIVE NBA SCORE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_nba_games() -> list[dict]:
    """
    Fetches all NBA games currently in progress using nba_api's live scoreboard.

    Returns a list of game state dicts:
      home_team, away_team, home_score, away_score, period, clock

    Only includes games with status containing "Q" (Q1/Q2/Q3/Q4)
    or "Halftime" — excludes upcoming and final games.
    """
    try:
        from nba_api.live.nba.endpoints import scoreboard
        board      = scoreboard.ScoreBoard()
        games_data = board.get_dict()["scoreboard"]["games"]
    except ImportError:
        print("[ERROR] nba_api not installed. Run: pip install nba_api")
        return []
    except Exception as e:
        print(f"[ERROR] NBA live scoreboard: {e}")
        return []

    live_games = []

    for g in games_data:
        status = g.get("gameStatusText", "")

        # Filter to in-progress games only
        if "Q" not in status and "Halftime" not in status:
            continue

        period    = g.get("period", 1)
        raw_clock = g.get("gameClock", "PT12M00.00S")

        # Convert ISO 8601 duration "PT5M32.00S" → "5:32"
        try:
            raw_clock = raw_clock.replace("PT", "").replace("S", "")
            mins, secs = raw_clock.split("M")
            clock = f"{int(float(mins))}:{int(float(secs)):02d}"
        except Exception:
            clock = "12:00"

        home = g["homeTeam"]
        away = g["awayTeam"]

        live_games.append({
            "game_id":    g.get("gameId", ""),
            "home_team":  home["teamName"],
            "away_team":  away["teamName"],
            "home_city":  home.get("teamCity", ""),
            "away_city":  away.get("teamCity", ""),
            "home_score": int(home.get("score", 0)),
            "away_score": int(away.get("score", 0)),
            "period":     period,
            "clock":      clock,
            "status":     status,
        })

    print(f"[NBA Live] {len(live_games)} game(s) in progress")
    return live_games


# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKET MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_game_to_markets(game: dict, markets: list[dict]) -> list[tuple[dict, str]]:
    """
    Matches a live NBA game to its corresponding Polymarket markets.

    Polymarket event titles are typically formatted as:
      "Los Angeles Lakers vs Golden State Warriors"
      "Lakers vs Warriors"

    We match by checking if either the team name or city appears
    in the event title. Returns a list of (market, perspective) tuples
    where perspective is "home" or "away" indicating which team the
    market question is about.
    """
    home_name = game["home_team"].lower()
    away_name = game["away_team"].lower()
    home_city = game["home_city"].lower()
    away_city = game["away_city"].lower()

    matched = []

    for mkt in markets:
        title     = mkt["event_title"].lower()
        question  = mkt["question"].lower()

        home_in_title = home_name in title or home_city in title
        away_in_title = away_name in title or away_city in title

        # Only consider markets where both teams appear in the event title
        # This reduces false matches (e.g. "Lakers season wins" matching a game)
        if not (home_in_title and away_in_title):
            continue

        # Determine if this market is asking about the home or away team winning
        if home_name in question or home_city in question:
            matched.append((mkt, "home"))
        elif away_name in question or away_city in question:
            matched.append((mkt, "away"))
        else:
            # Generic "will X win" — default to checking both
            matched.append((mkt, "home"))

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# 8. EDGE CALCULATION FOR A SINGLE GAME
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge(
    model,
    game: dict,
    markets: list[dict],
) -> list[dict]:
    """
    For a single live game, finds matching Polymarket markets,
    fetches the latest CLOB price, computes model win probability,
    and returns alerts for any market where |edge| >= MIN_EDGE.

    Args:
      model   : trained XGBoost model
      game    : live game state dict from fetch_live_nba_games()
      markets : all active Polymarket NBA markets

    Returns:
      list of alert dicts (empty if no edge found)
    """
    if model is None:
        return []

    # Get model's home win probability
    home_prob = predict_win_prob(
        model,
        home_score = game["home_score"],
        away_score = game["away_score"],
        period     = game["period"],
        clock      = game["clock"],
    )
    away_prob = 1.0 - home_prob

    matched_markets = match_game_to_markets(game, markets)

    if not matched_markets:
        return []

    alerts = []

    for mkt, perspective in matched_markets:
        model_prob = home_prob if perspective == "home" else away_prob

        # Try real-time CLOB price first, fall back to cached Gamma price
        yes_token   = mkt.get("clob_yes_id")
        live_yes    = fetch_clob_midpoint(yes_token)
        price_source = "CLOB (live)"

        if live_yes is None:
            live_yes     = mkt["yes_price"]
            price_source = "Gamma (cached)"

        edge = model_prob - live_yes

        # Skip if edge below threshold
        if abs(edge) < MIN_EDGE:
            continue

        # Determine trade direction
        if edge > 0:
            direction = "BUY YES"
            stake     = kelly_stake(edge, live_yes)
        else:
            # Edge is negative → YES is overpriced → buy NO
            no_price  = 1.0 - live_yes
            direction = "BUY NO"
            stake     = kelly_stake(abs(edge), no_price)

        alerts.append({
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "game":         f"{game['away_city']} {game['away_team']} @ {game['home_city']} {game['home_team']}",
            "score":        f"{game['away_score']}-{game['home_score']}",
            "period":       game["period"],
            "clock":        game["clock"],
            "market":       mkt["question"],
            "model_prob":   round(model_prob, 4),
            "poly_price":   round(live_yes, 4),
            "edge":         round(edge, 4),
            "edge_pct":     f"{round(edge * 100, 2)}%",
            "direction":    direction,
            "kelly_stake":  f"{round(stake * 100, 2)}% of bankroll",
            "raw_stake":    stake,
            "liquidity":    mkt["liquidity"],
            "price_source": price_source,
            "url":          mkt["url"],
        })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# 9. ALERT PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_alert(alert: dict):
    """Pretty-prints a single edge alert to the console."""
    edge_val = alert["edge"]
    bar      = "▲" if edge_val > 0 else "▼"

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  EDGE ALERT  {alert['timestamp']}                          ")
    print("  ╚══════════════════════════════════════════════════╝")
    print(f"  Game:         {alert['game']}")
    print(f"  Score:        {alert['score']}  |  Q{alert['period']}  {alert['clock']}")
    print(f"  Market:       {alert['market']}")
    print(f"  Model prob:   {alert['model_prob'] * 100:.1f}%")
    print(f"  Poly price:   {alert['poly_price'] * 100:.1f}%  ({alert['price_source']})")
    print(f"  Edge:         {bar} {alert['edge_pct']}")
    print(f"  Signal:       >>>  {alert['direction']}")
    print(f"  Kelly stake:  {alert['kelly_stake']}")
    print(f"  Liquidity:    ${alert['liquidity']:,.0f}")
    print(f"  Link:         {alert['url']}")
    print()


def print_no_edge(n_games: int, n_markets: int):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] Scanned {n_games} game(s) across {n_markets} market(s) — no edge >= {MIN_EDGE*100:.0f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN CONTINUOUS SCANNER
# ─────────────────────────────────────────────────────────────────────────────

def run_scanner(model_path: str = "xgb_win_prob_model.pkl", interval: int = SCAN_INTERVAL_SEC):
    """
    Main loop. Runs indefinitely, scanning for edge every `interval` seconds.

    Each iteration:
      1. Fetches all active NBA Polymarket markets (Gamma REST)
      2. Fetches all live NBA game scores (nba_api live scoreboard)
      3. For each live game, computes edge vs Polymarket price
      4. Prints alert if edge >= MIN_EDGE

    Polymarket markets are re-fetched each scan so prices stay current.
    For true real-time prices, use polymarket_ws_scanner.py instead.

    Stop with Ctrl+C.
    """
    print()
    print("=" * 60)
    print("  NBA × POLYMARKET  —  Live Edge Scanner")
    print("=" * 60)
    print(f"  Min edge threshold : {MIN_EDGE * 100:.0f}%")
    print(f"  Min liquidity      : ${MIN_LIQUIDITY:,}")
    print(f"  Kelly fraction     : {KELLY_FRACTION * 100:.0f}%  (quarter Kelly)")
    print(f"  Scan interval      : {interval}s")
    print(f"  Price source       : CLOB REST (live mid-point)")
    print("=" * 60)
    print()

    model      = load_model(model_path)
    scan_count = 0

    try:
        while True:
            scan_count += 1
            print(f"[Scan #{scan_count}]  {datetime.now().strftime('%H:%M:%S')}")

            # Fetch markets and live scores
            markets    = fetch_nba_markets()
            live_games = fetch_live_nba_games()

            if not markets:
                print("  No active NBA markets on Polymarket. Waiting...")

            elif not live_games:
                print("  No NBA games currently in progress.")

            else:
                all_alerts = []

                for game in live_games:
                    alerts = compute_edge(model, game, markets)
                    all_alerts.extend(alerts)

                if all_alerts:
                    # Sort by edge size descending
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


# ─────────────────────────────────────────────────────────────────────────────
# 11. MANUAL TEST MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_manual_test():
    """
    Tests the full pipeline using a hardcoded game state.
    Works any time — no live NBA game required.

    Scenario: Lakers lead Warriors 88-84 with 3:20 left in Q4.
    If Polymarket is pricing Lakers YES at 0.64, your model
    (if trained) should show edge worth buying.
    """
    print()
    print("=" * 60)
    print("  MANUAL TEST MODE")
    print("=" * 60)

    model = load_model()

    test_game = {
        "game_id":    "TEST",
        "home_team":  "Lakers",
        "away_team":  "Warriors",
        "home_city":  "Los Angeles",
        "away_city":  "Golden State",
        "home_score": 88,
        "away_score": 84,
        "period":     4,
        "clock":      "3:20",
        "status":     "Q4",
    }

    print(f"\n  Game:    {test_game['away_team']} @ {test_game['home_team']}")
    print(f"  Score:   {test_game['away_score']}-{test_game['home_score']}")
    print(f"  Period:  Q{test_game['period']}  {test_game['clock']}")

    # Compute win probability
    if model:
        prob = predict_win_prob(
            model,
            home_score = test_game["home_score"],
            away_score = test_game["away_score"],
            period     = test_game["period"],
            clock      = test_game["clock"],
        )
        print(f"\n  Model home win probability: {prob * 100:.1f}%")
        print(f"  Model away win probability: {(1 - prob) * 100:.1f}%")
    else:
        print("\n  [Skipping win probability — model not loaded]")
        prob = 0.73   # use a plausible placeholder

    # Test edge calculation against a simulated Polymarket price
    print("\n  Simulating Polymarket price of 0.64 for Lakers YES...")
    simulated_price = 0.64
    edge            = prob - simulated_price
    stake           = kelly_stake(edge, simulated_price)

    print(f"  Edge:         {edge * 100:.2f}%")
    print(f"  Direction:    {'BUY YES' if edge > 0 else 'BUY NO'}")
    print(f"  Kelly stake:  {stake * 100:.2f}% of bankroll")

    # Fetch real Polymarket markets (so you can see what's live)
    print("\n  Fetching real Polymarket NBA markets...")
    markets = fetch_nba_markets()

    if markets:
        print(f"\n  Active markets found ({len(markets)}):")
        for m in markets[:5]:
            print(f"    - {m['event_title']}  |  YES={m['yes_price']:.2f}  "
                  f"liq=${m['liquidity']:,.0f}")
        if len(markets) > 5:
            print(f"    ... and {len(markets) - 5} more")
    else:
        print("  No active NBA markets right now (try during game day)")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 12. LIST MARKETS MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_list_markets():
    """
    Fetches and prints all active NBA Polymarket markets with their
    current prices, liquidity, and URLs. Useful for sanity-checking
    before running the live scanner.
    """
    print()
    print("=" * 60)
    print("  ACTIVE NBA MARKETS ON POLYMARKET")
    print("=" * 60)

    markets = fetch_nba_markets()

    if not markets:
        print("  No active NBA game markets found.")
        print("  This is normal outside of game days.")
        return

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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    if mode == "test":
        run_manual_test()

    elif mode == "markets":
        run_list_markets()

    elif mode == "live":
        run_scanner()

    else:
        print(f"Unknown mode '{mode}'. Use: live | test | markets")
        sys.exit(1)
