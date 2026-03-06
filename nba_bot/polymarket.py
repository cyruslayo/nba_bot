"""
nba_bot/polymarket.py
=====================
Polymarket REST API functions — merged from both scanner doc scripts.

Provides:
  fetch_nba_markets()           — discover active NBA game markets
  fetch_clob_midpoint()         — real-time order-book mid-point price
  kelly_stake()                 — fractional Kelly criterion
  match_game_to_markets()       — match live NBA game to Polymarket markets
  compute_edge()                — edge computation using model.predict_home_win_prob
  print_alert()                 — pretty-print an edge alert
  print_no_edge()               — print "no edge found" summary line
"""

import json
import logging
from datetime import datetime

import requests

from nba_bot.config import (
    CLOB_API,
    GAMMA_API,
    GAME_TAG_ID,
    HEADERS,
    KELLY_FRACTION,
    MIN_EDGE,
    MIN_LIQUIDITY,
    NBA_SERIES_ID,
)
from nba_bot.model import predict_home_win_prob  # M4: no circular dependency

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Market discovery
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nba_markets(return_token_ids: bool = False):
    """
    Queries the Polymarket Gamma API for all active, open NBA game-winner markets.

    Args:
        return_token_ids: If False (default), returns list[dict] of market dicts.
                          If True, returns (markets, token_ids) — used by WS scanner
                          to subscribe to the Polymarket WebSocket.

    Market dicts contain:
        event_title, question, market_id, event_slug, outcomes,
        yes_price, no_price, clob_yes_id, clob_no_id, liquidity, url
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
        logger.error("Gamma API timed out. Will retry next scan.")
        return ([], []) if return_token_ids else []
    except requests.exceptions.HTTPError as e:
        logger.error("Gamma API HTTP error: %s", e)
        return ([], []) if return_token_ids else []
    except Exception as e:
        logger.error("Gamma API unexpected error: %s", e)
        return ([], []) if return_token_ids else []

    markets   = []
    token_ids = []
    offset    = 0
    PAGE_SIZE = 50

    while True:
        params["limit"]  = PAGE_SIZE
        params["offset"] = offset

        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            page_events = resp.json()
        except requests.exceptions.Timeout:
            logger.error("Gamma API timed out at offset=%d. Returning partial results.", offset)
            break
        except requests.exceptions.HTTPError as e:
            logger.error("Gamma API HTTP error at offset=%d: %s", offset, e)
            break
        except Exception as e:
            logger.error("Gamma API unexpected error at offset=%d: %s", offset, e)
            break

        if not page_events:
            break  # no more pages

        for event in page_events:
            for mkt in event.get("markets", []):
                try:
                    outcome_prices = json.loads(mkt.get("outcomePrices", "[]"))
                    outcomes       = json.loads(mkt.get("outcomes", "[]"))
                    clob_ids       = json.loads(mkt.get("clobTokenIds", "[]"))
                    liquidity      = float(mkt.get("liquidity", 0))

                    if len(outcome_prices) < 2 or liquidity < MIN_LIQUIDITY:
                        continue

                    yes_token = clob_ids[0] if len(clob_ids) > 0 else None
                    no_token  = clob_ids[1] if len(clob_ids) > 1 else None

                    mkt_data = {
                        "event_title": event.get("title", "Unknown"),
                        "question":    mkt.get("question", ""),
                        "market_id":   mkt.get("id"),
                        "event_slug":  event.get("slug", ""),
                        "outcomes":    outcomes,
                        "yes_price":   float(outcome_prices[0]),
                        "no_price":    float(outcome_prices[1]),
                        "clob_yes_id": yes_token,
                        "clob_no_id":  no_token,
                        "yes_token":   yes_token,
                        "no_token":    no_token,
                        "liquidity":   liquidity,
                        "url":         f"https://polymarket.com/event/{event.get('slug', '')}",
                    }
                    markets.append(mkt_data)

                    if return_token_ids:
                        if yes_token:
                            token_ids.append(yes_token)
                        if no_token:
                            token_ids.append(no_token)

                except (ValueError, KeyError, json.JSONDecodeError):
                    continue

        if len(page_events) < PAGE_SIZE:
            break  # last page
        offset += PAGE_SIZE

    logger.info("[Polymarket] %d active NBA game market(s) found", len(markets))
    return (markets, token_ids) if return_token_ids else markets


# ─────────────────────────────────────────────────────────────────────────────
# Live CLOB price
# ─────────────────────────────────────────────────────────────────────────────

def fetch_clob_midpoint(token_id: str) -> float | None:
    """
    Fetches the real-time order-book mid-point price for a single token
    from Polymarket's CLOB REST endpoint.

    More accurate than Gamma API cached outcomePrices (which can lag minutes).

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
# Kelly criterion
# ─────────────────────────────────────────────────────────────────────────────

def kelly_stake(
    edge: float,
    market_price: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    Computes recommended bet size as a fraction of bankroll (fractional Kelly).

    Full Kelly:  f* = edge / (1 - market_price)
    We apply `fraction` (default 0.25 = quarter Kelly) to reduce variance.

    Returns 0.0 if edge <= 0 or market_price is degenerate (0 or 1).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if edge <= 0:
        return 0.0

    full_kelly       = edge / (1 - market_price)
    fractional_kelly = full_kelly * fraction
    return round(max(fractional_kelly, 0.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Market matching
# ─────────────────────────────────────────────────────────────────────────────

def match_game_to_markets(game: dict, markets: list[dict]) -> list[tuple[dict, str]]:
    """
    Matches a live NBA game to its corresponding Polymarket markets.

    Polymarket event titles are typically formatted as:
        "Los Angeles Lakers vs Golden State Warriors"

    Matches by checking if either team name or city appears in event title.
    Returns list of (market, perspective) tuples where perspective is
    "home" or "away" indicating which team the market question is about.
    """
    home_name = game["home_team"].lower()
    away_name = game["away_team"].lower()
    home_city = game["home_city"].lower()
    away_city = game["away_city"].lower()

    matched = []

    for mkt in markets:
        title    = mkt["event_title"].lower()
        question = mkt["question"].lower()

        home_in_title = home_name in title or home_city in title
        away_in_title = away_name in title or away_city in title

        # Only match markets where BOTH teams appear in the event title
        # (reduces false matches like "Lakers season wins" in a game market)
        if not (home_in_title and away_in_title):
            continue

        if home_name in question or home_city in question:
            matched.append((mkt, "home"))
        elif away_name in question or away_city in question:
            matched.append((mkt, "away"))
        else:
            matched.append((mkt, "home"))  # default to home for generic titles

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# Edge computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge(
    model,
    game: dict,
    markets: list[dict],
    feature_cols: list | None = None,
    advanced_ctx: dict | None = None,
    price_cache: dict | None = None,
) -> list[dict]:
    """
    For a single live game, finds matching Polymarket markets, fetches the
    latest price, computes model win probability, and returns alert dicts
    for any market where |edge| >= MIN_EDGE.

    Args:
        model:        Trained win probability model (sklearn/XGBoost object).
        game:         Live game state dict from nba_live.fetch_live_nba_games().
        markets:      All active Polymarket NBA markets from fetch_nba_markets().
        feature_cols: Feature column list from model load (for order validation).
        advanced_ctx: Optional Tier-2 context dict {starttype_encoded,
                      player_quality_home, player_quality_away}.
        price_cache:  Optional dict {token_id: float} from ws_stream module.
                      If provided, WS prices are used instead of CLOB REST calls.

    Returns:
        List of alert dicts (empty if no edge found or model is None).
    """
    if model is None:
        return []

    home_prob = predict_home_win_prob(
        model       = model,
        feature_cols = feature_cols,
        home_score  = game["home_score"],
        away_score  = game["away_score"],
        period      = game["period"],
        clock       = game["clock"],
        advanced_ctx = advanced_ctx,
    )
    away_prob = 1.0 - home_prob

    matched_markets = match_game_to_markets(game, markets)
    if not matched_markets:
        return []

    alerts = []

    for mkt, perspective in matched_markets:
        model_prob = home_prob if perspective == "home" else away_prob

        yes_token   = mkt.get("clob_yes_id") or mkt.get("yes_token")
        live_yes    = None
        price_source = "Gamma (cached)"

        # Try WS cache first (if provided by ws scanner)
        if price_cache is not None and yes_token in price_cache:
            live_yes     = price_cache[yes_token]
            price_source = "WS (live)"

        # Fall back to CLOB REST
        if live_yes is None:
            live_yes = fetch_clob_midpoint(yes_token)
            if live_yes is not None:
                price_source = "CLOB (live)"

        # Final fallback: use Gamma cached price
        if live_yes is None:
            live_yes = mkt["yes_price"]

        edge = model_prob - live_yes

        if abs(edge) < MIN_EDGE:
            continue

        if edge > 0:
            direction   = "BUY YES"
            stake       = kelly_stake(edge, live_yes)
            enter_price = round(live_yes, 4)
        else:
            no_price    = 1.0 - live_yes
            direction   = "BUY NO"
            stake       = kelly_stake(abs(edge), no_price)
            enter_price = round(no_price, 4)

        alerts.append({
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "game":         f"{game['away_city']} {game['away_team']} @ {game['home_city']} {game['home_team']}",
            "score":        f"{game['away_score']}-{game['home_score']}",
            "period":       game["period"],
            "clock":        game["clock"],
            "market":       mkt["question"],
            "market_id":    mkt["market_id"],
            "event_slug":   mkt["event_slug"],
            "model_prob":   round(model_prob, 4),
            "poly_price":   round(live_yes, 4),
            "enter_price":  enter_price,
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
# Alert printers
# ─────────────────────────────────────────────────────────────────────────────

def print_alert(alert: dict):
    """Pretty-prints a single edge alert to the console."""
    edge_val = alert["edge"]
    bar      = "▲" if edge_val > 0 else "▼"

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  EDGE ALERT  {alert['timestamp']}")
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
    """Prints a one-line summary when no edge was found in a scan."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] Scanned {n_games} game(s) across {n_markets} market(s) "
          f"— no edge >= {MIN_EDGE * 100:.0f}%")
