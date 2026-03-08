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
import re
import time
from datetime import datetime

import requests

from nba_bot.config import (
    CLOB_API,
    EDGE_CONFIDENCE_THRESHOLD,
    EDGE_DECAY_FACTOR,
    GAMMA_API,
    GAME_TAG_ID,
    HARDENED_MIN_LIQUIDITY,
    HEADERS,
    KELLY_FRACTION,
    MAX_STAKE_LIQUIDITY_PCT,
    MIN_EDGE,
    MIN_LIQUIDITY,
    NBA_SERIES_ID,
    PRICE_STALE_THRESHOLD_SEC,
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


def fetch_order_book(token_id: str) -> dict | None:
    if not token_id:
        return None

    try:
        resp = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            headers=HEADERS,
            timeout=5,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception as e:
        logger.debug("CLOB /book error for token %s: %s", token_id, e)
        return None


def compute_vwap_from_book(book: dict, side: str, stake: float) -> tuple[float | None, float]:
    levels = book.get("asks", []) if side == "BUY" else book.get("bids", [])
    if not isinstance(levels, list) or not levels:
        return None, 0.0

    total_cost = 0.0
    total_shares = 0.0
    remaining = stake

    for level in levels:
        try:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
        except (AttributeError, TypeError, ValueError):
            continue

        if price <= 0 or price >= 1 or size <= 0:
            continue

        level_cost = price * size
        if level_cost <= remaining:
            total_cost += level_cost
            total_shares += size
            remaining -= level_cost
        else:
            shares_from_level = remaining / price
            total_cost += remaining
            total_shares += shares_from_level
            remaining = 0.0
            break

    if total_shares <= 0 or remaining > 1e-6:
        return None, total_shares

    return total_cost / total_shares, total_shares


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


def _normalize_market_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_market_type(question: str) -> str:
    normalized = _normalize_market_text(question)
    if "1h" in normalized or "1st half" in normalized or "first half" in normalized:
        return "first_half"
    if "spread" in normalized:
        return "spread"
    if "o u" in normalized or "over under" in normalized or "total" in normalized:
        return "total"
    return "moneyline"


def _team_aliases(game: dict, side: str) -> set[str]:
    aliases = {
        game.get(f"{side}_team", ""),
        game.get(f"{side}_city", ""),
        f"{game.get(f'{side}_city', '')} {game.get(f'{side}_team', '')}".strip(),
    }
    return {
        _normalize_market_text(alias)
        for alias in aliases
        if alias
    }


def _outcome_matches_team(label: str, aliases: set[str]) -> bool:
    normalized_label = _normalize_market_text(label)
    if not normalized_label:
        return False
    return any(
        alias and (alias == normalized_label or alias in normalized_label or normalized_label in alias)
        for alias in aliases
    )


def _resolve_team_outcome_mapping(game: dict, market: dict) -> dict[str, int] | None:
    outcomes = market.get("outcomes") or []
    if len(outcomes) < 2:
        return None

    home_aliases = _team_aliases(game, "home")
    away_aliases = _team_aliases(game, "away")
    home_matches = [idx for idx, label in enumerate(outcomes[:2]) if _outcome_matches_team(label, home_aliases)]
    away_matches = [idx for idx, label in enumerate(outcomes[:2]) if _outcome_matches_team(label, away_aliases)]

    if len(home_matches) != 1 or len(away_matches) != 1:
        return None
    if home_matches[0] == away_matches[0]:
        return None

    return {
        "home_index": home_matches[0],
        "away_index": away_matches[0],
    }


def _load_outcome_price(token_id: str | None, fallback_price: float | None, price_cache: dict | None) -> tuple[float | None, str, float | None]:
    live_price = None
    price_source = "Gamma (cached)"
    price_timestamp = None

    if price_cache is not None and token_id in price_cache:
        cached_quote = price_cache[token_id]
        if isinstance(cached_quote, dict):
            try:
                live_price = float(cached_quote.get("price"))
            except (TypeError, ValueError):
                live_price = None
            timestamp_value = cached_quote.get("timestamp")
            if isinstance(timestamp_value, (int, float)):
                price_timestamp = float(timestamp_value)
            cached_source = cached_quote.get("source")
            if isinstance(cached_source, str) and cached_source:
                price_source = f"WS ({cached_source})"
        else:
            try:
                live_price = float(cached_quote)
            except (TypeError, ValueError):
                live_price = None
        if live_price is not None:
            if price_timestamp is None:
                price_timestamp = time.time()
            if price_source == "Gamma (cached)":
                price_source = "WS (live)"

    if live_price is None:
        live_price = fetch_clob_midpoint(token_id)
        if live_price is not None:
            price_source = "CLOB (live)"
            price_timestamp = time.time()

    if live_price is None and fallback_price is not None:
        try:
            live_price = float(fallback_price)
        except (TypeError, ValueError):
            live_price = None

    return live_price, price_source, price_timestamp


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
    hardened: bool = False,
    bankroll: float = 0.0,
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
        hardened:     If True, applies additional risk controls (stale price checks,
                      edge calibration, liquidity-based stake caps).
        bankroll:     If > 0, applies liquidity-based stake caps.

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

    for mkt, _ in matched_markets:
        market_type = _classify_market_type(mkt.get("question", ""))
        if market_type != "moneyline":
            if hardened:
                logger.debug(
                    "Skipping hardened market — unsupported market type | market_id=%s | market_type=%s | question=%s",
                    mkt.get("market_id"),
                    market_type,
                    mkt.get("question", ""),
                )
            continue

        outcome_mapping = _resolve_team_outcome_mapping(game, mkt)
        if outcome_mapping is None:
            if hardened:
                logger.debug(
                    "Skipping hardened market — unresolved team outcome mapping | market_id=%s | outcomes=%s | question=%s",
                    mkt.get("market_id"),
                    mkt.get("outcomes"),
                    mkt.get("question", ""),
                )
            continue

        yes_token   = mkt.get("clob_yes_id") or mkt.get("yes_token")
        no_token    = mkt.get("clob_no_id") or mkt.get("no_token")
        quotes = {
            0: _load_outcome_price(yes_token, mkt.get("yes_price"), price_cache),
            1: _load_outcome_price(no_token, mkt.get("no_price"), price_cache),
        }

        home_index = outcome_mapping["home_index"]
        away_index = outcome_mapping["away_index"]
        home_price, home_price_source, home_price_timestamp = quotes[home_index]
        away_price, away_price_source, away_price_timestamp = quotes[away_index]

        if home_price is None or away_price is None:
            if hardened:
                logger.debug(
                    "Skipping hardened market — missing live outcome price | market_id=%s | home_price=%s | away_price=%s | question=%s",
                    mkt.get("market_id"),
                    home_price,
                    away_price,
                    mkt.get("question", ""),
                )
            continue

        candidates = [
            {
                "team_side": "home",
                "outcome_index": home_index,
                "model_prob": home_prob,
                "market_price": home_price,
                "price_source": home_price_source,
                "price_timestamp": home_price_timestamp,
                "edge": home_prob - home_price,
            },
            {
                "team_side": "away",
                "outcome_index": away_index,
                "model_prob": away_prob,
                "market_price": away_price,
                "price_source": away_price_source,
                "price_timestamp": away_price_timestamp,
                "edge": away_prob - away_price,
            },
        ]
        candidates = [candidate for candidate in candidates if candidate["edge"] >= MIN_EDGE]
        if not candidates:
            if hardened:
                best_edge = max(home_prob - home_price, away_prob - away_price)
                logger.debug(
                    "Skipping hardened market — below min edge after outcome mapping | market_id=%s | best_edge=%.4f | threshold=%.4f | question=%s",
                    mkt.get("market_id"),
                    best_edge,
                    MIN_EDGE,
                    mkt.get("question", ""),
                )
            continue

        selected = max(candidates, key=lambda candidate: candidate["edge"])
        model_prob = selected["model_prob"]
        selected_price = selected["market_price"]
        price_source = selected["price_source"]
        price_timestamp = selected["price_timestamp"]
        edge = selected["edge"]
        selected_outcome_index = selected["outcome_index"]
        trade_token = yes_token if selected_outcome_index == 0 else no_token
        direction = "BUY YES" if selected_outcome_index == 0 else "BUY NO"

        if hardened:
            price_age = None if price_timestamp is None else (time.time() - price_timestamp)
            if price_age is None or price_age > PRICE_STALE_THRESHOLD_SEC:
                logger.info(
                    "Skipping hardened market — stale quote | market_id=%s | source=%s | age=%s | threshold=%ss | question=%s",
                    mkt.get("market_id"),
                    price_source,
                    "unknown" if price_age is None else f"{price_age:.1f}s",
                    PRICE_STALE_THRESHOLD_SEC,
                    mkt.get("question", ""),
                )
                continue

        if hardened:
            calibrated_edge = edge * EDGE_DECAY_FACTOR
            calibrated_edge = max(
                min(calibrated_edge, EDGE_CONFIDENCE_THRESHOLD),
                -EDGE_CONFIDENCE_THRESHOLD,
            )
        else:
            calibrated_edge = edge

        liquidity = float(mkt.get("liquidity", 0) or 0)
        if hardened:
            if liquidity < HARDENED_MIN_LIQUIDITY:
                logger.debug(
                    "Skipping hardened market — below hardened liquidity | market_id=%s | liquidity=%.2f | threshold=%s | question=%s",
                    mkt.get("market_id"),
                    liquidity,
                    HARDENED_MIN_LIQUIDITY,
                    mkt.get("question", ""),
                )
                continue

        stake = kelly_stake(abs(calibrated_edge), selected_price)
        enter_price = round(selected_price, 4)

        if hardened and bankroll > 0:
            max_stake_fraction = (liquidity * MAX_STAKE_LIQUIDITY_PCT) / bankroll
            stake = min(stake, max(max_stake_fraction, 0.0))

        if stake <= 0:
            if hardened:
                logger.debug(
                    "Skipping hardened market — zero stake after caps | market_id=%s | liquidity=%.2f | bankroll=%.2f | question=%s",
                    mkt.get("market_id"),
                    liquidity,
                    bankroll,
                    mkt.get("question", ""),
                )
            continue

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
            "poly_price":   round(selected_price, 4),
            "enter_price":  enter_price,
            "edge":         round(edge, 4),
            "calibrated_edge": round(calibrated_edge, 4),
            "edge_pct":     f"{round(edge * 100, 2)}%",
            "direction":    direction,
            "kelly_stake":  f"{round(stake * 100, 2)}% of bankroll",
            "raw_stake":    stake,
            "liquidity":    liquidity,
            "price_source": price_source,
            "yes_token":    yes_token,
            "no_token":     no_token,
            "trade_token":  trade_token,
            "selected_outcome": (mkt.get("outcomes") or [None, None])[selected_outcome_index],
            "clob_yes_id":  mkt.get("clob_yes_id"),
            "clob_no_id":   mkt.get("clob_no_id"),
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
