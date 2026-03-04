"""
Polymarket WebSocket Live Edge Scanner
========================================
Real-time price streaming version of polymarket_live_scanner.py.
Instead of polling every 60s, this receives price updates the instant
they happen on Polymarket's order book — typically within milliseconds.

Architecture:
  Thread 1 (WS daemon)  — connects to Polymarket CLOB WebSocket, maintains
                          a local order book per token, updates price_cache
                          on every book / price_change / last_trade_price event
  Thread 2 (main loop)  — every NBA_POLL_INTERVAL seconds: fetches live NBA
                          scores, reads price_cache, computes model edge,
                          fires alerts

Shared state:
  price_cache     dict  { token_id -> float }   protected by price_lock
  orderbook_cache dict  { token_id -> {bids: [], asks: []} }

WebSocket endpoints (confirmed from Polymarket docs, Dec 2025):
  Market channel : wss://ws-subscriptions-clob.polymarket.com/ws/market
  No authentication required for the public market channel.

Message types handled:
  book             — full order book snapshot (on subscribe + after trades)
  price_change     — incremental order book update (on order place/cancel)
  last_trade_price — price of most recent trade execution

Dependencies:
  pip install websocket-client requests numpy joblib xgboost nba_api

Files required in same directory:
  xgb_win_prob_model.pkl   (trained by nba_win_probability.py)

Usage:
  python polymarket_ws_scanner.py            # full live scanner
  python polymarket_ws_scanner.py test       # test WS connection only
  python polymarket_ws_scanner.py markets    # list markets then exit
"""

import json
import time
import sys
import threading
import numpy as np
import joblib
import requests
from datetime import datetime
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Polymarket WebSocket — public market channel, no auth required
WSS_URL            = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Polymarket REST APIs
GAMMA_API          = "https://gamma-api.polymarket.com"
CLOB_API           = "https://clob.polymarket.com"

# NBA series + game-winner tag IDs on Polymarket
NBA_SERIES_ID      = 10345
GAME_TAG_ID        = 100639

# Edge detection
MIN_EDGE           = 0.05    # Alert when |model_prob - poly_price| >= 5%
MIN_LIQUIDITY      = 500     # Skip markets with < $500 liquidity
KELLY_FRACTION     = 0.25   # 25% of full Kelly (quarter Kelly)

# Timing
NBA_POLL_INTERVAL  = 30     # Fetch live NBA scores every 30 seconds
WS_PING_INTERVAL   = 20     # Send WebSocket keepalive ping every 20s
WS_PING_TIMEOUT    = 10     # Seconds to wait for pong before reconnecting
WS_RECONNECT_DELAY = 5      # Seconds to wait before reconnecting after drop

# Polymarket price logic:
# If spread (ask - bid) <= TIGHT_SPREAD_THRESHOLD, use midpoint (best_bid + best_ask) / 2
# If spread is wide, fall back to last_trade_price as a cleaner signal
TIGHT_SPREAD_THRESHOLD = 0.05

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (written by WS thread, read by main thread)
# ─────────────────────────────────────────────────────────────────────────────

price_lock  = threading.Lock()

# token_id -> float (0–1): the best current price estimate per token
price_cache: dict[str, float] = {}

# token_id -> {"bids": [(price, size), ...], "asks": [(price, size), ...]}
# Maintained as a local order book so we can recompute midpoint after
# each incremental price_change event without needing a REST call
orderbook_cache: dict[str, dict] = defaultdict(lambda: {"bids": {}, "asks": {}})

# token_id -> float: last trade price (used when spread is wide)
last_trade_cache: dict[str, float] = {}

# token_id -> market metadata dict (populated during market discovery)
token_to_market: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. POLYMARKET MARKET DISCOVERY (REST)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nba_markets() -> tuple[list[dict], list[str]]:
    """
    Queries Gamma API for all active NBA game-winner markets.

    Returns:
      markets   : list of market metadata dicts
      token_ids : list of YES token IDs to subscribe to on the WebSocket

    Also populates:
      price_cache     with REST-cached prices as fallback baseline
      token_to_market with metadata lookup by token ID
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
        print("[ERROR] Gamma API timed out during market discovery.")
        return [], []
    except Exception as e:
        print(f"[ERROR] Gamma API: {e}")
        return [], []

    markets   = []
    token_ids = []

    for event in events:
        for mkt in event.get("markets", []):
            try:
                outcome_prices = json.loads(mkt.get("outcomePrices", "[]"))
                clob_ids       = json.loads(mkt.get("clobTokenIds", "[]"))
                liquidity      = float(mkt.get("liquidity", 0))

                if len(outcome_prices) < 2 or liquidity < MIN_LIQUIDITY:
                    continue

                yes_token = clob_ids[0] if len(clob_ids) > 0 else None
                no_token  = clob_ids[1] if len(clob_ids) > 1 else None
                yes_price = float(outcome_prices[0])
                no_price  = float(outcome_prices[1])

                mkt_data = {
                    "event_title": event.get("title", "Unknown"),
                    "question":    mkt.get("question", ""),
                    "market_id":   mkt.get("id"),
                    "event_slug":  event.get("slug", ""),
                    "yes_token":   yes_token,
                    "no_token":    no_token,
                    "yes_price":   yes_price,    # REST-cached fallback
                    "no_price":    no_price,
                    "liquidity":   liquidity,
                    "url":         f"https://polymarket.com/event/{event.get('slug', '')}",
                }
                markets.append(mkt_data)

                # Seed price_cache with REST values so edge checks work
                # immediately before WS delivers its first book snapshot
                with price_lock:
                    if yes_token:
                        price_cache[yes_token] = yes_price
                        token_to_market[yes_token] = {**mkt_data, "token_side": "yes"}
                        token_ids.append(yes_token)
                    if no_token:
                        price_cache[no_token] = no_price
                        token_to_market[no_token] = {**mkt_data, "token_side": "no"}
                        token_ids.append(no_token)

            except (ValueError, KeyError, json.JSONDecodeError):
                continue

    print(f"[Gamma] {len(markets)} NBA market(s) | {len(token_ids)} token(s) to stream")
    return markets, token_ids


# ─────────────────────────────────────────────────────────────────────────────
# 2. ORDER BOOK PRICE DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_price(token_id: str) -> tuple[float | None, str]:
    """
    Derives the best current price for a token using Polymarket's
    own pricing logic:

      1. If best_bid and best_ask both exist AND spread <= TIGHT_SPREAD_THRESHOLD:
           price = (best_bid + best_ask) / 2   [midpoint]
      2. Else if last_trade_price exists:
           price = last_trade_price
      3. Else if only one side of book exists:
           price = that side's best price
      4. Else: return None

    Returns:
      (price, source_label) where source_label is one of:
        "midpoint", "last_trade", "best_bid_only", "best_ask_only"
    """
    with price_lock:
        book       = orderbook_cache.get(token_id, {"bids": {}, "asks": {}})
        last_trade = last_trade_cache.get(token_id)

    bids = book["bids"]   # dict { price_str -> size_float }
    asks = book["asks"]

    best_bid = max((float(p) for p in bids if float(bids[p]) > 0), default=None)
    best_ask = min((float(p) for p in asks if float(asks[p]) > 0), default=None)

    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
        if spread <= TIGHT_SPREAD_THRESHOLD:
            return (best_bid + best_ask) / 2, "midpoint"
        elif last_trade is not None:
            return last_trade, "last_trade (wide spread)"
        else:
            return (best_bid + best_ask) / 2, "midpoint (wide spread)"

    elif last_trade is not None:
        return last_trade, "last_trade"
    elif best_bid is not None:
        return best_bid, "best_bid_only"
    elif best_ask is not None:
        return best_ask, "best_ask_only"

    return None, "no_data"


# ─────────────────────────────────────────────────────────────────────────────
# 3. WEBSOCKET PRICE STREAM
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketPriceStream:
    """
    Maintains a persistent WebSocket connection to Polymarket's CLOB
    market channel. Updates orderbook_cache, last_trade_cache, and
    price_cache on every incoming message.

    Handles:
      book             — full order book snapshot
                         Emitted on subscribe + after every trade execution.
                         Contains bids[] and asks[] as aggregate levels.

      price_change     — incremental level update
                         Emitted when orders are placed or cancelled.
                         Contains price_changes[] with per-token updates.
                         Each entry has: asset_id, price, size, side ("BUY"/"SELL")
                         size=0 means that price level was removed from the book.

      last_trade_price — price of the most recently executed trade.
                         Used as fallback when spread is wide.

    Auto-reconnects with exponential backoff on disconnect.
    """

    def __init__(self, token_ids: list[str]):
        self.token_ids    = token_ids
        self.ws           = None
        self.connected    = False
        self._stop        = threading.Event()
        self._reconnect_delay = WS_RECONNECT_DELAY
        self.messages_received = 0
        self.last_message_time = None

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_open(self, ws):
        self.connected            = True
        self._reconnect_delay     = WS_RECONNECT_DELAY   # reset backoff on success
        print(f"[WS] Connected → {WSS_URL}")
        print(f"[WS] Subscribing to {len(self.token_ids)} token(s)...")

        subscribe_msg = {
            "assets_ids": self.token_ids,
            "type":       "market",
        }
        ws.send(json.dumps(subscribe_msg))
        print("[WS] Subscription message sent. Awaiting book snapshots...")

    def on_message(self, ws, raw: str):
        """
        Dispatches incoming WebSocket messages to the appropriate handler.

        Messages can arrive as a single JSON object OR a JSON array of objects.
        We normalise both into a list before processing.
        """
        self.messages_received += 1
        self.last_message_time  = datetime.now()

        try:
            data     = json.loads(raw)
            messages = data if isinstance(data, list) else [data]

            for msg in messages:
                event_type = msg.get("event_type", "")

                if event_type == "book":
                    self._handle_book(msg)
                elif event_type == "price_change":
                    self._handle_price_change(msg)
                elif event_type == "last_trade_price":
                    self._handle_last_trade(msg)
                # tick_size_change, market_resolved etc. are ignored

        except json.JSONDecodeError:
            pass   # silently discard malformed frames
        except Exception:
            pass   # never crash the WS thread on a bad message

    def on_error(self, ws, error):
        print(f"[WS ERROR] {error}")
        self.connected = False

    def on_close(self, ws, code, reason):
        self.connected = False
        print(f"[WS] Disconnected — code={code}  reason={reason or 'none'}")

        if not self._stop.is_set():
            print(f"[WS] Reconnecting in {self._reconnect_delay}s...")
            time.sleep(self._reconnect_delay)
            # Exponential backoff up to 60s
            self._reconnect_delay = min(self._reconnect_delay * 2, 60)
            self._connect()

    # ── Message handlers ──────────────────────────────────────────────────────

    def _handle_book(self, msg: dict):
        """
        Full order book snapshot. Replaces the local book entirely.

        Message structure:
          {
            "event_type": "book",
            "asset_id":   "<token_id>",
            "bids": [{"price": "0.48", "size": "30"}, ...],
            "asks": [{"price": "0.52", "size": "25"}, ...]
          }
        """
        token_id = msg.get("asset_id", "")
        if not token_id:
            return

        bids_raw = msg.get("bids", [])
        asks_raw = msg.get("asks", [])

        # Build dict { price_str -> size_float } for O(1) level lookup
        bids = {b["price"]: float(b["size"]) for b in bids_raw if float(b.get("size", 0)) > 0}
        asks = {a["price"]: float(a["size"]) for a in asks_raw if float(a.get("size", 0)) > 0}

        with price_lock:
            orderbook_cache[token_id] = {"bids": bids, "asks": asks}

        # Immediately recompute and cache the derived price
        price, source = derive_price(token_id)
        if price is not None:
            with price_lock:
                price_cache[token_id] = price

    def _handle_price_change(self, msg: dict):
        """
        Incremental order book update. Applies level-by-level changes to the
        local book, then recomputes the derived price.

        Message structure:
          {
            "event_type": "price_change",
            "price_changes": [
              {
                "asset_id": "<token_id>",
                "price":    "0.51",
                "size":     "40",      ← new aggregate size at this level
                "side":     "SELL"     ← "BUY" (bid) or "SELL" (ask)
              },
              ...
            ]
          }

        size = "0" means that level was fully cancelled — remove it from book.
        """
        for change in msg.get("price_changes", []):
            try:
                token_id = change.get("asset_id", "")
                price_str = change.get("price", "")
                size      = float(change.get("size", 0))
                side      = change.get("side", "").upper()

                if not token_id or not price_str:
                    continue

                with price_lock:
                    book = orderbook_cache[token_id]

                    if side == "BUY":
                        if size > 0:
                            book["bids"][price_str] = size
                        else:
                            book["bids"].pop(price_str, None)
                    elif side == "SELL":
                        if size > 0:
                            book["asks"][price_str] = size
                        else:
                            book["asks"].pop(price_str, None)

                # Recompute derived price after update
                price, source = derive_price(token_id)
                if price is not None:
                    with price_lock:
                        price_cache[token_id] = price

            except (ValueError, KeyError):
                continue

    def _handle_last_trade(self, msg: dict):
        """
        Records the most recently executed trade price.
        Used as a pricing fallback when the spread is wide.

        Message structure:
          {
            "event_type": "last_trade_price",
            "asset_id":   "<token_id>",
            "price":      "0.63",
            "size":       "100",
            "side":       "BUY"
          }
        """
        token_id    = msg.get("asset_id", "")
        price_str   = msg.get("price", "")

        if not token_id or not price_str:
            return

        try:
            trade_price = float(price_str)
            with price_lock:
                last_trade_cache[token_id] = trade_price

            # If no order book data yet, use last trade as current price
            if token_id not in price_cache or price_cache[token_id] == 0:
                with price_lock:
                    price_cache[token_id] = trade_price

        except ValueError:
            pass

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self):
        """Opens the WebSocket. Blocks — always call via start() which runs in a thread."""
        try:
            import websocket as ws_lib
        except ImportError:
            print("[ERROR] websocket-client not installed. Run: pip install websocket-client")
            return

        self.ws = ws_lib.WebSocketApp(
            WSS_URL,
            on_open    = self.on_open,
            on_message = self.on_message,
            on_error   = self.on_error,
            on_close   = self.on_close,
        )
        self.ws.run_forever(
            ping_interval = WS_PING_INTERVAL,
            ping_timeout  = WS_PING_TIMEOUT,
        )

    def start(self) -> threading.Thread:
        """
        Launches the WebSocket listener in a background daemon thread.
        Daemon threads are killed automatically when the main process exits.
        """
        t = threading.Thread(
            target = self._connect,
            daemon = True,
            name   = "polymarket-ws-listener",
        )
        t.start()
        print("[WS] Listener thread started")
        return t

    def stop(self):
        """Cleanly closes the WebSocket connection."""
        self._stop.set()
        if self.ws:
            self.ws.close()

    def status(self) -> str:
        """Returns a one-line status string for display in the scanner loop."""
        age = ""
        if self.last_message_time:
            secs = (datetime.now() - self.last_message_time).seconds
            age  = f", last msg {secs}s ago"
        state = "CONNECTED" if self.connected else "DISCONNECTED"
        return f"{state} | {self.messages_received} msgs received{age}"


# ─────────────────────────────────────────────────────────────────────────────
# 4. WIN PROBABILITY MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str = "xgb_win_prob_model.pkl"):
    """Loads trained XGBoost model from disk."""
    try:
        model = joblib.load(path)
        print(f"[Model] Loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"[WARN] Model not found at '{path}'.")
        print("       Run nba_win_probability.py first to train the model.")
        return None


def predict_win_prob(
    model,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
) -> float:
    """
    Returns home-team win probability (0–1) given live game state.

    Features:
      score_diff    = home_score - away_score
      time_remaining = total seconds left in regulation
      time_pressure  = score_diff / sqrt(time_remaining + 1)
      game_progress  = 1 - (time_remaining / 2880)
      period         = current quarter
    """
    try:
        mins, secs     = map(int, clock.split(":"))
        time_in_period = mins * 60 + secs
    except Exception:
        time_in_period = 720

    if period <= 4:
        time_remaining = (4 - period) * 720 + time_in_period
    else:
        time_remaining = max(0, time_in_period)

    score_diff    = home_score - away_score
    time_pressure = score_diff / np.sqrt(time_remaining + 1)
    game_progress = 1 - (time_remaining / 2880)

    X = np.array([[score_diff, time_remaining, time_pressure, game_progress, period]])
    return float(model.predict_proba(X)[0][1])


# ─────────────────────────────────────────────────────────────────────────────
# 5. KELLY CRITERION
# ─────────────────────────────────────────────────────────────────────────────

def kelly_stake(edge: float, market_price: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Fractional Kelly stake as fraction of bankroll.

    Full Kelly:  f* = edge / (1 - market_price)
    We use 25% of full Kelly to reduce variance significantly.
    """
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0.0
    return round(max((edge / (1 - market_price)) * fraction, 0.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 6. LIVE NBA SCORE FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_live_nba_games() -> list[dict]:
    """
    Fetches all NBA games currently in progress via nba_api live scoreboard.
    Returns a list of game state dicts.
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
        if "Q" not in status and "Halftime" not in status:
            continue

        period    = g.get("period", 1)
        raw_clock = g.get("gameClock", "PT12M00.00S")

        # Convert ISO 8601 "PT5M32.00S" → "5:32"
        try:
            raw_clock  = raw_clock.replace("PT", "").replace("S", "")
            mins, secs = raw_clock.split("M")
            clock      = f"{int(float(mins))}:{int(float(secs)):02d}"
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

    return live_games


# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKET MATCHING
# ─────────────────────────────────────────────────────────────────────────────

def match_game_to_markets(game: dict, markets: list[dict]) -> list[tuple[dict, str]]:
    """
    Matches a live NBA game to its Polymarket markets by team name/city.
    Returns list of (market, perspective) where perspective is "home" or "away".
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

        if not (home_in_title and away_in_title):
            continue

        if home_name in question or home_city in question:
            matched.append((mkt, "home"))
        elif away_name in question or away_city in question:
            matched.append((mkt, "away"))
        else:
            matched.append((mkt, "home"))

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# 8. EDGE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge(model, game: dict, markets: list[dict]) -> list[dict]:
    """
    For a single live game, computes edge vs all matching Polymarket markets.
    Uses WebSocket-streamed prices from price_cache.

    Returns list of alert dicts where |edge| >= MIN_EDGE.
    """
    if model is None:
        return []

    home_prob = predict_win_prob(
        model,
        home_score = game["home_score"],
        away_score = game["away_score"],
        period     = game["period"],
        clock      = game["clock"],
    )
    away_prob = 1.0 - home_prob

    matched = match_game_to_markets(game, markets)
    if not matched:
        return []

    alerts = []

    for mkt, perspective in matched:
        model_prob = home_prob if perspective == "home" else away_prob
        yes_token  = mkt.get("yes_token")

        # Get the freshest price from WS cache
        price, price_source = derive_price(yes_token) if yes_token else (None, "none")

        # Fall back to REST-cached price if WS hasn't delivered a snapshot yet
        if price is None:
            with price_lock:
                price = price_cache.get(yes_token, mkt["yes_price"])
            price_source = "REST fallback"

        edge = model_prob - price

        if abs(edge) < MIN_EDGE:
            continue

        if edge > 0:
            direction = "BUY YES"
            stake     = kelly_stake(edge, price)
        else:
            no_price  = 1.0 - price
            direction = "BUY NO"
            stake     = kelly_stake(abs(edge), no_price)

        alerts.append({
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "game":         f"{game['away_city']} {game['away_team']} @ "
                            f"{game['home_city']} {game['home_team']}",
            "score":        f"{game['away_score']}-{game['home_score']}",
            "period":       game["period"],
            "clock":        game["clock"],
            "market":       mkt["question"],
            "model_prob":   round(model_prob, 4),
            "poly_price":   round(price, 4),
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
    bar = "▲" if alert["edge"] > 0 else "▼"
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


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN SCANNER LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_scanner(model_path: str = "xgb_win_prob_model.pkl"):
    """
    Main entry point. Starts the WebSocket listener in a background thread,
    then polls NBA live scores every NBA_POLL_INTERVAL seconds and checks
    for edge using the WS-streamed prices.

    Price updates arrive in real time via WebSocket.
    NBA score updates arrive every 30 seconds via REST.
    The bottleneck is therefore the NBA poll, not the prices.
    """
    print()
    print("=" * 60)
    print("  NBA × POLYMARKET  —  WebSocket Edge Scanner")
    print("=" * 60)
    print(f"  Min edge threshold : {MIN_EDGE * 100:.0f}%")
    print(f"  Min liquidity      : ${MIN_LIQUIDITY:,}")
    print(f"  Kelly fraction     : {KELLY_FRACTION * 100:.0f}%")
    print(f"  NBA score poll     : every {NBA_POLL_INTERVAL}s")
    print(f"  Price updates      : real-time WebSocket push")
    print("=" * 60)
    print()

    model = load_model(model_path)

    # ── Step 1: Discover markets via REST ─────────────────────────────────────
    print("[INIT] Discovering active NBA markets...")
    markets, token_ids = fetch_nba_markets()

    if not token_ids:
        print("[WARN] No active NBA markets found on Polymarket.")
        print("       This is normal outside of game day. Try again later.")
        return

    # ── Step 2: Start WebSocket stream ────────────────────────────────────────
    stream = PolymarketPriceStream(token_ids)
    stream.start()

    # Wait for WS to connect and receive initial book snapshots
    print("[INIT] Waiting 4s for WebSocket book snapshots...")
    time.sleep(4)

    cached_count = len(price_cache)
    print(f"[INIT] {cached_count} price(s) in cache. Starting scan loop.\n")

    scan_count = 0

    try:
        while True:
            scan_count += 1
            ts = datetime.now().strftime("%H:%M:%S")

            with price_lock:
                n_prices = len(price_cache)

            print(f"[Scan #{scan_count}]  {ts}  |  WS: {stream.status()}  |  "
                  f"{n_prices} prices cached")

            live_games = fetch_live_nba_games()

            if not live_games:
                print("  No NBA games currently in progress.\n")

            else:
                all_alerts = []

                for game in live_games:
                    game_label = (f"{game['away_team']} @ {game['home_team']} "
                                  f"{game['away_score']}-{game['home_score']} "
                                  f"Q{game['period']} {game['clock']}")
                    print(f"  Checking: {game_label}")

                    alerts = compute_edge(model, game, markets)
                    all_alerts.extend(alerts)

                if all_alerts:
                    all_alerts.sort(key=lambda x: abs(x["edge"]), reverse=True)
                    print(f"\n  >>> {len(all_alerts)} ALERT(S) THIS SCAN <<<")
                    for alert in all_alerts:
                        print_alert(alert)
                else:
                    print(f"  No edge >= {MIN_EDGE * 100:.0f}% found "
                          f"across {len(live_games)} game(s).")

            print(f"\n  Sleeping {NBA_POLL_INTERVAL}s...  (Ctrl+C to stop)\n")
            time.sleep(NBA_POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[Scanner stopped]")
        stream.stop()


# ─────────────────────────────────────────────────────────────────────────────
# 11. TEST MODE — WS CONNECTION ONLY
# ─────────────────────────────────────────────────────────────────────────────

def run_ws_test():
    """
    Subscribes to real Polymarket NBA markets and prints the first 15
    raw WebSocket messages. Verifies connectivity without needing a trained
    model or live NBA game.

    If no NBA markets are currently active, falls back to a known
    high-volume token to confirm the connection works.
    """
    try:
        import websocket as ws_lib
    except ImportError:
        print("[ERROR] websocket-client not installed.")
        print("        Run: pip install websocket-client")
        return

    print()
    print("=" * 60)
    print("  WEBSOCKET CONNECTION TEST")
    print("=" * 60)

    print("[TEST] Fetching NBA market tokens from Gamma API...")
    markets, token_ids = fetch_nba_markets()

    if not token_ids:
        print("[TEST] No active NBA markets. Using a known test token...")
        # Fallback: a frequently active Polymarket token for testing
        token_ids = [
            "21742633143463906290569050155826241533067272736897614950488156847949938836455"
        ]

    test_tokens  = token_ids[:3]    # subscribe to first 3 only for the test
    msg_count    = [0]
    target_msgs  = 15

    def on_open(ws):
        print(f"[TEST] Connected to {WSS_URL}")
        print(f"[TEST] Subscribing to {len(test_tokens)} token(s)...")
        ws.send(json.dumps({"assets_ids": test_tokens, "type": "market"}))
        print(f"[TEST] Waiting for {target_msgs} messages...\n")

    def on_message(ws, raw):
        msg_count[0] += 1
        try:
            data     = json.loads(raw)
            messages = data if isinstance(data, list) else [data]
            for msg in messages:
                event_type = msg.get("event_type", "unknown")
                asset_id   = msg.get("asset_id", "?")

                if event_type == "book":
                    bids = msg.get("bids", [])
                    asks = msg.get("asks", [])
                    best_bid = max((float(b["price"]) for b in bids), default=None)
                    best_ask = min((float(a["price"]) for a in asks), default=None)
                    mid      = round((best_bid + best_ask) / 2, 4) if best_bid and best_ask else None
                    print(f"  [{msg_count[0]:02d}] book          "
                          f"token=...{asset_id[-8:]}  "
                          f"bid={best_bid}  ask={best_ask}  mid={mid}")

                elif event_type == "price_change":
                    changes = msg.get("price_changes", [])
                    for c in changes[:2]:   # show first 2 changes only
                        print(f"  [{msg_count[0]:02d}] price_change  "
                              f"token=...{c.get('asset_id','?')[-8:]}  "
                              f"side={c.get('side')}  "
                              f"price={c.get('price')}  "
                              f"size={c.get('size')}")

                elif event_type == "last_trade_price":
                    print(f"  [{msg_count[0]:02d}] last_trade    "
                          f"token=...{asset_id[-8:]}  "
                          f"price={msg.get('price')}  "
                          f"size={msg.get('size')}")

                else:
                    print(f"  [{msg_count[0]:02d}] {event_type:<14} "
                          f"token=...{asset_id[-8:]}")

        except Exception as e:
            print(f"  [{msg_count[0]:02d}] parse error: {e}  raw={raw[:60]}")

        if msg_count[0] >= target_msgs:
            print(f"\n[TEST] {target_msgs} messages received — WebSocket is working correctly.")
            ws.close()

    def on_error(ws, error):
        print(f"[TEST ERROR] {error}")

    def on_close(ws, code, reason):
        print(f"[TEST] Connection closed (code={code})")

    ws = ws_lib.WebSocketApp(
        WSS_URL,
        on_open    = on_open,
        on_message = on_message,
        on_error   = on_error,
        on_close   = on_close,
    )
    ws.run_forever(ping_interval=WS_PING_INTERVAL, ping_timeout=WS_PING_TIMEOUT)


# ─────────────────────────────────────────────────────────────────────────────
# 12. MARKETS LIST MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_list_markets():
    """Lists all active NBA Polymarket markets with token IDs and prices."""
    print()
    print("=" * 60)
    print("  ACTIVE NBA MARKETS ON POLYMARKET")
    print("=" * 60)

    markets, token_ids = fetch_nba_markets()

    if not markets:
        print("  No active NBA game markets right now.")
        print("  Normal outside of game days.")
        return

    for i, m in enumerate(markets, 1):
        print(f"\n  [{i}] {m['event_title']}")
        print(f"       Question  : {m['question']}")
        print(f"       YES token : {m.get('yes_token', 'N/A')}")
        print(f"       NO token  : {m.get('no_token', 'N/A')}")
        print(f"       YES price : {m['yes_price']:.3f}  "
              f"(implied {m['yes_price']*100:.1f}%)")
        print(f"       Liquidity : ${m['liquidity']:,.0f}")
        print(f"       URL       : {m['url']}")

    print(f"\n  Total: {len(token_ids)} token(s) to stream via WebSocket\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "live"

    if mode == "test":
        run_ws_test()

    elif mode == "markets":
        run_list_markets()

    elif mode == "live":
        run_scanner()

    else:
        print(f"Unknown mode '{mode}'.")
        print("Usage: python polymarket_ws_scanner.py [live|test|markets]")
        sys.exit(1)
