"""
nba_bot/ws_stream.py
====================
Polymarket WebSocket price stream — extracted verbatim from
docs/polymarket_ws_scanner.py.

Module-level shared state (price_cache, orderbook_cache, etc.) is
intentional: the WS daemon thread writes to these, and the main
scanner thread reads from them. All access is guarded by price_lock.

Exported:
  price_lock        — threading.Lock protecting all cache dicts
  price_cache       — {token_id: float} best current price per token
  orderbook_cache   — {token_id: {"bids": {...}, "asks": {...}}}
  last_trade_cache  — {token_id: float} last traded price
  token_to_market   — {token_id: market_dict} populated during discovery
  derive_price()    — compute best price from order book state
  PolymarketPriceStream — WebSocket listener class
"""

import json
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime

from nba_bot.config import TIGHT_SPREAD_THRESHOLD, WS_PING_INTERVAL, WS_PING_TIMEOUT, WS_RECONNECT_DELAY

logger = logging.getLogger(__name__)

WSS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED STATE  (written by WS thread, read by main thread)
# ─────────────────────────────────────────────────────────────────────────────

price_lock = threading.Lock()

# token_id -> float (0–1): best current price estimate per token
price_cache: dict[str, float] = {}

# token_id -> {"bids": {price_str: size_float}, "asks": {...}}
# Maintained as a local order book for incremental price_change updates
orderbook_cache: dict[str, dict] = defaultdict(lambda: {"bids": {}, "asks": {}})

# token_id -> float: price of last executed trade (wide-spread fallback)
last_trade_cache: dict[str, float] = {}

# token_id -> market metadata dict (populated during market discovery)
token_to_market: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Price derivation
# ─────────────────────────────────────────────────────────────────────────────

def derive_price(token_id: str) -> tuple[float | None, str]:
    """
    Derives the best current price for a token using Polymarket's pricing logic:

    1. If best_bid + best_ask both exist AND spread <= TIGHT_SPREAD_THRESHOLD:
         price = (best_bid + best_ask) / 2   [midpoint]
    2. Else if last_trade_price exists:
         price = last_trade_price
    3. Else if only one side of book exists:
         price = that side's best price
    4. Else: return (None, "no_data")

    Returns:
        (price, source_label) where source_label is one of:
          "midpoint", "last_trade", "best_bid_only", "best_ask_only", "no_data"
    """
    with price_lock:
        book       = orderbook_cache.get(token_id, {"bids": {}, "asks": {}})
        last_trade = last_trade_cache.get(token_id)

    bids = book["bids"]   # dict {price_str -> size_float}
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


def _store_price(token_id: str, price: float, source: str) -> None:
    with price_lock:
        price_cache[token_id] = {
            "price": price,
            "timestamp": time.time(),
            "source": source,
        }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket listener class
# ─────────────────────────────────────────────────────────────────────────────

class PolymarketPriceStream:
    """
    Maintains a persistent WebSocket connection to Polymarket's CLOB market
    channel. Updates orderbook_cache, last_trade_cache, and price_cache on
    every incoming message.

    Handles:
      book             — full order book snapshot (on subscribe + after trades)
      price_change     — incremental level update (order placed/cancelled)
      last_trade_price — price of most recently executed trade (wide-spread fallback)

    Auto-reconnects with exponential backoff on disconnect.

    Usage:
        stream = PolymarketPriceStream(token_ids)
        stream.start()  # launches daemon thread, returns immediately
        # Main thread reads price_cache / calls derive_price()
        stream.stop()   # cleanly closes WebSocket
    """

    def __init__(self, token_ids: list[str]):
        self.token_ids         = token_ids
        self.ws                = None
        self.connected         = False
        self._stop             = threading.Event()
        self._reconnect_delay  = WS_RECONNECT_DELAY
        self.messages_received = 0
        self.last_message_time = None

    # ── Event handlers ────────────────────────────────────────────────────────

    def on_open(self, ws):
        self.connected           = True
        self._reconnect_delay    = WS_RECONNECT_DELAY  # reset backoff on success
        logger.info("[WS] Connected → %s", WSS_URL)
        print(f"[WS] Connected")
        logger.info("[WS] Subscribing to %d token(s)...", len(self.token_ids))

        subscribe_msg = {
            "assets_ids": self.token_ids,
            "type":       "market",
        }
        ws.send(json.dumps(subscribe_msg))
        logger.info("[WS] Subscription message sent. Awaiting book snapshots...")

    def on_message(self, ws, raw: str):
        """
        Dispatches incoming WebSocket messages to the appropriate handler.
        Messages can arrive as a single JSON object OR a JSON array of objects.
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
                # tick_size_change, market_resolved, etc. are silently ignored

        except json.JSONDecodeError:
            pass   # discard malformed frames
        except Exception:
            pass   # never crash the WS thread on a bad message

    def on_error(self, ws, error):
        logger.error("[WS ERROR] %s", error)
        self.connected = False

    def on_close(self, ws, code, reason):
        self.connected = False
        logger.warning("[WS] Disconnected — code=%s  reason=%s", code, reason or "none")

        if not self._stop.is_set():
            logger.info("[WS] Reconnecting in %ds...", self._reconnect_delay)
            time.sleep(self._reconnect_delay)
            # Exponential backoff, capped at 60s
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
            "asks": [{"price": "0.52", "size": "25"}, ...],
          }
        """
        token_id = msg.get("asset_id", "")
        if not token_id:
            return

        bids_raw = msg.get("bids", [])
        asks_raw = msg.get("asks", [])

        bids = {b["price"]: float(b["size"]) for b in bids_raw if float(b.get("size", 0)) > 0}
        asks = {a["price"]: float(a["size"]) for a in asks_raw if float(a.get("size", 0)) > 0}

        with price_lock:
            orderbook_cache[token_id] = {"bids": bids, "asks": asks}

        price, source = derive_price(token_id)
        if price is not None:
            _store_price(token_id, price, source)

    def _handle_price_change(self, msg: dict):
        """
        Incremental order book update. Applies level-by-level changes,
        then recomputes the derived price.

        size="0" means that level was fully cancelled — remove from book.
        """
        for change in msg.get("price_changes", []):
            try:
                token_id  = change.get("asset_id", "")
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

                price, source = derive_price(token_id)
                if price is not None:
                    _store_price(token_id, price, source)

            except (ValueError, KeyError):
                continue

    def _handle_last_trade(self, msg: dict):
        """
        Records the most recently executed trade price.
        Used as pricing fallback when the spread is wide.
        """
        token_id  = msg.get("asset_id", "")
        price_str = msg.get("price", "")

        if not token_id or not price_str:
            return

        try:
            trade_price = float(price_str)
            with price_lock:
                last_trade_cache[token_id] = trade_price

            # If no order book data yet, seed price_cache with last trade
            cached_quote = price_cache.get(token_id)
            if not isinstance(cached_quote, dict) or float(cached_quote.get("price", 0) or 0) <= 0:
                _store_price(token_id, trade_price, "last_trade")

        except ValueError:
            pass

    # ── Connection management ─────────────────────────────────────────────────

    def _connect(self):
        """Opens the WebSocket. Blocks — always call via start()."""
        try:
            import websocket as ws_lib
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
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
        Returns the thread object.
        """
        t = threading.Thread(
            target = self._connect,
            daemon = True,
            name   = "polymarket-ws-listener",
        )
        t.start()
        logger.info("[WS] Listener thread started")
        return t

    def stop(self):
        """Cleanly closes the WebSocket connection."""
        self._stop.set()
        if self.ws:
            self.ws.close()

    def status(self) -> str:
        """Returns a one-line status string for display in the scanner loop."""
        age   = ""
        if self.last_message_time:
            secs = (datetime.now() - self.last_message_time).seconds
            age  = f", last msg {secs}s ago"
        state = "CONNECTED" if self.connected else "DISCONNECTED"
        return f"{state} | {self.messages_received} msgs received{age}"
