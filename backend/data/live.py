import asyncio
import json
import logging
import time
from collections import deque
import websockets
from backend.data.history import BinanceREST

log = logging.getLogger(__name__)


class OrderBook:
    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_id = 0
        self.valid = False
        self.bridged = False

    def snapshot(self, data: dict) -> None:
        self.bids = {float(p): float(q) for p, q in data["bids"]}
        self.asks = {float(p): float(q) for p, q in data["asks"]}
        self.last_id = int(data["lastUpdateId"])
        self.valid = True
        self.bridged = False

    def delta(self, event: dict) -> bool:
        if not self.valid:
            return False
        if event["u"] <= self.last_id:
            return True
        if not (event["U"] <= self.last_id + 1 <= event["u"]):
            self.valid = False
            return False
        for side, key in [(self.bids, "b"), (self.asks, "a")]:
            for p, q in event[key]:
                price, quantity = float(p), float(q)
                if quantity == 0:
                    side.pop(price, None)
                else:
                    side[price] = quantity
        self.last_id = int(event["u"])
        self.bridged = True
        return True

    def features(self) -> dict:
        if not self.valid or not self.bridged or not self.bids or not self.asks:
            return {"book_valid": False}
        bid, ask = max(self.bids), min(self.asks)
        if bid >= ask:
            self.valid = False
            return {"book_valid": False}
        bv = sum(p * self.bids[p] for p in sorted(self.bids, reverse=True)[:20])
        av = sum(p * self.asks[p] for p in sorted(self.asks)[:20])
        return {
            "book_valid": True,
            "spread_bps": (ask - bid) / ((ask + bid) / 2) * 1e4,
            "imbalance": (bv - av) / (bv + av),
            "depth_usdt": bv + av,
        }


class LiveMarket:
    def __init__(self, symbol: str, rest: BinanceREST, ws_base: str):
        self.symbol, self.rest, self.ws_base = symbol, rest, ws_base
        self.book = OrderBook()
        self.trades: deque = deque(maxlen=5000)
        self.aggregate_events: deque = deque(maxlen=100000)
        self.last_received = 0.0
        self.connected = False

    def status(self) -> dict:
        age = time.time() - self.last_received if self.last_received else None
        trades = [t for t in self.trades if time.time() - t[0] < 60]
        signed = sum(t[1] for t in trades)
        total = sum(abs(t[1]) for t in trades)
        return {
            "connected": self.connected,
            "age_seconds": age,
            **self.book.features(),
            "trade_flow_imbalance": signed / total if total else None,
        }

    def orderflow_snapshot(self, now_ms: int | None = None) -> dict:
        """Return anonymous, venue-level aggressor flow and the current sequenced L2 book."""
        now_ms = now_ms or int(time.time() * 1000)
        observations = []
        oldest_needed = now_ms - 15 * 60_000
        for event in reversed(self.aggregate_events):
            try:
                timestamp = int(event.get("T", event.get("E", 0)))
                if timestamp and timestamp < oldest_needed:
                    break
                price, quantity = float(event["p"]), float(event["q"])
                buyer_maker = event["m"] is True
                if timestamp <= 0 or timestamp > now_ms + 1000 or price <= 0 or quantity <= 0:
                    continue
                observations.append(
                    {
                        "aggregate_id": int(event["a"]),
                        "time": timestamp,
                        "price": price,
                        "quantity": quantity,
                        "notional_usdt": price * quantity,
                        "side": "SELL" if buyer_maker else "BUY",
                        "buyer_is_maker": buyer_maker,
                        "underlying_trade_count": max(1, int(event.get("l", 0)) - int(event.get("f", 0)) + 1),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
        observations.reverse()

        trailing = observations[-1000:]
        notionals = sorted(item["notional_usdt"] for item in trailing)
        threshold = None
        if len(notionals) >= 20:
            index = min(len(notionals) - 1, int(0.95 * (len(notionals) - 1)))
            threshold = notionals[index]
        for item in trailing:
            item["relative_large"] = threshold is not None and item["notional_usdt"] >= threshold

        flow_windows = {}
        for minutes in (1, 5, 15):
            cutoff = now_ms - minutes * 60_000
            window = [item for item in observations if cutoff <= item["time"] <= now_ms]
            buy = sum(item["notional_usdt"] for item in window if item["side"] == "BUY")
            sell = sum(item["notional_usdt"] for item in window if item["side"] == "SELL")
            total = buy + sell
            flow_windows[f"{minutes}m"] = {
                "buy_notional_usdt": buy,
                "sell_notional_usdt": sell,
                "net_taker_delta_usdt": buy - sell,
                "imbalance": (buy - sell) / total if total else None,
                "aggregate_trade_count": len(window),
            }

        book_features = self.book.features()
        depth_buckets = []
        if book_features.get("book_valid"):
            bid, ask = max(self.book.bids), min(self.book.asks)
            mid = (bid + ask) / 2
            bid_size, ask_size = self.book.bids[bid], self.book.asks[ask]
            microprice = (ask * bid_size + bid * ask_size) / max(bid_size + ask_size, 1e-12)
            bucket_bps, range_bps = 5.0, 50.0
            bucket_count = int(range_bps / bucket_bps)
            depth_buckets = [
                {
                    "distance_bps": (index + 0.5) * bucket_bps,
                    "bid_band_low": mid * (1 - (index + 1) * bucket_bps / 10_000),
                    "bid_band_high": mid * (1 - index * bucket_bps / 10_000),
                    "ask_band_low": mid * (1 + index * bucket_bps / 10_000),
                    "ask_band_high": mid * (1 + (index + 1) * bucket_bps / 10_000),
                    "bid_notional_usdt": 0.0,
                    "ask_notional_usdt": 0.0,
                }
                for index in range(bucket_count)
            ]
            for price, quantity in self.book.bids.items():
                distance = (mid - price) / mid * 10_000
                if 0 <= distance < range_bps:
                    index = min(bucket_count - 1, int(distance / bucket_bps))
                    depth_buckets[index]["bid_notional_usdt"] += price * quantity
            for price, quantity in self.book.asks.items():
                distance = (price - mid) / mid * 10_000
                if 0 <= distance < range_bps:
                    index = min(bucket_count - 1, int(distance / bucket_bps))
                    depth_buckets[index]["ask_notional_usdt"] += price * quantity
            book_features.update(
                {
                    "mid": mid,
                    "bid": bid,
                    "ask": ask,
                    "best_bid_size": bid_size,
                    "best_ask_size": ask_size,
                    "microprice": microprice,
                    "microprice_offset_bps": (microprice / mid - 1) * 10_000,
                    "top_of_book_imbalance": (bid_size - ask_size) / max(bid_size + ask_size, 1e-12),
                    "sequence_id": self.book.last_id,
                }
            )

        age = time.time() - self.last_received if self.last_received else None
        recent_trades = [
            {**item, "relative_large": threshold is not None and item["notional_usdt"] >= threshold}
            for item in reversed(observations[-100:])
        ]
        return {
            "generated_at": now_ms,
            "symbol": self.symbol,
            "venue": "Binance Spot",
            "connected": self.connected,
            "feed_age_seconds": age,
            "book": book_features,
            "depth_map": {
                "status": "live sequenced snapshot" if depth_buckets else "unavailable; book is not synchronized",
                "range_bps": 50,
                "bucket_bps": 5,
                "basis": "current aggregate L2 quantity by price band; not historical resting-liquidity replay",
                "buckets": depth_buckets,
            },
            "flow_windows": flow_windows,
            "large_print_threshold_usdt": threshold,
            "large_print_basis": "95th percentile of up to 1,000 recent Binance aggregate-trade notionals; relative size, not trader identity",
            "large_print_sample_count": len(trailing),
            "identity_basis": "anonymous aggregate trades; side identifies the taker aggressor from buyer-is-maker, not a person or account",
            "recent_trades": recent_trades,
        }

    async def run(self) -> None:
        streams = f"{self.symbol.lower()}@depth@100ms/{self.symbol.lower()}@aggTrade"
        url = self.ws_base.rsplit("/ws", 1)[0] + "/stream?streams=" + streams
        delay = 1
        while True:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, max_queue=4096) as ws:
                    self.connected = True
                    self.book.valid = False
                    # Buffer messages while the REST snapshot is fetched.
                    snapshot = asyncio.create_task(
                        self.rest.get("/api/v3/depth", {"symbol": self.symbol, "limit": 1000})
                    )
                    pending: list[dict] = []
                    deadline = time.monotonic() + 23.5 * 3600
                    while time.monotonic() < deadline:
                        message = json.loads(await asyncio.wait_for(ws.recv(), timeout=45))["data"]
                        self.last_received = time.time()
                        if snapshot.done() and not self.book.valid:
                            self.book.snapshot(snapshot.result())
                            for event in pending:
                                if not self.book.delta(event):
                                    raise RuntimeError("Depth gap during snapshot sync")
                            pending.clear()
                        if message.get("e") == "depthUpdate":
                            if not self.book.valid:
                                pending.append(message)
                                if len(pending) > 4096:
                                    raise RuntimeError("Depth snapshot buffer exceeded")
                            elif not self.book.delta(message):
                                raise RuntimeError("Depth sequence gap; rebuilding book")
                        elif message.get("e") == "aggTrade":
                            self.aggregate_events.append(message)
                            value = float(message["p"]) * float(message["q"])
                            self.trades.append((time.time(), -value if message["m"] else value))
                        delay = 1
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Market WebSocket disconnected; retrying")
            finally:
                self.connected = False
                self.book.valid = False
                if "snapshot" in locals() and not snapshot.done():
                    snapshot.cancel()
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30)
