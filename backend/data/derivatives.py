"""Optional futures observations; never conflate them with spot tradeable prices."""
import asyncio
import json
import logging
import time
import httpx
import websockets
from backend.store import Store

log = logging.getLogger(__name__)


class DerivativesIngestor:
    def __init__(self, symbol: str, store: Store):
        self.symbol, self.store = symbol, store

    async def poll_open_interest(self):
        while True:
            try:
                async with httpx.AsyncClient(timeout=20) as client:
                    response = await client.get("https://fapi.binance.com/fapi/v1/openInterest",
                                                params={"symbol": self.symbol})
                    response.raise_for_status()
                    d = response.json()
                    now = int(time.time()*1000)
                    self.store.append("derivatives", f"oi:{now}", {"type": "open_interest",
                                      "market": "futures", "symbol": self.symbol, "available_at": now,
                                      "event_time": d["time"], "value": float(d["openInterest"])})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Open interest feed unavailable")
            await asyncio.sleep(60)

    async def streams(self):
        while True:
            try:
                streams = f"{self.symbol.lower()}@markPrice@1s/{self.symbol.lower()}@forceOrder"
                async with websockets.connect("wss://fstream.binance.com/market/stream?streams="+streams,
                                               ping_interval=20, ping_timeout=20) as ws:
                    async for message in ws:
                        d = json.loads(message)["data"]
                        now = int(time.time()*1000)
                        self.store.append("derivatives", f"futures:{d['e']}:{d['E']}",
                                          {"market": "futures", "available_at": now, "event_time": d["E"],
                                           "type": d["e"], "raw": d,
                                           "coverage": "liquidations are sampled snapshots, not complete totals"})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Futures stream disconnected")
            await asyncio.sleep(5)

