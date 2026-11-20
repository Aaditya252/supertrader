import hashlib
import math
from datetime import datetime, timezone
import calendar
from typing import Any
import feedparser
import httpx
from backend.store import Store

TAGS = {"ETF": ["etf", "exchange-traded"], "hack": ["hack", "exploit"],
        "upgrade": ["upgrade", "alpenglow"], "regulation": ["sec", "regulation"],
        "outage": ["outage", "halt"], "listing": ["listing", "listed"]}


def normalize_article(title: str, link: str, published_ms: int | None, observed_ms: int,
                      source: str, score: float | None = None) -> dict:
    normalized = " ".join(title.lower().split())
    key = hashlib.sha256(normalized.encode()).hexdigest()
    return {"id": key, "title": title, "url": link, "source": source,
            "published_at": published_ms, "observed_at": observed_ms,
            "available_at": max(published_ms or observed_ms, observed_ms),
            "sentiment": score, "sentiment_status": "scored" if score is not None else "unavailable",
            "credibility_weight": 1., "credibility_basis": "allowlisted RSS source; not learned",
            "novelty": 1., "novelty_basis": "exact normalized-title deduplication only",
            "tags": [tag for tag, words in TAGS.items() if any(word in normalized for word in words)]}


class NewsIngestor:
    def __init__(self, store: Store, urls: list[str], enabled: bool, model: str, revision: str):
        self.store, self.urls, self.enabled = store, urls, enabled
        self.model, self.revision = model, revision
        self.pipeline: Any = None

    def score(self, text: str) -> float | None:
        if not self.enabled:
            return None
        if self.pipeline is None:
            from transformers import pipeline
            self.pipeline = pipeline("text-classification", model=self.model, revision=self.revision,
                                     top_k=None)
        values = self.pipeline(text[:1500], truncation=True)[0]
        scores = {row["label"].lower(): row["score"] for row in values}
        return float(scores.get("positive", 0)-scores.get("negative", 0))

    async def poll(self) -> int:
        count = 0
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            for url in self.urls:
                response = await client.get(url)
                response.raise_for_status()
                feed = feedparser.parse(response.text)
                for entry in feed.entries[:50]:
                    title = entry.get("title", "")
                    if not any(word in title.lower() for word in ["solana", "sol ", "crypto", "bitcoin"]):
                        continue
                    timestamp = entry.get("published_parsed")
                    published = calendar.timegm(timestamp)*1000 if timestamp else None
                    observed = int(datetime.now(timezone.utc).timestamp()*1000)
                    article = normalize_article(title, entry.get("link", ""), published, observed,
                                                feed.feed.get("title", url), self.score(title))
                    count += self.store.append("news", "news:"+article["id"], article)
        return count


def rolling_sentiment(articles: list[dict], decision_ms: int, half_life_hours: float = 6) -> dict:
    rows = [a for a in articles if a["available_at"] <= decision_ms and a.get("sentiment") is not None]
    if not rows:
        return {"score": None, "count": 0, "status": "no point-in-time sentiment coverage"}
    weights = [a["credibility_weight"]*math.exp(-math.log(2)*(decision_ms-a["available_at"])/3600000/half_life_hours) for a in rows]
    total = sum(weights)
    return {"score": sum(w*a["sentiment"] for w, a in zip(weights, rows))/total,
            "count": len(rows), "status": "available"}

