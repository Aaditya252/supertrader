"""Public strategy discovery. Retrieved content is evidence, never executable code."""

import asyncio
import base64
import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx


def candidate(source: str, identity: str, title: str, url: str, text: str, **metadata) -> dict:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()[:24000]
    families = {
        "trend": r"momentum|trend|moving average|breakout",
        "mean reversion": r"mean.rever|bollinger|z.score",
        "arbitrage": r"arbitrage|cointegration|pairs trading",
        "machine learning": r"reinforcement|neural|machine learning|lightgbm",
    }
    mentions = {name: bool(re.search(pattern, text, re.I)) for name, pattern in {
        "entry": r"entry|enter|buy when", "exit": r"exit|sell when|take.profit",
        "risk": r"stop.loss|position.siz|risk.per.trade", "costs": r"slippage|commission|transaction.cost|fees",
    }.items()}
    return {
        "id": hashlib.sha256(f"{source}:{identity}".encode()).hexdigest()[:24],
        "source": source, "title": title[:240], "url": url,
        "summary": text[:1000], "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "families": [name for name, pattern in families.items() if re.search(pattern, text, re.I)],
        "rule_mentions": mentions,
        "review_gaps": [name for name, found in mentions.items() if not found],
        "assessment": "Keyword triage only; mentions do not establish executable rules or performance",
        "status": "discovered_unvalidated", "approved": False,
        "observed_at": datetime.now(timezone.utc).isoformat(), **metadata,
    }


async def github(client: httpx.AsyncClient, query: str) -> list[dict]:
    response = await client.get("https://api.github.com/search/repositories", params={
        "q": f"{query} trading strategy archived:false", "per_page": 5, "sort": "updated",
    })
    response.raise_for_status()
    results = []
    for repo in response.json().get("items", [])[:5]:
        name = repo.get("full_name", "")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
            continue
        text = repo.get("description") or ""
        basis = "repository description"
        try:
            readme = await client.get(f"https://api.github.com/repos/{name}/readme")
            readme.raise_for_status()
            data = readme.json()
            if data.get("encoding") == "base64" and data.get("size", 0) <= 100000:
                text += " " + base64.b64decode(data.get("content", "")).decode("utf-8", errors="replace")
                basis = "repository description and README"
        except (httpx.HTTPError, ValueError):
            pass
        license_info = repo.get("license") or {}
        results.append(candidate("github", name.lower(), name, f"https://github.com/{name}", text,
                                 license=license_info.get("spdx_id") or "unknown; review required",
                                 updated_at=repo.get("updated_at"), content_basis=basis))
    return results


async def arxiv(client: httpx.AsyncClient, query: str) -> list[dict]:
    # User words become quoted literals rather than API query operators.
    words = re.findall(r"[A-Za-z0-9]+", query)[:10]
    search = " AND ".join(f'(ti:"{word}" OR abs:"{word}")' for word in words)
    response = await client.get("https://export.arxiv.org/api/query", params={
        "search_query": f"({search}) AND (cat:q-fin.TR OR cat:q-fin.ST OR cat:q-fin.PM OR cat:q-fin.CP)",
        "max_results": 5, "sortBy": "submittedDate", "sortOrder": "descending",
    })
    response.raise_for_status()
    root = ET.fromstring(response.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        identity = entry.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?", identity):
            continue
        results.append(candidate("arxiv", identity, entry.findtext("a:title", "Untitled", ns),
                                 f"https://arxiv.org/abs/{identity}", entry.findtext("a:summary", "", ns),
                                 license="metadata only; full-text rights require review",
                                 updated_at=entry.findtext("a:updated", None, ns), content_basis="paper abstract"))
    return results


class StrategyFinder:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.cache: dict[tuple[str, str], tuple[float, dict]] = {}
        self.last_request: dict[str, float] = {}

    async def search(self, query: str, source: str) -> dict:
        key = (query.casefold(), source)
        async with self.lock:
            now = time.monotonic()
            cached = self.cache.get(key)
            if cached and now - cached[0] < 600:
                return {**cached[1], "cached": True}
            wait = max(0.0, 7 - (now - self.last_request.get(source, -1000)))
            if wait:
                return {"status": "rate_limited", "items": [], "retry_after_seconds": round(wait + 0.5),
                        "error": "Wait before another provider search", "cached": False}
            self.last_request[source] = now
            try:
                async with httpx.AsyncClient(timeout=12, follow_redirects=False, headers={
                    "User-Agent": "Supertrader-Research/0.1", "Accept": "application/json, application/atom+xml",
                }) as client:
                    items = await (github(client, query) if source == "github" else arxiv(client, query))
                result = {"status": "ready", "items": list({item["id"]: item for item in items}.values()),
                          "cached": False, "fetched_at": datetime.now(timezone.utc).isoformat(),
                          "scope": "Public web discovery; no strategy has been backtested or approved"}
                if len(self.cache) >= 100:
                    self.cache.pop(next(iter(self.cache)))
                self.cache[key] = (time.monotonic(), result)
                return result
            except (httpx.HTTPError, ValueError, ET.ParseError):
                return {"status": "unavailable", "items": [], "cached": False,
                        "error": "Provider unavailable or request quota exhausted; retry later"}
