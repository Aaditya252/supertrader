import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.config import Settings
from backend.strategies import StrategyFinder, arxiv, candidate, github


@pytest.mark.asyncio
async def test_github_collects_readme_and_rejects_untrusted_identity():
    def respond(request):
        if request.url.path == "/search/repositories":
            return httpx.Response(200, json={"items": [
                {"full_name": "owner/strategy", "description": "momentum", "license": {"spdx_id": "MIT"}},
                {"full_name": "../../evil"},
            ]})
        assert request.url.host == "api.github.com"
        assert request.url.path == "/repos/owner/strategy/readme"
        return httpx.Response(200, json={"encoding": "base64", "size": 60,
            "content": base64.b64encode(b"Entry breakout; exit stop loss. Slippage included.").decode()})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        items = await github(client, "SOL")
    assert len(items) == 1
    assert items[0]["license"] == "MIT"
    assert items[0]["review_gaps"] == []
    assert items[0]["approved"] is False
    assert items[0]["status"] == "discovered_unvalidated"


@pytest.mark.asyncio
async def test_arxiv_extracts_metadata_without_executing_markup():
    xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <id>http://arxiv.org/abs/2601.12345v1</id><title>Trading momentum</title>
    <summary>Entry rules for trend trading</summary></entry></feed>'''
    def respond(request):
        query = request.url.params["search_query"]
        assert "cat:q-fin.TR" in query
        assert 'ti:"SOL"' in query
        return httpx.Response(200, text=xml)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        items = await arxiv(client, "SOL momentum")
    assert items[0]["url"] == "https://arxiv.org/abs/2601.12345v1"
    assert "costs" in items[0]["review_gaps"]
    assert items[0]["content_basis"] == "paper abstract"


@pytest.mark.asyncio
async def test_cache_dedup_and_provider_throttle(monkeypatch):
    calls = []

    async def provider(client, query):
        calls.append(query)
        item = candidate("github", "x", "x", "https://github.com/a/b", "momentum")
        return [item, item]

    monkeypatch.setattr("backend.strategies.github", provider)
    finder = StrategyFinder()
    first = await finder.search("SOL", "github")
    assert len(first["items"]) == 1
    assert (await finder.search("sol", "github"))["cached"]
    assert (await finder.search("BTC", "github"))["status"] == "rate_limited"
    assert calls == ["SOL"]


@pytest.mark.asyncio
async def test_provider_error_does_not_fabricate_results(monkeypatch):
    async def fail(client, query):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr("backend.strategies.github", fail)
    result = await StrategyFinder().search("SOL", "github")
    assert result["status"] == "unavailable"
    assert result["items"] == []


def test_api_validation_and_comparison(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, artifact_dir=tmp_path / "models",
                              db_url="sqlite:///:memory:"), online=False)
    with TestClient(app) as client:
        assert client.get("/strategies/discover", params={"q": "https://localhost"}).status_code == 422
        assert client.get("/strategies/discover", params={"q": "   "}).status_code == 422
        assert client.get("/strategies/discover?source=other").status_code == 422
        assert client.get("/strategies/comparison").json()["rows"] == []
        root = tmp_path / "models"
        (root / "test-v1").mkdir(parents=True)
        (root / "latest.json").write_text(json.dumps({"version": "test-v1"}))
        (root / "test-v1/report.json").write_text(json.dumps({"approved": False,
            "candidate": {"metrics": {"total_return": 0}},
            "baselines": {"momentum": {"metrics": {"total_return": .1}}}}))
        result = client.get("/strategies/comparison").json()
        assert len(result["rows"]) == 2
        assert result["approved"] is False
        (root / "latest.json").write_text(json.dumps({"version": "../outside"}))
        assert client.get("/strategies/comparison").status_code == 422
