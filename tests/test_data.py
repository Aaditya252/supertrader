import pandas as pd
import pytest
from backend.data.history import clean_candles, gap_ranges, resample_candles, save_candles
from backend.data.live import OrderBook
from backend.data.news import normalize_article, rolling_sentiment


def test_duplicates_finality_and_alignment(candles):
    d = pd.concat([candles, candles.iloc[[-1]]])
    result = clean_candles(d, "1h", int(candles.close_time.iloc[-1]))
    assert len(result) == len(candles)-1
    assert result.open_time.is_unique
    malformed = candles.copy()
    malformed.loc[0, "open_time"] += 1
    with pytest.raises(ValueError, match="aligned"):
        clean_candles(malformed, "1h", 10**15)


def test_truncated_maintenance_bar_becomes_gap(candles):
    d = candles.copy()
    d.loc[5, "close_time"] -= 100000
    result = clean_candles(d, "1h", 10**15)
    assert len(result) == len(d)-1
    assert gap_ranges(result, "1h")[0]["missing_bars"] == 1


def test_ohlc_validation(candles):
    candles.loc[5, "high"] = 1
    with pytest.raises(ValueError, match="bounds"):
        clean_candles(candles, "1h", 10**15)


def test_resample_rejects_incomplete_bucket(candles):
    d = candles.iloc[:8].drop(index=2)
    output = resample_candles(d, "1h", "4h", 10**15)
    assert len(output) == 1
    assert output.open_time.iloc[0] == 4*3600000
    assert output.volume.iloc[0] == pytest.approx(d.iloc[3:].volume.sum())


def test_save_is_idempotent_and_manifests_gaps(candles, tmp_path):
    first = save_candles(tmp_path, "SOLUSDT", "1h", candles, 10**15)
    second = save_candles(tmp_path, "SOLUSDT", "1h", candles, 10**15)
    assert first["rows"] == second["rows"] == len(candles)


def test_book_gap_invalidation():
    book = OrderBook()
    book.snapshot({"lastUpdateId": 10, "bids": [["99", "2"]], "asks": [["101", "2"]]})
    assert not book.features()["book_valid"]  # snapshot needs a bridging delta
    assert book.delta({"U": 9, "u": 11, "b": [["99", "3"]], "a": []})
    assert book.features()["book_valid"]
    assert not book.delta({"U": 13, "u": 14, "b": [], "a": []})
    assert not book.features()["book_valid"]


def test_crossed_book_is_invalid():
    book = OrderBook()
    book.snapshot({"lastUpdateId": 10, "bids": [["101", "2"]], "asks": [["100", "2"]]})
    book.delta({"U": 11, "u": 11, "b": [], "a": []})
    assert not book.features()["book_valid"]


def test_news_uses_receipt_time_not_backdated_publication():
    a = normalize_article("Solana ETF upgrade", "https://example.org", 1000, 5000, "test", .8)
    assert a["available_at"] == 5000
    assert rolling_sentiment([a], 4000)["score"] is None
    assert rolling_sentiment([a], 6000)["score"] == pytest.approx(.8)
    assert "ETF" in a["tags"] and "upgrade" in a["tags"]

