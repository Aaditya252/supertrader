"""Checksum-verified monthly 1m archives; resample only complete UTC buckets."""
import asyncio
import hashlib
import io
import json
import zipfile
from pathlib import Path
import httpx
import pandas as pd
from backend.data.history import BinanceREST, COLUMNS, save_candles, resample_candles, download_history


def parse_archive(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected one CSV per exchange archive")
        raw = pd.read_csv(archive.open(names[0]), header=None, usecols=range(7))
    raw = raw[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    raw.columns = COLUMNS
    for c in COLUMNS:
        raw[c] = pd.to_numeric(raw[c])
    for c in ["open_time", "close_time"]:
        # Binance spot archives switched to microsecond timestamps in 2025.
        raw[c] = raw[c].astype("int64")
        raw.loc[raw[c] > 100000000000000, c] //= 1000
    return raw


async def download_archives(root: Path, symbol: str, base: str,
                            start: str = "2020-08-01", workers: int = 4) -> dict:
    exchange = BinanceREST(base)
    now = await exchange.now_ms()
    current = pd.to_datetime(now, unit="ms", utc=True)
    periods = pd.period_range(pd.Timestamp(start).strftime("%Y-%m"), current.strftime("%Y-%m"), freq="M")[:-1]
    directory = root/"archive_partitions"/symbol/"1m"
    directory.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(workers)
    missing = []
    async with httpx.AsyncClient(timeout=90, follow_redirects=True) as client:
        async def month(period):
            output = directory/f"{period}.parquet"
            if output.exists() and output.with_suffix(".sha256").exists():
                return
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{period}.zip"
            async with semaphore:
                for attempt in range(4):
                    try:
                        response = await client.get(url)
                        if response.status_code == 404:
                            missing.append(str(period))
                            return
                        response.raise_for_status()
                        checksum = await client.get(url+".CHECKSUM")
                        checksum.raise_for_status()
                        expected = checksum.text.split()[0]
                        if hashlib.sha256(response.content).hexdigest() != expected:
                            raise ValueError("Archive checksum mismatch")
                        frame = await asyncio.to_thread(parse_archive, response.content)
                        await asyncio.to_thread(frame.to_parquet, output, index=False)
                        output.with_suffix(".sha256").write_text(expected)
                        print(f"Archive {period}: {len(frame)} rows", flush=True)
                        return
                    except (httpx.HTTPError, ValueError):
                        if attempt == 3:
                            raise
                        await asyncio.sleep(2**attempt)
        await asyncio.gather(*(month(p) for p in periods))
    frames = [pd.read_parquet(f) for f in sorted(directory.glob("*.parquet"))]
    if not frames:
        raise RuntimeError("No archives available")
    minute = pd.concat(frames, ignore_index=True)
    save_candles(root, symbol, "1m", minute, now)
    # REST catches the current month; no synthetic current-month archive.
    await download_history(root, symbol, "1m", start+"T00:00:00Z" if "T" not in start else start, base)
    from backend.data.history import candle_path
    minute = pd.read_parquet(candle_path(root, symbol, "1m"))
    manifests = {}
    for target in ["5m", "15m", "1h", "4h", "1d"]:
        resampled = resample_candles(minute, "1m", target, now)
        manifests[target] = save_candles(root, symbol, target, resampled, now)
    result = {"archive_months": len(frames), "missing_archive_months": missing,
              "minute_rows": len(minute), "resampled": manifests}
    (root/"archive-download.json").write_text(json.dumps(result, indent=2))
    return result

