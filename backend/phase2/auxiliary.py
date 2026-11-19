"""Discover public derivative/cross-asset archives; record unavailable coverage."""

import concurrent.futures
import io
import json
from typing import Any
import zipfile
from pathlib import Path
import httpx
import pandas as pd
from backend.phase2.acquire import download, sha
from backend.data.archives import parse_archive


def auxiliary_archive(root, kind, period, symbol="SOLUSDT", frequency="daily"):
    folder = root / "aux" / kind / symbol
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"{period}.parquet"
    receipt = file.with_suffix(".receipt.json")
    if file.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if sha(file) == saved["parquet_sha256"]:
            return saved
    interval = "1h/" if kind in ["spot_klines", "markPriceKlines", "indexPriceKlines"] else ""
    route = "spot" if kind == "spot_klines" else "futures/um"
    path_kind = "klines" if kind == "spot_klines" else kind
    filename = f"{symbol}-1h-{period}.zip" if interval else f"{symbol}-{kind}-{period}.zip"
    url = f"https://data.binance.vision/data/{route}/{frequency}/{path_kind}/{symbol}/{interval}{filename}"
    zfile = folder / f"{period}.zip"
    with httpx.Client(timeout=90, follow_redirects=True) as client:
        checksum = client.get(url + ".CHECKSUM")
        if checksum.status_code == 404:
            return {
                "kind": kind,
                "symbol": symbol,
                "period": period,
                "status": "unavailable",
                "url": url,
            }
        checksum.raise_for_status()
        expected = checksum.text.split()[0]
        if not zfile.exists() or sha(zfile) != expected:
            download(client, url, zfile)
        if sha(zfile) != expected:
            raise ValueError("Auxiliary archive checksum mismatch")
    if interval:
        frame = parse_archive(zfile.read_bytes())
        frame["available_at"] = frame.close_time + 1
    else:
        with zipfile.ZipFile(zfile) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            frame = pd.read_csv(io.BytesIO(z.read(name)))
        if kind == "fundingRate":
            frame["available_at"] = pd.to_numeric(frame.calc_time).astype("int64")
        else:
            column = "create_time" if "create_time" in frame else "timestamp"
            frame["available_at"] = pd.to_datetime(frame[column], utc=True).astype("int64") // 1000000
    frame["availability_basis"] = "archive exchange timestamp; receipt latency unverified"
    frame = frame.sort_values("available_at").drop_duplicates()
    frame.to_parquet(file, index=False, compression="zstd")
    result = {
        "kind": kind,
        "symbol": symbol,
        "period": period,
        "status": "complete",
        "rows": len(frame),
        "first_ms": int(frame.available_at.min()),
        "last_ms": int(frame.available_at.max()),
        "url": url,
        "archive_sha256": expected,
        "parquet_sha256": sha(file),
        "bytes": file.stat().st_size,
    }
    receipt.write_text(json.dumps(result, indent=2))
    return result


def acquire_auxiliary(root=Path("data/phase2"), start="2020-09", workers=8):
    root = Path(root)
    now = pd.Timestamp.now(tz="UTC")
    months = pd.period_range(start, now.strftime("%Y-%m"), freq="M")[:-1]
    days = pd.date_range(start + "-01", now.strftime("%Y-%m-%d"), freq="D")[:-1]
    tasks = []
    for month in months:
        tasks.append(("fundingRate", str(month), "SOLUSDT", "monthly"))
        for kind in ["markPriceKlines", "indexPriceKlines"]:
            tasks.append((kind, str(month), "SOLUSDT", "monthly"))
        for symbol in ["BTCUSDT", "ETHUSDT"]:
            tasks.append(("spot_klines", str(month), symbol, "monthly"))
    for date in days:
        for kind in ["metrics", "bookDepth"]:
            tasks.append((kind, date.strftime("%Y-%m-%d"), "SOLUSDT", "daily"))
    report: dict[str, Any] = {
        "status": "downloading",
        "archives": [],
        "expected_archives": len(tasks),
        "liquidations": {
            "status": "true historical events unavailable in checked Binance archive; provider required",
            "tardis": "supported types; no subscription credentials supplied",
            "coinglass_cryptoquant": "not connected; coverage not asserted",
        },
        "bookDepth": "percentage depth summaries, not sequenced L2; historical OFI/microprice unavailable",
        "optional_sources": {
            "DXY": "not connected",
            "VIX": "not connected",
            "Deribit_SOL_IV": "not connected",
        },
    }
    file = root / "auxiliary-coverage.json"

    def save():
        tmp = file.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(file)

    save()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for i, result in enumerate(pool.map(lambda task: auxiliary_archive(root, *task), tasks)):
                report["archives"].append(result)
                if (i + 1) % 50 == 0:
                    save()
                    print(f"Auxiliary archives checked: {i + 1}/{len(tasks)}", flush=True)
        for kind in [
            "fundingRate",
            "metrics",
            "bookDepth",
            "markPriceKlines",
            "indexPriceKlines",
            "spot_klines",
        ]:
            directory = root / "aux" / kind
            for folder in directory.glob("*"):
                frames = [
                    pd.read_parquet(p)
                    for p in sorted(folder.glob("*.parquet"))
                    if p.name != "history.parquet"
                ]
                if frames:
                    d = pd.concat(frames, ignore_index=True).sort_values("available_at").drop_duplicates()
                    d.to_parquet(folder / "history.parquet", index=False, compression="zstd")
        report["status"] = "complete_with_reported_coverage_gaps"
        report["finished_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        save()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        save()
        raise
    return report


if __name__ == "__main__":
    acquire_auxiliary()
