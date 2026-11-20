"""Bounded-memory, resumable official SOLUSDT archive download (1s through 1w)."""
import concurrent.futures
import hashlib
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(r'D:\supertrader\data\full_history\spot\SOLUSDT')
COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_volume', 'trade_count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
BASE = 'https://data.binance.vision/data/spot'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fetch(client, url, file):
    for attempt in range(6):
        try:
            with client.stream('GET', url) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with file.open('wb') as f:
                    for b in r.iter_bytes(1024*1024):
                        f.write(b)
            return True
        except httpx.HTTPError:
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 30))


def normalize(d):
    for c in COLS:
        d[c] = pd.to_numeric(d[c], errors='raise')
    for c in ['open_time', 'close_time', 'trade_count']:
        d[c] = d[c].astype('int64')
    for c in ['open_time', 'close_time']:
        d.loc[d[c] > 100000000000000, c] //= 1000
    return d


def archive(task):
    kind, period = task
    out = ROOT/'1s'/f'{period}.parquet'
    receipt = out.with_suffix('.receipt.json')
    if out.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if sha(out) == saved['parquet_sha256']:
            print(f'RESUME {period}: {saved["rows"]:,} rows', flush=True)
            return saved
    zipfile_path = ROOT/'archives'/f'SOLUSDT-1s-{period}.zip'
    url = f'{BASE}/{kind}/klines/SOLUSDT/1s/{zipfile_path.name}'
    with httpx.Client(timeout=180, follow_redirects=True) as client:
        check = client.get(url+'.CHECKSUM')
        if check.status_code == 404:
            print(f'UNAVAILABLE {period}', flush=True)
            return {'period': period, 'status': 'unavailable', 'url': url}
        check.raise_for_status()
        expected = check.text.split()[0]
        if not zipfile_path.exists() or sha(zipfile_path) != expected:
            if not fetch(client, url, zipfile_path):
                return {'period': period, 'status': 'unavailable', 'url': url}
        if sha(zipfile_path) != expected:
            raise ValueError(f'Checksum mismatch: {period}')
    temp = out.with_suffix('.tmp.parquet')
    writer = None
    rows = gaps = rejected = 0
    first = last = previous = None
    try:
        with zipfile.ZipFile(zipfile_path) as z:
            names = [n for n in z.namelist() if n.endswith('.csv')]
            if len(names) != 1:
                raise ValueError('Expected one CSV')
            for d in pd.read_csv(z.open(names[0]), header=None, names=COLS, chunksize=200000):
                d = normalize(d)
                valid = ((d.open_time % 1000 == 0) & (d.close_time == d.open_time+999)
                         & (d[['open','high','low','close']] > 0).all(axis=1) & (d.volume >= 0)
                         & (d.high >= d[['open','low','close']].max(axis=1))
                         & (d.low <= d[['open','high','close']].min(axis=1)))
                rejected += int((~valid).sum())
                d = d.loc[valid].copy()
                if d.empty:
                    continue
                t = d.open_time.to_numpy()
                if np.any(np.diff(t) <= 0) or (previous is not None and t[0] <= previous):
                    raise ValueError(f'Unordered or duplicate archive: {period}')
                gaps += int(np.sum(np.diff(t) > 1000))
                if previous is not None:
                    gaps += int(t[0]-previous > 1000)
                previous = int(t[-1])
                first = int(t[0]) if first is None else first
                last = int(d.close_time.iloc[-1])
                rows += len(d)
                table = pa.Table.from_pandas(d, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression='zstd')
                writer.write_table(table)
        if writer is not None:
            writer.close()
            writer = None
        if not rows:
            raise ValueError(f'Empty archive {period}')
        temp.replace(out)
    finally:
        if writer is not None:
            writer.close()
    result = {'period': period, 'status': 'complete', 'rows': rows, 'first_ms': first,
              'last_ms': last, 'gap_segments': gaps, 'rejected_rows': rejected,
              'source_url': url, 'archive_sha256': expected, 'parquet_sha256': sha(out),
              'parquet_bytes': out.stat().st_size, 'archive_bytes': zipfile_path.stat().st_size}
    receipt.write_text(json.dumps(result, indent=2))
    print(f'COMPLETE {period}: {rows:,} rows, {out.stat().st_size/1e6:.1f} MB', flush=True)
    return result


def larger_intervals(now_ms):
    minute = pd.read_parquet(Path(r'D:\supertrader\data\candles\spot\SOLUSDT\1m.parquet'))
    # Refresh the snapshot tail without modifying the dashboard's active dataset.
    with httpx.Client(timeout=60) as client:
        cursor = int(minute.open_time.iloc[-1])+60000
        pages = []
        while cursor < now_ms:
            response = client.get('https://data-api.binance.vision/api/v3/klines',params={
                'symbol':'SOLUSDT','interval':'1m','startTime':cursor,'endTime':now_ms-1,'limit':1000})
            response.raise_for_status()
            values = response.json()
            if not values:
                break
            page = normalize(pd.DataFrame(values,columns=COLS))
            page = page[(page.close_time < now_ms) & (page.close_time == page.open_time+59999)]
            if page.empty:
                break
            pages.append(page[minute.columns.intersection(page.columns)])
            following = int(page.open_time.iloc[-1])+60000
            if following <= cursor:
                raise RuntimeError('Minute REST failed to advance')
            cursor = following
        if pages:
            minute = pd.concat([minute,*pages],ignore_index=True).drop_duplicates('open_time').sort_values('open_time')
    intervals = {'1m':60, '3m':180, '5m':300, '15m':900, '30m':1800, '1h':3600,
                 '2h':7200, '4h':14400, '6h':21600, '8h':28800, '12h':43200,
                 '1d':86400, '3d':259200, '1w':604800}
    results = {}
    for name, seconds in intervals.items():
        step = seconds*1000
        # Weekly exchange candles start Monday 00:00 UTC (epoch Thursday).
        origin = {'1w':4*86400000,'3d':2*86400000}.get(name,0)
        d = minute.copy()
        d['bucket'] = ((d.open_time-origin)//step)*step+origin
        g = d.groupby('bucket', sort=True).agg(open=('open','first'),high=('high','max'),
             low=('low','min'),close=('close','last'),volume=('volume','sum'),count=('open_time','nunique'))
        g = g[g['count'] == seconds//60].drop(columns='count').reset_index().rename(columns={'bucket':'open_time'})
        g['close_time'] = g.open_time+step-1
        g = g[g.close_time < now_ms]
        folder = ROOT/name
        folder.mkdir(exist_ok=True)
        file = folder/'history.parquet'
        g.to_parquet(file,index=False,compression='zstd')
        results[name] = {'rows':len(g),'bytes':file.stat().st_size,'source':'resampled verified 1m history',
                         'incomplete_buckets':'excluded; never filled synthetically'}
        print(f'RESAMPLED {name}: {len(g):,} rows',flush=True)
    return results


def today_rest(start_ms, end_ms):
    file = ROOT/'1s'/'current-day.parquet'
    frames = []
    with httpx.Client(timeout=60) as client:
        cursor = start_ms
        while cursor < end_ms:
            for attempt in range(6):
                r = client.get('https://data-api.binance.vision/api/v3/klines',params={
                    'symbol':'SOLUSDT','interval':'1s','startTime':cursor,'endTime':end_ms-1,'limit':1000})
                if r.status_code in [418,429]:
                    time.sleep(min(float(r.headers.get('Retry-After',2**attempt)),60))
                    continue
                r.raise_for_status()
                break
            else:
                raise RuntimeError('REST rate limit exhausted')
            values = r.json()
            if not values:
                break
            d = normalize(pd.DataFrame(values,columns=COLS))
            d = d[d.close_time < end_ms]
            if d.empty:
                break
            frames.append(d)
            following = int(d.open_time.iloc[-1])+1000
            if following <= cursor:
                raise RuntimeError('REST failed to advance')
            cursor = following
            time.sleep(.12)
    if not frames:
        return {'rows':0}
    frame = pd.concat(frames,ignore_index=True).drop_duplicates('open_time').sort_values('open_time')
    frame.to_parquet(file,index=False,compression='zstd')
    return {'rows':len(frame),'first_ms':int(frame.open_time.iloc[0]),'last_ms':int(frame.close_time.iloc[-1]),'bytes':file.stat().st_size,'source':'REST; observed at download time'}


def main():
    for p in [ROOT,ROOT/'archives',ROOT/'1s']:
        p.mkdir(parents=True,exist_ok=True)
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp()*1000)
    current_month = now.strftime('%Y-%m')
    months = pd.period_range('2020-08',current_month,freq='M')[:-1]
    days = pd.date_range(now.strftime('%Y-%m-01'),now.strftime('%Y-%m-%d'),freq='D')[:-1]
    tasks = [('monthly',str(p)) for p in months]+[('daily',d.strftime('%Y-%m-%d')) for d in days]
    manifest = {'symbol':'SOLUSDT','market':'spot','snapshot_end_utc':now.isoformat(),
                'status':'downloading','archives':[],'expected_archives':len(tasks)}
    manifest_file = ROOT/'manifest.json'
    def save():
        temporary = manifest_file.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(manifest,indent=2))
        temporary.replace(manifest_file)
    save()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(archive,tasks):
                manifest['archives'].append(result)
                save()
        midnight = int(pd.Timestamp(now.strftime('%Y-%m-%d'),tz='UTC').timestamp()*1000)
        manifest['current_day'] = today_rest(midnight,now_ms)
        manifest['larger_intervals'] = larger_intervals(now_ms)
        manifest['missing_archives'] = [a['period'] for a in manifest['archives'] if a['status']!='complete']
        manifest['one_second_rows'] = sum(a.get('rows',0) for a in manifest['archives'])+manifest['current_day']['rows']
        manifest['total_bytes_including_archive_zips'] = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        manifest['status'] = 'complete' if not manifest['missing_archives'] else 'complete_with_missing_archives'
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({k:v for k,v in manifest.items() if k not in ['archives']},indent=2),flush=True)
    except Exception as exc:
        manifest.update(status='failed',error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
