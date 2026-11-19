import argparse
import asyncio
import json
import logging
import pandas as pd
from backend.config import load_settings
from backend.data.history import download_history, candle_path, INTERVAL_MS


def research_frame(settings):
    from backend.features.engine import build_features, FEATURE_COLUMNS
    from backend.features.labels import triple_barrier
    d = build_features(pd.read_parquet(candle_path(settings.data_dir, settings.symbol, settings.timeframe)),
                       INTERVAL_MS[settings.timeframe])
    labels = triple_barrier(d, settings.label_horizon, interval_ms=INTERVAL_MS[settings.timeframe])
    return pd.concat([d, labels], axis=1).dropna(subset=FEATURE_COLUMNS+["label", "forward_return"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Supertrader research and signals")
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--timeframes", nargs="+", default=list(INTERVAL_MS))
    download.add_argument("--start")
    download.add_argument("--ccxt", action="store_true")
    sub.add_parser("download-archives")
    training = sub.add_parser("train")
    training.add_argument("--mlflow", action="store_true")
    mlops = sub.add_parser("mlops", help="run or inspect the promotion-gated MLOps pipeline")
    mlops_actions = mlops.add_subparsers(dest="mlops_action", required=True)
    mlops_run = mlops_actions.add_parser("run", help="snapshot data, train, evaluate, register, and monitor a candidate")
    mlops_run.add_argument("--no-mlflow", action="store_true")
    mlops_actions.add_parser("status", help="show the latest local MLOps run")
    sequence = sub.add_parser("train-sequence")
    sequence.add_argument("--epochs", type=int, default=5)
    rl = sub.add_parser("train-rl")
    rl.add_argument("--timesteps", type=int, default=10000)
    rl.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    rl.add_argument("--recurrent", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = load_settings(args.config)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "download":
        async def run():
            for timeframe in args.timeframes:
                result = await download_history(settings.data_dir, settings.symbol, timeframe,
                                                args.start or settings.history_start, settings.api_base, args.ccxt)
                print(json.dumps(result, indent=2), flush=True)
        asyncio.run(run())
    elif args.command == "download-archives":
        from backend.data.archives import download_archives
        result = asyncio.run(download_archives(settings.data_dir, settings.symbol, settings.api_base))
        print(json.dumps({"minute_rows": result["minute_rows"], "archive_months": result["archive_months"],
                          "missing_archive_months": result["missing_archive_months"]}, indent=2))
    elif args.command == "train":
        from backend.models.train import train
        report = train(settings, args.mlflow)
        print(json.dumps({k: report[k] for k in ["version", "rows", "approved", "promotion_failures"]}, indent=2))
    elif args.command == "mlops":
        from backend.mlops.pipeline import pipeline_status, run_pipeline
        if args.mlops_action == "run":
            result = run_pipeline(settings, record_mlflow=not args.no_mlflow)
            summary = {
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "candidate_version": result.get("candidate", {}).get("version"),
                "approved": result.get("promotion", {}).get("approved", False),
                "promotion_failures": result.get("promotion", {}).get("failures", []),
                "candidate_artifact_path": result.get("registration", {}).get("artifact_path"),
                "manifest": result.get("manifest_path"),
            }
        else:
            result = pipeline_status(settings)
            candidate = result.get("candidate") or {}
            summary = {
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "candidate_version": candidate.get("version"),
                "approved": (result.get("promotion") or {}).get("approved", False),
                "promotion_failures": (result.get("promotion") or {}).get("failures", []),
                "manifest": result.get("manifest"),
            }
        print(json.dumps(summary, indent=2, allow_nan=False))
    elif args.command == "train-sequence":
        from backend.models.sequence import train_sequence
        print(train_sequence(research_frame(settings), settings.artifact_dir/"experimental", settings.seed, args.epochs))
    elif args.command == "train-rl":
        from backend.rl.train import train_agents
        print(train_agents(research_frame(settings), settings.artifact_dir/"experimental", args.timesteps, args.seeds, args.recurrent))
    elif args.command == "serve":
        import uvicorn
        from backend.api.app import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

