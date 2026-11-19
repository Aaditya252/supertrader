import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from backend.features.engine import FEATURE_COLUMNS
from backend.rl.environment import TradingEnv
from backend.backtest.engine import metrics, block_bootstrap


def train_agents(d, output: Path, timesteps: int = 10000, seeds=(42, 43, 44), recurrent: bool = False):
    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.env_checker import check_env
    boundary = int(len(d)*.7)
    split = boundary-8
    scaler = StandardScaler().fit(d[FEATURE_COLUMNS].iloc[:split])
    x = np.clip(scaler.transform(d[FEATURE_COLUMNS]), -10, 10).astype(np.float32)
    prices = d.open.to_numpy()
    step = int(np.median(np.diff(d.open_time)))
    def longest_contiguous(start, end):
        times = d.open_time.iloc[start:end].to_numpy()
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) != step)+1, len(times)]
        lengths = np.diff(boundaries)
        best = int(np.argmax(lengths))
        return slice(start+int(boundaries[best]), start+int(boundaries[best+1]))
    training_slice = longest_contiguous(0, split)
    testing_slice = longest_contiguous(boundary, len(d))
    output.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(scaler, output/"rl-scaler.joblib")
    algorithms = {"PPO": PPO, "SAC": SAC}
    if recurrent:
        from sb3_contrib import RecurrentPPO
        algorithms["RecurrentPPO"] = RecurrentPPO
    reports = []
    for name, cls in algorithms.items():
        for seed in seeds:
            env = TradingEnv(x[training_slice], prices[training_slice])
            check_env(env, warn=True)
            policy = "MlpLstmPolicy" if name == "RecurrentPPO" else "MlpPolicy"
            args = {"learning_starts": 1000, "buffer_size": 50000, "batch_size": 128} if name == "SAC" else {"n_steps": 512, "batch_size": 128}
            model = cls(policy, env, seed=seed, verbose=0, device="cpu", **args)
            model.learn(total_timesteps=timesteps)
            model.save(str(output/f"{name}-{seed}"))
            test = TradingEnv(x[testing_slice], prices[testing_slice], randomize_costs=False)
            observation, _ = test.reset(seed=seed)
            done, state = False, None
            pnls = []
            exposures, turnovers = [], []
            while not done:
                if name == "RecurrentPPO":
                    action, state = model.predict(observation, state=state,
                                                  episode_start=np.array([state is None]), deterministic=True)
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, _, done, _, info = test.step(action)
                pnls.append(info["net_return"])
                exposures.append(info["exposure"])
                turnovers.append(info["turnover"])
            reports.append({"algorithm": name, "seed": seed, "equity": info["equity"],
                            "drawdown": info["drawdown"], "bars": len(pnls),
                            "metrics": metrics(np.array(pnls), np.array(exposures), np.array(turnovers)),
                            "block_bootstrap": block_bootstrap(np.array(pnls), seed)})
    summary = {"runs": reports, "variance": {name: float(np.var([r["equity"] for r in reports if r["algorithm"] == name])) for name in algorithms},
               "training_rows": training_slice.stop-training_slice.start,
               "testing_rows": testing_slice.stop-testing_slice.start,
               "gap_policy": "longest contiguous segment per split; no synthetic cross-outage rewards",
               "timesteps": timesteps, "approved": False,
               "status": "experimental RL; not calibrated; baseline/robustness gates still required"}
    (output/"rl-report.json").write_text(json.dumps(summary, indent=2))
    return summary

