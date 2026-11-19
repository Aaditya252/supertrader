"""SB3 online algorithms with pretrained multi-timeframe attention extraction."""

import copy
import json
import gymnasium as gym
import numpy as np
from typing import Any
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from backend.phase2.simulator import PerpetualEnv
from backend.phase2.gpu import training_device, gpu_details


class MultiTimeframeEnv(PerpetualEnv):
    def __init__(self, windows, state, bars, **kwargs):
        self.windows = windows
        self.static = np.asarray(state, dtype=np.float32)
        self.timeframes = list(windows)
        self.encoder_window = next(iter(windows.values())).shape[1]
        self.encoder_features = next(iter(windows.values())).shape[2]
        super().__init__(np.zeros((len(bars), 1), dtype=np.float32), bars, **kwargs)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (len(self.timeframes) * self.encoder_window * self.encoder_features + self.static.shape[1] + 5,),
            dtype=np.float32,
        )

    def _observation(self):
        return np.concatenate(
            [self.windows[tf][self.i].ravel() for tf in self.timeframes]
            + [
                self.static[self.i],
                np.array(
                    [
                        self.position / self.cap,
                        self.equity - 1,
                        self.age / 100,
                        self.equity / self.peak - 1,
                        self.mean,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, pretrained, encoder_window=24, encoder_features=8):
        width = pretrained.value.in_features
        super().__init__(observation_space, features_dim=width)
        self.model = copy.deepcopy(pretrained).cpu()
        self.window = encoder_window
        self.count = encoder_features

    def forward(self, observation):
        windows = {}
        offset = 0
        length = self.window * self.count
        for tf in self.model.timeframes:
            windows[tf] = observation[:, offset : offset + length].reshape(-1, self.window, self.count)
            offset += length
        return self.model.encode(windows, observation[:, offset:])[0]


class DiscreteExposure(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.levels = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.levels))

    def action(self, action):
        return np.array([self.levels[int(action)]], dtype=np.float32)


class CurriculumEnv(gym.Env):
    """Sample contiguous historical segments; gaps never become invented fills."""

    def __init__(self, windows, state, bars, funding=None, regimes=None):
        boundaries = np.flatnonzero(np.diff(bars.open_time.to_numpy()) != 3600000) + 1
        self.parts = [p for p in np.split(np.arange(len(bars)), boundaries) if len(p) >= 27]
        if not self.parts:
            raise ValueError("No contiguous curriculum episode available")
        self.windows, self.state, self.bars, self.funding = windows, state, bars, funding
        self.regimes = np.asarray(regimes) if regimes is not None else np.zeros(len(bars))
        self.env = self._make(self.parts[0])
        self.observation_space, self.action_space = self.env.observation_space, self.env.action_space

    def _make(self, p):
        return MultiTimeframeEnv(
            {k: v[p] for k, v in self.windows.items()},
            self.state[p],
            self.bars.iloc[p],
            funding=self.funding,
            randomize=True,
            episode_bars=min(512, len(p) - 26),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Balance segment selection by the dominant training-filtered regime.
        labels = np.array(
            [np.bincount(self.regimes[p].astype(int), minlength=4).argmax() for p in self.parts]
        )
        selected = self.np_random.choice(np.unique(labels))
        p = self.parts[int(self.np_random.choice(np.flatnonzero(labels == selected)))]
        self.env = self._make(p)
        return self.env.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))

    def step(self, action):
        return self.env.step(action)


def train_online(
    pretrained,
    windows,
    state,
    bars,
    destination,
    timesteps=10000,
    seeds=(42, 43, 44),
    funding=None,
    regimes=None,
):
    from stable_baselines3 import SAC
    from sb3_contrib import RecurrentPPO, QRDQN
    from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % 1000 == 0:
                print(f"RL {self.model.__class__.__name__}: {self.num_timesteps:,} steps", flush=True)
            return True

    destination.mkdir(parents=True, exist_ok=True)
    device = training_device()
    devices = []
    models = []
    policy_kwargs = {
        "features_extractor_class": AttentionExtractor,
        "features_extractor_kwargs": {"pretrained": pretrained},
        "net_arch": [64, 64],
    }
    for algorithm in ["RecurrentPPO", "SAC", "QRDQN"]:
        for seed in seeds:
            print(f"RL {algorithm} seed={seed}: {timesteps:,} requested simulator steps", flush=True)
            env: Any = CurriculumEnv(windows, state, bars, funding, regimes)
            if algorithm == "RecurrentPPO":
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    env,
                    n_steps=256,
                    batch_size=64,
                    seed=seed,
                    device=device,
                    policy_kwargs={**policy_kwargs, "lstm_hidden_size": 64},
                    verbose=0,
                )
            elif algorithm == "SAC":
                model = SAC(
                    "MlpPolicy",
                    env,
                    buffer_size=5000,
                    learning_starts=256,
                    batch_size=64,
                    seed=seed,
                    device=device,
                    policy_kwargs=dict(policy_kwargs),
                    verbose=0,
                )
            else:
                env = DiscreteExposure(env)
                model = QRDQN(
                    "MlpPolicy",
                    env,
                    buffer_size=5000,
                    learning_starts=256,
                    batch_size=64,
                    seed=seed,
                    device=device,
                    policy_kwargs={**policy_kwargs, "n_quantiles": 25},
                    verbose=0,
                )
            if any(not p.is_cuda for p in model.policy.parameters()):
                raise RuntimeError(f"{algorithm} policy parameters were not placed on CUDA")
            print(
                f"RL {algorithm} seed={seed} training on {model.device}: {gpu_details()['name']}", flush=True
            )
            checkpoint = CheckpointCallback(
                save_freq=5000,
                save_path=str(destination / "checkpoints"),
                name_prefix=f"{algorithm}-{seed}",
                save_replay_buffer=algorithm != "RecurrentPPO",
            )
            model.learn(total_timesteps=timesteps, callback=[checkpoint, ProgressCallback()])
            path = destination / f"{algorithm}-{seed}"
            model.save(str(path))
            devices.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "device": str(model.device),
                    "steps": model.num_timesteps,
                    "policy_parameters_on_cuda": True,
                }
            )
            (destination / "training-devices.json").write_text(
                json.dumps({"gpu": gpu_details(), "models": devices}, indent=2)
            )
            print(f"RL saved {path}; actual steps={model.num_timesteps}", flush=True)
            models.append((algorithm, seed, model))
    return models
