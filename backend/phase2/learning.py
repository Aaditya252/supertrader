"""Supervised pretraining, simulator-transition IQL warm start and RL experiments."""

import copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from backend.phase2.network import DeepQuantAgent
from backend.phase2.gpu import training_device


def tensors(windows, state, index, device):
    return (
        {tf: torch.as_tensor(x[index], device=device) for tf, x in windows.items()},
        torch.as_tensor(state[index], device=device),
    )


def pretrain(
    windows,
    state,
    returns,
    volatility,
    regimes,
    train,
    validation,
    seed=42,
    epochs=3,
    risk_targets=None,
    checkpoint_dir=None,
):
    """Pinball quantiles + direction CE + volatility MSE + regime CE, training-only targets."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    device = training_device()
    model = DeepQuantAgent(list(windows), next(iter(windows.values())).shape[-1], state.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    history = []
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=device)
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in np.array_split(rng.permutation(train), max(1, len(train) // 256)):
            w, s = tensors(windows, state, batch, device)
            out = model(w, s)
            y = torch.as_tensor(returns[batch], device=device, dtype=torch.float32)
            label = torch.as_tensor(
                np.where(returns[batch] > 0.003, 2, np.where(returns[batch] < -0.003, 0, 1)),
                device=device,
            )
            error = y[:, None] - out["quantiles"]
            pinball = torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
            vol = torch.as_tensor(volatility[batch], device=device, dtype=torch.float32)
            regime = torch.as_tensor(regimes[batch], device=device, dtype=torch.int64)
            loss = (
                pinball
                + nn.functional.cross_entropy(out["direction_logits"], label)
                + 0.1 * nn.functional.mse_loss(out["volatility"].squeeze(), vol)
                + 0.1 * nn.functional.cross_entropy(out["regime_logits"], regime)
            )
            if risk_targets is not None:
                risk = torch.as_tensor(risk_targets[batch], device=device, dtype=torch.float32)
                loss += 0.1 * nn.functional.mse_loss(out["risk"], risk)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        print(
            f"Supervised seed={seed} epoch={epoch + 1}/{epochs} loss={history[-1]['train_loss']:.5f}",
            flush=True,
        )
        if checkpoint_dir is not None:
            destination = Path(checkpoint_dir)
            destination.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "seed": seed,
                    "torch_rng": torch.get_rng_state(),
                },
                destination / f"pretrain-{seed}-epoch-{epoch + 1}.pt",
            )
    model.eval()
    logits = []
    with torch.no_grad():
        for batch in np.array_split(validation, max(1, len(validation) // 512)):
            w, s = tensors(windows, state, batch, device)
            logits.append(model(w, s)["direction_logits"].cpu().numpy())
    return model, {
        "seed": seed,
        "epochs": epochs,
        "device": device,
        "history": history,
        "validation_logits": np.concatenate(logits),
    }


def iql_warm_start(model, windows, state, transitions, steps=1000, seed=42, expectile=0.7):
    """IQL on declared simulator transitions: asymmetric V loss, Bellman Q, weighted BC.

    Transitions must carry decision/next indices, action, reward and terminal flags.
    They are not represented as logged real trader behavior.
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    features = np.concatenate([x[:, -1] for x in windows.values()] + [state], axis=1)
    q = nn.Sequential(nn.Linear(features.shape[1] + 1, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    target = copy.deepcopy(q)
    qopt = torch.optim.Adam(q.parameters(), lr=3e-4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    for _ in range(steps):
        batch = [transitions[i] for i in rng.integers(0, len(transitions), size=min(128, len(transitions)))]
        index = np.array([b["index"] for b in batch])
        following = np.array([b["next_index"] for b in batch])
        action = torch.tensor([b["action"] for b in batch], device=device, dtype=torch.float32)[:, None]
        reward = torch.tensor([b["reward"] for b in batch], device=device, dtype=torch.float32)[:, None]
        done = torch.tensor([b["terminal"] for b in batch], device=device, dtype=torch.float32)[:, None]
        w, s = tensors(windows, state, index, device)
        wn, sn = tensors(windows, state, following, device)
        if "state" in batch[0]:
            s = torch.as_tensor(np.stack([b["state"] for b in batch]), device=device)
            sn = torch.as_tensor(np.stack([b["next_state"] for b in batch]), device=device)
        obs = torch.as_tensor(features[index], device=device)
        obs = obs.clone()
        obs[:, -state.shape[1] :] = s
        qa = q(torch.cat([obs, action], dim=1))
        with torch.no_grad():
            backup = reward + 0.99 * (1 - done) * model(wn, sn)["value"]
            target_q = target(torch.cat([obs, action], dim=1))
        qloss = nn.functional.mse_loss(qa, backup)
        qopt.zero_grad()
        qloss.backward()
        qopt.step()
        out = model(w, s)
        difference = target_q - out["value"]
        weight = torch.where(difference > 0, expectile, 1 - expectile)
        value_loss = (weight * difference.pow(2)).mean()
        advantage = torch.exp((3 * difference.detach()).clamp(max=float(np.log(100))))
        actor_loss = (advantage * (out["action"] - action).pow(2)).mean()
        loss = value_loss + actor_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        with torch.no_grad():
            for target_p, source_p in zip(target.parameters(), q.parameters()):
                target_p.mul_(0.995).add_(source_p, alpha=0.005)
        losses.append(float(loss.detach()))
    return {
        "steps": steps,
        "final_loss": float(np.mean(losses[-100:])),
        "source": "simulator transitions, not actual trader logs",
        "calibration_status": "must recalibrate direction heads after shared-trunk updates",
    }
