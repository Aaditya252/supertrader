"""Seven causal timeframe encoders, attention fusion, policy/value/quantile/risk heads."""

import torch
from torch import nn


class CausalConv(nn.Module):
    def __init__(self, inputs, outputs, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class DeepQuantAgent(nn.Module):
    def __init__(self, timeframes, feature_count, state_count=8, width=32):
        super().__init__()
        self.timeframes = list(timeframes)
        self.encoders = nn.ModuleDict(
            {
                tf: nn.Sequential(
                    CausalConv(feature_count, width),
                    nn.GELU(),
                    CausalConv(width, width, dilation=2),
                    nn.GELU(),
                )
                for tf in self.timeframes
            }
        )
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.state = nn.Sequential(nn.Linear(state_count, width), nn.GELU())
        self.gate = nn.Linear(width * 2, width)
        self.trunk = nn.Sequential(nn.Linear(width * 2, width * 2), nn.GELU())
        self.policy = nn.Linear(width * 2, 1)
        self.value = nn.Linear(width * 2, 1)
        self.direction = nn.Linear(width * 2, 3)
        self.quantiles = nn.Linear(width * 2, 3)
        self.risk = nn.Linear(width * 2, 2)
        self.volatility = nn.Linear(width * 2, 1)
        self.regime = nn.Linear(width * 2, 4)
        self.interpretation = nn.Linear(width * 2, 5)

    def encode(self, windows, state, missing=None):
        embeddings = torch.stack(
            [self.encoders[tf](windows[tf].transpose(1, 2))[:, :, -1] for tf in self.timeframes],
            dim=1,
        )
        if missing is not None and missing.all(dim=1).any():
            raise ValueError("At least one observed timeframe required")
        attended, weights = self.attention(
            embeddings,
            embeddings,
            embeddings,
            key_padding_mask=missing,
            need_weights=True,
            average_attn_weights=True,
        )
        valid = (
            torch.ones(embeddings.shape[:2], device=embeddings.device)
            if missing is None
            else (~missing).float()
        )
        fused = (attended * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        static = self.state(state)
        gate = torch.sigmoid(self.gate(torch.cat([fused, static], dim=-1)))
        trunk = self.trunk(torch.cat([gate * fused, (1 - gate) * static], dim=-1))
        return trunk, (weights * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)

    def forward(self, windows, state, missing=None):
        trunk, attention = self.encode(windows, state, missing)
        base = self.quantiles(trunk)
        median = base[:, 1]
        q = torch.stack(
            [
                median - nn.functional.softplus(base[:, 0]),
                median,
                median + nn.functional.softplus(base[:, 2]),
            ],
            dim=-1,
        )
        return {
            "action": torch.tanh(self.policy(trunk)),
            "value": self.value(trunk),
            "direction_logits": self.direction(trunk),
            "quantiles": q,
            "risk": nn.functional.softplus(self.risk(trunk)),
            "volatility": nn.functional.softplus(self.volatility(trunk)),
            "regime_logits": self.regime(trunk),
            "family_logits": self.interpretation(trunk),
            "attention": attention,
        }


def integrated_gradients(model, windows, state, steps=16):
    """Post-hoc gradients from zero standardized training baseline; not causal explanation."""
    model.eval()
    acc = {tf: torch.zeros_like(x) for tf, x in windows.items()}
    for scale in torch.linspace(0, 1, steps, device=state.device):
        inputs = {tf: (x.detach() * scale).requires_grad_(True) for tf, x in windows.items()}
        score = model(inputs, state)["direction_logits"][:, 2].sum()
        gradients = torch.autograd.grad(score, list(inputs.values()))
        for (tf, _), gradient in zip(inputs.items(), gradients):
            acc[tf] += gradient
    return {tf: (windows[tf] * gradient / steps).detach() for tf, gradient in acc.items()}
