"""Experimental causal TCN, direction logits and return quantiles; never auto-promoted."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from backend.features.engine import FEATURE_COLUMNS


class CausalConv(nn.Module):
    def __init__(self, incoming: int, outgoing: int, dilation: int):
        super().__init__()
        self.padding = 2*dilation
        self.conv = nn.Conv1d(incoming, outgoing, 3, padding=self.padding, dilation=dilation)

    def forward(self, x):
        return torch.relu(self.conv(x)[..., :-self.padding])


class TCN(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.body = nn.Sequential(CausalConv(features, 32, 1), CausalConv(32, 32, 2),
                                  CausalConv(32, 32, 4))
        self.direction = nn.Linear(32, 3)
        self.quantiles = nn.Linear(32, 3)

    def forward(self, x):
        state = self.body(x.transpose(1, 2))[:, :, -1]
        # Sorted quantiles prevent crossing at inference.
        return self.direction(state), torch.sort(self.quantiles(state), dim=1).values


def train_sequence(d, output: Path, seed: int = 42, epochs: int = 5, window: int = 48):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    boundary = int(len(d)*.7)
    train_end = boundary-8
    scaler = StandardScaler().fit(d[FEATURE_COLUMNS].iloc[:train_end])
    z = scaler.transform(d[FEATURE_COLUMNS])
    indexes = np.arange(window-1, len(d))
    # Do not form sequences across candle gaps.
    step = int(np.median(np.diff(d.open_time)))
    indexes = np.array([i for i in indexes if (np.diff(d.open_time.iloc[i-window+1:i+1]) == step).all()])
    x = np.array([z[i-window+1:i+1] for i in indexes], dtype=np.float32)
    y = d.label.iloc[indexes].to_numpy(dtype=np.int64)+1
    r = d.forward_return.iloc[indexes].to_numpy(dtype=np.float32)
    mask = (indexes < train_end) & (d.label_end.iloc[indexes].to_numpy() < d.available_at.iloc[boundary])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCN(len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loader = DataLoader(TensorDataset(torch.tensor(x[mask]), torch.tensor(y[mask]), torch.tensor(r[mask])),
                        batch_size=256, shuffle=False)
    q = torch.tensor([.1, .5, .9], device=device)
    for _ in range(epochs):
        model.train()
        for xx, yy, rr in loader:
            logits, quantiles = model(xx.to(device))
            error = rr.to(device).unsqueeze(1)-quantiles
            loss = nn.functional.cross_entropy(logits, yy.to(device)) + torch.maximum(q*error, (q-1)*error).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    test = indexes >= boundary
    losses = []
    with torch.no_grad():
        for start in range(0, test.sum(), 256):
            xx = torch.tensor(x[test][start:start+256]).to(device)
            yy = torch.tensor(y[test][start:start+256]).to(device)
            logits, _ = model(xx)
            losses.append(float(nn.functional.cross_entropy(logits, yy)))
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "mean": scaler.mean_, "scale": scaler.scale_,
                "features": FEATURE_COLUMNS, "window": window}, output/"tcn.pt")
    report = {"seed": seed, "epochs": epochs, "device": device, "validation_cross_entropy": float(np.mean(losses)),
              "status": "experimental; probabilities uncalibrated; excluded from live ensemble"}
    (output/"tcn-report.json").write_text(json.dumps(report, indent=2))
    return report

