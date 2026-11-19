"""Causal quantitative diagnostics with explicit missing-source and fit status."""

import math
from typing import Any, Literal
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats


def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None


def volatility_features(close: pd.Series, n=24) -> pd.DataFrame:
    """RV=sum r²; BV=pi/2 sum|r*rprev|; BNS uses tripower quarticity."""
    r = np.log(close).diff()
    rv = r.pow(2).rolling(n).sum()
    bv = (math.pi / 2) * (r.abs() * r.shift().abs()).rolling(n - 1).sum() * n / (n - 1)
    mu = 2 ** (2 / 3) * math.gamma(7 / 6) / math.sqrt(math.pi)
    tpq = n * mu ** (-3) * (r.abs() * r.shift().abs() * r.shift(2).abs()).pow(4 / 3).rolling(n - 2).sum()
    scale = np.sqrt((math.pi**2 / 4 + math.pi - 5) / n * np.maximum(1, tpq / bv.pow(2).replace(0, np.nan)))
    z = (rv - bv) / rv.replace(0, np.nan) / scale
    return pd.DataFrame(
        {
            "return": r,
            "rv": rv,
            "bipower_variation": bv,
            "jump_variation": (rv - bv).clip(lower=0),
            "bns_z": z,
            "bns_p_value": 2 * stats.norm.sf(z.abs()),
            "down_semivariance": r.where(r < 0, 0).pow(2).rolling(n).sum(),
            "up_semivariance": r.where(r > 0, 0).pow(2).rolling(n).sum(),
            "vol_of_vol": np.sqrt(rv).rolling(n).std(),
            "skew": r.rolling(n).skew(),
            "excess_kurtosis": r.rolling(n).kurt(),
            "return_q05": r.rolling(n).quantile(0.05),
            "return_q95": r.rolling(n).quantile(0.95),
        }
    ).replace([np.inf, -np.inf], np.nan)


def har_forecast(daily_rv: pd.Series) -> dict:
    """RV(t+1)=b0+b1*RV(t)+b5*mean5+b22*mean22; fit completed targets only."""
    x = pd.DataFrame(
        {
            "one": daily_rv,
            "five": daily_rv.rolling(5).mean(),
            "month": daily_rv.rolling(22).mean(),
        }
    )
    d = x.assign(target=daily_rv.shift(-1)).dropna()
    if len(d) < 30 or x.iloc[-1].isna().any():
        return {"status": "insufficient daily history", "forecast": None}
    beta = np.linalg.lstsq(np.c_[np.ones(len(d)), d[["one", "five", "month"]]], d.target, rcond=None)[0]
    return {
        "status": "fitted expanding completed targets",
        "forecast": max(0, float(np.r_[1, x.iloc[-1]] @ beta)),
        "training_rows": len(d),
        "coefficients": beta.tolist(),
    }


def garch_forecasts(returns: np.ndarray) -> dict:
    """Trailing Student-t GARCH/GJR/EGARCH variance forecasts; not full-sample fits."""
    from arch import arch_model

    r = np.asarray(returns)
    r = r[np.isfinite(r)][-1000:] * 100
    if len(r) < 250 or np.std(r) < 1e-8:
        return {"status": "insufficient variable returns", "forecasts": {}}
    output = {}
    models: list[tuple[str, Literal["GARCH", "EGARCH"], int]] = [
        ("garch", "GARCH", 0),
        ("gjr_garch", "GARCH", 1),
        ("egarch", "EGARCH", 1),
    ]
    for name, vol, o in models:
        try:
            fitted = arch_model(r, mean="Zero", vol=vol, p=1, o=o, q=1, dist="t", rescale=False).fit(
                disp="off", show_warning=False
            )
            forecast = float(fitted.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]) / 10000
            output[name] = {
                "variance": finite(forecast),
                "converged": fitted.convergence_flag == 0,
                "rows": len(r),
            }
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            output[name] = {"variance": None, "status": str(exc)[:100]}
    return {"status": "trailing fit", "forecasts": output}


def evt_tail(returns, alpha=0.95, threshold_quantile=0.9) -> dict:
    """GPD loss tail: VaR=u+beta/xi*((p_u/(1-alpha))**xi-1)."""
    x = -np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)][-2000:]
    if len(x) < 300:
        return {"status": "insufficient tail history", "var": None, "cvar": None}
    u = float(np.quantile(x, threshold_quantile))
    ex = x[x > u] - u
    if len(ex) < 30 or np.std(ex) < 1e-10:
        return {
            "status": "insufficient variable exceedances",
            "var": None,
            "cvar": None,
        }
    shape, _, scale = stats.genpareto.fit(ex, floc=0)
    ratio = len(ex) / len(x) / (1 - alpha)
    if ratio < 1 or scale <= 0:
        return {
            "status": "requested quantile below fitted tail",
            "var": None,
            "cvar": None,
        }
    var = u + (scale / shape * (ratio**shape - 1) if abs(shape) > 1e-7 else scale * np.log(ratio))
    cvar = (var + scale - shape * u) / (1 - shape) if shape < 1 else None
    return {
        "status": "trailing GPD; extrapolation uncertain",
        "var": finite(var),
        "cvar": finite(cvar),
        "shape_xi": finite(shape),
        "tail_index": finite(1 / shape) if shape > 0 else None,
        "threshold": u,
        "exceedances": len(ex),
    }


def kalman_trend(prices, q=1e-5, r=1e-3):
    """Local linear trend x=[level,slope], prediction F*x then causal measurement update."""
    y = np.log(np.asarray(prices, dtype=float))
    out = np.full((len(y), 2), np.nan)
    if not len(y):
        return out
    state = np.array([y[0], 0.0])
    cov = np.eye(2)
    f = np.array([[1.0, 1.0], [0.0, 1.0]])
    h = np.array([1.0, 0.0])
    for i, value in enumerate(y):
        state = f @ state
        cov = f @ cov @ f.T + np.eye(2) * q
        gain = cov @ h / (h @ cov @ h + r)
        state += gain * (value - h @ state)
        cov = (np.eye(2) - np.outer(gain, h)) @ cov
        out[i] = state
    return out


def dynamic_beta(asset, benchmark, q=1e-5, r=1e-4):
    """Return observation y=alpha+beta*x; random-walk coefficient Kalman update."""
    a, b = np.asarray(asset), np.asarray(benchmark)
    state = np.zeros(2)
    cov = np.eye(2)
    out = np.full((len(a), 2), np.nan)
    for i, (y, x) in enumerate(zip(a, b)):
        if not np.isfinite([y, x]).all():
            continue
        cov += np.eye(2) * q
        h = np.array([1.0, x])
        gain = cov @ h / (h @ cov @ h + r)
        state += gain * (y - h @ state)
        cov = (np.eye(2) - np.outer(gain, h)) @ cov
        out[i] = state
    return out


def cusum(returns, reference_mean=0.0, reference_std=0.01, threshold=5.0, drift=0.5):
    """Training-reference standardized online two-sided CUSUM; reset on alarm."""
    positive = negative = 0.0
    out = []
    for value in returns:
        z = (value - reference_mean) / max(reference_std, 1e-8) if np.isfinite(value) else 0
        positive = max(0.0, positive + z - drift)
        negative = min(0.0, negative + z + drift)
        alarm = positive > threshold or negative < -threshold
        out.append(float(alarm))
        if alarm:
            positive = negative = 0.0
    return np.asarray(out)


def hurst(x):
    """Trailing log-lag variance slope/2, diagnostic not proof of persistence."""
    x = np.asarray(x)
    lags = np.array([2, 4, 8, 16, 32])
    variances = np.array([np.var(x[n:] - x[:-n]) for n in lags])
    if len(x) < 100 or (variances <= 1e-15).any():
        return None
    return finite(np.polyfit(np.log(lags), np.log(variances), 1)[0] / 2)


def permutation_entropy(x, order=3):
    if len(x) < order + 10 or np.std(x) < 1e-12:
        return 0.0
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(x), order)
    ranks = np.argsort(windows, axis=1, kind="stable")
    _, counts = np.unique(ranks, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))


def sample_entropy(x, m=2):
    """SampEn=-log(matches(m+1)/matches(m)); bounded trailing window <=300."""
    x = np.asarray(x)[-300:]
    tol = 0.2 * np.std(x)
    if len(x) < 20 or tol < 1e-12:
        return 0.0
    counts = []
    for length in [m, m + 1]:
        w = np.lib.stride_tricks.sliding_window_view(x, length)
        dist = np.max(np.abs(w[:, None, :] - w[None, :, :]), axis=2)
        counts.append(np.sum(np.triu(dist <= tol, k=1)))
    return finite(-np.log(counts[1] / counts[0])) if min(counts) > 0 else None


def structure(close):
    """Trailing variance ratio, OU half-life and entropy; constant series return null."""
    x = np.log(np.asarray(close, dtype=float))[-512:]
    r = np.diff(x)
    if len(r) < 100 or np.var(r) < 1e-15:
        return {
            "status": "insufficient variable history",
            "hurst": None,
            "ou_half_life_bars": None,
        }
    b = np.linalg.lstsq(np.c_[np.ones(len(r)), x[:-1]], r, rcond=None)[0][1]
    half = -np.log(2) / np.log(1 + b) if -1 < b < 0 else None
    return {
        "status": "trailing diagnostics",
        "hurst": hurst(x),
        "variance_ratio_8": finite(np.var(x[8:] - x[:-8]) / (8 * np.var(r))),
        "autocorrelation_1": finite(np.corrcoef(r[1:], r[:-1])[0, 1]),
        "ou_half_life_bars": finite(half),
        "permutation_entropy": permutation_entropy(r),
        "sample_entropy": sample_entropy(r),
    }


def ofi(previous: dict, current: dict) -> float:
    """Cont-Kukanov-Stoikov best-quote event OFI with signed bid/ask size changes."""
    b, bp = current["bid"], previous["bid"]
    a, ap = current["ask"], previous["ask"]
    return float(
        (current["bid_size"] if b >= bp else 0)
        - (previous["bid_size"] if b <= bp else 0)
        - (current["ask_size"] if a <= ap else 0)
        + (previous["ask_size"] if a >= ap else 0)
    )


def book_features(bids: dict, asks: dict) -> dict:
    if not bids or not asks:
        return {"status": "unavailable; missing sequenced book"}
    bid, ask = max(bids), min(asks)
    if bid >= ask:
        return {"status": "invalid crossed book"}
    mid = (bid + ask) / 2
    bq, aq = bids[bid], asks[ask]
    out = {
        "status": "observed sequenced L2",
        "spread_bps": (ask - bid) / mid * 10000,
        "microprice": (ask * bq + bid * aq) / max(bq + aq, 1e-12),
        "imbalance": (bq - aq) / max(bq + aq, 1e-12),
        "bid": bid,
        "ask": ask,
        "bid_size": bq,
        "ask_size": aq,
    }
    for width in [0.001, 0.005, 0.01]:
        name = str(width * 100)
        out[f"bid_depth_{name}pct"] = sum(p * q for p, q in bids.items() if p >= mid * (1 - width))
        out[f"ask_depth_{name}pct"] = sum(p * q for p, q in asks.items() if p <= mid * (1 + width))
    return out


def vpin(buy_volume, sell_volume, bucket_size=1000.0, window=50):
    """Volume-synchronous imbalance; fractional allocation across exact-volume buckets."""
    if bucket_size <= 0:
        raise ValueError("Positive bucket size required")
    filled = imbalance = 0.0
    buckets = []
    for buy, sell in zip(buy_volume, sell_volume):
        volume = float(buy + sell)
        if volume <= 0:
            continue
        direction = (buy - sell) / volume
        while volume > 1e-9:
            amount = min(volume, bucket_size - filled)
            imbalance += amount * direction
            filled += amount
            volume -= amount
            if filled >= bucket_size - 1e-9:
                buckets.append(abs(imbalance) / bucket_size)
                filled = imbalance = 0.0
    return {
        "value": finite(np.mean(buckets[-window:])) if len(buckets) >= window else None,
        "completed_buckets": len(buckets),
        "bucket_size": bucket_size,
        "basis": "bar-aggregated volume allocation approximation; intra-bar ordering unavailable",
    }


def hawkes(events_seconds, cutoff=None):
    """Exponential conditional likelihood: sum log(lambda_i)-mu*T-alpha/beta*sum(1-exp(-beta*(T-ti)))."""
    t = np.asarray(events_seconds, dtype=float)
    t = t[np.isfinite(t)]
    if cutoff is not None:
        t = t[t <= cutoff]
    t = np.sort(t)[-3000:]
    if len(t) < 30 or t[-1] - t[0] < 1:
        return {
            "status": "insufficient event history",
            "branching_ratio": None,
            "intensity": None,
        }
    t = t - t[0]
    duration = t[-1] + 1e-6
    rate = len(t) / duration

    def objective(params):
        mu, beta, branch = (
            np.exp(params[0]),
            np.exp(params[1]),
            1 / (1 + np.exp(-params[2])) * 0.995,
        )
        alpha = branch * beta
        history = 0.0
        logs = 0.0
        for i in range(len(t)):
            if i:
                history = np.exp(-beta * (t[i] - t[i - 1])) * (history + 1)
            logs += np.log(max(mu + alpha * history, 1e-12))
        integral = mu * duration + branch * np.sum(-np.expm1(-beta * (duration - t)))
        return integral - logs

    fit = optimize.minimize(
        objective,
        [np.log(max(rate * 0.5, 1e-8)), 0.0, 0.0],
        method="L-BFGS-B",
        bounds=[(-20, 15), (-12, 12), (-10, 10)],
    )
    mu, beta, branch = (
        np.exp(fit.x[0]),
        np.exp(fit.x[1]),
        1 / (1 + np.exp(-fit.x[2])) * 0.995,
    )
    alpha = branch * beta
    intensity = mu + alpha * np.exp(-beta * (duration - t)).sum()
    return {
        "status": "fitted" if fit.success else "nonconverged",
        "branching_ratio": finite(branch),
        "intensity": finite(intensity),
        "mu": finite(mu),
        "beta": finite(beta),
        "events": len(t),
        "interpretation": f"Branching ratio {branch:.2f}: "
        + (
            "high modeled persistence; fragile under fitted assumptions"
            if branch > 0.8
            else "subcritical modeled excitation; not proof of cascade safety"
        ),
    }


def liquidation_proxy(
    price, atr, oi_change=None, volume_z=None, funding_z=None, oi_notional=None, price_return=0.0
):
    """Heuristic OI drop/price/volume/funding pressure; tier map is an assumed scenario."""
    required = [oi_change, volume_z, funding_z]
    if any(v is None or not np.isfinite(v) for v in required) or atr <= 0:
        return {
            "status": "unavailable; OI/volume/funding inputs missing",
            "estimated": True,
            "pressure": None,
            "clusters": [],
        }
    pressure = float(
        np.clip(
            max(0, -oi_change) * 10
            + abs(price_return) * price / max(atr, 1e-8) / 4
            + max(0, volume_z) / 5
            + abs(funding_z) / 8,
            0,
            1,
        )
    )
    tiers = [(5, 0.3), (10, 0.4), (20, 0.2), (50, 0.1)]
    clusters = []
    if oi_notional is not None and oi_notional > 0:
        for leverage, weight in tiers:
            for side in ["long", "short"]:
                level = price * (1 - 1 / leverage + 0.005 if side == "long" else 1 + 1 / leverage - 0.005)
                clusters.append(
                    {
                        "side": side,
                        "price": float(level),
                        "leverage": leverage,
                        "estimated_notional": float(oi_notional * weight / 2),
                        "distance_atr": abs(level - price) / atr,
                    }
                )
    return {
        "status": "estimated proxy, not observed liquidations",
        "estimated": True,
        "pressure": pressure,
        "clusters": clusters,
        "assumptions": "OI split 50/50; tier weights 30/40/20/10%; 0.5% maintenance; entry anchored at current price; not account-level data",
    }


def cross_asset(asset, benchmark):
    """Trailing synchronized log-price diagnostics; Granger is predictive, not causal proof."""
    from statsmodels.tsa.stattools import coint, grangercausalitytests
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    a, b = (
        np.asarray(asset, dtype=float)[-512:],
        np.asarray(benchmark, dtype=float)[-512:],
    )
    if (
        len(a) != len(b)
        or len(a) < 100
        or np.std(np.diff(np.log(a))) < 1e-10
        or np.std(np.diff(np.log(b))) < 1e-10
    ):
        return {"status": "insufficient aligned variable history"}
    prices = np.c_[np.log(a), np.log(b)]
    returns = np.diff(prices, axis=0)
    _, p, _ = coint(prices[:, 0], prices[:, 1])
    johansen = coint_johansen(prices, det_order=0, k_ar_diff=1)
    granger = grangercausalitytests(returns, maxlag=3, verbose=False)
    z = (returns - returns.mean(axis=0)) / returns.std(axis=0)
    eigenvalues = np.linalg.eigvalsh(np.cov(z.T))
    tau = stats.kendalltau(returns[:, 0], returns[:, 1]).statistic
    rho = np.clip(np.sin(np.pi * tau / 2), -0.99, 0.99)
    u = np.c_[stats.rankdata(returns[:, 0]), stats.rankdata(returns[:, 1])] / (len(returns) + 1)

    def likelihood(nu):
        df = float(nu[0])
        scores = stats.t.ppf(u, df)
        joint = stats.multivariate_t.logpdf(scores, shape=[[1, rho], [rho, 1]], df=df)
        return -float(np.sum(joint - stats.t.logpdf(scores, df).sum(axis=1)))

    fit = optimize.minimize(likelihood, [8.0], bounds=[(2.01, 50.0)], method="L-BFGS-B")
    nu = float(fit.x[0])
    tail = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), df=nu + 1)
    return {
        "status": "trailing fitted diagnostics",
        "engle_granger_p": finite(p),
        "johansen_trace": johansen.lr1.tolist(),
        "granger_min_p": min(float(v[0]["ssr_ftest"][1]) for v in granger.values()),
        "pca_first_variance_share": finite(eigenvalues[-1] / eigenvalues.sum()),
        "student_t_copula_tail_dependence": finite(tail),
        "copula_df": nu,
        "dynamic_beta": finite(dynamic_beta(returns[:, 0], returns[:, 1])[-1, 1]),
        "interpretation": "Historical statistical association; multiple-testing and model risk remain.",
    }


def dcc_correlation(returns):
    """DCC(1,1): Q=(1-a-b)Qbar+a*zprev*zprev'+b*Qprev, a+b<1."""
    x = np.asarray(returns, dtype=float)[-512:]
    if x.ndim != 2 or len(x) < 100 or (x.std(axis=0) < 1e-10).any():
        return {
            "status": "insufficient variable synchronized returns",
            "correlation": None,
        }
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    qbar = np.cov(z.T)

    def recurse(params, keep=False):
        a, b = params
        q = qbar.copy()
        cost = 0.0
        correlation = qbar.copy()
        if a + b >= 0.999:
            return 1e10
        for i in range(1, len(z)):
            q = (1 - a - b) * qbar + a * np.outer(z[i - 1], z[i - 1]) + b * q
            scale = np.sqrt(np.maximum(np.diag(q), 1e-10))
            correlation = q / np.outer(scale, scale)
            correlation += np.eye(x.shape[1]) * 1e-8
            sign, logdet = np.linalg.slogdet(correlation)
            if sign <= 0:
                return 1e10
            cost += logdet + z[i] @ np.linalg.solve(correlation, z[i])
        return correlation if keep else cost

    fit = optimize.minimize(
        recurse,
        [0.03, 0.93],
        bounds=[(1e-6, 0.5), (1e-6, 0.99)],
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda p: 0.999 - p.sum()}],
    )
    matrix = recurse(fit.x, True)
    return {
        "status": "fitted" if fit.success else "nonconverged",
        "a": float(fit.x[0]),
        "b": float(fit.x[1]),
        "correlation": matrix.tolist() if isinstance(matrix, np.ndarray) else None,
        "innovation_basis": "trailing standardized returns; not separately GARCH-filtered residuals",
    }


def spectral(close):
    """Trailing-window endpoint Hilbert/wavelet summaries; no centered future samples."""
    x = np.log(np.asarray(close, dtype=float))[-256:]
    if len(x) < 128 or np.std(x) < 1e-10:
        return {"status": "insufficient variable history", "hilbert_phase": None}
    detrended = signal.detrend(x)
    analytic = signal.hilbert(detrended)
    result = {
        "status": "trailing endpoint estimate; boundary effects",
        "hilbert_phase": finite(np.angle(analytic[-1])),
    }
    try:
        import pywt

        coeff = pywt.wavedec(detrended, "db2", mode="constant", level=3)
        result["wavelet_band_energy"] = [float(np.mean(c * c)) for c in coeff]
    except ImportError:
        result["wavelet_status"] = "optional PyWavelets not installed"
    return result


def risk_size(
    edge,
    variance,
    vol,
    cvar,
    cap=0.2,
    risk_budget=0.005,
    kelly_fraction=0.25,
    target_vol=0.15,
):
    """min(cap, quarter-Kelly, vol target, CVaR loss budget); edge must be calibrated."""
    if not np.isfinite([edge, variance, vol, cvar]).all() or min(variance, vol, cvar) <= 0:
        return 0.0
    return float(
        max(
            0,
            min(
                cap,
                kelly_fraction * edge / variance,
                target_vol / vol,
                risk_budget / cvar,
            ),
        )
    )


def impact_cost(participation, signed_size, spread_bps=2.0, eta=0.001, gamma=0.0001):
    """Almgren-Chriss-inspired scenario spread/2+eta*participation+gamma*|size|; coefficients assumed."""
    return max(0, spread_bps / 20000 + eta * max(0, participation) + gamma * abs(signed_size))


def uniqueness_weights(start, end):
    """Average inverse concurrency over each discrete label interval, inclusive endpoints."""
    start, end = np.asarray(start, dtype=int), np.asarray(end, dtype=int)
    if len(start) != len(end) or (end < start).any():
        raise ValueError("Invalid label intervals")
    if not len(start):
        return np.array([])
    grid = np.unique(np.r_[start, end + 1])
    left = np.searchsorted(grid, start)
    right = np.searchsorted(grid, end + 1)
    changes = np.zeros(len(grid))
    np.add.at(changes, left, 1)
    np.add.at(changes, right, -1)
    counts = np.cumsum(changes)[:-1]
    durations = np.diff(grid)
    inverse = np.divide(durations, counts, out=np.zeros_like(durations, dtype=float), where=counts > 0)
    prefix = np.r_[0.0, np.cumsum(inverse)]
    return (prefix[right] - prefix[left]) / (end - start + 1)


def diagnostics(bars: pd.DataFrame, advanced=False, derivatives=None, trade_events=None) -> dict:
    """Latest-window diagnostics; all fitted data precede this completed-bar decision."""
    if len(bars) < 100:
        return {
            "status": "warming_up",
            "interpretation": "At least 100 completed bars required.",
        }
    close = bars.close
    v = volatility_features(close)
    r = v["return"].dropna().to_numpy()
    latest = v.iloc[-1]
    q = {k: finite(value) for k, value in latest.items()}
    daily = (
        pd.DataFrame({"day": bars.open_time // 86400000, "r2": v["return"].pow(2)}).groupby("day").r2.sum()
    )
    daily = daily[(daily.index + 1) * 86400000 <= int(bars.available_at.iloc[-1])]
    latent = kalman_trend(close)
    atr = float((bars.high - bars.low).rolling(14).mean().iloc[-1])
    volume_z = finite(
        (bars.volume.iloc[-1] - bars.volume.tail(100).mean()) / max(bars.volume.tail(100).std(), 1e-12)
    )
    derivative = derivatives or {}
    proxy = liquidation_proxy(
        float(close.iloc[-1]),
        atr,
        derivative.get("oi_change"),
        volume_z,
        derivative.get("funding_z"),
        derivative.get("oi_notional"),
        float(close.iloc[-1] / close.iloc[-2] - 1),
    )
    flow: dict[str, Any] = {"status": "unavailable; tick-derived buy/sell split required"}
    if "signed_volume_delta" in bars:
        dollar = bars.quote_volume.replace(0, np.nan)
        delta = bars.signed_volume_delta.tail(128).to_numpy()
        y = close.diff().tail(128).to_numpy()
        mask = np.isfinite(delta) & np.isfinite(y)
        slope = (
            np.linalg.lstsq(np.c_[np.ones(mask.sum()), delta[mask]], y[mask], rcond=None)[0][1]
            if mask.sum() > 20
            else np.nan
        )
        bucket = max(float(bars.volume.tail(100).median()) * 5, 1.0)
        flow = {
            "status": "observed tick-derived taker flow",
            "kyle_lambda": finite(slope),
            "amihud": finite((v["return"].abs() / dollar).tail(24).mean()),
            "vpin": vpin(bars.buy_volume, bars.sell_volume, bucket),
            "realized_spread": None,
            "realized_spread_status": "requires historical quote outcomes at delayed horizon",
        }
    else:
        flow["amihud"] = finite(
            (v["return"].abs() / (bars.volume * close).replace(0, np.nan)).tail(24).mean()
        )
    trend = finite(latent[-1, 1])
    regime = (
        "panic"
        if q.get("rv") is not None and q["rv"] > float(v.rv.tail(100).quantile(0.95))
        else "trend"
        if trend is not None and abs(trend) > 0.001
        else "range"
    )
    return {
        "status": "ready",
        "available_at": max(int(bars.available_at.iloc[-1]), int(derivative.get("available_at", 0))),
        "volatility": q,
        "har_rv": har_forecast(daily),
        "garch": garch_forecasts(r)
        if advanced
        else {"status": "not fitted in fast path; use advanced research diagnostics"},
        "tail": evt_tail(r),
        "structure": structure(close),
        "kalman_log_slope": trend,
        "change_point_alarm": float(cusum(r)[-1]),
        "regime": {
            "name": regime,
            "basis": "trailing heuristic rules; not HMM",
            "posterior": None,
        },
        "flow": flow,
        "trade_hawkes": hawkes(trade_events)
        if trade_events is not None
        else {
            "status": "unavailable; event timestamps required",
            "branching_ratio": None,
        },
        "liquidation_hawkes": {
            "status": "unavailable; true liquidation event history missing",
            "branching_ratio": None,
        },
        "liquidation": proxy,
        "derivatives": derivative,
        "spectral": spectral(close) if advanced else {"status": "not fitted in fast path"},
        "interpretation": "Trailing diagnostics describe market conditions; estimates and attribution are not causal proof.",
    }
