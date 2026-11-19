"""Training-only four-state HMM with forward filtering at inference."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class RegimeFilter:
    def fit(self, x, seed=42):
        from hmmlearn.hmm import GaussianHMM

        x = np.asarray(x, dtype=float)
        if len(x) < 500 or not np.isfinite(x).all():
            raise ValueError("At least 500 finite training observations required")
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), 1e-6)
        self.model = GaussianHMM(4, covariance_type="diag", n_iter=100, random_state=seed, min_covar=1e-4)
        self.model.fit((x - self.mean) / self.std)
        # Semantics are emission-based labels, not known ground truth.
        means = self.model.means_
        panic = int(np.argmax(means[:, 1]))
        squeeze = int(np.argmin(means[:, 1]))
        rest = [i for i in range(4) if i not in [panic, squeeze]]
        trend = max(rest, key=lambda i: abs(means[i, 0])) if rest else 0
        self.labels = {i: "range" for i in range(4)}
        self.labels.update({panic: "panic", squeeze: "squeeze", trend: "trend"})
        return self

    def filter(self, x):
        """p(s_t|x_<=t) proportional emission_t * T' p(s_(t-1)); no smoothing."""
        x = (np.asarray(x) - self.mean) / self.std
        posterior = np.log(np.maximum(self.model.startprob_, 1e-12))
        transition = np.log(np.maximum(self.model.transmat_, 1e-12))
        out = []
        for i, row in enumerate(x):
            emission = np.array(
                [
                    multivariate_normal.logpdf(row, mean=m, cov=c, allow_singular=True)
                    for m, c in zip(self.model.means_, self.model.covars_)
                ]
            )
            if i:
                posterior = logsumexp(posterior[:, None] + transition, axis=0)
            posterior += emission
            posterior -= logsumexp(posterior)
            out.append(np.exp(posterior))
        return np.asarray(out)
