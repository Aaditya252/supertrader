import numpy as np
from scipy.special import logsumexp


class CausalHMM:
    """Train on a past fold; recursively filter. Never smooth using future observations."""
    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, x: np.ndarray):
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(x)
        self.model = GaussianHMM(3, covariance_type="diag", n_iter=150, random_state=self.seed)
        self.model.fit(self.scaler.transform(x))
        return self

    def filter(self, x: np.ndarray, initial: np.ndarray | None = None) -> np.ndarray:
        z = self.scaler.transform(x)
        variance = np.maximum(self.model.covars_[:, np.arange(z.shape[1]), np.arange(z.shape[1])], 1e-8)
        means = self.model.means_
        prior = self.model.startprob_ if initial is None else initial
        output = []
        for row in z:
            loglik = -.5*np.sum(np.log(2*np.pi*variance)+(row-means)**2/variance, axis=1)
            logpost = np.log(np.maximum(prior, 1e-15))+loglik
            posterior = np.exp(logpost-logsumexp(logpost))
            output.append(posterior)
            prior = posterior @ self.model.transmat_
        return np.array(output)

