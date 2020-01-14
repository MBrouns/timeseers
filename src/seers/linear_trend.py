import numpy as np
from seers.utils import dot
import pymc3 as pm


class LinearTrend:
    def __init__(self, n_changepoints=None, changepoints_prior_scale=0.05, growth_prior_scale=1):
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale

    def definition(self, model, X):
        t = X['t'].values
        self.s = np.linspace(0, np.max(t), self.n_changepoints + 1)[1:]
        with model:
            A = (t[:, None] > self.s) * 1.
            # initial growth
            k = pm.Normal('k', 0, self.growth_prior_scale)

            # rate of change
            delta = pm.Laplace('delta', 0, self.changepoints_prior_scale, shape=self.n_changepoints)
            # offset
            m = pm.Normal('m', 0, 5)
            gamma = -self.s * delta

            g = (k + dot(A, delta)) * t + (m + dot(A, gamma))
        return g

    def plot(self, ax, trace, t):
        A = (t[:, None] > self.s) * 1

        k, m = trace['k'], trace['m']
        growth = (k + A @ trace['delta'].T)
        gamma = -self.s[:, None] * trace['delta'].T
        offset = m + A @ gamma
        trend = growth * t[:, None] + offset

        ax.set_title(str(self))
        ax.plot(t, trend)
        for changepoint in self.s:
            ax.axvline(changepoint, linestyle='--', alpha=0.2, c='k')

        return trend

    def __repr__(self):
        return f"LinearTrend(n_changepoints={self.n_changepoints}, changepoints_prior_scale={self.changepoints_prior_scale}, growth_prior_scale={self.growth_prior_scale})"
