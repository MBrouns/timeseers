import numpy as np
from seers.timeseries_model import TimeSeriesModel
from seers.utils import dot, add_subplot
import pymc3 as pm


class LinearTrend(TimeSeriesModel):
    def __init__(
        self, n_changepoints=None, changepoints_prior_scale=0.05, growth_prior_scale=1
    ):
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale
        super().__init__()

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        self.s = np.linspace(0, np.max(t), self.n_changepoints + 1)[1:]

        with model:
            A = (t[:, None] > self.s) * 1.0
            k = pm.Normal("k", 0, self.growth_prior_scale)
            delta = pm.Laplace(
                "delta", 0, self.changepoints_prior_scale, shape=self.n_changepoints
            )
            m = pm.Normal("m", 0, 5)
            gamma = -self.s * delta

            g = (k + dot(A, delta)) * t + (m + dot(A, gamma))
        return g

    def _predict(self, trace, t):
        A = (t[:, None] > self.s) * 1

        k, m = trace["k"], trace["m"]
        growth = k + A @ trace["delta"].T
        gamma = -self.s[:, None] * trace["delta"].T
        offset = m + A @ gamma
        return growth * t[:, None] + offset

    def plot(self, trace, t, y_scaler):
        ax = add_subplot()

        scaled_trend = self.predict(trace, t)
        trend = y_scaler.inv_transform(scaled_trend)

        ax.set_title(str(self))
        ax.plot(t, trend.mean(axis=1), c="lightblue")
        for changepoint in self.s:
            ax.axvline(changepoint, linestyle="--", alpha=0.2, c="k")

        return scaled_trend.mean(axis=1)

    def __repr__(self):
        return f"LinearTrend(n_changepoints={self.n_changepoints}, " \
               f"changepoints_prior_scale={self.changepoints_prior_scale}, " \
               f"growth_prior_scale={self.growth_prior_scale})"
