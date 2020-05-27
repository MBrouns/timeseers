import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import dot, add_subplot


class FourierSeasonality(TimeSeriesModel):
    def __init__(self, n: int = 10, period: pd.Timedelta = pd.Timedelta(days=365.25), pool_cols=None, pool_type=None):
        self.n = n
        self.period = period
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        super().__init__()

    @staticmethod
    def _X_t(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group = X[self.pool_cols].cat.codes.values
        n_groups = X[self.pool_cols].nunique()
        self.p_ = self.period / scale_factor['t']
        n_params = self.n * 2

        if self.pool_type == 'complete':
            with model:
                beta = pm.Normal("beta", 0, 1, shape=n_params)
                seasonality = dot(self._X_t(t, self.p_, self.n), beta)

        if self.pool_type == 'none':
            with model:
                beta = pm.Normal("beta", 0, 1, shape=(n_groups, n_params))
                seasonality = pm.math.sum(self._X_t(t, self.p_, self.n) * beta[group], axis=1)

        if self.pool_type == 'partial':
            with model:
                mu_beta = pm.Normal("mu_beta", 0, 1, shape=n_params)   # TODO: add as parameters
                sigma_beta = pm.HalfCauchy("sigma_beta", 1, shape=n_params)
                beta = pm.Normal("beta", mu_beta, sigma_beta, shape=(n_groups, n_params))
                seasonality = pm.math.sum(self._X_t(t, self.p_, self.n) * beta[group], axis=1)

        return seasonality

    def _predict(self, trace, t):
        return self._X_t(t, self.p_, self.n) @ trace["beta"].T

    def plot(self, trace, scaled_t, y_scaler):
        scaled_s = pd.Series(self._predict(trace, scaled_t).mean(axis=1), name='value')
        s = y_scaler.inv_transform(scaled_s)

        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])
        ax.plot(scaled_t, s, c="lightblue")
        return scaled_s

    def __repr__(self):
        return f"FourierSeasonality(n={self.n})"
