import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import dot, add_subplot, get_group_definition


class FourierSeasonality(TimeSeriesModel):
    def __init__(self, n: int = 10, period: pd.Timedelta = pd.Timedelta(days=365.25), pool_cols=None, pool_type='complete'):
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
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        self.p_ = self.period / scale_factor['t']
        n_params = self.n * 2

        with model:
            if self.pool_type == 'partial':
                mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, shape=n_params)  # TODO: add as parameters
                sigma_beta = pm.HalfCauchy("sigma_beta", 1, shape=n_params)
                offset_beta = pm.Normal("offset_beta", 0, 1, shape=(n_groups, n_params))

                beta = pm.Deterministic("beta", mu_beta + offset_beta * sigma_beta)
            else:
                beta = pm.Normal("beta", 0, 1, shape=(n_groups, n_params))

            seasonality = pm.math.sum(self._X_t(t, self.p_, self.n) * beta[group], axis=1)

        return seasonality

    def _predict(self, trace, t, pool_group=0):
        return self._X_t(t, self.p_, self.n) @ trace["beta"][:, pool_group].T

    def plot(self, trace, scaled_t, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])

        seasonality_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            scaled_s = self._predict(trace, scaled_t, group_code)
            s = y_scaler.inv_transform(scaled_s)
            ax.plot(scaled_t, s.mean(axis=1), label=group_name)
            seasonality_return[:, group_code] = scaled_s.mean(axis=1)

        return seasonality_return

    def __repr__(self):
        return f"FourierSeasonality(n={self.n})"
