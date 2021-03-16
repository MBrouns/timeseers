import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition, invert_dict


class FourierSeasonality(TimeSeriesModel):
    def __init__(
        self,
        name: str = None,
        n: int = 10,
        period: pd.Timedelta = pd.Timedelta(days=365.25),
        shrinkage_strength=100,
        pool_cols=None,
        pool_type='complete'
    ):
        self.n = n
        self.period = period
        self.shrinkage_strength = shrinkage_strength
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"FourierSeasonality(period={self.period})"
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

                mu_beta = pm.Normal(self._param_name("mu_beta"), mu=0, sigma=1, shape=n_params)
                sigma_beta = pm.HalfNormal(self._param_name("sigma_beta"), 0.1, shape=n_params)
                offset_beta = pm.Normal(
                    self._param_name("offset_beta"),
                    0,
                    1 / self.shrinkage_strength,
                    shape=(n_groups, n_params)
                )

                beta = pm.Deterministic(self._param_name("beta"), mu_beta + offset_beta * sigma_beta)
            else:
                beta = pm.Normal(self._param_name("beta"), 0, 1, shape=(n_groups, n_params))

            seasonality = pm.math.sum(self._X_t(t, self.p_, self.n) * beta[group], axis=1)

        return seasonality

    def _predict(self, trace, X):
        t = X['t']
        if self.pool_type == 'complete':
            pool_group = np.zeros(len(X), dtype=np.int)
        else:
            pool_group = X[self.pool_cols].map(invert_dict(self.groups_))
        l = self._X_t(t, self.p_, self.n)[..., None]
        r = trace[self._param_name("beta")][:, pool_group].transpose(1, 2, 0)

        return (l * r).sum(axis=1)

    def plot(self, trace, X, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))

        scaled_s = self._predict(trace, X)
        s = y_scaler.inv_transform(scaled_s)
        for group_code, group_name in self.groups_.items():
            mask = X[self.pool_cols] == group_name if self.pool_cols is not None else np.full(len(s), True)
            ax.plot(
                list(range(self.period.days)),
                s[mask].mean(axis=1)[:self.period.days],
                label=group_name
            )

        return scaled_s

    def __repr__(self):
        return f"FourierSeasonality(n={self.n}, " \
               f"period={self.period}," \
               f"pool_cols={self.pool_cols}, " \
               f"pool_type={self.pool_type}"
