import numpy as np
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition
import pymc3 as pm
from scipy.stats import mode
import theano.tensor as tt


class Constant(TimeSeriesModel):
    def __init__(self, name: str = None, lower=0, upper=1, pool_cols=None, pool_type='complete'):
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.lower = lower
        self.upper = upper
        self.name = name or f"Constant(lower={self.lower}, upper={self.upper} pool_cols='{self.pool_cols}', pool_type='{self.pool_type}')"
        super().__init__()

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)

        with model:
            if self.pool_type == "partial":

                mu_c = pm.Uniform(self._param_name('mu_c'), lower=self.lower, upper=self.upper)
                offset_c = pm.Normal(self._param_name('offset_c'), mu=0, sigma=1)
                c = pm.Deterministic(self._param_name('c'), mu_c + offset_c)
            else:
                c = pm.Uniform(self._param_name('c'), lower=self.lower, upper=self.upper, shape=n_groups)

        return c[group]

    def _predict(self, trace, t, pool_group=0):
        ind = trace[self._param_name("c")][:, pool_group]

        return np.ones_like(t)[:, None] * ind.reshape(1, -1)

    def plot(self, trace, scaled_t, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])
        trend_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            y_hat = np.mean(self._predict(trace, scaled_t, group_code), axis=1)
            ax.plot(scaled_t, y_scaler.inv_transform(y_hat), label=group_name)
            trend_return[:, group_code] = y_hat
        delta = self.upper - self.lower
        ax.set_ylim([self.lower - 0.1 * delta, self.upper + 0.1 * delta])
        ax.legend()
        return trend_return

    def __repr__(self):
        return f"Constant(pool_cols={self.pool_cols}, pool_type={self.pool_type})"
