import numpy as np
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition, invert_dict
import pymc3 as pm


class Regressor(TimeSeriesModel):
    def __init__(self, on: str, scale: float = 1., name: str = None, pool_cols=None, pool_type='complete'):
        self.on = on
        self.scale = scale
        self.pool_cols = pool_cols
        self.pool_type = pool_type

        self.name = name or f"LinearRegressor(on={self.on}, scale={self.scale}, " \
                            f"pool_cols='{self.pool_cols}', pool_type='{self.pool_type}')"
        super().__init__()

    def definition(self, model, X, scale_factor):
        on = X[self.on].cat.categories
        self.shape_ = len(on)
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)

        with model:
            if self.pool_type == "partial":
                sigma_k = pm.HalfCauchy(self._param_name('sigma_k'), beta=self.scale)
                offset_k = pm.Normal(self._param_name('offset_k'), mu=0, sd=1, shape=(n_groups, self.shape_))
                k = pm.Deterministic(self._param_name("k"), offset_k * sigma_k)

            else:
                k = pm.Normal(self._param_name('k'), mu=0, sigma=self.scale, shape=(n_groups, self.shape_))

        return k[group, X[self.on].cat.codes]

    def _predict(self, trace, X):
        t = X['t']
        if self.pool_type == 'complete':
            pool_group = np.zeros(len(X), dtype=np.int)
        else:
            pool_group = X[self.pool_cols].map(invert_dict(self.groups_))

        ind = trace[self._param_name("k")][np.arange(len(t)), pool_group]
        return np.ones_like(t)[:, None] * ind.reshape(1, -1) * X[self.on]

    def _plot_predict(self, trace, t, pool_group=0):
        ind = trace[self._param_name("k")][:, pool_group]

        return np.ones_like(t)[:, None] * ind.reshape(1, -1)

    def plot(self, trace, scaled_t, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        # ax.set_xticks([])
        trend_return = np.empty((len(scaled_t), len(self.groups_)))
        plot_data = []
        for group_code, group_name in self.groups_.items():
            y_hat = np.mean(self._plot_predict(trace, scaled_t, group_code), axis=1)
            trend_return[:, group_code] = y_hat
            plot_data.append((group_name, y_hat[0]))
        ax.bar(*zip(*plot_data))
        ax.axhline(0, c='k', linewidth=3)

        return trend_return

    def __repr__(self):
        return f"LinearRegressor(on={self.on}, pool_cols={self.pool_cols}, pool_type={self.pool_type})"
