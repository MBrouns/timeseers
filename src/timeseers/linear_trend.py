import numpy as np
import pandas as pd
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition, MinMaxScaler
import pymc as pm

class LinearTrend(TimeSeriesModel):
    def __init__(
            self, name: str = None, n_changepoints=None, changepoints_prior_scale=0.05, growth_prior_scale=1,
            pool_cols=None, pool_type='complete'
    ):
        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"LinearTrend(n_changepoints={n_changepoints})"
        super().__init__()

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        self.s = np.linspace(0, np.max(t), self.n_changepoints + 2)[1:-1]

        with model:
            A = (t[:, None] > self.s) * 1.0

            if self.pool_type == 'partial':
                sigma_k = pm.HalfCauchy(self._param_name('sigma_k'), beta=self.growth_prior_scale)
                offset_k = pm.Normal(self._param_name('offset_k'), 0, 1, shape=n_groups)
                k = pm.Deterministic(self._param_name("k"), offset_k * sigma_k)

                sigma_delta = pm.HalfCauchy(self._param_name('sigma_delta'), beta=self.changepoints_prior_scale)
                offset_delta = pm.Laplace(self._param_name('offset_delta'), 0, 1, shape=(n_groups, self.n_changepoints))
                delta = pm.Deterministic(self._param_name("delta"), offset_delta * sigma_delta)

            else:
                delta = pm.Laplace(
                    self._param_name("delta"), 0, self.changepoints_prior_scale, shape=(n_groups, self.n_changepoints)
                )
                k = pm.Normal(self._param_name("k"), 0, self.growth_prior_scale, shape=n_groups)

            m = pm.Normal(self._param_name("m"), 0, 5, shape=n_groups)

            gamma = -self.s * delta[group, :]

            g = (
                (k[group] + pm.math.sum(A * delta[group], axis=1)) * t
                + (m[group] + pm.math.sum(A * gamma, axis=1))
            )
        return g

    def _predict(self, trace, t, pool_group=0):
        A = (t[:, None] > self.s) * 1

        k, m = trace[self._param_name("k")][:, pool_group], trace[self._param_name("m")][:, pool_group]
        growth = k + A @ trace[self._param_name("delta")][:, pool_group].T
        gamma = -self.s[:, None] * trace[self._param_name("delta")][:, pool_group].T
        offset = m + A @ gamma
        return growth * t[:, None] + offset

    def predict_component(self, trace, X_scaled):
        
        preds = pd.DataFrame(columns=['g','t','preds'])
        for group_code, group_name in self.groups_.items():
            scaled_t = X_scaled[X_scaled.g==group_name]['t'].sort_values().reset_index(drop=True).to_numpy()
            scaled_trend = self._predict(trace, scaled_t, group_code)
            trend_pred = pd.DataFrame(
                {
                    'g': group_name,
                    't': scaled_t,
                    'preds': scaled_trend.mean(axis=1)
                }
            )
            preds = pd.concat([preds, trend_pred])
        
        # set index to have a dataframe which can be added or multiplied with other components
        preds.set_index(['g','t'], inplace=True)
        
        return preds

    def plot(self, trace, X, t_scaler, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        
        # plot each groups separately
        for group_code, group_name in self.groups_.items():

            # get t for this group and scale it
            t_grp = X[X.g==group_name][['t']].sort_values('t').reset_index(drop=True)
            t_grp_scaled = t_scaler.transform(t_grp)['t'].to_numpy()

            # predict trend and re-scale it
            scaled_trend = self._predict(trace, t_grp_scaled, group_code)
            trend = y_scaler.inv_transform(scaled_trend)
            
            # plot on full x-axis space
            ax.plot(t_grp, trend.mean(axis=1), label=group_name)

        # re-scale s with the t_scaler and plot changepoint locations
        rescaled_s = t_scaler.inv_transform(pd.DataFrame({'t': self.s}))['t'].to_numpy()
        for changepoint in rescaled_s:
            ax.axvline(changepoint, linestyle="--", alpha=0.2, c="k")
        
        # add legend to plot
        ax.legend()

    def __repr__(self):
        return f"LinearTrend(n_changepoints={self.n_changepoints}, " \
               f"changepoints_prior_scale={self.changepoints_prior_scale}, " \
               f"growth_prior_scale={self.growth_prior_scale})"
