import numpy as np
import pandas as pd
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition
import pymc as pm


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
                offset_k = pm.Normal(self._param_name('offset_k'), 0, 1, shape=(n_groups, self.shape_))
                k = pm.Deterministic(self._param_name("k"), offset_k * sigma_k)

            else:
                k = pm.Normal(self._param_name('k'), mu=0, sigma=self.scale, shape=(n_groups, self.shape_))

        return k[group, X[self.on].cat.codes]

    def _predict(self, trace, t, pool_group=0):
        ind = trace[self._param_name("k")][:, pool_group]

        return np.ones_like(t)[:, None] * ind.reshape(1, -1)

    def predict_component(self, trace, X_scaled):
        
        preds = pd.DataFrame(columns=['g','t','preds'])
        for group_code, group_name in self.groups_.items():
            scaled_t = X_scaled[X_scaled.g==group_name]['t'].sort_values().reset_index(drop=True).to_numpy()
            scaled_pred = self._predict(trace, scaled_t, group_code)
            trend_pred = pd.DataFrame(
                {
                    'g': group_name,
                    't': scaled_t,
                    'preds': scaled_pred.mean(axis=1)
                }
            )
            preds = pd.concat([preds, trend_pred])
        
        # set index to have a dataframe which can be added or multiplied with other components
        preds.set_index(['g','t'], inplace=True)

        return preds
        
    def plot(self, trace, X, t_scaler, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        
        plot_data = []
        for group_code, group_name in self.groups_.items():
            
            # get t for this group and scale it
            t_grp = X[X.g==group_name][['t']].sort_values('t').reset_index(drop=True)
            t_grp_scaled = t_scaler.transform(t_grp)['t'].to_numpy()
            
            # predict 
            y_hat = np.mean(self._predict(trace, t_grp_scaled, group_code), axis=1)
            
            # plot
            plot_data.append((group_name, y_hat[0]))

        ax.bar(*zip(*plot_data))
        ax.axhline(0, c='k', linewidth=3)

    def __repr__(self):
        return f"LinearRegressor(on={self.on}, pool_cols={self.pool_cols}, pool_type={self.pool_type})"
