import numpy as np
import pandas as pd
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition
import pymc as pm
from scipy.stats import mode


class Indicator(TimeSeriesModel):
    def __init__(self, name: str = None, pool_cols=None, pool_type='complete'):
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"Indicator(pool_cols='{self.pool_cols}', pool_type='{self.pool_type}')"
        super().__init__()

    def definition(self, model, X, scale_factor):
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)

        with model:
            if self.pool_type == "partial":
                raise ValueError("Indicator() component doesn't support partial pooling")

            _ind = pm.Beta(self._param_name('_ind'), alpha=0.5, beta=0.5, shape=n_groups)
            ind = pm.Deterministic(self._param_name('ind'), _ind * 2 - 1)

        return ind[group]

    def _predict(self, trace, t, pool_group=0):
        ind = trace[self._param_name("ind")][:, pool_group]

        return np.ones_like(t)[:, None] * ind.reshape(1, -1)

    def predict_component(self, trace, X_scaled, y_scaler):
        
        preds = pd.DataFrame(columns=['g','t','preds'])
        for group_code, group_name in self.groups_.items():
            scaled_t = X_scaled[X_scaled.g==group_name]['t'].sort_values().reset_index(drop=True).to_numpy()
            scaled_pred = self._predict(trace, scaled_t, group_code)
            pred = y_scaler.inv_transform(scaled_pred)
            trend_pred = pd.DataFrame(
                {
                    'g': group_name,
                    't': scaled_t,
                    'preds': pred.mean(axis=1)
                }
            )
            preds = pd.concat([preds, trend_pred])
        
        # set index to have a dataframe which can be added or multiplied with other components
        preds.set_index(['g','t'], inplace=True)

        return preds
        
    def plot(self, trace, scaled_t, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])
        trend_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            y_hat = mode(self._predict(trace, scaled_t, group_code), axis=1)[0][:, 0]
            ax.plot(scaled_t, y_hat, label=group_name)
            trend_return[:, group_code] = y_hat
        ax.set_ylim([-1.05, 1.05])
        ax.legend()
        return trend_return

    def __repr__(self):
        return f"Indicator(pool_cols={self.pool_cols}, pool_type={self.pool_type})"
