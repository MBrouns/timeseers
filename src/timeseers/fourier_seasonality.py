import numpy as np
import pandas as pd
import pymc as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition


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

    def _predict(self, trace, t, pool_group=0):
        return self._X_t(t, self.p_, self.n) @ trace[self._param_name("beta")][:, pool_group].T

    def predict_component(self, trace, scaled_t, y_scaler):
        seasonality_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, _ in self.groups_.items():
            scaled_s = self._predict(trace, scaled_t, group_code)
            s = y_scaler.inv_transform(scaled_s)
            seasonality_return[:, group_code] = s.mean(axis=1)
        return seasonality_return

    def predict_component(self, trace, X_scaled):
        
        preds = pd.DataFrame(columns=['g','t','preds'])
        for group_code, group_name in self.groups_.items():
            scaled_t = X_scaled[X_scaled.g==group_name]['t'].sort_values().reset_index(drop=True).to_numpy()
            scaled_seasonality = self._predict(trace, scaled_t, group_code)
            seasonality_pred = pd.DataFrame(
                {
                    'g': group_name,
                    't': scaled_t,
                    'preds': scaled_seasonality.mean(axis=1)
                }
            )
            preds = pd.concat([preds, seasonality_pred])
            
        # set index to have a dataframe which can be added or multiplied with other components
        preds.set_index(['g','t'], inplace=True)

        return preds

    def plot(self, trace, X, t_scaler, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))

        # get min and max date over all groups
        min_date = X['t'].min()
        max_date = min_date + self.period

        # based on min and max date, build a full seasonality period 
        seasonality_period = pd.DataFrame({'t': X[(X['t']>=min_date)&(X['t']<=max_date)]['t'].sort_values().unique()})

        for group_code, group_name in self.groups_.items():
            # scale seasonality period
            t_grp_scaled = t_scaler.transform(seasonality_period)['t'].to_numpy()
            
            # predict seasonality for that one seasonality period
            scaled_s = self._predict(trace, t_grp_scaled, group_code)

            # re-scale seasonality
            s = y_scaler.inv_transform(scaled_s)
            
            # print seasonality on that seasonality period but with differently formatted x-tick labels (only day and month)
            seasonality_period_formatted = [dt.strftime(format="%b-%d") for dt in seasonality_period.t]
            ax.plot(seasonality_period_formatted, s.mean(axis=1)[:len(seasonality_period)], label=group_name)

    def __repr__(self):
        return f"FourierSeasonality(n={self.n}, " \
               f"period={self.period}," \
               f"pool_cols={self.pool_cols}, " \
               f"pool_type={self.pool_type}"
