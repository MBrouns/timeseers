import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import add_subplot, get_group_definition, get_periodic_peaks


class RBFSeasonality(TimeSeriesModel):
    """
    Seasonality with radial basis functions. Periodic radial basis functions
    are placed to model seasonality. With ``peaks`` argument, RBF's can be placed
    arbitrarily. If peaks is not provided, 20 evenly placed RBF's are used 
    evenly spread out over `period` days
    """
    def __init__(
        self,
        name: str = None,
        peaks: np.ndarray = None,
        period: pd.Timedelta = pd.Timedelta(days=365.25),
        shrinkage_strength=100,
        pool_cols=None,
        sigma=0.1,
        pool_type='complete'
    ):
        if peaks is None:
            self.peaks = get_periodic_peaks(period=period)
        else:
            self.peaks = peaks
        self.period = period
        self.shrinkage_strength = shrinkage_strength
        self.sigma = sigma
        self.pool_cols = pool_cols
        self.pool_type = pool_type
        self.name = name or f"RBFSeasonality(period={self.period})"
        super().__init__()

    @staticmethod
    def _X_t(t, peaks, sigma, year):
        mod = (t % year)[:, None]
        left_difference = np.sqrt((mod - peaks[None, :]) ** 2)
        right_difference = np.abs(year - left_difference)
        return np.exp(-((np.minimum(left_difference, right_difference)) ** 2) / (2 * sigma**2))

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        group, n_groups, self.groups_ = get_group_definition(X, self.pool_cols, self.pool_type)
        self.p_ = self.period / scale_factor['t']
        self.peaks_ = self.peaks / scale_factor['t']
        n_params = len(self.peaks)
        self.factor_ = scale_factor["t"]
        with model:
            if self.pool_type == 'partial':

                mu_beta = pm.Normal(self._param_name("mu_beta"), mu=0, sigma=1, shape=n_params)
                sigma_beta = pm.HalfNormal(self._param_name("sigma_beta"), 0.1, shape=n_params)
                offset_beta = pm.Normal(
                    self._param_name("offset_beta"), 0, 1 / self.shrinkage_strength, shape=(n_groups, n_params))

                beta = pm.Deterministic(self._param_name("beta"), mu_beta + offset_beta * sigma_beta)
            else:
                beta = pm.Normal(self._param_name("beta"), 0, 1, shape=(n_groups, n_params))

            seasonality = pm.math.sum(self._X_t(t, self.peaks_, self.sigma, self.p_) * beta[group], axis=1)

        return seasonality

    def _predict(self, trace, t, pool_group=0):
        return self._X_t(t, self.peaks_, self.sigma, self.p_) @ trace[self._param_name("beta")][:, pool_group].T

    def plot(self, trace, scaled_t, y_scaler):
        ax = add_subplot()
        ax.set_title(str(self))

        seasonality_return = np.empty((len(scaled_t), len(self.groups_)))
        for group_code, group_name in self.groups_.items():
            scaled_s = self._predict(trace, scaled_t, group_code)
            s = y_scaler.inv_transform(scaled_s)
            ax.plot(list(range(self.period.days)), s.mean(axis=1)[:self.period.days], label=group_name)

            seasonality_return[:, group_code] = scaled_s.mean(axis=1)

        return seasonality_return

    def __repr__(self):
        return f"RBFSeasonality(n={self.period}, " \
               f"pool_cols={self.pool_cols}, " \
               f"pool_type={self.pool_type}"
