import numpy as np
from seers.timeseries_model import TimeSeriesModel
from seers.utils import dot, add_subplot
import pymc3 as pm


class FourierSeasonality(TimeSeriesModel):
    def __init__(self, n=10, period=365.25):
        self.n = n
        self.period = period
        super().__init__()

    @staticmethod
    def _X_t(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    def definition(self, model, X, scale_factor):
        t = X['t'].values
        self.p_ = self.period / scale_factor

        with model:
            beta = pm.Normal('beta', 0, 10, shape=self.n * 2)
            seasonality = dot(self._X_t(t, self.p_, self.n), beta)

        return seasonality

    def plot(self, trace, t, y_scaler):
        scaled_s = self._X_t(t, self.p_, self.n) @ trace['beta'].T
        s = y_scaler.inv_transform(scaled_s)

        ax = add_subplot()
        ax.set_title(str(self))
        ax.plot(t, s.mean(axis=1), c='lightblue')
        return scaled_s.mean(axis=1)

    def __repr__(self):
        return f"FourierSeasonality(n={self.n})"
