import numpy as np
import pandas as pd
import pymc3 as pm
from timeseers.timeseries_model import TimeSeriesModel
from timeseers.utils import dot, add_subplot


class FourierSeasonality(TimeSeriesModel):
    def __init__(self, n: int = 10, period: pd.Timedelta = pd.Timedelta(days=365.25)):
        self.n = n
        self.period = period
        super().__init__()

    @staticmethod
    def _X_t(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    def definition(self, model, X, scale_factor):
        t = X["t"].values
        # print(t)
        self.p_ = self.period / scale_factor
        # print(self.p_)

        with model:
            beta = pm.Normal("beta", 0, 1, shape=self.n * 2)
            seasonality = dot(self._X_t(t, self.p_, self.n), beta)

        return seasonality

    def _predict(self, trace, t):
        return self._X_t(t, self.p_, self.n) @ trace["beta"].T

    def plot(self, trace, scaled_t, y_scaler):
        scaled_s = self._predict(trace, scaled_t)
        s = y_scaler.inv_transform(scaled_s)

        ax = add_subplot()
        ax.set_title(str(self))
        ax.set_xticks([])
        ax.plot(scaled_t, s.mean(axis=1), c="lightblue")
        return scaled_s.mean(axis=1)

    def __repr__(self):
        return f"FourierSeasonality(n={self.n})"
