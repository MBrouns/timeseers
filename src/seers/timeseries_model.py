import pymc3 as pm
from seers.utils import MinMaxScaler, add_subplot
import numpy as np

import matplotlib.pyplot as plt


class TimeSeriesModel:
    def fit(self, X, y, **sample_kwargs):
        self._X_scaler_ = MinMaxScaler()
        self._y_scaler_ = MinMaxScaler()

        X_scaled = self._X_scaler.fit_transform(X)
        y_scaled = self._y_scaler.fit_transform(y)
        model = pm.Model()

        del X
        mu = self.definition(model, X_scaled, self._X_scaler.max_['t'] - self._X_scaler.min_['t'])
        with model:
            sigma = pm.HalfCauchy('sigma', 0.5)
            pm.Normal(
                'obs',
                mu=mu,
                sd=sigma,
                observed=y_scaled
            )
            self.trace_ = pm.sample(**sample_kwargs)

    def plot_components(self, fig=None):
        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        n_points = 1000
        t = np.linspace(self._X_scaler.min_['t'], self._X_scaler.max_['t'], n_points)

        scaled_t = np.linspace(0, 1, n_points)
        total = self.plot(self.trace_, scaled_t, self._y_scaler)
        ax = add_subplot()
        ax.plot(t, self._y_scaler.inv_transform(total))
        fig.tight_layout()
        return fig

    def plot(self, trace, t, y_scaler):
        raise NotImplemented

    def definition(self, model, X_scaled, scale_factor):
        raise NotImplemented

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) + self.right.definition(*args, **kwargs)

    def plot(self, *args, **kwargs):
        l = self.left.plot(*args, **kwargs)
        r = self.right.plot(*args, **kwargs)
        return l + r

    def __repr__(self):
        return f"AdditiveTimeSeries( \n" \
               f"    left={self.left} \n" \
               f"    right={self.right} \n" \
               f")"


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (1 + self.right.definition(*args, **kwargs))

    def plot(self, trace, t, y_scaler):
        l = self.left.plot(trace, t, y_scaler)
        r = self.right.plot(trace, t, y_scaler)
        return l + (l * r)

    def __repr__(self):
        return f"MultiplicativeTimeSeries( \n" \
               f"    left={self.left} \n" \
               f"    right={self.right} \n" \
               f")"
