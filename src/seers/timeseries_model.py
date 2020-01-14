import operator
from functools import reduce
import pymc3 as pm
from seers.utils import MinMaxScaler
import numpy as np


class TimeSeriesModel:

    def __init__(self):
        self._model = pm.Model()
        self._components = []
        self._X_scaler = MinMaxScaler()
        self._y_scaler = MinMaxScaler()

    def __add__(self, other):
        # TODO return copy
        self._components.append(other)
        return self

    def fit(self, X, y):
        X_scaled = self._X_scaler.fit_transform(X)
        y_scaled = self._y_scaler.fit_transform(y)

        del X
        with self._model:
            sigma = pm.HalfCauchy('sigma', 0.5)

            mu = reduce(operator.add, [c.definition(self._model, X_scaled) for c in self._components])
            pm.Normal(
                'obs',
                mu=mu,
                sd=sigma,
                observed=y_scaled
            )

            self.trace_ = pm.sample(tune=100, draws=100)

    def plot_components(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=len(self._components) + 1, figsize=(25, 8 * len(self._components) + 1))

        n_points = 100
        t = np.linspace(self._X_scaler.min_['t'], self._X_scaler.max_['t'], n_points)
        scaled_t = np.linspace(0, 1, n_points)
        sub_trace = self.trace_[0:100:5]
        overall = np.zeros((n_points, len(sub_trace) * len(sub_trace.chains)))

        for idx, component in enumerate(self._components, start=1):
            trend = component.plot(axes[idx], sub_trace, scaled_t)
            overall += trend

        axes[0].set_title('Overall model')
        axes[0].plot(t, overall)
        fig.tight_layout()
