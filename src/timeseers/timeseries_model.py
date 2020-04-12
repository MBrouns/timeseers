import pandas as pd
import pymc3 as pm
from timeseers.utils import MinMaxScaler, add_subplot
import numpy as np
from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    def fit(self, X, y, **sample_kwargs):
        self._X_scaler_ = MinMaxScaler()
        self._y_scaler_ = MinMaxScaler()

        X_scaled = self._X_scaler_.fit_transform(X)
        y_scaled = self._y_scaler_.fit_transform(y)
        model = pm.Model()

        del X
        mu = self.definition(
            model, X_scaled, self._X_scaler_.max_["t"] - self._X_scaler_.min_["t"]
        )
        with model:
            sigma = pm.HalfCauchy("sigma", 0.5)
            pm.Normal("obs", mu=mu, sd=sigma, observed=y_scaled)
            self.trace_ = pm.sample(**sample_kwargs)

    def plot_components(self, X_true=None, y_true=None, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        n_points = 1000
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        t = pd.date_range(t_min, t_max, periods=n_points)

        scaled_t = np.linspace(0, 1, n_points)
        total = self.plot(self.trace_, scaled_t, self._y_scaler_)

        ax = add_subplot()
        ax.set_title("overall")
        ax.plot(t, self._y_scaler_.inv_transform(total))
        if X_true is not None and y_true is not None:
            ax.scatter(X_true["t"], y_true, c="k")
        fig.tight_layout()
        return fig

    @abstractmethod
    def plot(self, trace, t, y_scaler):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

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
        return self.left.definition(*args, **kwargs) + self.right.definition(
            *args, **kwargs
        )

    def plot(self, *args, **kwargs):
        left = self.left.plot(*args, **kwargs)
        right = self.right.plot(*args, **kwargs)
        return left + right

    def __repr__(self):
        return (
            f"AdditiveTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )


class MultiplicativeTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def definition(self, *args, **kwargs):
        return self.left.definition(*args, **kwargs) * (
            1 + self.right.definition(*args, **kwargs)
        )

    def plot(self, trace, scaled_t, y_scaler):
        left = self.left.plot(trace, scaled_t, y_scaler)
        right = self.right.plot(trace, scaled_t, y_scaler)
        return left + (left * right)

    def __repr__(self):
        return (
            f"MultiplicativeTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )
