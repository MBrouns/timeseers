import pandas as pd
import pymc3 as pm
from timeseers.utils import MinMaxScaler, StdScaler, add_subplot
import numpy as np
from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    def fit(self, X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, **sample_kwargs):
        if not X.index.is_monotonic_increasing:
            raise ValueError('index of X is not monotonically increasing. You might want to call `.reset_index()`')

        X_to_scale = X.select_dtypes(exclude='category')
        self._X_scaler_ = X_scaler()
        self._y_scaler_ = y_scaler()

        X_scaled = self._X_scaler_.fit_transform(X_to_scale)
        y_scaled = self._y_scaler_.fit_transform(y)
        model = pm.Model()
        X_scaled = X_scaled.join(X.select_dtypes('category'))
        del X
        mu = self.definition(
            model, X_scaled, self._X_scaler_.scale_factor_
        )
        with model:
            sigma = pm.HalfCauchy("sigma", 0.5)
            pm.Normal("obs", mu=mu, sd=sigma, observed=y_scaled)
            self.trace_ = pm.sample(**sample_kwargs)

    def plot_components(self, X_true=None, y_true=None, groups=None, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        lookahead_scale = 0.3
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        t_max += (t_max - t_min) * lookahead_scale
        t = pd.date_range(t_min, t_max, freq='D')

        scaled_t = np.linspace(0, 1 + lookahead_scale, len(t))
        total = self.plot(self.trace_, scaled_t, self._y_scaler_)

        ax = add_subplot()
        ax.set_title("overall")
        ax.plot(t, self._y_scaler_.inv_transform(total))

        if X_true is not None and y_true is not None:
            if groups is not None:
                for group in groups.cat.categories:
                    mask = groups == group
                    ax.scatter(X_true["t"][mask], y_true[mask], label=group, marker='.', alpha=0.2)
            else:
                ax.scatter(X_true["t"], y_true, c="k", marker='.', alpha=0.2)
        fig.tight_layout()
        return fig

    @abstractmethod
    def plot(self, trace, t, y_scaler):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

    def _param_name(self, param):
        return f"{self.name}-{param}"

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
