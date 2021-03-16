import pandas as pd
import pymc3 as pm
from timeseers.utils import MinMaxScaler, StdScaler, add_subplot, cartesian_product
from timeseers.likelihood import Gaussian
import numpy as np
from abc import ABC, abstractmethod


class TimeSeriesModel(ABC):
    def fit(self, X, y, X_scaler=MinMaxScaler, y_scaler=StdScaler, likelihood=None, **sample_kwargs):
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
        if likelihood is None:
            likelihood = Gaussian()
        with model:
            likelihood.observed(mu, y_scaled)
            self.trace_ = pm.sample(**sample_kwargs)

    def predict(self, X, ci_percentiles=(0.08, 0.5, 0.92)):
        X_to_scale = X.select_dtypes(exclude='category')
        X_scaled = self._X_scaler_.transform(X_to_scale)
        X_scaled = X_scaled.join(X.select_dtypes('category'))
        y_hat_scaled = self._predict(self.trace_, X_scaled)

        # TODO: We only take the uncertainty of the parameters here, still need to add the uncertainty
        # from the likelihood as well

        y_hat = self._y_scaler_.inv_transform(y_hat_scaled)
        return pd.DataFrame(
            np.percentile(
                y_hat,
                ci_percentiles,
                axis=1
            ).T,
            columns=[f"perc_{p}" for p in ci_percentiles]
        )

    def plot_components(self, X_true=None, y_true=None, groups=None, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure(figsize=(18, 1))

        lookahead_scale = 0.3
        t_min, t_max = self._X_scaler_.min_["t"], self._X_scaler_.max_["t"]
        t_max += (t_max - t_min) * lookahead_scale
        t = pd.date_range(t_min, t_max, freq='D')

        scaled_t = np.linspace(0, 1 + lookahead_scale, len(t))
        X = pd.DataFrame({'t': scaled_t})

        for col, val in self._groups().items():
            X = cartesian_product(X, pd.DataFrame({col: list(val.values())}))

        total = self.plot(self.trace_, X, self._y_scaler_)

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
    def _predict(self, trace, X):
        pass

    @abstractmethod
    def definition(self, model, X_scaled, scale_factor):
        pass

    def _param_name(self, param):
        return f"{self.name}-{param}"

    def _groups(self):
        if self.pool_type == "complete":
            return {}
        return {self.pool_cols: self.groups_}

    def __add__(self, other):
        return AdditiveTimeSeries(self, other)

    def __mul__(self, other):
        return MultiplicativeTimeSeries(self, other)

    def __str__(self):
        return self.name


class AdditiveTimeSeries(TimeSeriesModel):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.name = "AdditiveTimeSeries"
        super().__init__()

    def definition(self, *args, **kwargs):
        return (
            self.left.definition(*args, **kwargs) +
            self.right.definition(*args, **kwargs)
        )

    def _predict(self, trace, x_scaled):
        return (
            self.left._predict(trace, x_scaled) +
            self.right._predict(trace, x_scaled)
        )

    def plot(self, trace, X, y_scaler):
        return (
                self.left.plot(trace, X, y_scaler) +
                self.right.plot(trace, X, y_scaler)
        )

    def _groups(self):
        return {**self.left._groups(), ** self.right._groups()}

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
        self.name = "MultiplicativeTimeSeries"
        super().__init__()

    def definition(self, *args, **kwargs):
        return (
            self.left.definition(*args, **kwargs) *
            (1 + self.right.definition(*args, **kwargs))
        )

    def _predict(self, trace, x_scaled):
        return (
            self.left._predict(trace, x_scaled) *
            (1 + self.right._predict(trace, x_scaled))
        )

    def plot(self, trace, X, y_scaler):
        return (
            self.left.plot(trace, X, y_scaler) *
            (1 + self.right.plot(trace, X, y_scaler))
        )

    def _groups(self):
        return {**self.left._groups(), ** self.right._groups()}

    def __repr__(self):
        return (
            f"MultiplicativeTimeSeries( \n"
            f"    left={self.left} \n"
            f"    right={self.right} \n"
            f")"
        )
