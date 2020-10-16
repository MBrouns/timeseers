import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def dot(a, b):
    return (a * b[None, :]).sum(axis=-1)


class IdentityScaler:
    def fit(self, data):
        self.scale_factor_ = 1
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return data


class MinMaxScaler:
    def fit(self, data):
        if isinstance(data, pd.DataFrame):
            self.max_ = data.max(axis=0)
            self.min_ = data.min(axis=0)
            self.scale_factor_ = (self.max_ - self.min_).where(
                self.max_ != self.min_, 1
            )
        if isinstance(data, np.ndarray):
            self.max_ = data.max(axis=0)[None, ...]
            self.min_ = data.min(axis=0)[None, ...]
            self.scale_factor_ = np.where(
                self.max_ != self.min_, self.max_ - self.min_, 1
            )
        if isinstance(data, pd.Series):
            self.max_ = data.max()
            self.min_ = data.min()
            self.scale_factor_ = self.max_ - self.min_

        return self

    def transform(self, series):
        return (series - self.min_) / self.scale_factor_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.scale_factor_ + self.min_


class StdScaler:
    def fit(self, data):
        if isinstance(data, pd.Series):
            self.mean_ = data.mean()
            self.std_ = data.std()

        return self

    def transform(self, series):
        return (series - self.mean_) / self.std_

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inv_transform(self, series):
        return series * self.std_ + self.mean_


def add_subplot(height=5):
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + height)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)


def trend_data(n_changepoints, location="spaced", noise=0.001):
    delta = np.random.laplace(size=n_changepoints)

    t = np.linspace(0, 1, 1000)

    if location == "random":
        s = np.sort(np.random.choice(t, n_changepoints, replace=False))
    elif location == "spaced":
        s = np.linspace(0, np.max(t), n_changepoints + 2)[1:-1]
    else:
        raise ValueError('invalid `location`, should be "random" or "spaced"')

    A = (t[:, None] > s) * 1

    k, m = 0, 0

    growth = k + A @ delta
    gamma = -s * delta
    offset = m + A @ gamma
    trend = growth * t + offset + np.random.randn(len(t)) * noise

    return (
        pd.DataFrame({"t": pd.date_range("2018-1-1", periods=len(t)), "value": trend}),
        delta,
    )


def seasonal_data(n_components, noise=0.001):
    def X(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    t = np.linspace(0, 1, 1000)
    beta = np.random.normal(size=2 * n_components)

    seasonality = (
        X(t, 365.25 / len(t), n_components) @ beta + np.random.randn(len(t)) * noise
    )

    return (
        pd.DataFrame(
            {"t": pd.date_range("2018-1-1", periods=len(t)), "value": seasonality}
        ),
        beta,
    )


def get_group_definition(X, pool_cols, pool_type):
    if pool_type == "complete":
        group = np.zeros(len(X), dtype="int")
        group_mapping = {0: "all"}
        n_groups = 1
    else:
        group = X[pool_cols].cat.codes.values
        group_mapping = dict(enumerate(X[pool_cols].cat.categories))
        n_groups = X[pool_cols].nunique()
    return group, n_groups, group_mapping
