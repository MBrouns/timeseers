import numpy as np
import matplotlib.pyplot as plt


def dot(a, b):
    return (a * b[None, :]).sum(axis=-1)


class MinMaxScaler():
    def fit(self, df):
        self.max_ = df.max()
        self.min_ = df.min()

    def transform(self, df):
        df = df.copy()
        return (df - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def inv_transform(self, df):
        return df * (self.max_ - self.min_) + self.min_


def generate_data():
    np.random.seed(43)

    n_changepoints = 15
    delta = np.random.laplace(size=n_changepoints) * 0.01

    t = np.arange(1000)
    s = np.sort(np.random.choice(t, n_changepoints, replace=False))

    A = (t[:, None] > s) * 1

    k, m = 0, 0

    growth = (k + A @ delta)
    gamma = -s * delta
    offset = m + A @ gamma
    trend = growth * t + offset

    def X(t, p=365.25, n=10):
        x = 2 * np.pi * (np.arange(n) + 1) * t[:, None] / p
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    n = 10
    t = np.arange(1000)
    beta = np.random.normal(size=2 * n) * 0.1

    seasonailty = X(t, 365.25, n) @ beta

    return trend * seasonailty


def add_subplot():
    fig = plt.gcf()
    n = len(fig.axes)
    for i in range(n):
        fig.axes[i].change_geometry(n + 1, 1, i + 1)
    w, h = fig.get_size_inches()
    fig.set_size_inches(w, h + 8)
    return fig.add_subplot(len(fig.axes) + 1, 1, len(fig.axes) + 1)
