import numpy as np


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


def generate_data():
    np.random.seed(42)

    n_changepoints = 15
    delta = np.random.normal(size=n_changepoints)

    t = np.arange(1000)
    s = np.sort(np.random.choice(t, n_changepoints, replace=False))

    A = (t[:, None] > s) * 1

    k, m = 0, 0

    growth = (k + A @ delta)
    gamma = -s * delta
    offset = m + A @ gamma
    trend = growth * t + offset

    return trend
