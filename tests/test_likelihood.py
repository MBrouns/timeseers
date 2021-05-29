import pandas as pd
import numpy as np

from timeseers import Constant
from timeseers.utils import IdentityScaler

def weighted_sigma(y, sample_weight):
    n_observations = len(y)
    m_w = weighted_mean(y, sample_weight)

    num = (y - m_w) ** 2
    num *= sample_weight
    num = num.sum()

    den = (n_observations - 1) * sample_weight.sum() / n_observations

    return np.sqrt(num / den)


def weighted_mean(y, sample_weight):
    return (y * sample_weight).sum() / sample_weight.sum()


def test_weights(weighted_data):
    df = weighted_data

    model = Constant(name="intercept", lower=-1, upper=1)
    model.fit(df[["t"]], df["value"], sample_weight=df['weight'], y_scaler=IdentityScaler, tune=2000)

    sigma_estimate = model.trace_["sigma"].mean()
    sigma_true = weighted_sigma(df["value"], df["weight"])

    np.testing.assert_allclose(sigma_estimate, sigma_true, atol=.01)
