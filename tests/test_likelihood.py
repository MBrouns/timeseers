import pandas as pd
import numpy as np

from timeseers import Constant
from timeseers import utils

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


def test_can_fit_generated_data(weighted_data):
    df = weighted_data

    model = Constant(lower=-1, upper=1)
    model.fit(df[["t"]], df["value"], sample_weight=df['weight'], y_scaler=utils.IdentityScaler, tune=2000)

    sigma_estimate = model.trace_["sigma"].mean()
    sigma_true = weighted_sigma(df["value"], df["weight"])

    np.testing.assert_allclose(sigma_estimate, sigma_true, atol=.01)


def test_generated_data_parameter_change():
    df_lower_weights = utils.weighted_data(weights=[1, 2])
    # Higher weight on larger sigma data
    df_higher_weights = utils.weighted_data(weights=[1, 5])

    model_lower = Constant(lower=-1, upper=1)
    model_higher = Constant(lower=-1, upper=1)

    model_lower.fit(df_lower_weights[["t"]],
    df_lower_weights["value"], sample_weight=df_lower_weights['weight'], y_scaler=utils.IdentityScaler, tune=2000)
    model_higher.fit(df_higher_weights[["t"]], df_higher_weights["value"], sample_weight=df_higher_weights['weight'], y_scaler=utils.IdentityScaler, tune=2000)

    sigma_lower = model_lower.trace_["sigma"].mean()
    sigma_higher = model_higher.trace_["sigma"].mean()

    assert sigma_lower < sigma_higher
