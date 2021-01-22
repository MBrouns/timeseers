from timeseers import LogisticGrowth
from timeseers.utils import MaxScaler
import numpy as np


def test_can_fit_generated_data(logistic_growth_data):
    data, true_delta, n_changepoints = logistic_growth_data
    model = LogisticGrowth(capacity=1, n_changepoints=n_changepoints)
    model.fit(data, data["value"], cores=1, chains=1, y_scaler=MaxScaler)
    model_delta = model.trace_[model._param_name("delta")].mean(axis=0)[0]
    np.testing.assert_allclose(model_delta, true_delta, atol=0.01)
