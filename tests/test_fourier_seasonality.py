from timeseers import FourierSeasonality
import numpy as np


def test_can_fit_generated_data(seasonal_data):
    data, true_beta, n_components = seasonal_data
    model = FourierSeasonality(n=n_components)
    model.fit(data, data["value"])

    y_scale_factor = model._y_scaler_.max_ - model._y_scaler_.min_
    model_beta = np.mean(model.trace_["beta"], axis=0) * y_scale_factor

    np.testing.assert_allclose(model_beta, true_beta, atol=0.01)
