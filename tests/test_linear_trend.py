from timeseers import LinearTrend
import numpy as np


def test_can_fit_generated_data(trend_data):
    data, true_delta, n_changepoints = trend_data
    model = LinearTrend(n_changepoints=n_changepoints)
    model.fit(data, data["value"])

    y_scale_factor = model._y_scaler_.std_
    model_delta = np.mean(model.trace_[model._param_name("delta")], axis=(0, 1)) * y_scale_factor

    np.testing.assert_allclose(model_delta, true_delta, atol=0.01)
