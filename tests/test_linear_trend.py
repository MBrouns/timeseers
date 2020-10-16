from timeseers import LinearTrend
import numpy as np
import pandas as pd


def test_can_fit_generated_data_with_changepoints(trend_data):
    data, true_delta, n_changepoints = trend_data
    model = LinearTrend(n_changepoints=n_changepoints)
    model.fit(data, data["value"])

    y_scale_factor = model._y_scaler_.max_ - model._y_scaler_.min_
    # No groups, so we can take mean over 1 as well
    model_delta = np.mean(model.trace_["delta"], axis=(0, 1)) * y_scale_factor

    np.testing.assert_allclose(model_delta, true_delta, atol=0.01)


def test_can_fit_generated_data_no_changepoints():
    t = np.linspace(0, 1, 1000)

    k, m = 1, 2

    trend = k * t + m

    data = pd.DataFrame(
        {"t": pd.date_range("2018-1-1", periods=len(t)), "value": trend}
    )

    model = LinearTrend(n_changepoints=0)
    model.fit(data, data["value"])

    y_scale_min = model._y_scaler_.min_
    y_scale_factor = model._y_scaler_.max_ - y_scale_min

    model_k = np.mean(model.trace_["k"]) * y_scale_factor
    model_m = np.mean(model.trace_["m"]) + y_scale_min

    np.testing.assert_allclose([model_k, model_m], [k, m], atol=0.01)
