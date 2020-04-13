from timeseers import FourierSeasonality
import numpy as np
from timeseers.utils import IdentityScaler


def test_can_fit_generated_data(seasonal_data):
    data, true_beta, n_components = seasonal_data
    model = FourierSeasonality(n=n_components)
    model.fit(data, data["value"], y_scaler=IdentityScaler, tune=1000, draws=10000)

    model_beta = np.mean(model.trace_["beta"], axis=0)

    np.testing.assert_allclose(model_beta, true_beta, atol=0.12)
