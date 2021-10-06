import numpy as np
import pandas as pd

from timeseers import RBFSeasonality
from timeseers.utils import IdentityScaler, get_periodic_peaks


def test_can_fit_generated_data(rbf_seasonal_data):
    data, true_beta, n_components = rbf_seasonal_data
    ps = get_periodic_peaks(n_components)
    model = RBFSeasonality(peaks=ps, period=pd.Timedelta(days=365.25), sigma=0.015)
    model.fit(data[['t']], data['value'], tune=1000, draws=10000, y_scaler=IdentityScaler)
    model_beta = np.mean(model.trace_[model._param_name("beta")], axis=(0, 1))
    np.testing.assert_allclose(model_beta, true_beta, atol=0.12)
