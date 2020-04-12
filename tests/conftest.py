import numpy as np
import pytest
from timeseers import utils


@pytest.fixture(params=[1, 5, 10])
def trend_data(request):
    np.random.seed(42)
    n_changepoints = request.param
    data, delta = utils.trend_data(n_changepoints, noise=0.0001)
    return data, delta, n_changepoints


@pytest.fixture(params=[1, 5, 10])
def seasonal_data(request):
    np.random.seed(42)
    n_components = request.param
    data, beta = utils.seasonal_data(n_components)
    return data, beta, n_components
