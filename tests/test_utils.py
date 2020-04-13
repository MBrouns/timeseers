import numpy as np

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import data_frames, column
from timeseers.utils import MinMaxScaler


@given(
    data_frames(
        [
            column(
                np.float,
                unique=True,
                elements=st.floats(
                    max_value=1e8, min_value=-1e8, allow_nan=False, allow_infinity=False
                ),
            )
        ]
    )
)
def test_minmax_scaler(array):
    scaler = MinMaxScaler()
    scaler.fit(array)

    assert all(scaler.transform(array).min() >= 0)
    assert all(scaler.transform(array).max() <= 1)

    np.testing.assert_allclose(array, scaler.inv_transform(scaler.transform(array)))
