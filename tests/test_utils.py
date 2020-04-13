import numpy as np

from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.extra.pandas import data_frames, column, series, range_indexes
from timeseers.utils import MinMaxScaler


@given(
    arrays(
        np.float,
        shape=array_shapes(min_dims=2, max_dims=2, min_side=2),
        unique=True,
        elements=st.floats(
            max_value=1e8, min_value=-1e8, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_minmax_scaler_np(array):
    scaler = MinMaxScaler()
    scaler.fit(array)

    assert (scaler.transform(array).min(axis=0) >= 0).all()
    assert (scaler.transform(array).max(axis=0) <= 1).all()
    np.testing.assert_allclose(scaler.fit(array).transform(array), scaler.fit_transform(array))
    np.testing.assert_allclose(array, scaler.inv_transform(scaler.transform(array)))


@given(
    series(
        unique=True,
        elements=st.floats(
            max_value=1e8, min_value=-1e8, allow_nan=False, allow_infinity=False
        ),
        index=range_indexes(min_size=2)
    )
)
def test_minmax_scaler_series(series):
    scaler = MinMaxScaler()
    scaler.fit(series)

    assert scaler.transform(series).min() >= 0
    assert scaler.transform(series).max() <= 1

    np.testing.assert_allclose(scaler.fit(series).transform(series), scaler.fit_transform(series))
    np.testing.assert_allclose(series, scaler.inv_transform(scaler.transform(series)), rtol=1e-06)


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
        ],
        index=range_indexes(min_size=2),
    )
)
def test_minmax_scaler_df(df):
    scaler = MinMaxScaler()
    scaler.fit(df)

    assert (scaler.transform(df).min(axis=0) >= 0).all()
    assert (scaler.transform(df).max(axis=0) <= 1).all()

    np.testing.assert_allclose(scaler.fit(df).transform(df), scaler.fit_transform(df))
    np.testing.assert_allclose(df, scaler.inv_transform(scaler.transform(df)))
