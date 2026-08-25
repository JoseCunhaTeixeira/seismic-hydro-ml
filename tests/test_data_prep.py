import datetime as dt

import numpy as np
import pytest

from seismic_hydro_ml.data_prep import daterange, remove_nan_rows, resamp


def test_daterange_is_inclusive_and_daily():
    days = daterange(dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 4))
    assert list(days) == [
        dt.datetime(2023, 1, 1),
        dt.datetime(2023, 1, 2),
        dt.datetime(2023, 1, 3),
        dt.datetime(2023, 1, 4),
    ]


def test_daterange_single_day():
    days = daterange(dt.datetime(2023, 1, 1), dt.datetime(2023, 1, 1))
    assert list(days) == [dt.datetime(2023, 1, 1)]


def test_resamp_recovers_constant_velocity():
    freqs = np.array([5.0, 10.0, 20.0, 40.0])
    velocities = np.full_like(freqs, 300.0)
    wavelengths = np.array([5.0, 10.0, 15.0])

    out_wavelengths, out_velocities = resamp(freqs, velocities, wavelengths)

    assert np.array_equal(out_wavelengths, wavelengths)
    assert np.allclose(out_velocities, 300.0)


def test_remove_nan_rows_drops_any_row_with_a_nan():
    X = np.array(
        [
            [1.0, 2.0],
            [np.nan, 2.0],
            [3.0, np.nan],
            [4.0, 5.0],
        ]
    )
    y = np.array([10.0, 20.0, 30.0, 40.0])

    X_clean, y_clean = remove_nan_rows(X, y)

    assert np.array_equal(X_clean, np.array([[1.0, 2.0], [4.0, 5.0]]))
    assert np.array_equal(y_clean, np.array([10.0, 40.0]))


def test_remove_nan_rows_keeps_x_and_y_aligned():
    X = np.random.default_rng(0).normal(size=(5, 3))
    X[2, 1] = np.nan
    y = np.arange(5, dtype=float)

    X_clean, y_clean = remove_nan_rows(X, y)

    assert len(X_clean) == len(y_clean) == 4
    assert 2.0 not in y_clean


if __name__ == "__main__":
    pytest.main([__file__])
