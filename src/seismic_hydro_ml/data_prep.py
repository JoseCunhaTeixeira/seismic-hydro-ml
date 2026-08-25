"""Build the training/validation dataset from raw Rayleigh-wave phase-velocity
cubes and piezometer time series.

This step consumes internal SNCF Reseau raw data (per-frequency Rayleigh-wave
velocity cubes produced by the ``ndimcube`` package, and piezometer CSV time
series) that is **not** distributed with this repository. It is included so
the full data lineage is documented, but it cannot be run without access to
that raw data and to ``ndimcube``. If you only want to train or evaluate the
model, use the pre-built arrays already committed under ``input/``.

Run with ``shml-prepare-data --help`` for the available options.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
import sys
import warnings
from pathlib import Path
from pickle import dump

import numpy as np
from scipy.interpolate import interp1d

from seismic_hydro_ml import config

logger = logging.getLogger(__name__)

# Instrument frequencies (Hz) at which Rayleigh-wave velocity was picked.
FREQUENCIES = [
    5.00, 5.21, 5.44, 5.67, 5.91, 6.16, 6.43, 6.70, 6.99, 7.29, 7.60, 7.92, 8.26, 8.62,
    8.98, 9.37, 9.77, 10.19, 10.62, 11.08, 11.55, 12.04, 12.56, 13.10, 13.66, 14.24, 14.85,
    15.48, 16.15, 16.84, 17.56, 18.31, 19.09, 19.91, 20.76, 21.64, 22.57, 23.53, 24.54, 25.59,
    26.68, 27.82, 29.01, 30.25, 31.55, 32.90, 34.30, 35.77, 37.30, 38.89, 40.56, 42.29, 44.10,
    45.98, 47.95, 50.00,
]

# Geophone-array points (local x, y in metres) closest to piezometers PZ3 and PZ5.
POINTS_PZ3 = [
    (63.00, 4.75), (66.00, 4.75), (69.00, 4.75),
    (63.00, 0.00), (66.00, 0.00), (69.00, 0.00),
]
POINTS_PZ5 = [
    (93.00, 19.00), (96.00, 19.00), (99.00, 19.00),
    (93.00, 14.25), (96.00, 14.25), (99.00, 14.25),
]


def resamp(
    freqs: np.ndarray, velocities: np.ndarray, wavelengths: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a Rayleigh-wave velocity curve from frequency to wavelength."""
    freqs = np.asarray(freqs, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    source_wavelengths = velocities / freqs
    interpolator = interp1d(source_wavelengths, velocities, fill_value="extrapolate")
    return wavelengths, interpolator(wavelengths)


def daterange(date_start: dt.datetime, date_end: dt.datetime) -> np.ndarray:
    """Return an array of daily timestamps from ``date_start`` to ``date_end`` inclusive."""
    days = []
    date = date_start
    step = dt.timedelta(days=1)
    while date <= date_end:
        days.append(date)
        date += step
    return np.array(days)


def _load_ndimcube(ndimcube_path: str | None):
    """Import ``ndimcube.NDimCube`` from an internal, non-public package."""
    if ndimcube_path:
        sys.path.append(ndimcube_path)
    try:
        from ndimcube.ndimcube import NDimCube
    except ImportError as exc:
        raise ImportError(
            "Could not import 'ndimcube'. This is an internal SNCF Reseau package "
            "not published with seismic-hydro-ml. Provide its location via "
            "--ndimcube-path or the SHML_NDIMCUBE_PATH environment variable."
        ) from exc
    return NDimCube


def build_vr_wavelength_database(
    raw_vr_dir: str,
    days: np.ndarray,
    wavelengths: np.ndarray,
    ndimcube_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw per-frequency Rayleigh-wave velocity cubes and resample them
    onto ``wavelengths`` for every day in ``days``.

    Returns ``(xs, ys, db_vr_wlgt)`` where ``db_vr_wlgt`` has shape
    ``(len(days), len(xs), len(ys), len(wavelengths))``.
    """
    NDimCube = _load_ndimcube(ndimcube_path)

    files = sorted(Path(raw_vr_dir).iterdir())
    first_cube = NDimCube.load(str(files[0]))
    dims = first_cube.get_dimensions_scale()
    xs = np.array(list(dims[0].values())[0])
    ys = np.array(list(dims[1].values())[0])
    fs = np.array(list(dims[2].values())[0])

    logger.info("Building Vr(t, x, y, f) database from %d raw cube files", len(files))
    db_vr_freq = np.full((len(days), len(xs), len(ys), len(FREQUENCIES)), np.nan)
    for file in files:
        cube = NDimCube.load(str(file))
        if cube.time_stamp not in days:
            continue
        day_i = np.where(days == cube.time_stamp)[0][0]
        for x_i in range(len(xs)):
            for y_i in range(len(ys)):
                for f_i in range(len(fs)):
                    db_vr_freq[day_i, x_i, y_i, f_i] = cube.data[x_i, y_i, f_i]

    logger.info("Resampling Vr(t, x, y, f) onto wavelengths")
    db_vr_wlgt = np.full((len(days), len(xs), len(ys), len(wavelengths)), np.nan)
    for day_i in range(len(days)):
        for x_i in range(len(xs)):
            for y_i in range(len(ys)):
                _, db_vr_wlgt[day_i, x_i, y_i, :] = resamp(
                    FREQUENCIES, db_vr_freq[day_i, x_i, y_i, :], wavelengths
                )
    db_vr_wlgt /= 2000  # normalize to a comparable scale for the network input

    return xs, ys, db_vr_wlgt


def stack_points(
    db_vr_wlgt: np.ndarray, xs: np.ndarray, ys: np.ndarray, points: list[tuple[float, float]]
) -> np.ndarray:
    """Stack the wavelength-domain series of several (x, y) points into one array."""
    series = [
        db_vr_wlgt[:, np.where(xs == x)[0][0], np.where(ys == y)[0][0], :] for x, y in points
    ]
    return np.vstack(series)


def load_piezo_series(
    csv_path: str, days: np.ndarray, date_start: dt.datetime, date_end: dt.datetime
) -> np.ndarray:
    """Read a daily piezometer level time series aligned to ``days`` (NaN where missing)."""
    y = np.full(len(days), np.nan)
    with open(csv_path, newline="") as csvfile:
        for row in csv.reader(csvfile, delimiter=",", quotechar="|"):
            if row[0] == "" or row[1] == "":
                continue
            date = dt.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
            if date_start <= date <= date_end and date.time() == dt.time(0, 0, 0):
                y[np.where(days == date)[0][0]] = float(row[1])
    return y


def remove_nan_rows(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop rows where any wavelength value is NaN, keeping ``X`` and ``y`` aligned."""
    mask = ~np.isnan(X).any(axis=1)
    return X[mask], y[mask]


def build_arrays(
    args: argparse.Namespace,
) -> None:
    warnings.filterwarnings("ignore")

    date_start = dt.datetime.strptime(args.date_start, "%Y-%m-%d")
    date_end = dt.datetime.strptime(args.date_end, "%Y-%m-%d")
    days = daterange(date_start, date_end)
    logger.info("From %s to %s -> %d days", date_start.date(), date_end.date(), len(days))

    wavelengths = np.arange(
        args.wavelength_min, args.wavelength_max + args.wavelength_step, args.wavelength_step
    )

    xs, ys, db_vr_wlgt = build_vr_wavelength_database(
        args.raw_vr_dir, days, wavelengths, args.ndimcube_path
    )

    X_PZ3 = stack_points(db_vr_wlgt, xs, ys, POINTS_PZ3)
    X_PZ5 = stack_points(db_vr_wlgt, xs, ys, POINTS_PZ5)

    raw_piezo_dir = Path(args.raw_piezo_dir)
    y_PZ3_daily = load_piezo_series(
        str(raw_piezo_dir / "PZ3_interp300s.csv"), days, date_start, date_end
    )
    y_PZ5_daily = load_piezo_series(
        str(raw_piezo_dir / "PZ5_interp300s.csv"), days, date_start, date_end
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "GWT_PZ3(t).npy", y_PZ3_daily)
    np.save(output_dir / "GWT_PZ5(t).npy", y_PZ5_daily)

    y_PZ3 = np.tile(np.abs(y_PZ3_daily), len(POINTS_PZ3))
    y_PZ5 = np.tile(np.abs(y_PZ5_daily), len(POINTS_PZ5))

    logger.info("Before removing NaNs: X_PZ3=%s X_PZ5=%s", X_PZ3.shape, X_PZ5.shape)
    X_PZ3, y_PZ3 = remove_nan_rows(X_PZ3, y_PZ3)
    X_PZ5, y_PZ5 = remove_nan_rows(X_PZ5, y_PZ5)
    logger.info("After removing NaNs: X_PZ3=%s X_PZ5=%s", X_PZ3.shape, X_PZ5.shape)

    dump(days, open(output_dir / "days.sav", "wb"))
    np.save(output_dir / "xs.npy", xs)
    np.save(output_dir / "ys.npy", ys)
    np.save(output_dir / "wavelengths.npy", wavelengths)

    np.save(output_dir / "X_train.npy", X_PZ3)
    np.save(output_dir / "y_train.npy", y_PZ3)
    np.save(output_dir / "X_validation.npy", X_PZ5)
    np.save(output_dir / "y_validation.npy", y_PZ5)

    np.save(output_dir / "Vr(t,x,y,wlgt).npy", db_vr_wlgt)
    logger.info("Saved dataset arrays to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-vr-dir", default=config.RAW_VR_DIR,
        help="Directory of raw Rayleigh-wave velocity cube files (env: SHML_RAW_VR_DIR).",
    )
    parser.add_argument(
        "--raw-piezo-dir", default=config.RAW_PIEZO_DIR,
        help="Directory containing PZ3/PZ5 piezometer CSV files (env: SHML_RAW_PIEZO_DIR).",
    )
    parser.add_argument(
        "--ndimcube-path", default=None,
        help="Directory to add to sys.path to import the internal 'ndimcube' package "
        "(env: SHML_NDIMCUBE_PATH).",
    )
    parser.add_argument("--output-dir", default=str(config.INPUT_DIR))
    parser.add_argument("--date-start", default="2022-12-30")
    parser.add_argument("--date-end", default="2023-09-03")
    parser.add_argument("--wavelength-min", type=float, default=4.0)
    parser.add_argument("--wavelength-max", type=float, default=15.0)
    parser.add_argument("--wavelength-step", type=float, default=0.5)
    args = parser.parse_args(argv)

    if args.ndimcube_path is None:
        import os

        args.ndimcube_path = os.environ.get("SHML_NDIMCUBE_PATH")

    if not args.raw_vr_dir or not args.raw_piezo_dir:
        parser.error(
            "--raw-vr-dir and --raw-piezo-dir are required (or set SHML_RAW_VR_DIR / "
            "SHML_RAW_PIEZO_DIR). This raw data is internal to SNCF Reseau and is not "
            "distributed with this repository."
        )
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    build_arrays(args)


if __name__ == "__main__":
    main()
