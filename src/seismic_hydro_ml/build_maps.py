"""Predict groundwater table level over the full survey grid for every day,
and check the resulting maps against the PZ3 / PZ5 piezometer time series.

Run with ``shml-build-maps --help`` for the available options.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm import tqdm

from seismic_hydro_ml import config
from seismic_hydro_ml.data_prep import POINTS_PZ3, POINTS_PZ5

logger = logging.getLogger(__name__)


def predict_grid(model, db_vr_wlgt: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Run the model over every (day, x, y) cell that has data, NaN elsewhere."""
    n_days = db_vr_wlgt.shape[0]
    db_piezo = np.full((n_days, len(xs), len(ys)), np.nan)

    for x_i in tqdm(range(len(xs)), colour="blue", leave=False):
        for y_i in range(len(ys)):
            valid_days = ~np.isnan(db_vr_wlgt[:, x_i, y_i, 0])
            for day_i in np.nonzero(valid_days)[0]:
                X = db_vr_wlgt[day_i, x_i, y_i, :].reshape(1, -1)
                db_piezo[day_i, x_i, y_i] = model.predict(X, verbose=0)
    return db_piezo


def check_point(
    db_piezo: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    point: tuple[float, float],
    days: np.ndarray,
    y_true_series: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Compare the gridded prediction at ``point`` against the observed series."""
    x_i, y_i = np.where(xs == point[0])[0][0], np.where(ys == point[1])[0][0]
    y_pred = db_piezo[:, x_i, y_i]
    mask = ~np.isnan(y_pred)

    metrics = {
        "rmse": float(
            root_mean_squared_error(y_true=-y_true_series[mask], y_pred=-y_pred[mask])
        ),
        "r2": float(r2_score(y_true=-y_true_series[mask], y_pred=-y_pred[mask])),
    }
    logger.info("Point %s -> RMSE=%.4f R2=%.4f", point, metrics["rmse"], metrics["r2"])

    fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
    ax.plot(days, y_true_series, c="green")
    ax.plot(days, y_pred, c="orange")
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("GWT level (m)")
    ax.legend(["Real", "Predicted"])
    ax.set_ylim([-4, -1])
    ax.set_title(f"point {point} | R2 {metrics['r2']:.3f} | RMSE {metrics['rmse']:.3f}")
    plt.tight_layout()
    fig.savefig(
        output_dir / f"point{point}_GWTprediction_fromMap.png",
        format="png",
        dpi="figure",
        bbox_inches="tight",
    )
    plt.close(fig)
    return metrics


def build_maps(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    days = load(open(input_dir / "days.sav", "rb"))
    xs = np.load(input_dir / "xs.npy")
    ys = np.load(input_dir / "ys.npy")
    y_PZ3 = np.load(input_dir / "GWT_PZ3(t).npy")
    y_PZ5 = np.load(input_dir / "GWT_PZ5(t).npy")
    db_vr_wlgt = np.load(input_dir / "Vr(t,x,y,wlgt).npy")

    map_path = output_dir / f"{args.model_name}_GWT(t,x,y).npy"

    if args.skip_predict:
        logger.info("Loading existing predictions from %s", map_path)
        db_piezo = np.load(map_path)
    else:
        model = load(open(Path(args.models_dir) / f"{args.model_name}.sav", "rb"))
        logger.info("Predicting GWT level over the full grid for every day")
        db_piezo = predict_grid(model, db_vr_wlgt, xs, ys)
        np.save(map_path, db_piezo)
        logger.info("Saved prediction maps to %s", map_path)

    db_piezo_signed = -db_piezo
    for point in POINTS_PZ3:
        check_point(db_piezo_signed, xs, ys, point, days, y_PZ3, output_dir)
    for point in POINTS_PZ5:
        check_point(db_piezo_signed, xs, ys, point, days, y_PZ5, output_dir)

    return map_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model filename without '.sav'.")
    parser.add_argument("--input-dir", default=str(config.INPUT_DIR))
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Reuse an existing prediction map instead of recomputing it.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    build_maps(args)


if __name__ == "__main__":
    main()
