"""Evaluate a trained model against the PZ3 / PZ5 piezometer time series and
plot predicted vs. observed groundwater table level for each survey point.

Run with ``shml-evaluate --help`` for the available options.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from pickle import load

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error

from seismic_hydro_ml import config
from seismic_hydro_ml.data_prep import POINTS_PZ3, POINTS_PZ5

logger = logging.getLogger(__name__)


def evaluate_point(
    model,
    db_vr_wlgt: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    point: tuple[float, float],
    days: np.ndarray,
    y_true_series: np.ndarray,
    output_dir: Path,
    suffix: str,
) -> dict[str, float]:
    """Predict GWT level at a single (x, y) point and save a real-vs-predicted plot."""
    X = db_vr_wlgt[:, np.where(xs == point[0])[0][0], np.where(ys == point[1])[0][0], :]
    mask = ~np.isnan(X[:, 0])
    X = X[mask]

    y_pred = model.predict(X, verbose=0).flatten()
    y_true = -y_true_series[mask]
    metrics = {
        "rmse": float(root_mean_squared_error(y_true=y_true, y_pred=y_pred)),
        "r2": float(r2_score(y_true=y_true, y_pred=y_pred)),
    }
    logger.info("Point %s -> RMSE=%.4f R2=%.4f", point, metrics["rmse"], metrics["r2"])

    fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
    ax.plot(days, y_true_series, c="green")
    ax.plot(days[mask], -y_pred, c="orange")
    ax.set_xlabel("Time (month)")
    ax.set_ylabel("GWT level (m)")
    ax.legend(["Real", "Predicted"])
    ax.set_ylim([-4, -1])
    ax.set_title(f"point {point} | R2 {metrics['r2']:.3f} | RMSE {metrics['rmse']:.3f}")
    plt.tight_layout()
    fig.savefig(
        output_dir / f"point{point}_GWTprediction{suffix}.png",
        format="png",
        dpi="figure",
        bbox_inches="tight",
    )
    plt.close(fig)
    return metrics


def evaluate(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    days = load(open(input_dir / "days.sav", "rb"))
    xs = np.load(input_dir / "xs.npy")
    ys = np.load(input_dir / "ys.npy")
    y_PZ3 = np.load(input_dir / "GWT_PZ3(t).npy")
    y_PZ5 = np.load(input_dir / "GWT_PZ5(t).npy")
    db_vr_wlgt = np.load(input_dir / "Vr(t,x,y,wlgt).npy")

    model_path = Path(args.models_dir) / f"{args.model_name}.sav"
    model = load(open(model_path, "rb"))

    for point in POINTS_PZ3:
        evaluate_point(model, db_vr_wlgt, xs, ys, point, days, y_PZ3, output_dir, "")
    for point in POINTS_PZ5:
        evaluate_point(model, db_vr_wlgt, xs, ys, point, days, y_PZ5, output_dir, "")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True, help="Model filename without '.sav'.")
    parser.add_argument("--input-dir", default=str(config.INPUT_DIR))
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument("--output-dir", default=str(config.OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
