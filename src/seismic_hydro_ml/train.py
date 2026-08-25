"""Train the MLP on the pre-built dataset and save the fitted model.

Run with ``shml-train --help`` for the available options.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

from seismic_hydro_ml import config
from seismic_hydro_ml.model import MLP

logger = logging.getLogger(__name__)


def load_dataset(input_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = np.load(input_dir / "X_train.npy")
    y_train = np.load(input_dir / "y_train.npy")
    X_validation = np.load(input_dir / "X_validation.npy")
    y_validation = np.load(input_dir / "y_validation.npy")
    return X_train, y_train, X_validation, y_validation


def plot_learning_curve(history, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
    epochs = range(len(history.history["loss"]))
    ax.semilogy(epochs, history.history["loss"], epochs, history.history["val_loss"])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("RMSE")
    ax.legend(["Training dataset", "Validation dataset"])
    fig.savefig(save_path, format="png", dpi="figure", bbox_inches="tight")
    plt.close(fig)


def train(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input_dir)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_validation, y_validation = load_dataset(input_dir)

    model = MLP(hidden_dim=args.hidden_dim)
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_validation, y_validation),
        shuffle=True,
        callbacks=[EarlyStopping(monitor="val_loss", patience=args.patience)],
    )
    model.summary()

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M")
    model_path = models_dir / f"{run_name}_MLP.sav"
    model.save(str(model_path))
    plot_learning_curve(history, models_dir / f"{run_name}_history.png")

    logger.info("Evaluation on training dataset: %s", model.evaluate_metrics(X_train, y_train))
    logger.info(
        "Evaluation on validation dataset: %s", model.evaluate_metrics(X_validation, y_validation)
    )
    logger.info("Saved model to %s", model_path)
    return model_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(config.INPUT_DIR))
    parser.add_argument("--models-dir", default=str(config.MODELS_DIR))
    parser.add_argument("--run-name", default=None, help="Defaults to the current timestamp.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patience", type=int, default=100, help="Early-stopping patience.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
