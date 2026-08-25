"""The Multilayer Perceptron used to regress groundwater table level from a
Rayleigh-wave dispersion curve."""

from __future__ import annotations

from pickle import dump
from typing import Any

import numpy as np
from keras import Model
from keras.layers import Dense
from sklearn.metrics import r2_score, root_mean_squared_error


class MLP(Model):
    """A 3-layer feed-forward network: two ReLU hidden layers and a linear output."""

    def __init__(self, hidden_dim: int = 32, output_dim: int = 1) -> None:
        super().__init__()
        self.dense1 = Dense(hidden_dim, activation="relu")
        self.dense2 = Dense(hidden_dim, activation="relu")
        self.dense3 = Dense(output_dim)

    def call(self, x: Any) -> Any:
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            dump(self, f)

    def evaluate_metrics(self, X: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
        """Predict on ``X`` and return RMSE / R2 against ``y_true``."""
        y_pred = self.predict(X, verbose=0)
        return {
            "rmse": float(root_mean_squared_error(y_true=y_true, y_pred=y_pred)),
            "r2": float(r2_score(y_true=y_true, y_pred=y_pred)),
        }
