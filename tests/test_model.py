import numpy as np
import pytest

keras = pytest.importorskip("keras")

from seismic_hydro_ml.model import MLP  # noqa: E402


def test_mlp_forward_pass_output_shape():
    model = MLP(hidden_dim=8, output_dim=1)
    X = np.random.default_rng(0).normal(size=(4, 6)).astype("float32")

    y = model(X)

    assert y.shape == (4, 1)


def test_mlp_evaluate_metrics_perfect_prediction_is_zero_rmse():
    model = MLP(hidden_dim=8, output_dim=1)
    model.compile(optimizer="adam", loss="mse")

    X = np.random.default_rng(0).normal(size=(4, 6)).astype("float32")
    y_true = model.predict(X, verbose=0)

    metrics = model.evaluate_metrics(X, y_true)

    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["r2"] == pytest.approx(1.0, abs=1e-4)
