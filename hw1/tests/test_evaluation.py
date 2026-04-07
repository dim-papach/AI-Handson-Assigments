import pytest

pytest.importorskip("torch")

import os
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from src.config import PipelineConfig
from src.evaluation import (
    calculate_metrics,
    evaluate_classical_model,
    evaluate_nn_model,
    plot_model_performance,
    plot_nn_evaluation,
    plot_nn_training_history,
)


@pytest.fixture
def sample_binary_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    y_ser = pd.Series(y)
    return X_df, y_ser


@pytest.fixture
def sample_multiclass_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic multiclass classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    y_ser = pd.Series(y)
    return X_df, y_ser


def test_calculate_metrics_binary() -> None:
    """Test metric calculation for binary classification."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    # Predicted probabilities for class 1
    y_prob = np.array([0.1, 0.4, 0.2, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert "Accuracy" in metrics
    assert "Precision" in metrics
    assert "Recall" in metrics
    assert "F1-score" in metrics
    assert "ROC-AUC" in metrics
    assert metrics["Accuracy"] == 0.75
    assert not np.isnan(metrics["ROC-AUC"])


def test_calculate_metrics_multiclass() -> None:
    """Test metric calculation for multiclass classification."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 0, 2])
    # Probabilities for 3 classes
    y_prob = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]
    )

    metrics = calculate_metrics(y_true, y_pred, y_prob)

    assert "Accuracy" in metrics
    assert "ROC-AUC" in metrics
    assert not np.isnan(metrics["ROC-AUC"])


def test_calculate_metrics_no_prob() -> None:
    """Test metric calculation when probabilities are missing."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 1])

    metrics = calculate_metrics(y_true, y_pred, y_prob=None)

    assert "Accuracy" in metrics
    assert np.isnan(metrics["ROC-AUC"])


def test_evaluate_classical_model(sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """Test evaluation wrapper for classical models."""
    X, y = sample_binary_data
    model = LogisticRegression()
    model.fit(X, y)

    metrics, y_pred = evaluate_classical_model(model, X, y)

    assert isinstance(metrics, dict)
    assert len(y_pred) == len(y)
    assert "Accuracy" in metrics


def test_evaluate_nn_model(sample_binary_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """Test evaluation wrapper for neural networks."""
    X, y = sample_binary_data

    # Mock SimpleNN-like model
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    model.output_dim = 1
    # Mock forward pass returning logits
    model.side_effect = lambda x: torch.tensor([[2.0]] * len(x))

    config = PipelineConfig()
    config.nn_output_activation = "Sigmoid"

    metrics, y_pred = evaluate_nn_model(model, X, y, config=config)

    assert "Accuracy" in metrics
    assert len(y_pred) == len(y)
    assert np.all(y_pred == 1)  # logit 2.0 -> sigmoid > 0.5


@patch("matplotlib.pyplot.savefig")
def test_plot_model_performance(mock_savefig: MagicMock, tmp_path: Any) -> None:
    """Test the unified model performance plotter."""
    model_results = {
        "TestModel": {
            "metrics": {"Accuracy": 0.9, "F1-score": 0.85},
            "cm": np.array([[10, 2], [1, 15]]),
        }
    }
    visuals_dir = str(tmp_path / "visuals")

    plot_model_performance(
        model_results,
        visuals_dir=visuals_dir,
        metrics_filename="metrics.png",
        cm_filename="cm.png",
    )

    assert os.path.exists(visuals_dir)
    assert mock_savefig.call_count == 2


@patch("matplotlib.pyplot.savefig")
def test_plot_nn_training_history(mock_savefig: MagicMock, tmp_path: Any) -> None:
    """Test the training history plotter."""
    history = {"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]}
    config = PipelineConfig()
    config.visuals_dir = str(tmp_path / "visuals")

    plot_nn_training_history(history, config=config)

    assert os.path.exists(config.visuals_dir)
    mock_savefig.assert_called_once()


@patch("src.evaluation.plot_model_performance")
def test_plot_nn_evaluation(mock_plot_perf: MagicMock) -> None:
    """Test the NN evaluation plotter wrapper."""
    metrics = {"Accuracy": 0.8}
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    config = PipelineConfig()
    config.visuals_dir = "mock_dir"

    plot_nn_evaluation(metrics, y_true, y_pred, config=config)

    mock_plot_perf.assert_called_once()
    args, kwargs = mock_plot_perf.call_args
    assert "Neural Network" in args[0]
    assert kwargs["visuals_dir"] == "mock_dir"


def test_evaluate_nn_model_softmax(sample_multiclass_data: Tuple[pd.DataFrame, pd.Series]) -> None:
    """Test evaluation wrapper for multiclass NN with Softmax."""
    X, y = sample_multiclass_data
    model = MagicMock(spec=nn.Module)
    model.eval = MagicMock()
    model.output_dim = 3
    # Return 3-class logits
    model.side_effect = lambda x: torch.tensor([[10.0, 0.0, 0.0]] * len(x))

    config = PipelineConfig()
    config.nn_output_activation = "Softmax"

    metrics, y_pred = evaluate_nn_model(model, X, y, config=config)

    assert "Accuracy" in metrics
    assert len(y_pred) == len(y)
    assert np.all(y_pred == 0)


def test_calculate_metrics_invalid_inputs() -> None:
    """Test metric calculation handles mismatched or empty inputs gracefully."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1, 0])  # Mismatched length

    with pytest.raises(ValueError):
        calculate_metrics(y_true, y_pred)
