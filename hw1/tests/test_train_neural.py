import os
import shutil
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn
from src.config import PipelineConfig
from src.train_neural import (
    SimpleNN,
    EarlyStopping,
    prepare_tensors,
    train_neural_network,
)
from src.evaluation import (
    evaluate_nn_model,
    plot_nn_training_history,
)

@pytest.fixture
def mock_data():
    """Generates small mock data for testing."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    return X, y

@pytest.fixture
def temp_config(tmp_path):
    """Creates a temporary config for testing purposes."""
    models_dir = tmp_path / "models"
    visuals_dir = tmp_path / "visuals"
    models_dir.mkdir()
    visuals_dir.mkdir()
    
    config = PipelineConfig()
    config.models_dir = str(models_dir)
    config.visuals_dir = str(visuals_dir)
    config.nn_epochs = 2
    config.nn_hidden_layers = [10, 5]
    config.nn_patience = 5
    return config

def test_simple_nn_init():
    """Tests if SimpleNN initializes with correct dimensions and activation."""
    input_dim = 10
    hidden_layers = [20, 10]
    output_dim = 1
    dropout = 0.2
    
    model = SimpleNN(input_dim, hidden_layers, output_dim, dropout, activation="ReLU")
    
    assert isinstance(model.network[0], nn.Linear)
    assert model.network[0].in_features == input_dim
    assert model.network[-1].out_features == output_dim

def test_prepare_tensors(mock_data):
    """Tests if prepare_tensors converts data correctly."""
    X, y = mock_data
    X_t, y_t = prepare_tensors(X, y)
    
    assert isinstance(X_t, torch.Tensor)
    assert isinstance(y_t, torch.Tensor)
    assert X_t.shape == (100, 5)
    assert y_t.shape == (100,)

def test_early_stopping_logic(temp_config):
    """Tests if EarlyStopping triggers correctly."""
    model = nn.Linear(5, 1)
    path = os.path.join(temp_config.models_dir, "test_check.pt")
    es = EarlyStopping(patience=2, path=path)
    
    # Improved
    es(1.0, model)
    assert es.best_score == -1.0
    assert os.path.exists(path)
    
    # Not improved
    es(1.1, model)
    assert es.counter == 1
    assert not es.early_stop
    
    # Still not improved
    es(1.2, model)
    assert es.counter == 2
    assert es.early_stop

def test_train_neural_network(mock_data, temp_config):
    """Tests the full training loop with mock data."""
    X, y = mock_data
    # Split into train and val
    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]
    
    model, history = train_neural_network(X_train, y_train, X_val, y_val, config=temp_config)
    
    assert isinstance(model, nn.Module)
    assert "train_loss" in history
    assert len(history["train_loss"]) == temp_config.nn_epochs
    assert os.path.exists(os.path.join(temp_config.models_dir, temp_config.nn_model_filename))

def test_evaluate_nn_model(mock_data, temp_config):
    """Tests NN evaluation function."""
    X, y = mock_data
    input_dim = X.shape[1]
    model = SimpleNN(input_dim, [10], 1, 0.2)
    
    metrics, y_pred = evaluate_nn_model(model, X, y, config=temp_config)
    
    assert "Accuracy" in metrics
    assert "ROC-AUC" in metrics
    assert len(y_pred) == len(y)

def test_plotting_functions(temp_config):
    """Tests if plotting functions generate files as expected."""
    history = {"train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]}
    plot_nn_training_history(history, config=temp_config)
    assert os.path.exists(os.path.join(temp_config.visuals_dir, temp_config.nn_loss_plot_filename))
    
    from src.train_neural import plot_nn_evaluation
    metrics = {"Accuracy": 0.8, "F1-score": 0.75}
    y_true = np.array([0, 1])
    y_pred = np.array([0, 0])
    plot_nn_evaluation(metrics, y_true, y_pred, config=temp_config)
    assert os.path.exists(os.path.join(temp_config.visuals_dir, temp_config.nn_metrics_filename))
    assert os.path.exists(os.path.join(temp_config.visuals_dir, temp_config.nn_cm_filename))
