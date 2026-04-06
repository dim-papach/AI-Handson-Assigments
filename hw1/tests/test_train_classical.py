import pytest
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Import the targets to test
from src.train_classical import evaluate_model, plot_model_evaluations, train_classical_models


@pytest.fixture
def sample_data():
    """Generates synthetic dataset to mimic binary classification structure."""
    X, y = make_classification(
        n_samples=250, 
        n_features=5, 
        n_informative=3, 
        n_redundant=1, 
        n_classes=2, 
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    
    # Simple split
    X_train, y_train = X_df.iloc[:200], y[:200]
    X_val, y_val = X_df.iloc[200:], y[200:]
    return X_train, y_train, X_val, y_val


def test_evaluate_model(sample_data):
    """Test target metric evaluations correctly identify standard structures."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Fit a dummy linear logic base
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Trigger mapping method
    metrics, y_pred = evaluate_model(model, X_val, y_val, is_multiclass=False)
    
    assert isinstance(metrics, dict)
    assert 'Accuracy' in metrics
    assert 'Precision' in metrics
    assert 'ROC-AUC' in metrics
    assert not np.isnan(metrics['ROC-AUC'])
    
    assert len(y_pred) == len(y_val)


@patch('matplotlib.pyplot.savefig')
def test_plot_model_evaluations(mock_savefig, tmp_path):
    """Test graphical plot mappings avoid crashing when triggered natively."""
    model_results = {
        'DummyModel': {
            'metrics': {'Accuracy': 0.85, 'ROC-AUC': 0.90},
            'cm': np.array([[15, 2], [3, 20]])
        }
    }
    
    visuals_dir = tmp_path / "visuals"
    
    # Try plotting
    plot_model_evaluations(model_results, visuals_dir=str(visuals_dir))
    
    # Check dir creation & mocking captures
    assert os.path.exists(str(visuals_dir))
    assert mock_savefig.call_count == 2 # 1 for metrics + 1 for confusion matrices


def test_train_classical_models(sample_data, tmp_path):
    """Test full training grid securely completes evaluations iteratively across all topologies."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Patch out file IO to ensure isolated testing functionality natively
    with patch('src.train_classical.plot_model_evaluations') as mock_plot:
        with patch('joblib.dump') as mock_dump:
            with patch('os.makedirs'):
                best_model = train_classical_models(X_train, y_train, X_val, y_val)
                
    # Evaluation Assertions
    assert best_model is not None
    assert hasattr(best_model, 'predict')
    
    mock_plot.assert_called_once()
    assert mock_dump.call_count >= 1
