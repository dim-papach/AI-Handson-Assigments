import pytest

pytest.importorskip("xgboost")
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.datasets import make_classification
from typing import Dict, Tuple, List, Optional, Any
from sklearn.linear_model import LogisticRegression

# Import the targets to test
from src.train_classical import (
    train_classical_models,
    build_model_grid,
    run_grid_search_for_model,
    report_feature_importances,
    save_best_model
)
from src.evaluation import evaluate_classical_model, plot_model_performance


@pytest.fixture
def sample_data() -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
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


def test_evaluate_model(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    """Test target metric evaluations correctly identify standard structures."""
    X_train, y_train, X_val, y_val = sample_data
    
    # Fit a dummy linear logic base
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Trigger mapping method
    metrics, y_pred = evaluate_classical_model(model, X_val, y_val)
    
    assert isinstance(metrics, dict)
    assert 'Accuracy' in metrics
    assert 'Precision' in metrics
    assert 'ROC-AUC' in metrics
    assert not np.isnan(metrics['ROC-AUC'])
    
    assert len(y_pred) == len(y_val)


@patch('matplotlib.pyplot.savefig')
def test_plot_model_evaluations(mock_savefig: MagicMock, tmp_path: Any) -> None:
    """Test graphical plot mappings avoid crashing when triggered natively."""
    model_results = {
        'DummyModel': {
            'metrics': {'Accuracy': 0.85, 'ROC-AUC': 0.90},
            'cm': np.array([[15, 2], [3, 20]])
        }
    }
    
    visuals_dir = tmp_path / "visuals"
    
    # Try plotting
    plot_model_performance(
        model_results, 
        visuals_dir=str(visuals_dir),
        metrics_filename="metrics.png",
        cm_filename="cm.png"
    )
    
    # Check dir creation & mocking captures
    assert os.path.exists(str(visuals_dir))
    assert mock_savefig.call_count == 2 # 1 for metrics + 1 for confusion matrices


def test_train_classical_models(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray], tmp_path: Any) -> None:
    """Test full training grid securely completes evaluations iteratively across all topologies."""
    X_train, y_train, X_val, y_val = sample_data
    models_dir = str(tmp_path / "models")
    visuals_dir = str(tmp_path / "visuals")

    # Patch out file IO to ensure isolated testing functionality natively
    with patch('src.train_classical.plot_model_performance') as mock_plot:
        with patch('joblib.dump') as mock_dump:
            with patch('os.makedirs'):
                best_model = train_classical_models(
                    X_train, y_train, X_val, y_val,
                    random_state=42,
                    models_dir=models_dir,
                    visuals_dir=visuals_dir,
                    metrics_filename="metrics.png",
                    cm_filename="cm.png",
                    model_filename="model.pkl",
                    xgb_early_stopping_rounds=5,
                    dt_max_depth=[None, 5],
                    dt_min_samples_split=[2],
                    lr_C=[1.0],
                    lr_max_iter=[200],
                    svm_C=[1.0],
                    svm_kernel=["rbf"],
                    rf_n_estimators=[10],
                    rf_max_depth=[None],
                    xgb_n_estimators=[10],
                    xgb_max_depth=[3],
                    xgb_learning_rate=[0.1],
                )

    # Evaluation Assertions
    assert best_model is not None
    assert hasattr(best_model, 'predict')

    mock_plot.assert_called_once()
    assert mock_dump.call_count >= 1


def _default_grid_kwargs() -> Dict[str, Any]:
    """Return a minimal but complete set of kwargs for build_model_grid."""
    return dict(
        random_state=42,
        dt_max_depth=[None, 5],
        dt_min_samples_split=[2],
        lr_C=[1.0],
        lr_max_iter=[100],
        svm_C=[1.0],
        svm_kernel=["rbf"],
        rf_n_estimators=[10],
        rf_max_depth=[None],
        xgb_n_estimators=[10],
        xgb_max_depth=[3],
        xgb_learning_rate=[0.1],
    )


def test_build_model_grid() -> None:
    """Test that build_model_grid returns correct structure and eval_metric by task type."""
    grid_binary = build_model_grid(is_multiclass=False, **_default_grid_kwargs())
    assert 'XGBoost' in grid_binary
    assert 'DecisionTree' in grid_binary
    assert 'LogisticRegression' in grid_binary
    assert 'SVM' in grid_binary
    assert 'RandomForest' in grid_binary
    assert grid_binary['XGBoost']['grid']['eval_metric'] == ['logloss']

    grid_multi = build_model_grid(is_multiclass=True, **_default_grid_kwargs())
    assert grid_multi['XGBoost']['grid']['eval_metric'] == ['mlogloss']


def test_build_model_grid_has_model_cls() -> None:
    """Verify every entry in the grid has a callable model_cls and non-empty grid."""
    grid = build_model_grid(is_multiclass=False, **_default_grid_kwargs())
    for name, entry in grid.items():
        assert callable(entry['model_cls']), f"{name} model_cls is not callable"
        assert isinstance(entry['grid'], dict), f"{name} grid is not a dict"
        assert len(entry['grid']) > 0, f"{name} grid is empty"


def test_run_grid_search_for_model(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    X_train, y_train, X_val, y_val = sample_data
    y_train_s = pd.Series(y_train)
    y_val_s = pd.Series(y_val)
    
    # We'll test with a simple model to keep it fast
    config = {
        'model_cls': LogisticRegression,
        'grid': {'C': [0.1, 1.0], 'max_iter': [100]}
    }
    
    best_model, best_metrics, best_preds, best_score = run_grid_search_for_model(
        'LogReg', config, X_train, y_train_s, X_val, y_val_s, is_multiclass=False
    )
    
    assert best_model is not None
    assert 'Accuracy' in best_metrics
    assert len(best_preds) == len(y_val)
    assert best_score >= 0


def test_report_feature_importances(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    X_train, y_train, _, _ = sample_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Linear model uses coef_
    with patch('sys.stdout', new=MagicMock()) as mock_out:
        report_feature_importances(model, X_train)
    
    # Tree model uses feature_importances_
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)
    with patch('sys.stdout', new=MagicMock()) as mock_out:
        report_feature_importances(tree, X_train)


def test_save_best_model(tmp_path: Any) -> None:
    model = LogisticRegression()
    models_dir = tmp_path / "models"
    
    path = save_best_model(model, models_dir=str(models_dir))
    assert os.path.exists(path)
    assert "classical_model.pkl" in path


def test_evaluate_model_multiclass() -> None:
    """Test evaluation logic for multiclass classification."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    
    model = LogisticRegression()
    model.fit(X_df, y)
    
    metrics, _ = evaluate_classical_model(model, X_df, y)
    assert 'ROC-AUC' in metrics
    assert not np.isnan(metrics['ROC-AUC'])


def test_evaluate_model_no_proba() -> None:
    """Test evaluation logic when model has no predict_proba."""
    X, y = make_classification(n_samples=50, n_features=5, random_state=42)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    
    # Create a mock model WITHOUT predict_proba
    model = MagicMock()
    model.predict.return_value = y
    del model.predict_proba # Ensure it doesn't have it
    
    metrics, _ = evaluate_classical_model(model, X_df, y)
    assert np.isnan(metrics['ROC-AUC'])


def test_evaluate_model_binary_auto_detect() -> None:
    """Test that binary multiclass flag is correctly auto-detected via classes_."""
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    model = LogisticRegression()
    model.fit(X_df, y)

    # Pass is_multiclass=False explicitly; model.classes_ has 2 entries → stays binary path
    metrics, y_pred = evaluate_classical_model(model, X_df, y)
    assert 'ROC-AUC' in metrics
    assert len(y_pred) == len(y)


def test_report_feature_importances_no_attrs(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    """Test that report_feature_importances is a no-op for models without importances or coef_."""
    X_train, y_train, _, _ = sample_data
    model = MagicMock(spec=[])  # No feature_importances_ or coef_ attributes
    # Should not raise
    report_feature_importances(model, X_train)


@patch('matplotlib.pyplot.savefig')
def test_plot_model_evaluations_multiple_models(mock_savefig: MagicMock, tmp_path: Any) -> None:
    """Test plot_model_evaluations handles multiple models without crashing."""
    model_results = {
        'ModelA': {
            'metrics': {'Accuracy': 0.80, 'ROC-AUC': np.nan},
            'cm': np.array([[10, 2], [3, 15]]),
        },
        'ModelB': {
            'metrics': {'Accuracy': 0.90, 'ROC-AUC': 0.95},
            'cm': np.array([[12, 1], [2, 15]]),
        },
    }
    visuals_dir = str(tmp_path / "visuals")
    plot_model_performance(
        model_results, 
        visuals_dir=visuals_dir,
        metrics_filename="metrics.png",
        cm_filename="cm.png"
    )
    assert mock_savefig.call_count == 2


def test_run_grid_search_xgboost(sample_data: Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]) -> None:
    """Test run_grid_search_for_model with XGBoost early stopping branch."""
    from xgboost import XGBClassifier

    X_train, y_train, X_val, y_val = sample_data
    y_train_s = pd.Series(y_train)
    y_val_s = pd.Series(y_val)

    config = {
        'model_cls': XGBClassifier,
        'grid': {
            'n_estimators': [10],
            'max_depth': [3],
            'learning_rate': [0.1],
            'random_state': [42],
            'eval_metric': ['logloss'],
        },
    }

    best_model, best_metrics, best_preds, best_score = run_grid_search_for_model(
        'XGBoost', config, X_train, y_train_s, X_val, y_val_s,
        is_multiclass=False,
        early_stopping_rounds=5,
    )

    assert best_model is not None
    assert 'Accuracy' in best_metrics
    assert best_score >= 0
