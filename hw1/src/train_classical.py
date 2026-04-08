import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import LabelEncoder
from typing import Dict, Tuple, Any, List, Optional
from src.config import PipelineConfig
from src.evaluation import evaluate_classical_model, plot_model_performance



def build_model_grid(
    is_multiclass: bool,
    random_state: int = PipelineConfig.random_state,
    dt_max_depth: Optional[List[Optional[int]]] = None,
    dt_min_samples_split: Optional[List[int]] = None,
    lr_C: Optional[List[float]] = None,
    lr_max_iter: Optional[List[int]] = None,
    svm_C: Optional[List[float]] = None,
    svm_kernel: Optional[List[str]] = None,
    rf_n_estimators: Optional[List[int]] = None,
    rf_max_depth: Optional[List[Optional[int]]] = None,
    xgb_n_estimators: Optional[List[int]] = None,
    xgb_max_depth: Optional[List[int]] = None,
    xgb_learning_rate: Optional[List[float]] = None,
    n_jobs: int = PipelineConfig.n_jobs,
) -> Dict[str, Any]:
    """Returns the model classes and hyperparameter grids to search over.

    Parameters
    ----------
    is_multiclass : bool
        Whether the classification task has more than two classes.
        Affects XGBoost's eval_metric selection.
    random_state : int
        Random seed applied to every model that accepts one.
    dt_max_depth : List[Optional[int]], optional
    dt_min_samples_split : List[int], optional
    lr_C : List[float], optional
    lr_max_iter : List[int], optional
    svm_C : List[float], optional
    svm_kernel : List[str], optional
    rf_n_estimators : List[int], optional
    rf_max_depth : List[Optional[int]], optional
    xgb_n_estimators : List[int], optional
    xgb_max_depth : List[int], optional
    xgb_learning_rate : List[float], optional
    n_jobs : int, optional
        Number of CPU cores to use for parallel models (LogisticRegression, RandomForest, XGBoost).
        -1 means use all available cores.

    Returns
    -------
    Dict[str, Any]
        Mapping of model name to a dict with keys ``model_cls`` and ``grid``.
    """
    cfg = PipelineConfig()
    dt_max_depth = dt_max_depth or cfg.dt_max_depth
    dt_min_samples_split = dt_min_samples_split or cfg.dt_min_samples_split
    lr_C = lr_C or cfg.lr_C
    lr_max_iter = lr_max_iter or cfg.lr_max_iter
    svm_C = svm_C or cfg.svm_C
    svm_kernel = svm_kernel or cfg.svm_kernel
    rf_n_estimators = rf_n_estimators or cfg.rf_n_estimators
    rf_max_depth = rf_max_depth or cfg.rf_max_depth
    xgb_n_estimators = xgb_n_estimators or cfg.xgb_n_estimators
    xgb_max_depth = xgb_max_depth or cfg.xgb_max_depth
    xgb_learning_rate = xgb_learning_rate or cfg.xgb_learning_rate
    return {
        'DecisionTree': {
            'model_cls': DecisionTreeClassifier,
            'grid': {
                'max_depth': dt_max_depth,
                'min_samples_split': dt_min_samples_split,
                'class_weight': ['balanced'],
                'random_state': [random_state],
            },
        },
        'LogisticRegression': {
            'model_cls': LogisticRegression,
            'grid': {
                'C': lr_C,
                'max_iter': lr_max_iter,
                'class_weight': ['balanced'],
                'random_state': [random_state],
            },
            'init_kwargs': {'n_jobs': n_jobs},
        },
        'SVM': {
            'model_cls': SVC,
            'grid': {
                'C': svm_C,
                'kernel': svm_kernel,
                'probability': [True],  # Required for predict_proba & AUC
                'class_weight': ['balanced'],
                'random_state': [random_state],
            },
        },
        'RandomForest': {
            'model_cls': RandomForestClassifier,
            'grid': {
                'n_estimators': rf_n_estimators,
                'max_depth': rf_max_depth,
                'class_weight': ['balanced'],
                'random_state': [random_state],
            },
            'init_kwargs': {'n_jobs': n_jobs},
        },
        'XGBoost': {
            'model_cls': XGBClassifier,
            'grid': {
                'n_estimators': xgb_n_estimators,
                'max_depth': xgb_max_depth,
                'learning_rate': xgb_learning_rate,
                'random_state': [random_state],
                'eval_metric': ['mlogloss' if is_multiclass else 'logloss'],
            },
            'init_kwargs': {'n_jobs': n_jobs},
        },
    }


def run_grid_search_for_model(
    model_name: str,
    config: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    is_multiclass: bool,
    early_stopping_rounds: int = PipelineConfig.xgb_early_stopping_rounds,
) -> Tuple[Any, Dict[str, float], np.ndarray, float]:
    """Exhaustively searches the parameter grid for one model type.

    Parameters
    ----------
    model_name : str
        Human-readable model identifier (e.g. ``'RandomForest'``).
    config : Dict[str, Any]
        Dict containing ``model_cls`` and ``grid`` keys as produced by
        :func:`build_model_grid`.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Validation labels.
    is_multiclass : bool
        Whether the task has more than two target classes.
    early_stopping_rounds : int, optional
        XGBoost early stopping patience, by default 10.

    Returns
    -------
    Tuple[Any, Dict[str, float], np.ndarray, float]
        ``(best_model, best_metrics, best_predictions, best_score)``
    """
    best_model: Any = None
    best_score: float = -1.0
    best_metrics: Dict[str, float] = {}
    best_predictions: np.ndarray = np.array([])

    for params in ParameterGrid(config['grid']):
        params = {**params, **config.get('init_kwargs', {})}
        model = config['model_cls'](**params)

        if model_name == 'XGBoost':
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        metrics, y_pred = evaluate_classical_model(model, X_val, y_val)
        score = metrics['ROC-AUC'] if not np.isnan(metrics['ROC-AUC']) else metrics['Accuracy']

        if score > best_score:
            best_score = score
            best_model = model
            best_metrics = metrics
            best_predictions = y_pred

    print(f"Best {model_name} Validation Score (AUC fallback to Acc): {best_score:.4f}")
    for metric_name, value in best_metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    return best_model, best_metrics, best_predictions, best_score


def report_feature_importances(model: Any, X_train: pd.DataFrame, top_n: int = 10) -> None:
    """Prints feature importances or linear coefficients for the given model.

    Parameters
    ----------
    model : Any
        A fitted sklearn-compatible model.
    X_train : pd.DataFrame
        Training feature matrix, used to retrieve column names.
    top_n : int, optional
        Number of top features to display, by default 10.
    """
    if hasattr(model, "feature_importances_"):
        print("\nFeature importances:")
        importances: np.ndarray = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        feat_len = X_train.shape[1]
        for idx in indices[:top_n]:
            if idx < feat_len:
                feat_name = X_train.columns[idx]
                print(f"  {feat_name}: {importances[idx]:.4f}")
    elif hasattr(model, "coef_"):
        print("\nCoefficients:")
        print(model.coef_)


def save_best_model(
    model: Any,
    models_dir: str = PipelineConfig.models_dir,
    model_filename: str = PipelineConfig.model_filename,
) -> str:
    """Persists the best model to disk via joblib.

    Parameters
    ----------
    model : Any
        Fitted model to save.
    models_dir : str, optional
        Directory to write the file into, by default ``"models"``.
    model_filename : str, optional
        Name of the output file, by default ``"classical_model.pkl"``.

    Returns
    -------
    str
        Absolute path of the saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Saved best classical model to {model_path}")
    return model_path


def train_classical_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int = PipelineConfig.random_state,
    models_dir: str = PipelineConfig.models_dir,
    visuals_dir: str = PipelineConfig.visuals_dir,
    metrics_filename: str = PipelineConfig.metrics_filename,
    cm_filename: str = PipelineConfig.cm_filename,
    model_filename: str = PipelineConfig.model_filename,
    xgb_early_stopping_rounds: int = PipelineConfig.xgb_early_stopping_rounds,
    n_jobs: int = PipelineConfig.n_jobs,
    dt_max_depth: Optional[List[Optional[int]]] = None,
    dt_min_samples_split: Optional[List[int]] = None,
    lr_C: Optional[List[float]] = None,
    lr_max_iter: Optional[List[int]] = None,
    svm_C: Optional[List[float]] = None,
    svm_kernel: Optional[List[str]] = None,
    rf_n_estimators: Optional[List[int]] = None,
    rf_max_depth: Optional[List[Optional[int]]] = None,
    xgb_n_estimators: Optional[List[int]] = None,
    xgb_max_depth: Optional[List[int]] = None,
    xgb_learning_rate: Optional[List[float]] = None,
    le: Optional[LabelEncoder] = None,
) -> Any:
    """Grid search over classical ML algorithms and return the best model.

    Searches Decision Tree, Random Forest, XGBoost, Logistic Regression,
    and SVM. Saves the winner and plots evaluation metrics.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        Validation labels.
    random_state : int, optional
    models_dir : str, optional
    visuals_dir : str, optional
    metrics_filename : str, optional
    cm_filename : str, optional
    model_filename : str, optional
    le : LabelEncoder, optional
        Target label encoder fitted on the training set.

    Returns
    -------
    Any
        The best fitted sklearn-compatible model found across all searches.
    """
    cfg = PipelineConfig()
    dt_max_depth = dt_max_depth or cfg.dt_max_depth
    dt_min_samples_split = dt_min_samples_split or cfg.dt_min_samples_split
    lr_C = lr_C or cfg.lr_C
    lr_max_iter = lr_max_iter or cfg.lr_max_iter
    svm_C = svm_C or cfg.svm_C
    svm_kernel = svm_kernel or cfg.svm_kernel
    rf_n_estimators = rf_n_estimators or cfg.rf_n_estimators
    rf_max_depth = rf_max_depth or cfg.rf_max_depth
    xgb_n_estimators = xgb_n_estimators or cfg.xgb_n_estimators
    xgb_max_depth = xgb_max_depth or cfg.xgb_max_depth
    xgb_learning_rate = xgb_learning_rate or cfg.xgb_learning_rate
    is_multiclass: bool = len(np.unique(y_train)) > 2
    model_grid = build_model_grid(
        is_multiclass=is_multiclass,
        random_state=random_state,
        dt_max_depth=dt_max_depth,
        dt_min_samples_split=dt_min_samples_split,
        lr_C=lr_C,
        lr_max_iter=lr_max_iter,
        svm_C=svm_C,
        svm_kernel=svm_kernel,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        n_jobs=n_jobs,
    )

    best_overall_model: Any = None
    best_overall_score: float = -1.0
    best_overall_name: str = ""
    model_evaluations: Dict[str, Any] = {}

    print("Starting Grid Search for Classical ML Models...")

    for model_name, config in model_grid.items():
        print(f"\n--- Training {model_name} ---")
        best_model, best_metrics, best_predictions, best_score = run_grid_search_for_model(
            model_name, config, X_train, y_train, X_val, y_val,
            is_multiclass=is_multiclass,
            early_stopping_rounds=xgb_early_stopping_rounds,
        )

        model_evaluations[model_name] = {
            'metrics': best_metrics,
            'cm': confusion_matrix(y_val, best_predictions),
        }

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_model = best_model
            best_overall_name = model_name

    print(f"\nBest Overall Classical Model: {best_overall_name} with score: {best_overall_score:.4f}")

    class_names = [str(c) for c in le.classes_] if le is not None else None
    plot_model_performance(
        model_evaluations,
        visuals_dir=visuals_dir,
        metrics_filename=metrics_filename,
        cm_filename=cm_filename,
        main_title="Validation Metrics Comparison",
        class_names=class_names,
    )
    save_best_model(best_overall_model, models_dir=models_dir, model_filename=model_filename)
    report_feature_importances(best_overall_model, X_train)

    return best_overall_model


if __name__ == "__main__":
    print("This module provides classical ML training logic.")
    print("Import and run `train_classical_models(X_train, y_train, X_val, y_val)`.")
