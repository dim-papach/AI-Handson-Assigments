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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from typing import Dict, Tuple, Any, List

def evaluate_model(model: Any, X_val: pd.DataFrame, y_val: pd.Series, is_multiclass: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluates the model and returns a dictionary of robust classification metrics."""
    y_pred = model.predict(X_val)
    
    # Core performance metrics 
    metrics = {
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_val, y_pred, average='weighted', zero_division=0),
        'F1-score': f1_score(y_val, y_pred, average='weighted', zero_division=0)
    }
    
    # Calculate ROC-AUC
    try:
        y_prob = model.predict_proba(X_val)
        if getattr(model, "classes_", None) is not None and len(model.classes_) > 2:
            is_multiclass = True
            
        if is_multiclass:
            metrics['ROC-AUC'] = roc_auc_score(y_val, y_prob, multi_class='ovr')
        else:
            metrics['ROC-AUC'] = roc_auc_score(y_val, y_prob[:, 1])
    except (AttributeError, ValueError, IndexError):
        metrics['ROC-AUC'] = np.nan
        
    return metrics, y_pred

def plot_model_evaluations(model_results: Dict[str, Any], visuals_dir: str = "visuals") -> None:
    """
    Creates bar plots for metrics and confusion matrices for each model.
    model_results is a dict: {'ModelName': {'metrics': metrics_dict, 'cm': confusion_matrix_array}}
    """
    os.makedirs(visuals_dir, exist_ok=True)
    num_models = len(model_results)
    
    # 1. Plot Metrics (Bar plot per model)
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 5), sharey=True)
    if num_models == 1:
        axes = [axes]
        
    for ax, (model_name, data) in zip(axes, model_results.items()):
        metrics = data['metrics']
        # Prune NaN values
        metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}
        
        ax.bar(metrics.keys(), metrics.values(), color=sns.color_palette("viridis", len(metrics)))
        ax.set_title(f"{model_name}", fontweight='bold')
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(metrics.values()):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        
    plt.suptitle('Validation Metrics Comparison', fontsize=14, y=1.05)
    plt.tight_layout()
    metrics_path = os.path.join(visuals_dir, "classical_models_metrics.png")
    plt.savefig(metrics_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Plot Confusion Matrices (1 heatmap per model)
    fig2, axes2 = plt.subplots(1, num_models, figsize=(4 * num_models, 4))
    if num_models == 1:
        axes2 = [axes2]
        
    for ax, (model_name, data) in zip(axes2, model_results.items()):
        cm = data['cm']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{model_name}", fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
    plt.suptitle('Confusion Matrices', fontsize=14, y=1.05)
    plt.tight_layout()
    cm_path = os.path.join(visuals_dir, "classical_models_confusion_matrices.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Saved evaluation graphs to {metrics_path} and {cm_path}")

def build_model_grid(is_multiclass: bool) -> Dict[str, Any]:
    """Returns the model classes and hyperparameter grids to search over.

    Parameters
    ----------
    is_multiclass : bool
        Whether the classification task has more than two classes.
        Affects XGBoost's eval_metric selection.

    Returns
    -------
    Dict[str, Any]
        Mapping of model name to a dict with keys ``model_cls`` and ``grid``.
    """
    return {
        'DecisionTree': {
            'model_cls': DecisionTreeClassifier,
            'grid': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced'],
                'random_state': [42],
            },
        },
        'LogisticRegression': {
            'model_cls': LogisticRegression,
            'grid': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000],
                'class_weight': ['balanced'],
                'random_state': [42],
            },
        },
        'SVM': {
            'model_cls': SVC,
            'grid': {
                'C': [0.1, 1.0],
                'kernel': ['rbf', 'linear'],
                'probability': [True],  # Required for predict_proba & AUC
                'class_weight': ['balanced'],
                'random_state': [42],
            },
        },
        'RandomForest': {
            'model_cls': RandomForestClassifier,
            'grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'class_weight': ['balanced'],
                'random_state': [42],
            },
        },
        'XGBoost': {
            'model_cls': XGBClassifier,
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'random_state': [42],
                'eval_metric': ['mlogloss' if is_multiclass else 'logloss'],
            },
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
        model = config['model_cls'](**params)

        if model_name == 'XGBoost':
            model.set_params(early_stopping_rounds=10)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        metrics, y_pred = evaluate_model(model, X_val, y_val, is_multiclass)
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


def save_best_model(model: Any, models_dir: str = "models") -> str:
    """Persists the best model to disk via joblib.

    Parameters
    ----------
    model : Any
        Fitted model to save.
    models_dir : str, optional
        Directory to write the file into, by default ``"models"``.

    Returns
    -------
    str
        Absolute path of the saved model file.
    """
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "classical_model.pkl")
    joblib.dump(model, model_path)
    print(f"Saved best classical model to {model_path}")
    return model_path


def train_classical_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    models_dir: str = "models",
    visuals_dir: str = "visuals",
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
    models_dir : str, optional
        Directory to save the best model, by default ``"models"``.
    visuals_dir : str, optional
        Directory to save evaluation plots, by default ``"visuals"``.

    Returns
    -------
    Any
        The best fitted sklearn-compatible model found across all searches.
    """
    is_multiclass: bool = len(np.unique(y_train)) > 2
    model_grid = build_model_grid(is_multiclass)

    best_overall_model: Any = None
    best_overall_score: float = -1.0
    best_overall_name: str = ""
    model_evaluations: Dict[str, Any] = {}

    print("Starting Grid Search for Classical ML Models...")

    for model_name, config in model_grid.items():
        print(f"\n--- Training {model_name} ---")
        best_model, best_metrics, best_predictions, best_score = run_grid_search_for_model(
            model_name, config, X_train, y_train, X_val, y_val, is_multiclass
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

    plot_model_evaluations(model_evaluations, visuals_dir=visuals_dir)
    save_best_model(best_overall_model, models_dir=models_dir)
    report_feature_importances(best_overall_model, X_train)

    return best_overall_model


if __name__ == "__main__":
    print("This module provides classical ML training logic.")
    print("Import and run `train_classical_models(X_train, y_train, X_val, y_val)`.")
