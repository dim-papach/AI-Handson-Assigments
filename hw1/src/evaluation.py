import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import PipelineConfig


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculates core classification metrics: Accuracy, Precision, Recall, F1, and ROC-AUC.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    y_prob : np.ndarray, optional
        Predicted probabilities for ROC-AUC calculation.

    Returns
    -------
    Dict[str, float]
        Dictionary of metric names and their calculated values.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None:
        try:
            # Check if multi-class (more than 2 unique values in y_true)
            unique_labels = np.unique(y_true)
            if len(unique_labels) > 2:
                metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
            else:
                # Handle binary case where y_prob might be 1D (probs for class 1) or 2D
                if y_prob.ndim == 2 and y_prob.shape[1] > 1:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        except (ValueError, IndexError):
            metrics["ROC-AUC"] = np.nan
    else:
        metrics["ROC-AUC"] = np.nan

    return metrics


def evaluate_classical_model(
    model: Any, X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates a classical ML model (sklearn-compatible) on a dataset.

    Parameters
    ----------
    model : Any
        The fitted classical model.
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target labels.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        A tuple containing (metrics_dict, y_pred).
    """
    y_pred = model.predict(X)
    y_prob = None
    try:
        y_prob = model.predict_proba(X)
    except (AttributeError, ValueError):
        pass

    y_vals = getattr(y, "values", y)
    metrics = calculate_metrics(y_vals, y_pred, y_prob)
    return metrics, y_pred


def display_detailed_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Classification Report",
) -> None:
    """
    Prints a detailed classification report using class names.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_pred : np.ndarray
        Predicted labels.
    class_names : List[str], optional
        Actual names of the classes.
    title : str, optional
        Title for the report.
    """
    print(f"\n--- {title} ---")
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)


def evaluate_nn_model(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[PipelineConfig] = None,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates a PyTorch neural network model on a dataset.

    Parameters
    ----------
    model : nn.Module
        The trained neural network.
    X : pd.DataFrame
        Input features.
    y : pd.Series
        Target labels.
    config : PipelineConfig, optional
        Configuration object.

    Returns
    -------
    Tuple[Dict[str, float], np.ndarray]
        A tuple containing (metrics_dict, y_pred).
    """
    if config is None:
        config = PipelineConfig()

    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        out_act = config.nn_output_activation

        # Determine output dim (fallback to 1 if not defined)
        output_dim = getattr(model, "output_dim", 1)

        if out_act is None:
            out_act = "Sigmoid" if output_dim == 1 else "Softmax"

        if out_act == "Sigmoid":
            probs = torch.sigmoid(logits).squeeze().numpy()
            y_pred = (probs >= 0.5).astype(int)
        elif out_act == "Softmax":
            probs = torch.softmax(logits, dim=1).numpy()
            y_pred = np.argmax(probs, axis=1)
        else:
            probs = logits.numpy()
            y_pred = (
                np.argmax(probs, axis=1) if output_dim > 1 else (probs >= 0).astype(int)
            )

    y_vals = getattr(y, "values", y)
    metrics = calculate_metrics(y_vals, y_pred, probs)
    return metrics, y_pred


def plot_model_performance(
    model_results: Dict[str, Any],
    visuals_dir: str,
    metrics_filename: str,
    cm_filename: str,
    main_title: str = "Model Performance Comparison",
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Generates bar plots for metrics and confusion matrices for multiple models.

    Parameters
    ----------
    model_results : Dict[str, Any]
        Dictionary where keys are model names and values are dicts containing:
        'metrics' (Dict[str, float]) and 'cm' (np.ndarray).
    visuals_dir : str
        Directory to save plots.
    metrics_filename : str
        Filename for the metrics bar plot.
    cm_filename : str
        Filename for the confusion matrices heatmap.
    main_title : str, optional
        Title for the plots.
    class_names : List[str], optional
        List of class names for confusion matrix labels.
    """
    os.makedirs(visuals_dir, exist_ok=True)
    num_models = len(model_results)
    if num_models == 0:
        return

    # 1. Plot Metrics (Bar plots)
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 5), sharey=True)
    if num_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, model_results.items()):
        metrics = {k: v for k, v in data["metrics"].items() if not np.isnan(v)}
        ax.bar(
            metrics.keys(),
            metrics.values(),
            color=sns.color_palette("viridis", len(metrics)),
        )
        ax.set_title(model_name, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(metrics.values()):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle(main_title, fontsize=14, y=1.05)
    plt.tight_layout()
    metrics_path = os.path.join(visuals_dir, metrics_filename)
    plt.savefig(metrics_path, bbox_inches="tight", dpi=300)
    plt.close()

    # 2. Plot Confusion Matrices
    fig2, axes2 = plt.subplots(1, num_models, figsize=(4 * num_models, 4))
    if num_models == 1:
        axes2 = [axes2]

    for ax, (model_name, data) in zip(axes2, model_results.items()):
        cm = data["cm"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=class_names if class_names is not None else "auto",
            yticklabels=class_names if class_names is not None else "auto",
        )
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle(f"{main_title} - Confusion Matrices", fontsize=14, y=1.05)
    plt.tight_layout()
    cm_path = os.path.join(visuals_dir, cm_filename)
    plt.savefig(cm_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved evaluation graphs to {metrics_path} and {cm_path}")


def plot_nn_evaluation(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Optional[PipelineConfig] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Plots evaluation metrics and confusion matrix for the NN using the unified plotter.

    Parameters
    ----------
    metrics : Dict[str, float]
    y_true : np.ndarray
    y_pred : np.ndarray
    config : PipelineConfig, optional
    class_names : List[str], optional
    """
    if config is None:
        config = PipelineConfig()

    model_results = {
        "Neural Network": {
            "metrics": metrics,
            "cm": confusion_matrix(y_true, y_pred),
        }
    }

    plot_model_performance(
        model_results,
        visuals_dir=config.visuals_dir,
        metrics_filename=config.nn_metrics_filename,
        cm_filename=config.nn_cm_filename,
        main_title="Neural Network Evaluation",
        class_names=class_names,
    )


def plot_nn_training_history(
    history: Dict[str, List[float]],
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Plots training and validation loss curves for a neural network.

    Parameters
    ----------
    history : Dict[str, List[float]]
        Dictionary containing 'train_loss' and 'val_loss' lists.
    config : PipelineConfig, optional
        Configuration object.
    """
    if config is None:
        config = PipelineConfig()

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Neural Network Training History")
    plt.legend()
    plt.grid(True)

    os.makedirs(config.visuals_dir, exist_ok=True)
    plot_path = os.path.join(config.visuals_dir, config.nn_loss_plot_filename)
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved training history plot to {plot_path}")


def save_evaluation_tables(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_classical: Any,
    nn_model: nn.Module,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Generates side-by-side comparison CSVs for Task 4.2.
    Includes Metrics Comparison, Classical Feature Importance, and PCA Loadings.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (for PCA).
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    best_classical : Any
        The fitted classical model.
    nn_model : nn.Module
        The trained neural network.
    config : PipelineConfig, optional
        Configuration object.
    """
    if config is None:
        config = PipelineConfig()

    eval_dir = config.evaluation_dir
    os.makedirs(eval_dir, exist_ok=True)

    # 1. Metrics Comparison
    metrics_cl, _ = evaluate_classical_model(best_classical, X_test, y_test)
    metrics_nn, _ = evaluate_nn_model(nn_model, X_test, y_test, config=config)

    metrics_to_save = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    comparison_data = []
    for m in metrics_to_save:
        comparison_data.append(
            {
                "Metric": m,
                "Best_classical": metrics_cl.get(m, np.nan),
                "Neural_Network": metrics_nn.get(m, np.nan),
            }
        )

    pd.DataFrame(comparison_data).to_csv(
        os.path.join(eval_dir, "metrics_comparison.csv"), index=False
    )

    # 2. Classical Feature Importance
    if hasattr(best_classical, "feature_importances_"):
        importances = best_classical.feature_importances_
    elif hasattr(best_classical, "coef_"):
        importances = np.abs(best_classical.coef_[0])
    else:
        importances = np.zeros(X_train.shape[1])

    pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values(
        by="Importance", ascending=False
    ).to_csv(os.path.join(eval_dir, "classical_feature_importance.csv"), index=False)

    # 3. PCA Loadings
    pca = PCA()
    pca.fit(X_train)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=X_train.columns,
    )
    loadings.to_csv(os.path.join(eval_dir, "pca_loadings.csv"), index=True)

    print(f"Saved Task 4.2 evaluation CSVs to '{eval_dir}/' directory.")
