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

def evaluate_model(model, X_val, y_val, is_multiclass=False):
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

def plot_model_evaluations(model_results, visuals_dir="visuals"):
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

def train_classical_models(X_train, y_train, X_val, y_val):
    """
    Grid search over Decision Tree, Random Forest, XGBoost, Logistic Regression, and SVM.
    Returns the best model across all types searched.
    """
    unique_classes = np.unique(y_train)
    is_multiclass = len(unique_classes) > 2

    models_and_grids = {
        'DecisionTree': {
            'model_cls': DecisionTreeClassifier,
            'grid': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced'],
                'random_state': [42]
            }
        },
        'LogisticRegression': {
            'model_cls': LogisticRegression,
            'grid': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000],
                'class_weight': ['balanced'],
                'random_state': [42]
            }
        },
        'SVM': {
            'model_cls': SVC,
            'grid': {
                'C': [0.1, 1.0],
                'kernel': ['rbf', 'linear'],
                'probability': [True],  # Required for predict_proba & AUC
                'class_weight': ['balanced'],
                'random_state': [42]
            }
        },
        'RandomForest': {
            'model_cls': RandomForestClassifier,
            'grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'class_weight': ['balanced'],
                'random_state': [42]
            }
        },
        'XGBoost': {
            'model_cls': XGBClassifier,
            'grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'random_state': [42],
                'eval_metric': ['logloss' if not is_multiclass else 'mlogloss']
            }
        }
    }

    best_overall_model = None
    best_overall_score = -1
    best_overall_name = ""
    
    # Dictionary to collect results for graphs
    model_evaluations = {}

    print("Starting Grid Search for Classical ML Models...")

    for model_name, config in models_and_grids.items():
        print(f"\n--- Training {model_name} ---")
        grid = list(ParameterGrid(config['grid']))
        best_model_for_type = None
        best_score_for_type = -1
        best_metrics_for_type = None
        best_preds_for_type = None
        
        for params in grid:
            model = config['model_cls'](**params)
            
            if model_name == 'XGBoost':
                # XGBoost specific handling for early stopping
                model.set_params(early_stopping_rounds=10)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Retrieve robust classification metrics instead of just simple accuracy
            metrics, y_pred = evaluate_model(model, X_val, y_val, is_multiclass)
            
            # Using ROC-AUC if possible to choose the model, fallback to Accuracy
            score = metrics['ROC-AUC'] if not np.isnan(metrics['ROC-AUC']) else metrics['Accuracy']
            
            if score > best_score_for_type:
                best_score_for_type = score
                best_model_for_type = model
                best_metrics_for_type = metrics
                best_preds_for_type = y_pred
                
        print(f"Best {model_name} Validation Score (AUC fallback to Acc): {best_score_for_type:.4f}")
        for k, v in best_metrics_for_type.items():
            print(f"  {k}: {v:.4f}")
            
        # Store for visualizations
        model_evaluations[model_name] = {
            'metrics': best_metrics_for_type,
            'cm': confusion_matrix(y_val, best_preds_for_type)
        }
        
        if best_score_for_type > best_overall_score:
            best_overall_score = best_score_for_type
            best_overall_model = best_model_for_type
            best_overall_name = model_name

    print(f"\nBest Overall Classical Model: {best_overall_name} with score: {best_overall_score:.4f}")
    
    # 5. Generate the specified plots for the metrics and confusion matrices
    plot_model_evaluations(model_evaluations, visuals_dir="visuals")
    
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    # Save best overall model
    model_path = os.path.join("models", "classical_model.pkl")
    joblib.dump(best_overall_model, model_path)
    print(f"Saved best classical model to {model_path}")
    
    # Print feature importances or coefficients for insights
    if hasattr(best_overall_model, "feature_importances_"):
        print("\nFeature importances:")
        importances = best_overall_model.feature_importances_
        # Sort importances
        indices = np.argsort(importances)[::-1]
        
        # Determine valid feature length
        feat_len = X_train.shape[1] if hasattr(X_train, "shape") else len(X_train.columns)
        for idx in indices[:10]: # Print top 10
            if idx < feat_len:
                feat_name = X_train.columns[idx] if hasattr(X_train, "columns") else f"Feature {idx}"
                print(f"{feat_name}: {importances[idx]:.4f}")
                
    elif hasattr(best_overall_model, "coef_"):
        print("\nCoefficients:")
        print(best_overall_model.coef_)

    return best_overall_model

if __name__ == "__main__":
    print("This module provides classical ML training logic.")
    print("Import and run `train_classical_models(X_train, y_train, X_val, y_val)`.")
