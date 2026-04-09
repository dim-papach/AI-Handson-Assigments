import os
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from src.config import PipelineConfig
from src.preprocessing import (
    compute_colors,
    filter_error_ratios,
    filter_important_columns,
    drop_specific_group,
    select_metal_flag,
    drop_missing_targets,
    fit_target_encoder,
    apply_target_encoder,
    compute_iqr_bounds,
    apply_iqr_capping,
    save_outlier_histograms,
    split_data,
    build_preprocessing_pipeline,
    apply_imputation,
    get_fitted_scaler,
    apply_scaling,
    generate_pca_insights,
    print_class_ratios,
    apply_smote,
)
from src.train_classical import train_classical_models
from src.train_neural import train_neural_network, SimpleNN
from src.evaluation import (
    evaluate_classical_model,
    evaluate_nn_model,
    plot_nn_training_history,
    plot_nn_evaluation,
    plot_model_performance,
    save_evaluation_tables,
    display_detailed_report,
)


# ---------------------------------------------------------------------------
# Plot style (isolated from module-level side-effects)
# ---------------------------------------------------------------------------

def configure_plot_style() -> None:
    """Apply a consistent matplotlib style used throughout the pipeline.

    Keeps all ``plt`` global-state mutations in one explicit call site
    rather than as silent module-level side-effects.
    """
    plt.style.use("bmh")
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        "color", plt.cm.viridis(np.linspace(0, 1, 4))
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_and_filter_data(config: PipelineConfig) -> pd.DataFrame:
    """Load raw CSV and apply column selection and metal-flag subsampling.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe ready for splitting.
    """
    raw_df = pd.read_csv(config.filepath)
    return (
        raw_df
        .pipe(
            filter_important_columns,
            key_vars=config.key_vars,
            err_vars=config.err_vars,
            no_err_vars=config.no_err_vars,
            flags_vars=config.flags_vars,
            color_vars=config.color_vars,
        )
        .pipe(
            drop_specific_group,
            target_col=config.target_col,
            group_name=config.drop_group,
        )
        .pipe(select_metal_flag)
    )


def fit_preprocessing_params(
    train_df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[Dict[str, Tuple[float, float]], StandardScaler, pd.Index, Pipeline, LabelEncoder]:
    """Fit all preprocessing artefacts exclusively on the training split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Raw training split (before any transformation).
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    Tuple[Dict[str, Tuple[float, float]], StandardScaler, pd.Index, Pipeline, LabelEncoder]
        ``(iqr_bounds, scaler, num_cols, imputation_pipeline, label_encoder)``
    """
    le = fit_target_encoder(
        train_df,
        config.target_col,
        models_dir=config.models_dir,
        encoder_filename=config.encoder_filename,
    )

    iqr_bounds = compute_iqr_bounds(
        train_df.drop(columns=[config.target_col]), factor=1.5
    )

    train_temp = (
        train_df
        .pipe(drop_missing_targets, target_col=config.target_col)
        .pipe(apply_target_encoder, target_col=config.target_col, le=le)
        .pipe(apply_iqr_capping, bounds=iqr_bounds)
    )

    # 1. Fit Imputation Pipeline on UN-SCALED data
    pipeline = build_preprocessing_pipeline(
        train_temp.drop(columns=[config.target_col])
    )
    pipeline.fit(
        train_temp.drop(columns=[config.target_col]),
        train_temp[config.target_col],
    )

    # 2. Apply Imputation first
    train_temp_imputed = train_temp.pipe(
        apply_imputation, pipeline=pipeline, target_col=config.target_col
    )

    # 3. Filter Error Ratios AFTER imputation (SNR > 3 check)
    train_temp_filtered = train_temp_imputed.pipe(
        filter_error_ratios, key_vars=config.key_vars, err_vars=config.err_vars
    )

    # 4. Fit Scaler ONLY AFTER filtering and imputation
    scaler, num_cols = get_fitted_scaler(train_temp_filtered.drop(columns=[config.target_col]))

    os.makedirs(config.models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.models_dir, config.scaler_filename))
    joblib.dump(pipeline, os.path.join(config.models_dir, "imputation_pipeline.pkl"))
    joblib.dump(iqr_bounds, os.path.join(config.models_dir, "iqr_bounds.pkl"))

    return iqr_bounds, scaler, num_cols, pipeline, le


def preprocess_split(
    df: pd.DataFrame,
    split_name: str,
    iqr_bounds: Dict[str, Tuple[float, float]],
    scaler: StandardScaler,
    num_cols: pd.Index,
    pipeline: Pipeline,
    le: LabelEncoder,
    config: PipelineConfig,
) -> pd.DataFrame:
    """Apply the full preprocessing chain to a single data split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw split dataframe (train, val, or test).
    split_name : str
        Human-readable label used when saving diagnostic plots (e.g. ``'train'``).
    iqr_bounds : Dict[str, Tuple[float, float]]
        IQR capping thresholds fitted on the training set.
    scaler : StandardScaler
        Feature scaler fitted on the training set.
    num_cols : pd.Index
        Numeric column names used when applying scaling.
    pipeline : Pipeline
        Imputation and encoding pipeline fitted on the training set.
    le : LabelEncoder
        Target label encoder fitted on the training set.
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed split.
    """
    return (
        df
        .pipe(drop_missing_targets, target_col=config.target_col)
        .pipe(print_class_ratios, target_col=config.target_col, title=f"{split_name} (Before Encoding)")
        .pipe(apply_target_encoder, target_col=config.target_col, le=le)
        .pipe(print_class_ratios, target_col=config.target_col, title=f"{split_name} (After Encoding)")
        .pipe(apply_iqr_capping, bounds=iqr_bounds)
        .pipe(apply_imputation, pipeline=pipeline, target_col=config.target_col)
        .pipe(filter_error_ratios, key_vars=config.key_vars, err_vars=config.err_vars)
        .pipe(apply_scaling, scaler=scaler, num_cols=num_cols, target_col=config.target_col)
        .pipe(save_outlier_histograms, prefix=split_name, out_dir=config.visuals_dir)
        .pipe(compute_colors)
    )


def detach_targets(
    clean_df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate feature matrix from target labels.

    Parameters
    ----------
    clean_df : pd.DataFrame
        Preprocessed dataframe containing both features and target column.
    target_col : str
        Name of the target column to detach.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` — feature matrix and label series.
    """
    y = clean_df.pop(target_col)
    return clean_df, y


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def run_data_ingestion_and_preprocessing(
    config: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder]:
    """Execute the data loading, splitting, and preprocessing pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder]
        (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
    """
    df_prepared = load_and_filter_data(config)

    train_df, val_df, test_df = split_data(
        df_prepared,
        target_col=config.target_col,
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    iqr_bounds, scaler, num_cols, pipeline, le = fit_preprocessing_params(
        train_df, config
    )

    preprocess_kwargs: Dict[str, Any] = dict(
        iqr_bounds=iqr_bounds,
        scaler=scaler,
        num_cols=num_cols,
        pipeline=pipeline,
        le=le,
        config=config,
    )

    train_clean = preprocess_split(train_df, "train", **preprocess_kwargs)
    val_clean = preprocess_split(val_df, "val", **preprocess_kwargs)
    test_clean = preprocess_split(test_df, "test", **preprocess_kwargs)

    X_train, y_train = detach_targets(train_clean, config.target_col)
    X_val, y_val = detach_targets(val_clean, config.target_col)
    X_test, y_test = detach_targets(test_clean, config.target_col)

    if config.use_smote:
        X_train, y_train = apply_smote(
            X_train,
            y_train,
            sampling_strategy=config.smote_sampling_strategy,
            k_neighbors=config.smote_k_neighbors,
            random_state=config.smote_random_state,
            target_distribution=config.target_distribution
        )
        # Print ratios after SMOTE
        smote_df = X_train.assign(**{config.target_col: y_train})
        print_class_ratios(smote_df, target_col=config.target_col, title="Train Set (After SMOTE)")

    if config.visuals_dir:
        generate_pca_insights(
            X_train,
            y_train,
            out_dir=config.visuals_dir,
            scree_filename=config.scree_filename,
            projection_filename=config.projection_filename,
        )

    return X_train, X_val, X_test, y_train, y_val, y_test, le


def run_classical_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig,
    le: LabelEncoder,
) -> Any:
    """Train and optimize classical machine learning models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation targets.
    config : PipelineConfig
        Pipeline configuration object.
    le : LabelEncoder
        Target label encoder fitted on the training set.

    Returns
    -------
    Any
        The best fitted classical model found during grid search.
    """
    model_path = os.path.join(config.models_dir, config.model_filename)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Check for dimension mismatch if possible
        if hasattr(model, "n_features_in_") and model.n_features_in_ != X_train.shape[1]:
            print(f"\n[WARN] Dimension mismatch in {model_path} ({model.n_features_in_} vs {X_train.shape[1]}). Re-training...")
        else:
            print(f"\n[SKIP] Classical model found at {model_path}. Skipping training...")
            return model

    print("\n" + "=" * 50)
    print("PHASE: Classical ML Training & Grid Search")
    print("=" * 50)
    return train_classical_models(
        X_train,
        y_train,
        X_val,
        y_val,
        random_state=config.random_state,
        models_dir=config.models_dir,
        visuals_dir=config.visuals_dir,
        metrics_filename=config.metrics_filename,
        cm_filename=config.cm_filename,
        model_filename=config.model_filename,
        xgb_early_stopping_rounds=config.xgb_early_stopping_rounds,
        n_jobs=config.n_jobs,
        dt_max_depth=config.dt_max_depth,
        dt_min_samples_split=config.dt_min_samples_split,
        lr_C=config.lr_C,
        lr_max_iter=config.lr_max_iter,
        svm_C=config.svm_C,
        svm_kernel=config.svm_kernel,
        rf_n_estimators=config.rf_n_estimators,
        rf_max_depth=config.rf_max_depth,
        xgb_n_estimators=config.xgb_n_estimators,
        xgb_max_depth=config.xgb_max_depth,
        xgb_learning_rate=config.xgb_learning_rate,
        le=le,
    )


def run_neural_training(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: PipelineConfig,
    le: LabelEncoder,
) -> torch.nn.Module:
    """Train and evaluate the neural network model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training targets.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation targets.
    config : PipelineConfig
        Pipeline configuration object.
    le : LabelEncoder
        Target label encoder fitted on the training set.

    Returns
    -------
    torch.nn.Module
        The trained or loaded neural network model.
    """
    model_path = os.path.join(config.models_dir, config.nn_model_filename)
    if os.path.exists(model_path):
        num_classes = len(np.unique(y_train))
        is_binary = num_classes <= 2
        output_dim = 1 if is_binary else num_classes
        
        nn_model = SimpleNN(
            input_dim=X_train.shape[1],
            hidden_layers=config.nn_hidden_layers,
            output_dim=output_dim,
            dropout=config.nn_dropout,
            activation=config.nn_activation,
        )
        try:
            nn_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            nn_model.eval()
            print(f"\n[SKIP] Neural network found at {model_path}. Skipping training...")
            return nn_model
        except Exception as e:
            print(f"\n[WARN] Failed to load Neural Network from {model_path}: {e}")
            print("Falling back to training a new model...")

    print("\n" + "=" * 50)
    print("PHASE: Neural Network Training")
    print("=" * 50)
    nn_model, nn_history = train_neural_network(
        X_train, y_train, X_val, y_val, config=config
    )

    # Plot training history
    plot_nn_training_history(nn_history, config=config)

    # Evaluate NN on validation set for comparison
    nn_val_metrics, nn_y_pred_val = evaluate_nn_model(
        nn_model, X_val, y_val, config=config
    )
    class_names = [str(c) for c in le.classes_]
    plot_nn_evaluation(
        nn_val_metrics, y_val.values, nn_y_pred_val, config=config, class_names=class_names
    )

    return nn_model


def evaluate_and_save_best_model(
    best_classical: Any,
    nn_model: torch.nn.Module,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    le: LabelEncoder,
    config: PipelineConfig,
) -> Any:
    """Evaluate both types of models on the test set and save the overall winner.

    Parameters
    ----------
    best_classical : Any
        The best fitted classical model.
    nn_model : torch.nn.Module
        The trained neural network model.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test targets.
    config : PipelineConfig
        Pipeline configuration object.

    Returns
    -------
    Any
        The overall best performing model (either classical or neural).
    """
    print("\n" + "=" * 50)
    print("PHASE: Final Test Set Evaluation & Comparison")
    print("=" * 50)
    
    # After Inversing Check
    y_test_inv = le.inverse_transform(y_test)
    y_test_inv_df = pd.DataFrame({config.target_col: y_test_inv})
    print_class_ratios(y_test_inv_df, target_col=config.target_col, title="Test Set (After Inversing)")

    print("\n--- Classical Model (Best) ---")
    test_metrics_classical, y_pred_cl = evaluate_classical_model(best_classical, X_test, y_test)
    for metric, value in test_metrics_classical.items():
        print(f"  {metric:10s}: {value:.4f}")

    class_names = [str(c) for c in le.classes_]
    display_detailed_report(y_test, y_pred_cl, class_names=class_names, title="Classical Model Report")

    print("\n--- Neural Network ---")
    test_metrics_nn, y_pred_nn = evaluate_nn_model(nn_model, X_test, y_test, config=config)
    for metric, value in test_metrics_nn.items():
        print(f"  {metric:10s}: {value:.4f}")

    display_detailed_report(y_test, y_pred_nn, class_names=class_names, title="Neural Network Report")

    # Determine best overall model
    classical_score = test_metrics_classical.get("ROC-AUC", 0)
    if np.isnan(classical_score):
        classical_score = test_metrics_classical.get("Accuracy", 0)

    nn_score = test_metrics_nn.get("ROC-AUC", 0)
    if np.isnan(nn_score):
        nn_score = test_metrics_nn.get("Accuracy", 0)

    print("\n" + "-" * 30)
    if nn_score > classical_score:
        print(f"Winner: Neural Network (Score: {nn_score:.4f} vs {classical_score:.4f})")
        best_overall = nn_model
        best_model_path = os.path.join(config.models_dir, "best_model.pt")
        torch.save(nn_model.state_dict(), best_model_path)
    else:
        print(
            f"Winner: Classical Model (Score: {classical_score:.4f} vs {nn_score:.4f})"
        )
        best_overall = best_classical
        best_model_path = os.path.join(config.models_dir, "best_model.pkl")
        joblib.dump(best_classical, best_model_path)

    print(f"Designated best model saved to {best_model_path}")

    # Generate final comparison plots for Task 4
    all_labels = np.arange(len(le.classes_))
    model_results = {
        "Classical (Best)": {
            "metrics": test_metrics_classical,
            "cm": confusion_matrix(y_test, y_pred_cl, labels=all_labels),
        },
        "Neural Network": {
            "metrics": test_metrics_nn,
            "cm": confusion_matrix(y_test, y_pred_nn, labels=all_labels),
        },
    }
    class_names = [str(c) for c in le.classes_]
    plot_model_performance(
        model_results,
        visuals_dir=config.visuals_dir,
        metrics_filename=config.comparison_metrics_filename,
        cm_filename=config.comparison_cm_filename,
        main_title="Final Model Comparison on Test Set",
        class_names=class_names,
    )

    # Task 4.2: Side-by-side comparison CSVs
    save_evaluation_tables(
        X_train, X_test, y_test, best_classical, nn_model, config=config
    )

    return best_overall


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(
    config: Optional[PipelineConfig] = None,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder, Any
]:
    """Run the full machine learning pipeline.

    This function coordinates data loading, preprocessing, classical model training,
    neural network training, evaluation, and saving the best overall model.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. A default :class:`PipelineConfig` is used
        when not provided.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, LabelEncoder, Any]
        (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, best_model)
    """
    if config is None:
        config = PipelineConfig()

    configure_plot_style()

    # Phase 1: Data Ingestion and Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, le = (
        run_data_ingestion_and_preprocessing(config)
    )

    # Phase 2: Classical ML Training
    best_classical = run_classical_training(X_train, y_train, X_val, y_val, config, le)

    # Phase 3: Neural Network Training
    nn_model = run_neural_training(X_train, y_train, X_val, y_val, config, le)

    # Phase 4: Final Evaluation and Saving
    best_overall = evaluate_and_save_best_model(
        best_classical, nn_model, X_train, X_test, y_test, le, config
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le, best_overall


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = PipelineConfig()

    print(f"Loading data from: {cfg.filepath}")
    print(f"Original dataset shape: {pd.read_csv(cfg.filepath).shape}")

    X_train, X_val, X_test, y_train, y_val, y_test, le, best_model = main(cfg)

    print("\n--- Pipeline Execution Complete ---")
    print(f"Train Features Shape: {X_train.shape}")
    print(f"Train Columns: {X_train.columns.tolist()}")
    print(f"Val Features Shape:   {X_val.shape}")
    print(f"Test Features Shape:  {X_test.shape}")
    print(f"Combined Features Shape: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")