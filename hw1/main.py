import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from src.config import PipelineConfig
from src.preprocessing import (
    compute_colors,
    filter_error_ratios,
    filter_important_columns,
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
)
from src.train_classical import train_classical_models, evaluate_model


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

    scaler, num_cols = get_fitted_scaler(train_temp.drop(columns=[config.target_col]))

    os.makedirs(config.models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(config.models_dir, config.scaler_filename))

    train_temp_scaled = train_temp.pipe(
        apply_scaling, scaler=scaler, num_cols=num_cols, target_col=config.target_col
    )

    pipeline = build_preprocessing_pipeline(
        train_temp_scaled.drop(columns=[config.target_col])
    )
    pipeline.fit(
        train_temp_scaled.drop(columns=[config.target_col]),
        train_temp_scaled[config.target_col],
    )

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
        .pipe(apply_target_encoder, target_col=config.target_col, le=le)
        .pipe(apply_iqr_capping, bounds=iqr_bounds)
        .pipe(save_outlier_histograms, prefix=split_name, out_dir=config.visuals_dir)
        .pipe(apply_scaling, scaler=scaler, num_cols=num_cols, target_col=config.target_col)
        .pipe(apply_imputation, pipeline=pipeline, target_col=config.target_col)
        .pipe(filter_error_ratios, key_vars=config.key_vars, err_vars=config.err_vars)
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
# Main entry point
# ---------------------------------------------------------------------------

def main(
    config: Optional[PipelineConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Any]:
    """Run the full preprocessing and classical ML training pipeline.

    Parameters
    ----------
    config : PipelineConfig, optional
        Pipeline configuration. A default :class:`PipelineConfig` is used
        when not provided.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Any]
        ``(X_train, X_val, X_test, y_train, y_val, y_test, best_model)``
    """
    if config is None:
        config = PipelineConfig()

    configure_plot_style()

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

    if config.visuals_dir:
        generate_pca_insights(
            X_train,
            y_train,
            out_dir=config.visuals_dir,
            scree_filename=config.scree_filename,
            projection_filename=config.projection_filename,
        )

    print("\n" + "=" * 50)
    print("PHASE: Classical ML Training & Grid Search")
    print("=" * 50)
    best_model = train_classical_models(
        X_train, y_train, X_val, y_val,
        random_state=config.random_state,
        models_dir=config.models_dir,
        visuals_dir=config.visuals_dir,
        metrics_filename=config.metrics_filename,
        cm_filename=config.cm_filename,
        model_filename=config.model_filename,
        xgb_early_stopping_rounds=config.xgb_early_stopping_rounds,
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
    )

    print("\n" + "=" * 50)
    print("PHASE: Final Test Set Evaluation")
    print("=" * 50)
    test_metrics, _ = evaluate_model(best_model, X_test, y_test)

    print("\nBest Model Performance on Test Set:")
    for metric, value in test_metrics.items():
        print(f"  {metric:10s}: {value:.4f}")

    return X_train, X_val, X_test, y_train, y_val, y_test, best_model


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = PipelineConfig()

    print(f"Loading data from: {cfg.filepath}")
    print(f"Original dataset shape: {pd.read_csv(cfg.filepath).shape}")

    X_train, X_val, X_test, y_train, y_val, y_test, best_model = main(cfg)

    print("\n--- Pipeline Execution Complete ---")
    print(f"Train Features Shape: {X_train.shape}")
    print(f"Train Columns: {X_train.columns.tolist()}")
    print(f"Val Features Shape:   {X_val.shape}")
    print(f"Test Features Shape:  {X_test.shape}")
    print(f"Combined Features Shape: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")