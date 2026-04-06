import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('bmh')
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0, 1, 4)))
# ---------------------------------------------------------
# Global Configuration: Data Split Percentages
# ---------------------------------------------------------
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10
TARGET_COL = 'CLASS_SP'

KEY_VARS = ['T', 'WF1', 'WF2', 'WF3', 'WF4', 'UT', 'BT', 'VT', 'IT', 'U', 'R', 'G', 'I', 'Z']
ERR_VARS = [f'E_{var}' for var in KEY_VARS]
NO_ERR_VARS = ['CLASS_SP', 'AGN_HEC', 'logM_HEC', 'logSFR_HEC']
FLAGS_VARS = ['METAL']
COLOR_VARS = ['u-g', 'g-r', 'W3-UT', '(W3+UT)/W1']

if round(TRAIN_SIZE + VAL_SIZE + TEST_SIZE, 5) != 1.0:
    raise ValueError("The sum of TRAIN_SIZE, VAL_SIZE, and TEST_SIZE must be exactly equal to 1.0")

import joblib

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
    generate_pca_insights
)

def load_and_filter_data(filepath, key_vars, err_vars, no_err_vars, flags_vars, color_vars):
    """Loads raw data, filters important columns, and selects based on metal flag."""
    raw_df = pd.read_csv(filepath)
    df_prepared = (
        raw_df
        .pipe(filter_important_columns, key_vars=key_vars, err_vars=err_vars, no_err_vars=no_err_vars, flags_vars=flags_vars, color_vars=color_vars)
        .pipe(select_metal_flag)
    )
    return df_prepared

def fit_preprocessing_params(train_df, target_col):
    """Calculates bounds, fits independent scaling, and formulates imputation pipeline on training data."""
    le = fit_target_encoder(train_df, target_col)
    
    iqr_bounds = compute_iqr_bounds(train_df.drop(columns=[target_col]), factor=1.5)
    
    train_temp = train_df.pipe(drop_missing_targets, target_col=target_col).pipe(apply_target_encoder, target_col=target_col, le=le).pipe(apply_iqr_capping, bounds=iqr_bounds)
    
    scaler, num_cols = get_fitted_scaler(train_temp.drop(columns=[target_col]))
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    
    train_temp_scaled = train_temp.pipe(apply_scaling, scaler=scaler, num_cols=num_cols, target_col=target_col)
    
    pipeline = build_preprocessing_pipeline(train_temp_scaled.drop(columns=[target_col]))
    pipeline.fit(train_temp_scaled.drop(columns=[target_col]), train_temp_scaled[target_col])
    
    return iqr_bounds, scaler, num_cols, pipeline, le

def preprocess_pipeline(df, split_name, target_col, iqr_bounds, scaler, num_cols, pipeline, le, key_vars, err_vars, visuals_dir):
    """Unified post-split architecture natively executed entirely in strict Pipeline format."""
    return (
        df
        .pipe(drop_missing_targets, target_col=target_col)
        .pipe(apply_target_encoder, target_col=target_col, le=le)
        .pipe(apply_iqr_capping, bounds=iqr_bounds)
        .pipe(save_outlier_histograms, prefix=split_name, out_dir=visuals_dir)
        .pipe(apply_scaling, scaler=scaler, num_cols=num_cols, target_col=target_col)
        .pipe(apply_imputation, pipeline=pipeline, target_col=target_col)
        .pipe(filter_error_ratios, key_vars=key_vars, err_vars=err_vars)
        .pipe(compute_colors)
    )

def detach_targets(clean_df, target_col):
    """Detaches target features natively explicitly."""
    y = clean_df.pop(target_col)
    return clean_df, y

def main(
    filepath="data/HECATE.csv", 
    target_col=TARGET_COL, 
    train_size=TRAIN_SIZE, 
    val_size=VAL_SIZE, 
    test_size=TEST_SIZE,
    key_vars=KEY_VARS,
    err_vars=ERR_VARS,
    no_err_vars=NO_ERR_VARS,
    flags_vars=FLAGS_VARS,
    color_vars=COLOR_VARS,
    visuals_dir="visuals"
):
    """Main execution function matching proper separated execution strategies."""
    df_prepared = load_and_filter_data(filepath, key_vars, err_vars, no_err_vars, flags_vars, color_vars)
    
    train_df, val_df, test_df = split_data(
        df_prepared, 
        target_col=target_col, 
        train_size=train_size, 
        val_size=val_size, 
        test_size=test_size
    )
    
    iqr_bounds, scaler, num_cols, pipeline, le = fit_preprocessing_params(train_df, target_col)
    
    train_clean = preprocess_pipeline(train_df, 'train', target_col, iqr_bounds, scaler, num_cols, pipeline, le, key_vars, err_vars, visuals_dir)
    val_clean = preprocess_pipeline(val_df, 'val', target_col, iqr_bounds, scaler, num_cols, pipeline, le, key_vars, err_vars, visuals_dir)
    test_clean = preprocess_pipeline(test_df, 'test', target_col, iqr_bounds, scaler, num_cols, pipeline, le, key_vars, err_vars, visuals_dir)

    X_train, y_train = detach_targets(train_clean, target_col)
    X_val, y_val = detach_targets(val_clean, target_col)
    X_test, y_test = detach_targets(test_clean, target_col)
    
    if visuals_dir:
        generate_pca_insights(X_train, y_train, out_dir=visuals_dir)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "data/HECATE.csv")
    
    print(f"Loading data from: {data_path}")
    print(f"Original dataset shape: {pd.read_csv(data_path).shape}")
    X_train, X_val, X_test, y_train, y_val, y_test = main(data_path)
    
    print("\n--- Pipeline Execution Complete ---")
    print(f"Train Features Shape: {X_train.shape}")
    print(f"Train Columns: {X_train.columns.tolist()}")
    print(f"Val Features Shape:   {X_val.shape}")
    print(f"Test Features Shape:  {X_test.shape}")
    print(f"Combined Features Shape: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")