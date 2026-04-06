import pandas as pd
import numpy as np
import pytest
import os
from typing import Dict, Tuple, List, Optional, Any

# Adjusting import path assuming the script is run from project root or pytest config handles paths
from src.preprocessing import (
    compute_colors,
    filter_important_columns,
    filter_error_ratios,
    select_metal_flag,
    drop_missing_targets,
    compute_iqr_bounds,
    apply_iqr_capping,
    split_data,
    build_preprocessing_pipeline,
    fit_target_encoder,
    apply_target_encoder,
    apply_imputation,
    get_fitted_scaler,
    apply_scaling,
    save_outlier_histograms,
    generate_pca_insights
)
from main import KEY_VARS, ERR_VARS, NO_ERR_VARS, FLAGS_VARS, COLOR_VARS

@pytest.fixture
def sample_df() -> pd.DataFrame:
    data = {
        'U': [10.0, 11.0, 12.0, np.nan, 10.0],
        'G': [9.0, 10.0, 11.0, 9.0, 9.0],
        'R': [8.0, 9.0, 10.0, 8.0, 8.0],
        'WF3': [15.0, 16.0, 17.0, 15.0, 15.0],
        'UT': [14.0, 15.0, 16.0, 14.0, 14.0],
        'WF1': [11.0, 12.0, 13.0, 11.0, 11.0],
        'T': [1.0, 2.0, 3.0, 4.0, 5.0],
        'E_T': [0.1, 0.2, 0.3, 0.4, 0.5],
        'CLASS_SP': ['A', 'B', np.nan, 'A', 'B'],
        'METAL': [0.5, 0.6, 0.7, 0.8, 0.9],
        'FLAG_METAL': [-1, 0, -1, 1, -1] # Three -1s, one 0, one 1
    }
    return pd.DataFrame(data)


def test_compute_colors(sample_df: pd.DataFrame) -> None:
    df = compute_colors(sample_df)
    assert 'u-g' in df.columns
    assert 'g-r' in df.columns
    assert 'W3-UT' in df.columns
    assert '(W3+UT)/W1' in df.columns
    assert df['u-g'].iloc[0] == 1.0
    assert df['g-r'].iloc[0] == 1.0


def test_filter_important_columns(sample_df: pd.DataFrame) -> None:
    # compute colors first so that filter keeps them
    df = compute_colors(sample_df)
    filtered = filter_important_columns(df, KEY_VARS, ERR_VARS, NO_ERR_VARS, FLAGS_VARS, COLOR_VARS)
    
    # Check that required columns are kept
    assert 'T' in filtered.columns
    assert 'CLASS_SP' in filtered.columns
    
    # Check that METAL implicitly pulls FLAG_METAL
    assert 'FLAG_METAL' in filtered.columns
    
    # Check color columns were retained
    assert 'u-g' in filtered.columns


def test_filter_error_ratios(sample_df: pd.DataFrame) -> None:
    """
    sample_df has initial T/E_T values scaling linearly:
    Row 0: T:1, E_T:0.1 -> ratio is abs(1/0.1)*100 = 1000 (>3)
    Let's inject a row that should be DROPPED due to < 3 error ratio:
    """
    df = sample_df.copy()
    # Replace last row with a 0.02 / 1.0 (ratio 2)
    df.loc[df.index[-1], 'T'] = 0.02
    df.loc[df.index[-1], 'E_T'] = 1.0
    
    filtered_df = filter_error_ratios(df, KEY_VARS, ERR_VARS)
    
    # 5 rows original, last row drops due to ratio == 2 < 3. 
    # Therefore we expect 4 records.
    assert len(filtered_df) == 4


def test_select_metal_flag(sample_df: pd.DataFrame) -> None:
    # Using frac=1.0 ensures all -1 records are returned for predictable testing
    reduced = select_metal_flag(sample_df, frac=1.0)
    assert len(reduced) == 5
    
    # Using frac=0.0 excludes all -1 records
    reduced_small = select_metal_flag(sample_df, frac=0.0)
    assert len(reduced_small) == 2


def test_drop_missing_targets(sample_df: pd.DataFrame) -> None:
    dropped_df = drop_missing_targets(sample_df, target_col='CLASS_SP')
    assert len(dropped_df) == 4
    assert dropped_df['CLASS_SP'].isna().sum() == 0


def test_split_data(sample_df: pd.DataFrame) -> None:
    # Multiply the dataset so there are enough samples generated for the nested stratify splits
    # Notice this operates raw df natively.
    large_df = pd.concat([sample_df] * 4, ignore_index=True)
    
    # 4 clean instances originally per sample (out of 5), times 4 = 16 clean records technically, 
    # but initially splitting against full 20.
    train_df, val_df, test_df = split_data(
        large_df, target_col='CLASS_SP', train_size=0.5, val_size=0.25, test_size=0.25, random_state=42
    )
    
    assert len(train_df) == 10
    assert len(val_df) == 5
    assert len(test_df) == 5


def test_iqr_capping(sample_df: pd.DataFrame) -> None:
    # Set explicitly identifiable upper and lower boundary outliers
    df = sample_df.copy()
    
    # Normally distributed 1-5, but let's push T to extreme values
    df.loc[0, 'T'] = 1000.0   # Upper bound breaking
    df.loc[1, 'T'] = -1000.0  # Lower bound breaking
    
    X = df.drop(columns=['CLASS_SP'])
    
    bounds = compute_iqr_bounds(X, factor=1.5)
    
    assert 'T' in bounds
    lower, upper = bounds['T']
    
    # Ensure they represent capped intervals correctly (meaning our hardcoded extremes aren't retained natively)
    capped_X = apply_iqr_capping(X, bounds)
    assert capped_X.loc[0, 'T'] <= upper
    assert capped_X.loc[1, 'T'] >= lower
    
    # Confirm regular values inside parameters remained intact natively
    assert capped_X.loc[2, 'T'] == 3.0


def test_build_preprocessing_pipeline(sample_df: pd.DataFrame) -> None:
    X = sample_df.drop(columns=['CLASS_SP'])
    # Need mock y for TargetEncoder fit logic
    y = sample_df['CLASS_SP'].fillna('A')
    
    pipeline = build_preprocessing_pipeline(X)
    
    pipeline.fit(X, y)
    X_transformed = pipeline.transform(X)
    
    # Ensures transformation executed mathematically. Column counts might change marginally due to OneHot encoders 
    # dropping columns inherently or widening based on parameters natively.
    assert X_transformed.shape[0] == X.shape[0]
    
    raw_cols = pipeline.get_feature_names_out()
    out_columns = [c.replace('num__', '').replace('bin__', '').replace('nom__', '').replace('hc__', '') for c in raw_cols]
    
    df_transformed = pd.DataFrame(X_transformed, columns=out_columns)
    
    # Ensure missing numeric values (the 'U' column logic) were imputed correctly
    assert not df_transformed['U'].isnull().any()


def test_target_encoder(sample_df: pd.DataFrame, tmp_path: Any) -> None:
    # Use a temporary directory for model saving
    df = drop_missing_targets(sample_df, target_col='CLASS_SP')
    with open(os.devnull, 'w') as f: # Suppress print
        le = fit_target_encoder(df, target_col='CLASS_SP')
    
    assert le is not None
    assert 'A' in le.classes_
    
    encoded_df = apply_target_encoder(df, target_col='CLASS_SP', le=le)
    assert encoded_df['CLASS_SP'].iloc[0] == 0 # 'A' should be 0
    assert encoded_df['CLASS_SP'].dtype == np.int64 or encoded_df['CLASS_SP'].dtype == np.int32


def test_apply_imputation(sample_df: pd.DataFrame) -> None:
    X = sample_df.drop(columns=['CLASS_SP'])
    y = sample_df['CLASS_SP'].fillna('A')
    pipeline = build_preprocessing_pipeline(X)
    pipeline.fit(X, y)
    
    imputed_df = apply_imputation(sample_df, pipeline, target_col='CLASS_SP')
    assert 'CLASS_SP' in imputed_df.columns
    assert not imputed_df.drop(columns=['CLASS_SP']).isnull().any().any()


def test_scaling(sample_df: pd.DataFrame) -> None:
    # Prepare numeric data
    df = sample_df.drop(columns=['CLASS_SP'])
    # Add a mock class for application
    df['CLASS_SP'] = [0, 1, 0, 1, 0]
    
    scaler, num_cols = get_fitted_scaler(df.drop(columns=['CLASS_SP']))
    assert scaler is not None
    assert 'U' in num_cols
    
    scaled_df = apply_scaling(df, scaler, num_cols, target_col='CLASS_SP')
    assert 'CLASS_SP' in scaled_df.columns
    # Check if a value is scaled (U=10.0, etc, mean would be around 10.75, so scaled != 10.0)
    assert scaled_df['U'].iloc[0] != 10.0


def test_save_outlier_histograms(sample_df: pd.DataFrame, tmp_path: Any) -> None:
    # Should run without error and respect out_dir
    visuals_dir = tmp_path / "visuals"
    res = save_outlier_histograms(sample_df, prefix="test", out_dir=str(visuals_dir))
    assert res.equals(sample_df)
    # Check if directory was created
    assert os.path.exists(str(visuals_dir))


def test_pca_insights(sample_df: pd.DataFrame, tmp_path: Any) -> None:
    # Test PCA generation doesn't crash
    visuals_dir = tmp_path / "pca_test"
    X = sample_df.drop(columns=['CLASS_SP']).select_dtypes(include=[np.number]).fillna(0)
    y = sample_df['CLASS_SP'].fillna('A')
    
    generate_pca_insights(X, y, out_dir=str(visuals_dir))
    assert os.path.exists(str(visuals_dir))
