import pytest
import pandas as pd
import numpy as np
from main import main

def test_run_pipeline_integration(tmp_path) -> None:
    """
    Validates the end-to-end orchestration in main.py simulating the exact read/write logic flawlessly natively.
    """
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
        'FLAG_METAL': [-1, 0, -1, 1, -1]
    }
    sample_df = pd.DataFrame(data)
    
    # Multiply dataset reliably ensuring sufficient scale survives the end-to-end multi-drop filters safely.
    # drop_missing_targets and filter_error_ratios slice aggressively, so multiplying by 20 gives resilient buffers.
    large_df = pd.concat([sample_df] * 20, ignore_index=True)
    
    # Isolate testing completely functionally to safe temporary bounds
    csv_file = tmp_path / "mock_dataset.csv"
    large_df.to_csv(csv_file, index=False)
    
    X_train, X_val, X_test, y_train, y_val, y_test, best_model = main(
        filepath=str(csv_file),
        train_size=0.5, 
        val_size=0.25, 
        test_size=0.25,
        visuals_dir=None
    )
    
    # Validate final shapes natively and logically confirming split integrity correctly executed everything
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0
    assert best_model is not None
    
    assert len(X_train) == len(y_train)
    assert 'CLASS_SP' not in X_train.columns
    assert 'u-g' in X_train.columns # Ensures compute_colors successfully passed through
