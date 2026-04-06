import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Any
import os

from main import (
    main,
    PipelineConfig,
    load_and_filter_data,
    fit_preprocessing_params,
    preprocess_split,
    detach_targets,
    configure_plot_style
)

@pytest.fixture
def mock_config(tmp_path: Any) -> PipelineConfig:
    csv_file = tmp_path / "mock_dataset.csv"
    data = {
        'U': [10.0, 11.0, 12.0, 13.0, 10.0, 11.0, 12.0, 13.0] * 5,
        'G': [9.0, 10.0, 11.0, 12.0, 9.0, 10.0, 11.0, 12.0] * 5,
        'R': [8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0] * 5,
        'WF3': [15.0, 16.0, 17.0, 18.0, 15.0, 16.0, 17.0, 18.0] * 5,
        'UT': [14.0, 15.0, 16.0, 17.0, 14.0, 15.0, 16.0, 17.0] * 5,
        'WF1': [11.0, 12.0, 13.0, 14.0, 11.0, 12.0, 13.0, 14.0] * 5,
        'T': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 5,
        'E_T': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] * 5,
        'CLASS_SP': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'] * 5,
        'METAL': [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.8] * 5,
        'FLAG_METAL': [0, 1, 0, 1, 0, 1, 0, 1] * 5 # DONT USE -1 SO THEY ALL SURVIVE
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    
    return PipelineConfig(
        filepath=str(csv_file),
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        visuals_dir=str(tmp_path / "visuals"),
        models_dir=str(tmp_path / "models")
    )


def test_pipeline_config_validation() -> None:
    """Test PipelineConfig validation logic."""
    with pytest.raises(ValueError):
        PipelineConfig(train_size=0.5, val_size=0.2, test_size=0.2) # Sums to 0.9
        
    config = PipelineConfig(key_vars=["T", "WF1"])
    assert config.err_vars == ["E_T", "E_WF1"]


def test_load_and_filter_data(mock_config: PipelineConfig) -> None:
    """Test data loading and initial filtering."""
    df = load_and_filter_data(mock_config)
    assert isinstance(df, pd.DataFrame)
    assert "T" in df.columns
    assert "CLASS_SP" in df.columns
    # Check that it didn't crash and returned reasonably filtered columns
    assert len(df.columns) <= (len(mock_config.key_vars) + len(mock_config.err_vars) + 
                               len(mock_config.no_err_vars) + len(mock_config.flags_vars) + 
                               len(mock_config.color_vars) + 1) # +1 for FLAG_METAL if it survived


@patch('main.fit_target_encoder')
@patch('main.compute_iqr_bounds')
@patch('main.get_fitted_scaler')
@patch('main.build_preprocessing_pipeline')
@patch('joblib.dump')
def test_fit_preprocessing_params(
    mock_dump: MagicMock,
    mock_build_pipe: MagicMock,
    mock_get_scaler: MagicMock,
    mock_iqr: MagicMock,
    mock_fit_target: MagicMock,
    mock_config: PipelineConfig
) -> None:
    """Test fitting preprocessing parameters."""
    train_df = pd.read_csv(mock_config.filepath)
    
    # Mock scaler with a transform method that returns a numpy array
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((len(train_df), 1))
    
    # Mock returns
    mock_fit_target.return_value = MagicMock()
    mock_iqr.return_value = {"T": (0, 10)}
    mock_get_scaler.return_value = (mock_scaler, pd.Index(['T']))
    mock_build_pipe.return_value = MagicMock()
    
    iqr, scaler, num_cols, pipe, le = fit_preprocessing_params(train_df, mock_config)
    
    assert iqr == {"T": (0, 10)}
    assert num_cols.tolist() == ['T']
    mock_dump.assert_called()


def test_style_configuration() -> None:
    """Test plot style configuration."""
    with patch('matplotlib.pyplot.style.use') as mock_style:
        configure_plot_style()
        mock_style.assert_called_once_with('bmh')


def test_preprocess_split(mock_config: PipelineConfig) -> None:
    """Test the preprocessing of a single split."""
    data = {
        'T': [1.0, 2.0], 'WF1': [10.0, 20.0], 'E_T': [0.1, 0.2], 'E_WF1': [1.0, 2.0],
        'CLASS_SP': ['A', 'B'], 'METAL': [0.5, 0.6]
    }
    df = pd.DataFrame(data)
    
    iqr_bounds = {'T': (0, 10)}
    scaler = MagicMock()
    # scaler.transform(X) should return a numpy array
    scaler.transform.return_value = np.array([[1.0, 10.0], [2.0, 20.0]])
    
    num_cols = pd.Index(['T', 'WF1'])
    
    # Mocking the pipeline to return what it's given
    pipeline = MagicMock()
    pipeline.get_feature_names_out.return_value = ['num__T', 'num__WF1']
    pipeline.transform.return_value = np.array([[1.0, 10.0], [2.0, 20.0]])
    
    le = MagicMock()
    le.transform.return_value = [0, 1]
    
    processed_df = preprocess_split(
        df, "test", iqr_bounds, scaler, num_cols, pipeline, le, mock_config
    )
    
    assert "CLASS_SP" in processed_df.columns
    assert "u-g" not in processed_df.columns # No color columns in input, but compute_colors handles it gracefully
    assert len(processed_df) == 2


@patch('main.load_and_filter_data')
@patch('main.split_data')
@patch('main.fit_preprocessing_params')
@patch('main.preprocess_split')
@patch('main.train_classical_models')
@patch('main.evaluate_model')
def test_main_orchestration(
    mock_eval: MagicMock,
    mock_train: MagicMock,
    mock_preprocess: MagicMock,
    mock_fit: MagicMock,
    mock_split: MagicMock,
    mock_load: MagicMock,
    mock_config: PipelineConfig
) -> None:
    """Test end-to-end orchestration in main function."""
    df = pd.DataFrame({'T': [1]*10, 'WF1': [2]*10, 'WF2': [3]*10, 'CLASS_SP': ['A']*10})
    mock_load.return_value = df
    mock_split.return_value = (df.iloc[:6], df.iloc[6:8], df.iloc[8:])
    mock_fit.return_value = ({}, MagicMock(), pd.Index(['T', 'WF1', 'WF2']), MagicMock(), MagicMock())
    
    # Use side_effect to return new copies of the DataFrame to avoid pop() affecting subsequent calls
    def mock_preprocess_side_effect(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({'T': [1]*5, 'WF1': [2]*5, 'WF2': [3]*5, 'CLASS_SP': [0]*5})
    
    mock_preprocess.side_effect = mock_preprocess_side_effect
    
    mock_eval.return_value = ({'Accuracy': 1.0}, np.array([0]*5))
    
    X_train, X_val, X_test, y_train, y_val, y_test, best_model = main(mock_config)
    
    assert mock_load.called
    assert mock_split.called
    assert mock_train.called
    assert mock_eval.called
    assert len(X_train) == 5
    assert len(X_val) == 5
    assert len(X_test) == 5


def test_detach_targets() -> None:
    """Test detaching target column."""
    df = pd.DataFrame({'feat': [1, 2], 'target': [0, 1]})
    X, y = detach_targets(df, 'target')
    assert 'feat' in X.columns
    assert 'target' not in X.columns
    assert list(y) == [0, 1]


@patch('main.load_and_filter_data')
@patch('main.split_data')
@patch('main.fit_preprocessing_params')
@patch('main.preprocess_split')
@patch('main.train_classical_models')
@patch('main.evaluate_model')
def test_main_no_config(
    mock_eval: MagicMock,
    mock_train: MagicMock,
    mock_preprocess: MagicMock,
    mock_fit: MagicMock,
    mock_split: MagicMock,
    mock_load: MagicMock,
) -> None:
    """Test main function when no config is provided."""
    df = pd.DataFrame({'T': [1]*30, 'WF1': [2]*30, 'WF2': [3]*30, 'CLASS_SP': ['A']*30})
    mock_load.return_value = df
    mock_split.return_value = (df.iloc[:20], df.iloc[20:25], df.iloc[25:])
    mock_fit.return_value = ({}, MagicMock(), pd.Index(['T', 'WF1', 'WF2']), MagicMock(), MagicMock())
    mock_preprocess.return_value = pd.DataFrame({'T': [1]*5, 'WF1': [2]*5, 'WF2': [3]*5, 'CLASS_SP': [0]*5})
    mock_eval.return_value = ({'Accuracy': 1.0}, np.array([0]*5))
    
    # This should call main() which will create a default PipelineConfig
    def mock_preprocess_side_effect(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({'T': [1]*5, 'WF1': [2]*5, 'WF2': [3]*5, 'CLASS_SP': [0]*5})
    
    mock_preprocess.side_effect = mock_preprocess_side_effect
    
    with patch('pandas.read_csv') as mock_read:
        mock_read.return_value = df
        main(config=None)
    
    assert mock_load.called


def test_main_integration(mock_config: PipelineConfig) -> None:
    """Test main function end-to-end without mocking internal pipeline steps.
    
    This ensures that load -> split -> fit -> preprocess -> train -> evaluate
    all work together with real data structures.
    """
    # Adjust config for faster integration testing if needed, 
    # but the current mock_config is small enough.
    
    # Use patch context managers to mock I/O/visuals but let main logic run naturally.
    with patch('matplotlib.pyplot.savefig'): 
        with patch('sys.stdout', new=MagicMock()): 
             X_train, X_val, X_test, y_train, y_val, y_test, best_model = main(mock_config)
    
    # Check outputs are correct types and shapes
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert len(X_train) > 0
    assert best_model is not None
    # Verify files were 'saved' (models_dir should have been created and used)
    assert os.path.exists(mock_config.models_dir)
    assert os.path.exists(os.path.join(mock_config.models_dir, "scaler.pkl"))
    assert os.path.exists(os.path.join(mock_config.models_dir, "classical_model.pkl"))
