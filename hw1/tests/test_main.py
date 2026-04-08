import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from typing import Any
import os

pytest.importorskip("torch")
pytest.importorskip("xgboost")

from main import (
    main,
    load_and_filter_data,
    fit_preprocessing_params,
    preprocess_split,
    detach_targets,
    configure_plot_style,
    run_data_ingestion_and_preprocessing,
    run_classical_training,
    run_neural_training,
    evaluate_and_save_best_model,
)
from src.config import PipelineConfig

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

def test_load_and_filter_data(mock_config: PipelineConfig) -> None:
    """Test data loading and initial filtering."""
    df = load_and_filter_data(mock_config)
    assert isinstance(df, pd.DataFrame)
    assert "T" in df.columns
    assert "CLASS_SP" in df.columns
    assert len(df.columns) <= (len(mock_config.key_vars) + len(mock_config.err_vars) + 
                               len(mock_config.no_err_vars) + len(mock_config.flags_vars) + 
                               len(mock_config.color_vars) + 1)

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
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((len(train_df), 1))
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
    scaler.transform.return_value = np.array([[1.0, 10.0], [2.0, 20.0]])
    num_cols = pd.Index(['T', 'WF1'])
    pipeline = MagicMock()
    pipeline.get_feature_names_out.return_value = ['num__T', 'num__WF1']
    pipeline.transform.return_value = np.array([[1.0, 10.0], [2.0, 20.0]])
    le = MagicMock()
    le.transform.return_value = [0, 1]
    processed_df = preprocess_split(
        df, "test", iqr_bounds, scaler, num_cols, pipeline, le, mock_config
    )
    assert "CLASS_SP" in processed_df.columns
    assert len(processed_df) == 2

@patch('main.load_and_filter_data')
@patch('main.split_data')
@patch('main.fit_preprocessing_params')
@patch('main.preprocess_split')
@patch('main.train_classical_models')
@patch('main.evaluate_classical_model')
@patch('main.train_neural_network')
@patch('main.evaluate_nn_model')
@patch('main.plot_nn_training_history')
@patch('main.plot_nn_evaluation')
@patch('main.plot_model_performance')
@patch('main.save_evaluation_tables')
@patch('joblib.dump')
@patch('torch.save')
def test_main_orchestration(
    mock_torch_save: MagicMock,
    mock_joblib_dump: MagicMock,
    mock_save_tables: MagicMock,
    mock_plot_perf: MagicMock,
    mock_plot_nn_eval: MagicMock,
    mock_plot_nn_hist: MagicMock,
    mock_eval_nn: MagicMock,
    mock_train_nn: MagicMock,
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
    
    def mock_preprocess_side_effect(*args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame({'T': [1]*5, 'WF1': [2]*5, 'WF2': [3]*5, 'CLASS_SP': [0]*5})
    mock_preprocess.side_effect = mock_preprocess_side_effect
    
    mock_eval.return_value = ({'Accuracy': 1.0, 'ROC-AUC': 0.9}, np.array([0]*5))
    mock_train_nn.return_value = (MagicMock(), {"train_loss": [0.1], "val_loss": [0.1]})
    mock_eval_nn.return_value = ({'Accuracy': 0.9, 'ROC-AUC': 0.85}, np.array([0]*5))

    X_train, X_val, X_test, y_train, y_val, y_test, le_out, best_model = main(mock_config)
    
    assert mock_load.called
    assert len(X_train) == 5
    assert best_model is not None

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
@patch('main.evaluate_classical_model')
@patch('main.train_neural_network')
@patch('main.evaluate_nn_model')
@patch('main.plot_nn_training_history')
@patch('main.plot_nn_evaluation')
@patch('main.plot_model_performance')
@patch('main.save_evaluation_tables')
@patch('joblib.dump')
@patch('torch.save')
def test_main_no_config(
    mock_torch_save: MagicMock,
    mock_joblib_dump: MagicMock,
    mock_save_tables: MagicMock,
    mock_plot_perf: MagicMock,
    mock_plot_nn_eval: MagicMock,
    mock_plot_nn_hist: MagicMock,
    mock_eval_nn: MagicMock,
    mock_train_nn: MagicMock,
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
    mock_train_nn.return_value = (MagicMock(), {"train_loss": [0.1], "val_loss": [0.1]})
    mock_eval_nn.return_value = ({'Accuracy': 0.9, 'ROC-AUC': 0.85}, np.array([0]*5))
    
    with patch('pandas.read_csv') as mock_read:
        mock_read.return_value = df
        main(config=None)
    assert mock_load.called

def test_main_integration(mock_config: PipelineConfig) -> None:
    """Test main function end-to-end without mocking internal pipeline steps."""
    with patch('matplotlib.pyplot.savefig'): 
        mock_stdout = MagicMock()
        mock_stdout.encoding = 'utf-8'
        with patch('sys.stdout', new=mock_stdout): 
             X_train, X_val, X_test, y_train, y_val, y_test, le_out, best_model = main(mock_config)
    assert isinstance(X_train, pd.DataFrame)
    assert best_model is not None

def test_pipeline_config_explicit_err_vars() -> None:
    """PipelineConfig should NOT override err_vars when they are explicitly provided."""
    explicit = ['E_custom1', 'E_custom2']
    config = PipelineConfig(key_vars=['T', 'WF1'], err_vars=explicit)
    assert config.err_vars == explicit

def test_load_and_filter_data_no_flag_metal(tmp_path: Any) -> None:
    """load_and_filter_data should not crash when FLAG_METAL column is absent."""
    csv_file = tmp_path / "no_flag.csv"
    data = {'T': [1.0, 2.0], 'E_T': [0.1, 0.2], 'CLASS_SP': ['A', 'B'], 'METAL': [0.5, 0.6]}
    pd.DataFrame(data).to_csv(csv_file, index=False)
    config = PipelineConfig(filepath=str(csv_file), key_vars=['T'], no_err_vars=['CLASS_SP'], flags_vars=['METAL'], color_vars=[])
    df = load_and_filter_data(config)
    assert 'FLAG_METAL' not in df.columns

@patch("main.load_and_filter_data")
@patch("main.split_data")
@patch("main.fit_preprocessing_params")
@patch("main.preprocess_split")
@patch("main.generate_pca_insights")
def test_run_data_ingestion_and_preprocessing(
    mock_pca: MagicMock,
    mock_preprocess: MagicMock,
    mock_fit: MagicMock,
    mock_split: MagicMock,
    mock_load: MagicMock,
    mock_config: PipelineConfig,
) -> None:
    """Test the data ingestion and preprocessing phase."""
    df = pd.DataFrame({"T": [1] * 10, "CLASS_SP": [0] * 10})
    mock_load.return_value = df
    mock_split.return_value = (df.iloc[:6], df.iloc[6:8], df.iloc[8:])
    mock_fit.return_value = ({}, MagicMock(), pd.Index(["T"]), MagicMock(), MagicMock())
    mock_preprocess.return_value = pd.DataFrame({"T": [1] * 5, "CLASS_SP": [0] * 5})
    results = run_data_ingestion_and_preprocessing(mock_config)
    assert len(results) == 7
    assert mock_load.called

@patch("main.train_classical_models")
def test_run_classical_training(mock_train: MagicMock, mock_config: PipelineConfig) -> None:
    """Test the classical training phase."""
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1])
    le = MagicMock()
    run_classical_training(X, y, X, y, mock_config, le)
    assert mock_train.called

@patch("main.train_neural_network")
@patch("main.plot_nn_training_history")
@patch("main.evaluate_nn_model")
@patch("main.plot_nn_evaluation")
def test_run_neural_training(
    mock_plot_eval: MagicMock,
    mock_eval: MagicMock,
    mock_plot_hist: MagicMock,
    mock_train: MagicMock,
    mock_config: PipelineConfig,
) -> None:
    """Test the neural network training phase."""
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1])
    mock_train.return_value = (MagicMock(), {"loss": [0.1]})
    mock_eval.return_value = ({"Accuracy": 0.9}, np.array([0, 1]))
    le = MagicMock()
    run_neural_training(X, y, X, y, mock_config, le)
    assert mock_train.called

@patch("main.evaluate_classical_model")
@patch("main.evaluate_nn_model")
@patch("main.plot_model_performance")
@patch("main.save_evaluation_tables")
@patch("joblib.dump")
@patch("torch.save")
def test_evaluate_and_save_best_model(
    mock_torch_save: MagicMock,
    mock_joblib_dump: MagicMock,
    mock_save_tables: MagicMock,
    mock_plot_perf: MagicMock,
    mock_eval_nn: MagicMock,
    mock_eval_cl: MagicMock,
    mock_config: PipelineConfig,
) -> None:
    """Test evaluation and saving the best model."""
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1])
    le = MagicMock()
    mock_eval_cl.return_value = ({"ROC-AUC": 0.9}, np.array([0, 1]))
    mock_eval_nn.return_value = ({"ROC-AUC": 0.8}, np.array([0, 1]))
    evaluate_and_save_best_model(MagicMock(), MagicMock(), X, X, y, le, mock_config)
    assert mock_plot_perf.called
