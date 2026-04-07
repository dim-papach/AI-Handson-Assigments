import pytest
from src.config import PipelineConfig, DataConfig, SplitConfig, TuningConfig

def test_pipeline_config_defaults() -> None:
    """Test that PipelineConfig initializes with default values."""
    config = PipelineConfig()
    assert config.target_col == PipelineConfig.target_col
    assert config.train_size == PipelineConfig.train_size
    assert config.val_size == PipelineConfig.val_size
    assert config.test_size == PipelineConfig.test_size
    assert config.random_state == PipelineConfig.random_state
    # Ensure err_vars are auto-derived
    assert len(config.err_vars) == len(config.key_vars)
    assert config.err_vars[0] == f"E_{config.key_vars[0]}"

def test_pipeline_config_validation_pass() -> None:
    """Test that PipelineConfig validates correctly when sizes sum to 1.0."""
    # Should not raise
    PipelineConfig(train_size=0.7, val_size=0.2, test_size=0.1)

def test_pipeline_config_validation_fail() -> None:
    """Test that PipelineConfig raises ValueError when sizes do not sum to 1.0."""
    with pytest.raises(ValueError, match=r"train_size \+ val_size \+ test_size must equal exactly 1.0"):
        PipelineConfig(train_size=0.5, val_size=0.2, test_size=0.2)

def test_pipeline_config_explicit_err_vars() -> None:
    """Test that PipelineConfig respects explicitly provided err_vars."""
    custom_errs = ["err_1", "err_2"]
    config = PipelineConfig(key_vars=["feat1", "feat2"], err_vars=custom_errs)
    assert config.err_vars == custom_errs

def test_pipeline_config_field_factories() -> None:
    """Test that list fields use factories and are independent."""
    c1 = PipelineConfig()
    c2 = PipelineConfig()
    assert c1.key_vars is not c2.key_vars
    c1.key_vars.append("new_feat")
    assert "new_feat" not in c2.key_vars

def test_data_config_derivation() -> None:
    """Test that DataConfig can derive err_vars independently."""
    data = DataConfig(key_vars=["A", "B"])
    data._derive_err_vars()
    assert data.err_vars == ["E_A", "E_B"]

def test_split_config_validation() -> None:
    """Test that SplitConfig can validate splits independently."""
    split = SplitConfig(train_size=0.5, val_size=0.2, test_size=0.3)
    split._validate_splits() # Should not raise
    
    with pytest.raises(ValueError, match=r"train_size \+ val_size \+ test_size must equal exactly 1.0"):
        invalid_split = SplitConfig(train_size=0.1, val_size=0.1, test_size=0.1)
        invalid_split._validate_splits()

def test_tuning_config_defaults() -> None:
    """Test that TuningConfig initializes grids correctly."""
    tune = TuningConfig()
    assert tune.xgb_early_stopping_rounds == 10
    assert len(tune.dt_max_depth) > 0

def test_nn_config_defaults() -> None:
    """Test that NNConfig initializes with default values."""
    config = PipelineConfig()
    assert config.nn_hidden_layers == [128, 64, 32]
    assert config.nn_activation == PipelineConfig.nn_activation
    assert config.nn_dropout == PipelineConfig.nn_dropout
    assert config.nn_epochs == PipelineConfig.nn_epochs
    assert config.nn_patience == PipelineConfig.nn_patience
    assert config.nn_checkpoint_path == PipelineConfig.nn_checkpoint_path
    assert config.nn_output_activation == PipelineConfig.nn_output_activation
