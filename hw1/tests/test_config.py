import pytest
from src.config import PipelineConfig

def test_pipeline_config_defaults() -> None:
    """Test that PipelineConfig initializes with default values."""
    config = PipelineConfig()
    assert config.target_col == "CLASS_SP"
    assert config.train_size == 0.80
    assert config.val_size == 0.10
    assert config.test_size == 0.10
    assert config.random_state == 42
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
