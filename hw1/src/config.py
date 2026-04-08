import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
@dataclass
class DataConfig:
    """Configuration for data paths, column names, and filenames."""

    filepath: str = os.path.join(current_dir, "../data/HECATE_test.csv")
    target_col: str = "CLASS_SP"
    key_vars: List[str] = field(
        default_factory=lambda: [
            "T", "WF1", "WF2", "WF3", "WF4", "UT", "BT", "VT", "IT",
            "U", "R", "G", "I", "Z",
        ]
    )
    err_vars: List[str] = field(default_factory=list)
    no_err_vars: List[str] = field(
        default_factory=lambda: ["CLASS_SP", "AGN_HEC", "logM_HEC", "logSFR_HEC"]
    )
    flags_vars: List[str] = field(default_factory=lambda: ["METAL"])
    color_vars: List[str] = field(
        default_factory=lambda: ["u-g", "g-r", "W3-UT", "(W3+UT)/W1"]
    )
    visuals_dir: str = os.path.join(current_dir, "../visuals")
    models_dir: str = os.path.join(current_dir, "../models")
    evaluation_dir: str = os.path.join(current_dir, "../evaluation_tables")
    scaler_filename: str = "scaler.pkl"
    encoder_filename: str = "label_encoder.pkl"
    model_filename: str = "classical_model.pkl"
    metrics_filename: str = "classical_models_metrics.png"
    cm_filename: str = "classical_models_confusion_matrices.png"
    scree_filename: str = "pca_scree_plot.png"
    projection_filename: str = "pca_2d_projection.png"

    nn_model_filename: str = "neural_network.pt"
    nn_metrics_filename: str = "nn_metrics.png"
    nn_loss_plot_filename: str = "nn_loss_curves.png"
    nn_cm_filename: str = "nn_confusion_matrix.png"
    comparison_metrics_filename: str = "final_comparison_metrics.png"
    comparison_cm_filename: str = "final_comparison_confusion_matrices.png"

    def _derive_err_vars(self) -> None:
        """Derives ``err_vars`` list if left empty."""
        if not self.err_vars:
            self.err_vars = [f"E_{var}" for var in self.key_vars]

@dataclass
class SplitConfig:
    """Configuration for dataset splitting and consistency."""

    train_size: float = 0.80
    val_size: float = 0.10
    test_size: float = 0.10
    random_state: int = 42

    def _validate_splits(self) -> None:
        """Validates that split fractions sum to unity."""
        if round(self.train_size + self.val_size + self.test_size, 5) != 1.0:
            raise ValueError(
                "train_size + val_size + test_size must equal exactly 1.0 "
                f"(got {self.train_size + self.val_size + self.test_size})."
            )

@dataclass
class TuningConfig:
    """Configuration for model training and hyperparameter search spaces."""

    xgb_early_stopping_rounds: int = 10

    # Hyperparameter search spaces
    dt_max_depth: List[Optional[int]] = field(default_factory=lambda: [None, 5, 10, 20])
    dt_min_samples_split: List[int] = field(default_factory=lambda: [2, 5, 10])

    lr_C: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    lr_max_iter: List[int] = field(default_factory=lambda: [1000])

    svm_C: List[float] = field(default_factory=lambda: [0.1, 1.0])
    svm_kernel: List[str] = field(default_factory=lambda: ["rbf", "linear"])

    rf_n_estimators: List[int] = field(default_factory=lambda: [50, 100])
    rf_max_depth: List[Optional[int]] = field(default_factory=lambda: [None, 10, 20])

    xgb_n_estimators: List[int] = field(default_factory=lambda: [50, 100, 200])
    xgb_max_depth: List[int] = field(default_factory=lambda: [3, 5, 7])
    xgb_learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.1])

@dataclass
class NNConfig:
    """Configuration for Neural Network architecture and training."""

    #nn_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    nn_hidden_layers: List[int] = field(default_factory=lambda: [32, 16])
    nn_activation: str = "ReLU"  # Options: ReLU, LeakyRFelu, ELU, Tanh
    nn_checkpoint_path: str = os.path.join(current_dir, "../models/nn_best_model.pth")
    nn_dropout: float = 0.2
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
    nn_batch_size: int = 64
    nn_patience: int = 10
    nn_output_activation: Optional[str] = None  # Sigmoid, Softmax, or None (auto)
    nn_random_state: int = 42

@dataclass
class PipelineConfig(DataConfig, SplitConfig, TuningConfig, NNConfig):
    """Unified configuration class inheriting from specialized sub-configs."""

    def __post_init__(self) -> None:
        """Execute validation and derivation logic from base classes."""
        self._derive_err_vars()
        self._validate_splits()
