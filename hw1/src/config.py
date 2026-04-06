import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional

@dataclass
class PipelineConfig:
    """Central configuration for the full ML pipeline.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV dataset.
    target_col : str
        Name of the target (label) column in the dataset.
    train_size : float
        Fraction of data reserved for training. Must sum to 1.0 with
        ``val_size`` and ``test_size``.
    val_size : float
        Fraction of data reserved for validation.
    test_size : float
        Fraction of data reserved for testing.
    key_vars : List[str]
        Primary feature columns used for modelling.
    err_vars : List[str]
        Error columns corresponding to ``key_vars``.  When left empty the
        list is derived automatically as ``['E_<var>' for var in key_vars]``.
    no_err_vars : List[str]
        Feature columns that have no associated error columns.
    flags_vars : List[str]
        Flag columns included in feature selection.
    color_vars : List[str]
        Computed colour-index columns.
    visuals_dir : str
        Directory where all plots are saved.
    models_dir : str
        Directory where serialised models are saved.
    random_state : int
        Global random seed for reproducibility.
    """

    filepath: str = "data/HECATE.csv"
    target_col: str = "CLASS_SP"
    train_size: float = 0.80
    val_size: float = 0.10
    test_size: float = 0.10
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
    visuals_dir: str = "visuals"
    models_dir: str = "models"
    scaler_filename: str = "scaler.pkl"
    encoder_filename: str = "label_encoder.pkl"
    model_filename: str = "classical_model.pkl"
    metrics_filename: str = "classical_models_metrics.png"
    cm_filename: str = "classical_models_confusion_matrices.png"
    scree_filename: str = "pca_scree_plot.png"
    projection_filename: str = "pca_2d_projection.png"
    random_state: int = 42

    # XGBoost training
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

    def __post_init__(self) -> None:
        """Validate split fractions and auto-derive ``err_vars`` if needed."""
        if round(self.train_size + self.val_size + self.test_size, 5) != 1.0:
            raise ValueError(
                "train_size + val_size + test_size must equal exactly 1.0 "
                f"(got {self.train_size + self.val_size + self.test_size})."
            )
        if not self.err_vars:
            self.err_vars = [f"E_{var}" for var in self.key_vars]
