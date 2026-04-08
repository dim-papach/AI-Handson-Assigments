import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from src.config import PipelineConfig
from src.evaluation import (
    evaluate_nn_model,
    plot_model_performance,
    plot_nn_training_history,
)


class SimpleNN(nn.Module):
    """
    A simple feedforward neural network with configurable hidden layers.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_layers : List[int]
        List of hidden layer neurons.
    output_dim : int
        Number of output neurons.
    dropout : float
        Dropout rate for regularization.
    activation : str, optional
        Activation function name from ``torch.nn``, by default "ReLU".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        dropout: float,
        activation: str = PipelineConfig.nn_activation,
    ) -> None:
        super(SimpleNN, self).__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim

        # Get activation function class from torch.nn
        act_fn_cls = getattr(nn, activation, nn.ReLU)

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn_cls())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        # Note: No output activation here as BCEWithLogitsLoss or CrossEntropyLoss will handle it.
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.

    Derived from common PyTorch early stopping implementations.
    """

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        path: str = PipelineConfig.nn_checkpoint_path,
    ) -> None:
        """
        Parameters
        ----------
        patience : int
            How long to wait after last time validation loss improved.
        verbose : bool
            If True, prints a message for each validation loss improvement.
        delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
        path : str
            Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def prepare_tensors(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts pandas DataFrame/Series to PyTorch tensors.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels/values.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        (X_tensor, y_tensor)
    """
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    # y_tensor logic: if classification, use LongTensor (for CrossEntropy) or float (for BCE)
    # Here we assume classification.
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return X_tensor, y_tensor


def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Optional[PipelineConfig] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """
    Trains a Feedforward Neural Network using PyTorch.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation labels.
    config : PipelineConfig, optional
        Configuration object. If None, default PipelineConfig is used.

    Returns
    -------
    Tuple[nn.Module, Dict[str, List[float]]]
        The trained model (best weights restored) and training history.
    """
    if config is None:
        config = PipelineConfig()

    torch.manual_seed(config.nn_random_state)
    np.random.seed(config.nn_random_state)

    torch.set_num_threads(config.nn_num_threads)
    torch.set_num_interop_threads(config.nn_num_threads)

    # Determine dimensions
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    is_binary = num_classes <= 2
    output_dim = 1 if is_binary else num_classes

    # Prepare data
    X_train_t, y_train_t = prepare_tensors(X_train, y_train)
    X_val_t, y_val_t = prepare_tensors(X_val, y_val)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_ds, batch_size=config.nn_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=config.nn_batch_size, shuffle=False)

    # Initialize model
    model = SimpleNN(
        input_dim=input_dim,
        hidden_layers=config.nn_hidden_layers,
        output_dim=output_dim,
        dropout=config.nn_dropout,
        activation=config.nn_activation,
    )

    # Loss and optimizer
    if is_binary:
        criterion = nn.BCEWithLogitsLoss()
    else:
        # For multi-class, target needs to be LongTensor
        # We'll handle this in the training loop to keep prepare_tensors simple
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.nn_learning_rate)

    # Early stopping
    checkpoint_path = config.nn_checkpoint_path
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    early_stopping = EarlyStopping(
        patience=config.nn_patience, verbose=True, path=checkpoint_path
    )

    history = {"train_loss": [], "val_loss": []}

    print(f"Starting Neural Network Training (Total Epochs: {config.nn_epochs})...")
    for epoch in range(1, config.nn_epochs + 1):
        # Training phase
        model.train()
        train_running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            if is_binary:
                loss = criterion(outputs.squeeze(), targets)
            else:
                loss = criterion(outputs, targets.long())

            loss.backward()
            optimizer.step()
            train_running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = train_running_loss / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                if is_binary:
                    loss = criterion(outputs.squeeze(), targets)
                else:
                    loss = criterion(outputs, targets.long())
                val_running_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{config.nn_epochs} | "
                f"Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f}"
            )

        # Early stopping check
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best weights
    model.load_state_dict(torch.load(checkpoint_path))
    # Save final model
    final_path = os.path.join(config.models_dir, config.nn_model_filename)
    torch.save(model.state_dict(), final_path)
    print(f"Restored best weights and saved model to {final_path}")

    return model, history


def plot_nn_evaluation(
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Plots evaluation metrics and confusion matrix for the NN using the unified plotter.

    Parameters
    ----------
    metrics : Dict[str, float]
    y_true : np.ndarray
    y_pred : np.ndarray
    config : PipelineConfig, optional
    """
    if config is None:
        config = PipelineConfig()

    model_results = {
        "Neural Network": {
            "metrics": metrics,
            "cm": confusion_matrix(y_true, y_pred),
        }
    }

    plot_model_performance(
        model_results,
        visuals_dir=config.visuals_dir,
        metrics_filename=config.nn_metrics_filename,
        cm_filename=config.nn_cm_filename,
        main_title="Neural Network Evaluation",
    )

if __name__ == "__main__":
    print("This module provides Neural Network training logic.")
    print("Import and run `train_neural_network(X_train, y_train, X_val, y_val)`.")
