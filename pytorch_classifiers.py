"""
PyTorch-based classifiers for trojan detection.

Implements gradient descent-optimized logistic regression with three regularization variants:
1. No regularization - Standard maximum likelihood
2. L1 regularization (Lasso) - Promotes sparsity
3. L2 regularization (Ridge) - Prevents overfitting via weight decay
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Tuple
from tqdm import tqdm


class PyTorchLogisticRegression(nn.Module):
    """
    Logistic regression implemented in PyTorch with gradient descent optimization.

    Supports three regularization types:
    - 'none': No regularization (standard logistic regression)
    - 'l1': L1 regularization (Lasso) - Loss = CrossEntropy + λ * ||W||_1
    - 'l2': L2 regularization (Ridge) - Loss = CrossEntropy + λ * ||W||_2^2
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        reg_type: str = 'none',
        reg_lambda: float = 1e-3,
    ):
        """
        Args:
            input_dim: Number of input features
            num_classes: Number of output classes (default: 2 for binary classification)
            reg_type: Regularization type - 'none', 'l1', or 'l2'
            reg_lambda: Regularization strength (lambda parameter)
        """
        super(PyTorchLogisticRegression, self).__init__()

        self.linear = nn.Linear(input_dim, num_classes)
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda

        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """Forward pass through linear layer."""
        return self.linear(x)

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization penalty.

        Returns:
            Regularization loss (scalar tensor)
        """
        if self.reg_type == 'none':
            return torch.tensor(0.0, device=self.linear.weight.device)
        elif self.reg_type == 'l1':
            # L1 regularization: sum of absolute values
            return self.reg_lambda * torch.norm(self.linear.weight, p=1)
        elif self.reg_type == 'l2':
            # L2 regularization: sum of squared values
            return self.reg_lambda * torch.norm(self.linear.weight, p=2) ** 2
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")


def train_pytorch_logistic_regression(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: Optional[torch.Tensor] = None,
    val_labels: Optional[torch.Tensor] = None,
    reg_type: str = 'none',
    reg_lambda: float = 1e-3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 5,
    device: str = 'cuda',
    verbose: bool = True,
) -> Tuple[PyTorchLogisticRegression, Dict]:
    """
    Train PyTorch logistic regression with early stopping.

    Args:
        train_features: Training features [N, D]
        train_labels: Training labels [N]
        val_features: Validation features [M, D] (optional)
        val_labels: Validation labels [M] (optional)
        reg_type: Regularization type - 'none', 'l1', or 'l2'
        reg_lambda: Regularization strength
        learning_rate: Adam optimizer learning rate
        batch_size: Mini-batch size for training
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience (epochs without improvement)
        device: Device to train on ('cuda' or 'cpu')
        verbose: Print training progress

    Returns:
        model: Trained PyTorchLogisticRegression model
        history: Dictionary containing training history
    """
    # Convert to tensors if needed
    if not isinstance(train_features, torch.Tensor):
        train_features = torch.from_numpy(train_features).float()
    if not isinstance(train_labels, torch.Tensor):
        train_labels = torch.from_numpy(train_labels).long()

    # Move to device
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)

    # Handle validation data
    has_validation = val_features is not None and val_labels is not None
    if has_validation:
        if not isinstance(val_features, torch.Tensor):
            val_features = torch.from_numpy(val_features).float()
        if not isinstance(val_labels, torch.Tensor):
            val_labels = torch.from_numpy(val_labels).long()
        val_features = val_features.to(device)
        val_labels = val_labels.to(device)

    # Create model
    input_dim = train_features.shape[1]
    num_classes = len(torch.unique(train_labels))
    model = PyTorchLogisticRegression(
        input_dim=input_dim,
        num_classes=num_classes,
        reg_type=reg_type,
        reg_lambda=reg_lambda,
    ).to(device)

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Create data loader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    iterator = tqdm(range(max_epochs), desc=f"Training PyTorch LR ({reg_type})") if verbose else range(max_epochs)

    for epoch in iterator:
        # Training phase
        model.train()
        train_loss_epoch = 0.0
        train_correct = 0
        train_total = 0

        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(batch_features)
            ce_loss = criterion(logits, batch_labels)
            reg_loss = model.compute_regularization_loss()
            loss = ce_loss + reg_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            train_loss_epoch += loss.item() * batch_features.size(0)
            _, predicted = torch.max(logits.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_loss = train_loss_epoch / train_total
        train_acc = train_correct / train_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation phase
        if has_validation:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_features)
                val_ce_loss = criterion(val_logits, val_labels)
                val_reg_loss = model.compute_regularization_loss()
                val_loss = val_ce_loss + val_reg_loss

                _, val_predicted = torch.max(val_logits.data, 1)
                val_acc = (val_predicted == val_labels).float().mean().item()

            history['val_loss'].append(val_loss.item())
            history['val_acc'].append(val_acc)

            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'val_loss': f'{val_loss.item():.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break
        else:
            # No validation - just update progress bar with training metrics
            if verbose:
                iterator.set_postfix({
                    'train_loss': f'{train_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}'
                })

    # Restore best model if we had validation
    if has_validation and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


def evaluate_pytorch_logistic_regression(
    model: PyTorchLogisticRegression,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate PyTorch logistic regression model.

    Args:
        model: Trained PyTorchLogisticRegression model
        features: Features to evaluate [N, D]
        labels: True labels [N]
        device: Device to run evaluation on

    Returns:
        Dictionary with metrics: accuracy, precision, recall, f1, auc_roc
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Convert to tensors if needed
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features).float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).long()

    features = features.to(device)
    labels = labels.to(device)

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(features)
        probs = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)

    # Convert to numpy for sklearn metrics
    y_true = labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()
    y_probs = probs.cpu().numpy()

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }

    # AUC-ROC (handle binary and multiclass)
    try:
        if probs.shape[1] == 2:
            # Binary classification - use probability of positive class
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs[:, 1])
        else:
            # Multiclass - use ovr strategy
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except:
        metrics['auc_roc'] = 0.0

    return metrics


def train_and_evaluate_pytorch_lr(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    reg_type: str = 'none',
    reg_lambda: float = 1e-3,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 5,
    device: str = 'cuda',
    verbose: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    High-level function to train and evaluate PyTorch logistic regression.

    This function matches the interface expected by ExperimentRunner.

    Returns:
        Dictionary with 'train' and 'val' metrics
    """
    # Train model
    model, history = train_pytorch_logistic_regression(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        reg_type=reg_type,
        reg_lambda=reg_lambda,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        verbose=verbose,
    )

    # Evaluate on training set
    train_metrics = evaluate_pytorch_logistic_regression(
        model, train_features, train_labels, device
    )

    results = {'train': train_metrics}

    # Evaluate on validation set if provided
    if val_features is not None and val_labels is not None:
        val_metrics = evaluate_pytorch_logistic_regression(
            model, val_features, val_labels, device
        )
        results['val'] = val_metrics

    return results, model
