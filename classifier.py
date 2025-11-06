import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, Tuple, Optional
import wandb


def train_trojan_classifier(
    sae_latents: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    random_state: int = 42,
    max_iter: int = 1000,
    use_wandb: bool = False,
    layer_idx: Optional[int] = None,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    """
    Train a logistic regression classifier on SAE latents.

    Assumes inputs are torch.Tensor and converts to numpy for sklearn.
    """
    X = sae_latents.numpy()
    y = labels.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight='balanced',
        solver='lbfgs',
        verbose=0
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    if use_wandb:
        log_dict = {f"classifier/{k}": v for k, v in metrics.items()}
        if layer_idx is not None:
            log_dict['layer'] = layer_idx
        wandb.log(log_dict)

    return clf, metrics


def evaluate_classifier(
    classifier: LogisticRegression,
    sae_latents: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate a trained classifier on SAE latents.

    Assumes inputs are torch.Tensor and converts to numpy for sklearn.
    """
    X = sae_latents.numpy()
    y = labels.numpy()

    y_pred = classifier.predict(X)
    y_proba = classifier.predict_proba(X)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auc_roc' : roc_auc_score(y, y_proba)
    }

    return metrics
