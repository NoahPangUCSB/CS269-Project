"""
Decision Boundary Visualization for Overfitting Analysis

This module provides functions to visualize classifier decision boundaries
in 2D space using PCA, t-SNE, and LDA dimensionality reduction techniques.
Used to compare overfitting behavior across layers and classifiers.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import pickle


def plot_decision_boundary_pca(
    classifier: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    classifier_name: str,
    layer_idx: int,
    feature_type: str,
    save_path: Path,
    resolution: int = 500
) -> Dict[str, float]:
    """
    Plot decision boundary using PCA reduction to 2D.

    Args:
        classifier: Trained classifier object (may be tuple for PCA/GMM/KMeans)
        train_features: Training features (num_samples, num_features)
        train_labels: Training labels (num_samples,)
        val_features: Validation features
        val_labels: Validation labels
        classifier_name: Name of the classifier
        layer_idx: Layer index
        feature_type: 'raw_activation' or 'sae_latent'
        save_path: Path to save the plot
        resolution: Resolution of decision boundary mesh

    Returns:
        Dict with train_accuracy, val_accuracy, overfitting_gap
    """
    # Convert to numpy if torch tensors
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(val_features, torch.Tensor):
        val_features = val_features.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2, random_state=42)
    train_2d = pca.fit_transform(train_features)
    val_2d = pca.transform(val_features)

    # Create meshgrid for decision boundary
    x_min, x_max = train_2d[:, 0].min() - 1, train_2d[:, 0].max() + 1
    y_min, y_max = train_2d[:, 1].min() - 1, train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Inverse transform meshgrid to original feature space
    mesh_2d = np.c_[xx.ravel(), yy.ravel()]
    mesh_original = pca.inverse_transform(mesh_2d)

    # Get predictions on mesh
    Z = _predict_with_classifier(classifier, mesh_original, classifier_name)
    Z = Z.reshape(xx.shape)

    # Calculate metrics
    train_pred = _predict_with_classifier(classifier, train_features, classifier_name)
    val_pred = _predict_with_classifier(classifier, val_features, classifier_name)

    train_accuracy = np.mean(train_pred == train_labels)
    val_accuracy = np.mean(val_pred == val_labels)
    overfitting_gap = train_accuracy - val_accuracy

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, alpha=0.4, levels=1, colors=['#ff7f0e', '#1f77b4'])

    # Plot training points
    train_class_0 = train_2d[train_labels == 0]
    train_class_1 = train_2d[train_labels == 1]
    ax.scatter(train_class_0[:, 0], train_class_0[:, 1],
               c='blue', marker='o', s=20, alpha=0.6, label='Train Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(train_class_1[:, 0], train_class_1[:, 1],
               c='orange', marker='o', s=20, alpha=0.6, label='Train Class 1', edgecolors='k', linewidths=0.5)

    # Plot validation points
    val_class_0 = val_2d[val_labels == 0]
    val_class_1 = val_2d[val_labels == 1]
    ax.scatter(val_class_0[:, 0], val_class_0[:, 1],
               c='blue', marker='s', s=20, alpha=0.8, label='Val Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(val_class_1[:, 0], val_class_1[:, 1],
               c='orange', marker='s', s=20, alpha=0.8, label='Val Class 1', edgecolors='k', linewidths=0.5)

    # Labels and title
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=12)
    ax.set_title(
        f'{classifier_name.replace("_", " ").title()} - Layer {layer_idx} ({feature_type.replace("_", " ").title()})\n'
        f'PCA Projection | Train Acc: {train_accuracy:.3f} | Val Acc: {val_accuracy:.3f} | '
        f'Overfitting Gap: {overfitting_gap:.3f}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'overfitting_gap': float(overfitting_gap),
        'variance_explained': float(pca.explained_variance_ratio_.sum())
    }


def plot_decision_boundary_tsne(
    classifier: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    classifier_name: str,
    layer_idx: int,
    feature_type: str,
    save_path: Path,
    perplexity: int = 30,
    resolution: int = 500
) -> Dict[str, float]:
    """
    Plot decision boundary using t-SNE reduction to 2D.

    Args:
        classifier: Trained classifier object
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        classifier_name: Name of the classifier
        layer_idx: Layer index
        feature_type: 'raw_activation' or 'sae_latent'
        save_path: Path to save the plot
        perplexity: t-SNE perplexity parameter
        resolution: Resolution of decision boundary mesh

    Returns:
        Dict with train_accuracy, val_accuracy, overfitting_gap
    """
    # Convert to numpy if torch tensors
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(val_features, torch.Tensor):
        val_features = val_features.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # Combine train and val for joint t-SNE embedding
    combined_features = np.vstack([train_features, val_features])
    n_train = len(train_features)

    # Apply t-SNE to reduce to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(combined_features) - 1))
    combined_2d = tsne.fit_transform(combined_features)

    train_2d = combined_2d[:n_train]
    val_2d = combined_2d[n_train:]

    # For t-SNE, we can't inverse transform, so we'll create a KNN-based interpolation
    # for the decision boundary visualization
    from scipy.interpolate import griddata

    # Get predictions on training data in original space
    train_pred = _predict_with_classifier(classifier, train_features, classifier_name)

    # Create meshgrid in t-SNE space
    x_min, x_max = train_2d[:, 0].min() - 1, train_2d[:, 0].max() + 1
    y_min, y_max = train_2d[:, 1].min() - 1, train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # Interpolate predictions onto mesh using nearest neighbor
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = griddata(train_2d, train_pred, mesh_points, method='nearest')
    Z = Z.reshape(xx.shape)

    # Calculate metrics in original space
    val_pred = _predict_with_classifier(classifier, val_features, classifier_name)

    train_accuracy = np.mean(train_pred == train_labels)
    val_accuracy = np.mean(val_pred == val_labels)
    overfitting_gap = train_accuracy - val_accuracy

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, alpha=0.4, levels=1, colors=['#ff7f0e', '#1f77b4'])

    # Plot training points
    train_class_0 = train_2d[train_labels == 0]
    train_class_1 = train_2d[train_labels == 1]
    ax.scatter(train_class_0[:, 0], train_class_0[:, 1],
               c='blue', marker='o', s=20, alpha=0.6, label='Train Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(train_class_1[:, 0], train_class_1[:, 1],
               c='orange', marker='o', s=20, alpha=0.6, label='Train Class 1', edgecolors='k', linewidths=0.5)

    # Plot validation points
    val_class_0 = val_2d[val_labels == 0]
    val_class_1 = val_2d[val_labels == 1]
    ax.scatter(val_class_0[:, 0], val_class_0[:, 1],
               c='blue', marker='s', s=20, alpha=0.8, label='Val Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(val_class_1[:, 0], val_class_1[:, 1],
               c='orange', marker='s', s=20, alpha=0.8, label='Val Class 1', edgecolors='k', linewidths=0.5)

    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(
        f'{classifier_name.replace("_", " ").title()} - Layer {layer_idx} ({feature_type.replace("_", " ").title()})\n'
        f't-SNE Projection | Train Acc: {train_accuracy:.3f} | Val Acc: {val_accuracy:.3f} | '
        f'Overfitting Gap: {overfitting_gap:.3f}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'overfitting_gap': float(overfitting_gap)
    }


def plot_decision_boundary_lda(
    classifier: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    layer_idx: int,
    feature_type: str,
    save_path: Path,
    resolution: int = 500
) -> Dict[str, float]:
    """
    Plot decision boundary using LDA's native projection.

    Args:
        classifier: Trained LDA classifier
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        layer_idx: Layer index
        feature_type: 'raw_activation' or 'sae_latent'
        save_path: Path to save the plot
        resolution: Resolution of decision boundary mesh

    Returns:
        Dict with train_accuracy, val_accuracy, overfitting_gap
    """
    # Convert to numpy if torch tensors
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(val_features, torch.Tensor):
        val_features = val_features.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # LDA transform to get 1D projection (binary classification)
    train_1d = classifier.transform(train_features)
    val_1d = classifier.transform(val_features)

    # For visualization, we'll use LDA projection as x-axis and a random second dimension
    # or use PCA for the second dimension
    pca = PCA(n_components=1, random_state=42)
    train_pca = pca.fit_transform(train_features)
    val_pca = pca.transform(val_features)

    train_2d = np.column_stack([train_1d, train_pca])
    val_2d = np.column_stack([val_1d, val_pca])

    # Create meshgrid
    x_min, x_max = train_2d[:, 0].min() - 1, train_2d[:, 0].max() + 1
    y_min, y_max = train_2d[:, 1].min() - 1, train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )

    # For LDA, decision boundary is primarily based on LDA projection (x-axis)
    # We need to reconstruct approximate features from the 2D projection
    from scipy.interpolate import griddata

    train_pred = classifier.predict(train_features)
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = griddata(train_2d, train_pred, mesh_points, method='nearest')
    Z = Z.reshape(xx.shape)

    # Calculate metrics
    val_pred = classifier.predict(val_features)

    train_accuracy = np.mean(train_pred == train_labels)
    val_accuracy = np.mean(val_pred == val_labels)
    overfitting_gap = train_accuracy - val_accuracy

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot decision boundary
    contour = ax.contourf(xx, yy, Z, alpha=0.4, levels=1, colors=['#ff7f0e', '#1f77b4'])

    # Plot training points
    train_class_0 = train_2d[train_labels == 0]
    train_class_1 = train_2d[train_labels == 1]
    ax.scatter(train_class_0[:, 0], train_class_0[:, 1],
               c='blue', marker='o', s=20, alpha=0.6, label='Train Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(train_class_1[:, 0], train_class_1[:, 1],
               c='orange', marker='o', s=20, alpha=0.6, label='Train Class 1', edgecolors='k', linewidths=0.5)

    # Plot validation points
    val_class_0 = val_2d[val_labels == 0]
    val_class_1 = val_2d[val_labels == 1]
    ax.scatter(val_class_0[:, 0], val_class_0[:, 1],
               c='blue', marker='s', s=20, alpha=0.8, label='Val Class 0', edgecolors='k', linewidths=0.5)
    ax.scatter(val_class_1[:, 0], val_class_1[:, 1],
               c='orange', marker='s', s=20, alpha=0.8, label='Val Class 1', edgecolors='k', linewidths=0.5)

    # Labels and title
    ax.set_xlabel('LDA Projection', fontsize=12)
    ax.set_ylabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=12)
    ax.set_title(
        f'LDA - Layer {layer_idx} ({feature_type.replace("_", " ").title()})\n'
        f'Native LDA Projection | Train Acc: {train_accuracy:.3f} | Val Acc: {val_accuracy:.3f} | '
        f'Overfitting Gap: {overfitting_gap:.3f}',
        fontsize=13, fontweight='bold'
    )
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'overfitting_gap': float(overfitting_gap)
    }


def _predict_with_classifier(classifier: Any, features: np.ndarray, classifier_name: str) -> np.ndarray:
    """
    Helper function to handle different classifier types and their prediction interfaces.

    Args:
        classifier: The classifier object (may be tuple for special cases)
        features: Features to predict on
        classifier_name: Name of the classifier type

    Returns:
        Predicted labels as numpy array
    """
    if classifier_name == 'pca':
        # PCA classifier is tuple: (pca_obj, logistic_regression)
        pca_obj, logreg = classifier
        features_pca = pca_obj.transform(features)
        return logreg.predict(features_pca)

    elif classifier_name == 'gmm':
        # GMM is tuple: (gmm_0, gmm_1)
        gmm_0, gmm_1 = classifier
        log_likelihood_0 = gmm_0.score_samples(features)
        log_likelihood_1 = gmm_1.score_samples(features)
        return (log_likelihood_1 > log_likelihood_0).astype(int)

    elif classifier_name == 'kmeans':
        # KMeans is tuple: (kmeans, cluster_to_label)
        kmeans, cluster_to_label = classifier
        cluster_predictions = kmeans.predict(features)
        return np.array([cluster_to_label[c] for c in cluster_predictions])

    else:
        # Standard sklearn classifiers
        return classifier.predict(features)


def plot_1d_distribution_comparison(
    classifier: Any,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    classifier_name: str,
    layer_idx: int,
    feature_type: str,
    save_path: Path,
) -> Dict[str, float]:
    """
    Plot 1D histogram distributions comparing train vs validation sets.
    Similar to the style in lda_stuff/visualize_lda_train_test.py

    Uses the first principal component or classifier's decision function for projection.

    Args:
        classifier: Trained classifier object
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        classifier_name: Name of the classifier
        layer_idx: Layer index
        feature_type: 'raw_activation' or 'sae_latent'
        save_path: Path to save the plot

    Returns:
        Dict with train_accuracy, val_accuracy, overfitting_gap
    """
    # Convert to numpy if torch tensors
    if isinstance(train_features, torch.Tensor):
        train_features = train_features.cpu().numpy()
    if isinstance(train_labels, torch.Tensor):
        train_labels = train_labels.cpu().numpy()
    if isinstance(val_features, torch.Tensor):
        val_features = val_features.cpu().numpy()
    if isinstance(val_labels, torch.Tensor):
        val_labels = val_labels.cpu().numpy()

    # Get 1D projection based on classifier type
    if classifier_name == 'lda':
        # Use LDA's native projection
        train_1d = classifier.transform(train_features).flatten()
        val_1d = classifier.transform(val_features).flatten()
        projection_name = "LDA Projection"
    else:
        # Use PCA for 1D projection
        pca = PCA(n_components=1, random_state=42)
        train_1d = pca.fit_transform(train_features).flatten()
        val_1d = pca.transform(val_features).flatten()
        projection_name = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)"

    # Get predictions
    train_pred = _predict_with_classifier(classifier, train_features, classifier_name)
    val_pred = _predict_with_classifier(classifier, val_features, classifier_name)

    # Calculate metrics
    train_accuracy = np.mean(train_pred == train_labels)
    val_accuracy = np.mean(val_pred == val_labels)
    overfitting_gap = train_accuracy - val_accuracy

    # Calculate threshold from training data
    train_class_0_1d = train_1d[train_labels == 0]
    train_class_1_1d = train_1d[train_labels == 1]
    threshold = (train_class_0_1d.mean() + train_class_1_1d.mean()) / 2

    # Separate validation data by class
    val_class_0_1d = val_1d[val_labels == 0]
    val_class_1_1d = val_1d[val_labels == 1]

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Training set distribution
    axes[0, 0].hist(train_class_0_1d, bins=50, alpha=0.7, label='Class 0 (Train)',
                    color='blue', density=True, edgecolor='darkblue', linewidth=0.5)
    axes[0, 0].hist(train_class_1_1d, bins=50, alpha=0.7, label='Class 1 (Train)',
                    color='orange', density=True, edgecolor='darkorange', linewidth=0.5)
    axes[0, 0].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.3f})')
    axes[0, 0].set_xlabel(projection_name, fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title(f'Training Set Distribution\n'
                         f'Accuracy: {train_accuracy:.3f}',
                         fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Validation set distribution
    axes[0, 1].hist(val_class_0_1d, bins=50, alpha=0.7, label='Class 0 (Val)',
                    color='blue', density=True, edgecolor='darkblue', linewidth=0.5)
    axes[0, 1].hist(val_class_1_1d, bins=50, alpha=0.7, label='Class 1 (Val)',
                    color='orange', density=True, edgecolor='darkorange', linewidth=0.5)
    axes[0, 1].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold ({threshold:.3f})')
    axes[0, 1].set_xlabel(projection_name, fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title(f'Validation Set Distribution\n'
                         f'Accuracy: {val_accuracy:.3f} | Overfitting Gap: {overfitting_gap:+.3f}',
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Class 0: Train vs Val overlay
    axes[1, 0].hist(train_class_0_1d, bins=50, alpha=0.5, label='Class 0 (Train)',
                    color='blue', density=True, edgecolor='blue', linewidth=1.5)
    axes[1, 0].hist(val_class_0_1d, bins=50, alpha=0.5, label='Class 0 (Val)',
                    color='cyan', density=True, edgecolor='darkblue', linewidth=1.5, histtype='step')
    axes[1, 0].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                       label='Threshold')
    axes[1, 0].set_xlabel(projection_name, fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Class 0: Train vs Validation', fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='best', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Class 1: Train vs Val overlay
    axes[1, 1].hist(train_class_1_1d, bins=50, alpha=0.5, label='Class 1 (Train)',
                    color='orange', density=True, edgecolor='orange', linewidth=1.5)
    axes[1, 1].hist(val_class_1_1d, bins=50, alpha=0.5, label='Class 1 (Val)',
                    color='red', density=True, edgecolor='darkred', linewidth=1.5, histtype='step')
    axes[1, 1].axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                       label='Threshold')
    axes[1, 1].set_xlabel(projection_name, fontsize=12)
    axes[1, 1].set_ylabel('Density', fontsize=12)
    axes[1, 1].set_title('Class 1: Train vs Validation', fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    # Overall title
    plt.suptitle(
        f'{classifier_name.replace("_", " ").title()} - Layer {layer_idx} ({feature_type.replace("_", " ").title()})\n'
        f'Train Accuracy: {train_accuracy:.3f} | Val Accuracy: {val_accuracy:.3f} | '
        f'Overfitting Gap: {overfitting_gap:+.3f}',
        fontsize=16, fontweight='bold', y=0.995
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    return {
        'train_accuracy': float(train_accuracy),
        'val_accuracy': float(val_accuracy),
        'overfitting_gap': float(overfitting_gap),
        'threshold': float(threshold)
    }


def visualize_classifier_overfitting(
    classifier_path: Path,
    classifier_name: str,
    train_features: Union[np.ndarray, torch.Tensor],
    train_labels: Union[np.ndarray, torch.Tensor],
    val_features: Union[np.ndarray, torch.Tensor],
    val_labels: Union[np.ndarray, torch.Tensor],
    layer_idx: int,
    feature_type: str,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Main function to generate all decision boundary visualizations for a classifier.

    Args:
        classifier_path: Path to saved classifier .pkl file
        classifier_name: Name of the classifier
        train_features: Training features
        train_labels: Training labels
        val_features: Validation features
        val_labels: Validation labels
        layer_idx: Layer index
        feature_type: 'raw_activation' or 'sae_latent'
        output_dir: Directory to save visualizations

    Returns:
        Dict with paths to generated plots and metrics
    """
    # Load classifier
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'classifier_name': classifier_name,
        'layer_idx': layer_idx,
        'feature_type': feature_type,
        'plots': {}
    }

    # Generate 1D distribution comparison (similar to lda_stuff style)
    dist_path = output_dir / f'{feature_type}_distribution.png'
    dist_metrics = plot_1d_distribution_comparison(
        classifier, train_features, train_labels,
        val_features, val_labels, classifier_name,
        layer_idx, feature_type, dist_path
    )
    results['plots']['distribution'] = str(dist_path)
    results['metrics_distribution'] = dist_metrics

    # Generate visualizations based on classifier type
    if classifier_name == 'lda':
        # LDA uses its own projection for 2D plot
        lda_path = output_dir / f'{feature_type}_lda.png'
        lda_metrics = plot_decision_boundary_lda(
            classifier, train_features, train_labels,
            val_features, val_labels, layer_idx, feature_type, lda_path
        )
        results['plots']['lda'] = str(lda_path)
        results['metrics_lda'] = lda_metrics
    else:
        # All other classifiers get PCA visualization only
        pca_path = output_dir / f'{feature_type}_pca.png'
        pca_metrics = plot_decision_boundary_pca(
            classifier, train_features, train_labels,
            val_features, val_labels, classifier_name,
            layer_idx, feature_type, pca_path
        )
        results['plots']['pca'] = str(pca_path)
        results['metrics_pca'] = pca_metrics

    return results
