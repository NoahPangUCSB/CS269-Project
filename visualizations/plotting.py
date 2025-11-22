import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color schemes
CLASSIFIER_COLORS = {
    'logistic_regression': '#1f77b4',
    'random_forest': '#ff7f0e',
    'pca': '#2ca02c',
    'lda': '#d62728',
    'naive_bayes': '#9467bd',
    'gmm': '#8c564b',
    'kmeans': '#e377c2',
}

FEATURE_COLORS = {
    'raw_activation': '#3498db',
    'sae_latent': '#e74c3c',
}

TRIGGER_COLORS = {
    'actual': '#2ecc71',
    'approximate': '#f39c12',
}


def load_experiment_results(results_dir: Path) -> List[Dict]:
    """Load all_results.json from the experiment directory."""
    results_path = results_dir / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with open(results_path, 'r') as f:
        return json.load(f)


def plot_classifier_performance_across_layers(
    results: List[Dict],
    metric: str = 'f1',
    split: str = 'val',
    feature_type: Optional[str] = None,
    trigger_type: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Plot performance of each classifier across different layers.

    Args:
        results: List of experiment result dictionaries
        metric: Metric to plot (accuracy, precision, recall, f1, auc_roc)
        split: Data split to plot (train, val, test)
        feature_type: Filter by feature type (raw_activation, sae_latent)
        trigger_type: Filter by trigger type (actual, approximate)
        save_path: Path to save the figure
    """
    # Filter results
    filtered_results = results
    if feature_type:
        filtered_results = [r for r in filtered_results if r['feature_type'] == feature_type]
    if trigger_type:
        filtered_results = [r for r in filtered_results if r['trigger_type'] == trigger_type]

    # Group by classifier and layer
    classifier_layer_data = {}
    for result in filtered_results:
        classifier = result['classifier']
        layer = result['layer_idx']

        if split in result and metric in result[split]:
            if classifier not in classifier_layer_data:
                classifier_layer_data[classifier] = {}
            classifier_layer_data[classifier][layer] = result[split][metric]

    if not classifier_layer_data:
        print(f"No data available for metric={metric}, split={split}")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for classifier, layer_data in classifier_layer_data.items():
        layers = sorted(layer_data.keys())
        values = [layer_data[l] for l in layers]

        color = CLASSIFIER_COLORS.get(classifier, '#808080')
        ax.plot(layers, values, marker='o', linewidth=2, markersize=8,
                label=classifier.replace('_', ' ').title(), color=color)

    ax.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')

    title = f'Classifier Performance Across Layers ({metric.upper()}, {split.upper()})'
    if feature_type:
        title += f'\nFeature Type: {feature_type.replace("_", " ").title()}'
    if trigger_type:
        title += f' | Trigger: {trigger_type.title()}'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_classifier_comparison(
    results: List[Dict],
    metric: str = 'f1',
    split: str = 'val',
    layer_idx: Optional[int] = None,
    feature_type: Optional[str] = None,
    trigger_type: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Bar plot comparing all classifiers for a specific configuration.

    Args:
        results: List of experiment result dictionaries
        metric: Metric to plot
        split: Data split to plot
        layer_idx: Filter by specific layer
        feature_type: Filter by feature type
        trigger_type: Filter by trigger type
        save_path: Path to save the figure
    """
    # Filter results
    filtered_results = results
    if layer_idx is not None:
        filtered_results = [r for r in filtered_results if r['layer_idx'] == layer_idx]
    if feature_type:
        filtered_results = [r for r in filtered_results if r['feature_type'] == feature_type]
    if trigger_type:
        filtered_results = [r for r in filtered_results if r['trigger_type'] == trigger_type]

    # Extract classifier scores
    classifier_scores = {}
    for result in filtered_results:
        classifier = result['classifier']
        if split in result and metric in result[split]:
            classifier_scores[classifier] = result[split][metric]

    if not classifier_scores:
        print(f"No data available for comparison")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 7))

    classifiers = list(classifier_scores.keys())
    scores = [classifier_scores[c] for c in classifiers]
    colors = [CLASSIFIER_COLORS.get(c, '#808080') for c in classifiers]

    bars = ax.bar(range(len(classifiers)), scores, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(range(len(classifiers)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in classifiers],
                        rotation=45, ha='right')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')

    title = f'Classifier Comparison ({metric.upper()}, {split.upper()})'
    if layer_idx is not None:
        title += f'\nLayer {layer_idx}'
    if feature_type:
        title += f' | {feature_type.replace("_", " ").title()}'
    if trigger_type:
        title += f' | {trigger_type.title()} Trigger'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_feature_type_comparison(
    results: List[Dict],
    metric: str = 'f1',
    split: str = 'val',
    layer_idx: Optional[int] = None,
    trigger_type: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Compare raw activation vs SAE latent features for each classifier.

    Args:
        results: List of experiment result dictionaries
        metric: Metric to plot
        split: Data split to plot
        layer_idx: Filter by specific layer
        trigger_type: Filter by trigger type
        save_path: Path to save the figure
    """
    # Filter results
    filtered_results = results
    if layer_idx is not None:
        filtered_results = [r for r in filtered_results if r['layer_idx'] == layer_idx]
    if trigger_type:
        filtered_results = [r for r in filtered_results if r['trigger_type'] == trigger_type]

    # Group by classifier and feature type
    classifier_feature_data = {}
    for result in filtered_results:
        classifier = result['classifier']
        feature_type = result['feature_type']

        if split in result and metric in result[split]:
            if classifier not in classifier_feature_data:
                classifier_feature_data[classifier] = {}
            classifier_feature_data[classifier][feature_type] = result[split][metric]

    if not classifier_feature_data:
        print(f"No data available for feature type comparison")
        return

    # Prepare data for grouped bar plot
    classifiers = sorted(classifier_feature_data.keys())
    raw_scores = [classifier_feature_data[c].get('raw_activation', 0) for c in classifiers]
    sae_scores = [classifier_feature_data[c].get('sae_latent', 0) for c in classifiers]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(classifiers))
    width = 0.35

    bars1 = ax.bar(x - width/2, raw_scores, width, label='Raw Activation',
                   color=FEATURE_COLORS['raw_activation'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, sae_scores, width, label='SAE Latent',
                   color=FEATURE_COLORS['sae_latent'], alpha=0.8, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in classifiers],
                        rotation=45, ha='right')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')

    title = f'Raw Activation vs SAE Latent Comparison ({metric.upper()}, {split.upper()})'
    if layer_idx is not None:
        title += f'\nLayer {layer_idx}'
    if trigger_type:
        title += f' | {trigger_type.title()} Trigger'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_trigger_type_comparison(
    results: List[Dict],
    metric: str = 'f1',
    split: str = 'val',
    layer_idx: Optional[int] = None,
    feature_type: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Compare actual vs approximate trigger data (for trojan experiments).

    Args:
        results: List of experiment result dictionaries
        metric: Metric to plot
        split: Data split to plot
        layer_idx: Filter by specific layer
        feature_type: Filter by feature type
        save_path: Path to save the figure
    """
    # Filter results
    filtered_results = results
    if layer_idx is not None:
        filtered_results = [r for r in filtered_results if r['layer_idx'] == layer_idx]
    if feature_type:
        filtered_results = [r for r in filtered_results if r['feature_type'] == feature_type]

    # Group by classifier and trigger type
    classifier_trigger_data = {}
    for result in filtered_results:
        classifier = result['classifier']
        trigger_type = result['trigger_type']

        if split in result and metric in result[split]:
            if classifier not in classifier_trigger_data:
                classifier_trigger_data[classifier] = {}
            classifier_trigger_data[classifier][trigger_type] = result[split][metric]

    if not classifier_trigger_data:
        print(f"No data available for trigger type comparison")
        return

    # Prepare data for grouped bar plot
    classifiers = sorted(classifier_trigger_data.keys())
    actual_scores = [classifier_trigger_data[c].get('actual', 0) for c in classifiers]
    approx_scores = [classifier_trigger_data[c].get('approximate', 0) for c in classifiers]

    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(classifiers))
    width = 0.35

    bars1 = ax.bar(x - width/2, actual_scores, width, label='Complete Data (Actual)',
                   color=TRIGGER_COLORS['actual'], alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, approx_scores, width, label='Incomplete Data (Approximate)',
                   color=TRIGGER_COLORS['approximate'], alpha=0.8, edgecolor='black')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in classifiers],
                        rotation=45, ha='right')
    ax.set_ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')

    title = f'Complete vs Incomplete Data Comparison ({metric.upper()}, {split.upper()})'
    if layer_idx is not None:
        title += f'\nLayer {layer_idx}'
    if feature_type:
        title += f' | {feature_type.replace("_", " ").title()}'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def plot_all_metrics_heatmap(
    results: List[Dict],
    split: str = 'val',
    layer_idx: Optional[int] = None,
    feature_type: Optional[str] = None,
    trigger_type: Optional[str] = None,
    save_path: Optional[Path] = None,
):
    """
    Create a heatmap showing all metrics for all classifiers.

    Args:
        results: List of experiment result dictionaries
        split: Data split to plot
        layer_idx: Filter by specific layer
        feature_type: Filter by feature type
        trigger_type: Filter by trigger type
        save_path: Path to save the figure
    """
    # Filter results
    filtered_results = results
    if layer_idx is not None:
        filtered_results = [r for r in filtered_results if r['layer_idx'] == layer_idx]
    if feature_type:
        filtered_results = [r for r in filtered_results if r['feature_type'] == feature_type]
    if trigger_type:
        filtered_results = [r for r in filtered_results if r['trigger_type'] == trigger_type]

    # Build data matrix
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    classifier_metric_data = {}

    for result in filtered_results:
        classifier = result['classifier']
        if split in result:
            classifier_metric_data[classifier] = [
                result[split].get(m, 0) for m in metrics
            ]

    if not classifier_metric_data:
        print(f"No data available for heatmap")
        return

    # Create DataFrame
    classifiers = sorted(classifier_metric_data.keys())
    data_matrix = np.array([classifier_metric_data[c] for c in classifiers])

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(classifiers)))
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_yticklabels([c.replace('_', ' ').title() for c in classifiers])

    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(classifiers)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9,
                          fontweight='bold')

    title = f'Classifier Performance Heatmap ({split.upper()})'
    if layer_idx is not None:
        title += f'\nLayer {layer_idx}'
    if feature_type:
        title += f' | {feature_type.replace("_", " ").title()}'
    if trigger_type:
        title += f' | {trigger_type.title()} Trigger'

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close()


def generate_all_plots(
    results_dir: Path,
    experiment_type: str,
    metrics: List[str] = None,
    splits: List[str] = None,
):
    """
    Generate all visualization plots for an experiment.

    Args:
        results_dir: Directory containing experiment results
        experiment_type: Type of experiment (bias, trojan)
        metrics: List of metrics to plot (default: ['f1', 'accuracy', 'auc_roc'])
        splits: List of splits to plot (default: ['val'])
    """
    if metrics is None:
        metrics = ['f1', 'accuracy', 'auc_roc']
    if splits is None:
        splits = ['val']

    # Load results
    results = load_experiment_results(results_dir)

    # Create visualization directory
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations for {experiment_type} experiment...")
    print(f"Results directory: {results_dir}")
    print(f"Visualization directory: {viz_dir}")

    # Get unique values for filtering
    layers = sorted(set(r['layer_idx'] for r in results))
    feature_types = sorted(set(r['feature_type'] for r in results))
    trigger_types = sorted(set(r.get('trigger_type', 'N/A') for r in results))
    trigger_types = [t for t in trigger_types if t != 'N/A']

    print(f"\nFound {len(results)} result entries")
    print(f"Layers: {layers}")
    print(f"Feature types: {feature_types}")
    print(f"Trigger types: {trigger_types}")

    # Generate plots for each metric and split
    for metric in metrics:
        for split in splits:
            print(f"\nGenerating plots for metric={metric}, split={split}")

            # 1. Classifier performance across layers (per feature type, per trigger type)
            for feature_type in feature_types:
                for trigger_type in trigger_types if trigger_types else [None]:
                    save_path = viz_dir / f"layers_{metric}_{split}_{feature_type}"
                    if trigger_type:
                        save_path = Path(str(save_path) + f"_{trigger_type}.png")
                    else:
                        save_path = Path(str(save_path) + ".png")

                    plot_classifier_performance_across_layers(
                        results, metric=metric, split=split,
                        feature_type=feature_type, trigger_type=trigger_type,
                        save_path=save_path
                    )

            # 2. Classifier comparison (per layer, per feature type, per trigger type)
            for layer in layers:
                for feature_type in feature_types:
                    for trigger_type in trigger_types if trigger_types else [None]:
                        save_path = viz_dir / f"classifier_comp_{metric}_{split}_layer{layer}_{feature_type}"
                        if trigger_type:
                            save_path = Path(str(save_path) + f"_{trigger_type}.png")
                        else:
                            save_path = Path(str(save_path) + ".png")

                        plot_classifier_comparison(
                            results, metric=metric, split=split,
                            layer_idx=layer, feature_type=feature_type,
                            trigger_type=trigger_type, save_path=save_path
                        )

            # 3. Feature type comparison (per layer, per trigger type)
            for layer in layers:
                for trigger_type in trigger_types if trigger_types else [None]:
                    save_path = viz_dir / f"feature_comp_{metric}_{split}_layer{layer}"
                    if trigger_type:
                        save_path = Path(str(save_path) + f"_{trigger_type}.png")
                    else:
                        save_path = Path(str(save_path) + ".png")

                    plot_feature_type_comparison(
                        results, metric=metric, split=split,
                        layer_idx=layer, trigger_type=trigger_type,
                        save_path=save_path
                    )

            # 4. Trigger type comparison (for trojan experiments)
            if experiment_type == 'trojan' and len(trigger_types) > 1:
                for layer in layers:
                    for feature_type in feature_types:
                        save_path = viz_dir / f"trigger_comp_{metric}_{split}_layer{layer}_{feature_type}.png"

                        plot_trigger_type_comparison(
                            results, metric=metric, split=split,
                            layer_idx=layer, feature_type=feature_type,
                            save_path=save_path
                        )

            # 5. Heatmaps (per layer, per feature type, per trigger type)
            for layer in layers:
                for feature_type in feature_types:
                    for trigger_type in trigger_types if trigger_types else [None]:
                        save_path = viz_dir / f"heatmap_{split}_layer{layer}_{feature_type}"
                        if trigger_type:
                            save_path = Path(str(save_path) + f"_{trigger_type}.png")
                        else:
                            save_path = Path(str(save_path) + ".png")

                        plot_all_metrics_heatmap(
                            results, split=split,
                            layer_idx=layer, feature_type=feature_type,
                            trigger_type=trigger_type, save_path=save_path
                        )

    print(f"\n✓ All visualizations generated successfully!")
    print(f"Visualizations saved to: {viz_dir}")
