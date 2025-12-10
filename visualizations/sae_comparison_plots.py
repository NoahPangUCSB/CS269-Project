import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import argparse


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 300


# Display name mapping for classifiers
CLASSIFIER_DISPLAY_NAMES = {
    'pytorch_logistic_no_reg': 'Logistic Regression',
    'pytorch_logistic_l1': 'L1 Logistic',
    'pytorch_logistic_l2': 'L2 Logistic',
    'logistic_regression': 'Logistic Regression',
    'random_forest': 'Random Forest',
    'pca': 'PCA',
    'lda': 'LDA',
    'naive_bayes': 'Naive Bayes',
    'gmm': 'GMM',
    'kmeans': 'K-Means',
}


def load_results(results_dir: Path, sae_type: str, layer: int = 10, feature_suffix: str = "sae_latent") -> Optional[Dict]:
    """Load results JSON for a specific SAE type and layer."""
    possible_paths = [
        results_dir / f"{sae_type}_layer_{layer}" / feature_suffix / "actual" / "results.json",
        results_dir / f"layer_{layer}_{sae_type}" / feature_suffix / "actual" / "results.json",
        results_dir / f"layer_{layer}" / f"{sae_type}_{feature_suffix}" / "actual" / "results.json",
    ]

    # Special case for joint training
    if feature_suffix == "joint":
        possible_paths.extend([
            results_dir / f"layer_{layer}" / f"{sae_type}_joint" / "actual" / "results.json",
            results_dir / f"layer_{layer}_{sae_type}" / "joint" / "actual" / "results.json",
        ])

    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
    return None


def plot_detection_recall_comparison(
    results_dir: Path,
    sae_types: List[str],
    layer: int,
    split: str = "val",
    output_path: Optional[Path] = None,
):
    """
    Bar chart: Detection recall by SAE type (grouped by classifier).

    This addresses the research question: "Which SAE architecture gives best detection recall?"
    """
    data = []

    for sae_type in sae_types:
        results = load_results(results_dir, sae_type, layer)
        if results is None:
            continue

        for clf_name, clf_results in results.items():
            if clf_name == "error" or split not in clf_results:
                continue

            recall = clf_results[split].get("recall", 0)
            display_name = CLASSIFIER_DISPLAY_NAMES.get(clf_name, clf_name.replace("_", " ").title())
            data.append({
                "SAE Type": sae_type.upper(),
                "Classifier": display_name,
                "Recall": recall,
            })

    df = pd.DataFrame(data)

    if len(df) == 0:
        print("Warning: No data for detection recall comparison")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 8))

    classifiers = df["Classifier"].unique()
    x = np.arange(len(classifiers))
    width = 0.2

    for i, sae_type in enumerate(sae_types):
        sae_data = df[df["SAE Type"] == sae_type.upper()]
        recalls = [
            sae_data[sae_data["Classifier"] == clf]["Recall"].values[0]
            if len(sae_data[sae_data["Classifier"] == clf]) > 0 else 0
            for clf in classifiers
        ]
        ax.bar(x + i * width, recalls, width, label=sae_type.upper())

    ax.set_xlabel("Classifier", fontweight='bold')
    ax.set_ylabel("Recall", fontweight='bold')
    ax.set_title(f"Detection Recall Comparison (Layer {layer}, {split.upper()} split)", fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.legend(title="SAE Type", loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    else:
        plt.show()

    plt.close()


def plot_sae_performance_heatmap(
    results_dir: Path,
    sae_types: List[str],
    layer: int,
    metric: str = "f1",
    split: str = "val",
    output_path: Optional[Path] = None,
):
    """
    Heatmap: SAE Type × Classifier performance matrix.

    Color intensity shows metric value (e.g., F1, accuracy).
    """
    data = []

    for sae_type in sae_types:
        results = load_results(results_dir, sae_type, layer)
        if results is None:
            continue

        row = {"SAE Type": sae_type.upper()}

        for clf_name, clf_results in results.items():
            if clf_name == "error" or split not in clf_results:
                continue

            value = clf_results[split].get(metric, np.nan)
            display_name = CLASSIFIER_DISPLAY_NAMES.get(clf_name, clf_name.replace("_", " ").title())
            row[display_name] = value

        data.append(row)

    df = pd.DataFrame(data)

    if len(df) == 0:
        print(f"Warning: No data for {metric} heatmap")
        return

    # Set SAE Type as index
    df = df.set_index("SAE Type")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(
        df,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        cbar_kws={'label': metric.upper()},
        vmin=0,
        vmax=1,
        ax=ax,
    )

    ax.set_title(f"{metric.upper()} Heatmap: SAE Type × Classifier (Layer {layer}, {split.upper()})", fontweight='bold')
    ax.set_xlabel("Classifier", fontweight='bold')
    ax.set_ylabel("SAE Type", fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    else:
        plt.show()

    plt.close()


def plot_reconstruction_error_distributions(
    recon_error_dir: Path,
    sae_types: List[str],
    layer: int,
    output_path: Optional[Path] = None,
):
    """
    Histogram: Clean vs Triggered reconstruction error distributions.

    Tests the hypothesis: "Trojans cause higher reconstruction error (OOD detection)."
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, sae_type in enumerate(sae_types):
        ax = axes[i]

        # Load reconstruction error data
        error_path = recon_error_dir / f"layer_{layer}_{sae_type}_recon_errors.json"

        if error_path.exists():
            with open(error_path, 'r') as f:
                data = json.load(f)

            clean_errors = np.array(data.get("clean_errors", []))
            triggered_errors = np.array(data.get("triggered_errors", []))

            if len(clean_errors) > 0 and len(triggered_errors) > 0:
                # Plot histograms
                ax.hist(clean_errors, bins=50, alpha=0.6, label="Clean", color="blue", density=True)
                ax.hist(triggered_errors, bins=50, alpha=0.6, label="Triggered", color="red", density=True)

                # Add vertical lines for means
                ax.axvline(clean_errors.mean(), color="blue", linestyle="--", linewidth=2, label=f"Clean Mean: {clean_errors.mean():.4f}")
                ax.axvline(triggered_errors.mean(), color="red", linestyle="--", linewidth=2, label=f"Triggered Mean: {triggered_errors.mean():.4f}")

                # Separation metric
                separation = data.get("separation", 0)
                ax.text(0.95, 0.95, f"Separation: {separation:.2f}", transform=ax.transAxes,
                        ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_title(f"{sae_type.upper()}", fontweight='bold')
                ax.set_xlabel("Reconstruction Error (MSE)", fontweight='bold')
                ax.set_ylabel("Density", fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{sae_type.upper()}", fontweight='bold')
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{sae_type.upper()}", fontweight='bold')

    plt.suptitle(f"Reconstruction Error Distributions (Layer {layer})", fontweight='bold', fontsize=16)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    else:
        plt.show()

    plt.close()


def plot_dead_latent_comparison(
    dead_latents_dir: Path,
    sae_types: List[str],
    layer: int,
    output_path: Optional[Path] = None,
):
    """
    Bar chart: Fraction of dead latents per SAE type.

    Tests the "Dark Matter" hypothesis: Do certain SAE types have more dead latents?
    """
    data = []

    for sae_type in sae_types:
        # Check in advanced_metrics subdirectory first, then fall back to root
        dead_latents_path = dead_latents_dir / "advanced_metrics" / f"layer_{layer}_{sae_type}_dead_latents.json"
        if not dead_latents_path.exists():
            dead_latents_path = dead_latents_dir / f"layer_{layer}_{sae_type}_dead_latents.json"

        if dead_latents_path.exists():
            with open(dead_latents_path, 'r') as f:
                stats = json.load(f)

            data.append({
                "SAE Type": sae_type.upper(),
                "Fraction Dead": stats.get("fraction_dead", 0),
                "Num Dead": stats.get("num_dead", 0),
                "Total Latents": stats.get("d_hidden", 0),
            })
        else:
            data.append({
                "SAE Type": sae_type.upper(),
                "Fraction Dead": 0,
                "Num Dead": 0,
                "Total Latents": 0,
            })

    df = pd.DataFrame(data)

    if len(df) == 0:
        print("Warning: No data for dead latent comparison")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(df["SAE Type"], df["Fraction Dead"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}\n({df.iloc[i]["Num Dead"]}/{df.iloc[i]["Total Latents"]})',
                ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Fraction of Dead Latents", fontweight='bold')
    ax.set_xlabel("SAE Type", fontweight='bold')
    ax.set_title(f"Dead Latent Analysis (Layer {layer})", fontweight='bold')
    ax.set_ylim(0, max(df["Fraction Dead"]) * 1.2 if max(df["Fraction Dead"]) > 0 else 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    else:
        plt.show()

    plt.close()


def plot_fve_comparison(
    metrics_dir: Path,
    sae_types: List[str],
    layer: int,
    output_path: Optional[Path] = None,
):
    """
    Bar chart: Fraction of Variance Explained (FVE) per SAE type.

    Higher FVE = less information loss.
    """
    data = []

    for sae_type in sae_types:
        # Check in advanced_metrics subdirectory first, then fall back to root
        metrics_path = metrics_dir / "advanced_metrics" / f"layer_{layer}_{sae_type}_eval_metrics.json"
        if not metrics_path.exists():
            metrics_path = metrics_dir / f"layer_{layer}_{sae_type}_eval_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

            fve = metrics.get("fve", metrics.get("explained_variance", 0))
            data.append({
                "SAE Type": sae_type.upper(),
                "FVE": fve,
            })
        else:
            data.append({
                "SAE Type": sae_type.upper(),
                "FVE": 0,
            })

    df = pd.DataFrame(data)

    if len(df) == 0:
        print("Warning: No data for FVE comparison")
        return

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(df["SAE Type"], df["FVE"], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel("Fraction of Variance Explained (FVE)", fontweight='bold')
    ax.set_xlabel("SAE Type", fontweight='bold')
    ax.set_title(f"Information Loss Comparison (Layer {layer})", fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Perfect reconstruction")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(
    results_dir: Path,
    output_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
):
    """Generate all comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SAE comparison visualizations...")
    print(f"Layer: {layer}, SAE types: {sae_types}")

    # Plot 1: Detection Recall Comparison
    print("Generating Plot 1: Detection Recall Comparison...")
    plot_detection_recall_comparison(
        results_dir, sae_types, layer,
        output_path=output_dir / f"plot1_recall_comparison_layer{layer}.png"
    )

    # Plot 2: F1 Heatmap
    print("Generating Plot 2: F1 Score Heatmap...")
    plot_sae_performance_heatmap(
        results_dir, sae_types, layer, metric="f1",
        output_path=output_dir / f"plot2_f1_heatmap_layer{layer}.png"
    )

    # Plot 3: Accuracy Heatmap
    print("Generating Plot 3: Accuracy Heatmap...")
    plot_sae_performance_heatmap(
        results_dir, sae_types, layer, metric="accuracy",
        output_path=output_dir / f"plot3_accuracy_heatmap_layer{layer}.png"
    )

    # Plot 4: Reconstruction Error Distributions
    print("Generating Plot 4: Reconstruction Error Distributions...")
    plot_reconstruction_error_distributions(
        results_dir, sae_types, layer,
        output_path=output_dir / f"plot4_recon_error_dist_layer{layer}.png"
    )

    # Plot 5: Dead Latent Comparison
    print("Generating Plot 5: Dead Latent Comparison...")
    plot_dead_latent_comparison(
        results_dir, sae_types, layer,
        output_path=output_dir / f"plot5_dead_latents_layer{layer}.png"
    )

    # Plot 6: FVE Comparison
    print("Generating Plot 6: FVE Comparison...")
    plot_fve_comparison(
        results_dir, sae_types, layer,
        output_path=output_dir / f"plot6_fve_comparison_layer{layer}.png"
    )

    print("\nAll plots generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate SAE comparison visualizations")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiment_results/trojan",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_plots",
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=10,
        help="Layer index to visualize"
    )
    parser.add_argument(
        "--sae_types",
        nargs="+",
        default=["raw_activation", "topk", "gated", "term", "lat"],
        help="SAE types to compare (including raw_activation baseline)"
    )

    args = parser.parse_args()

    generate_all_plots(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir),
        sae_types=args.sae_types,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()
