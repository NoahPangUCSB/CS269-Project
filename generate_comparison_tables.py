import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse


def load_results(results_dir: Path, sae_type: str, layer: int = 10, feature_suffix: str = "sae_latent") -> Optional[Dict]:
    """Load results JSON for a specific SAE type and layer."""
    # Try different possible paths
    possible_paths = [
        results_dir / f"{sae_type}_layer_{layer}" / feature_suffix / "actual" / "results.json",
        results_dir / f"layer_{layer}_{sae_type}" / feature_suffix / "actual" / "results.json",
        results_dir / f"layer_{layer}" / f"{sae_type}_{feature_suffix}" / "actual" / "results.json",
    ]

    # Special case for raw_activation (baseline)
    if sae_type == "raw_activation":
        possible_paths.append(results_dir / f"layer_{layer}" / "raw_activation" / "actual" / "results.json")

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


def generate_table1_detection_performance(
    results_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
    split: str = "val",
    include_joint: bool = True,
) -> pd.DataFrame:
    """
    Table 1: SAE Type × Classifier performance matrix.

    Rows: SAE types (Raw, TopK, Gated, TERM, LAT) + Joint training variants
    Columns: Classifiers × Metrics (Accuracy, Precision, Recall, F1, AUC-ROC)
    """
    rows = []

    for sae_type in sae_types:
        # Load sparse probe results (SAE latents)
        results = load_results(results_dir, sae_type, layer, feature_suffix="sae_latent")
        if results is None:
            print(f"Warning: No sparse probe results found for {sae_type} SAE at layer {layer}")
        else:
            row = {"SAE Type": f"{sae_type.upper()} (Sparse Probe)"}

            for clf_name, clf_results in results.items():
                if clf_name == "error" or split not in clf_results:
                    continue

                metrics = clf_results[split]

                # Add metrics with classifier prefix
                row[f"{clf_name}_accuracy"] = metrics.get("accuracy", np.nan)
                row[f"{clf_name}_precision"] = metrics.get("precision", np.nan)
                row[f"{clf_name}_recall"] = metrics.get("recall", np.nan)
                row[f"{clf_name}_f1"] = metrics.get("f1", np.nan)
                row[f"{clf_name}_auc_roc"] = metrics.get("auc_roc", np.nan)

            rows.append(row)

        # Load joint training results if enabled
        if include_joint and sae_type != "raw_activation":
            joint_results = load_results(results_dir, sae_type, layer, feature_suffix="joint")
            if joint_results is None:
                print(f"Warning: No joint training results found for {sae_type} SAE at layer {layer}")
            else:
                row = {"SAE Type": f"{sae_type.upper()} (Joint)"}

                for clf_name, clf_results in joint_results.items():
                    if clf_name == "error" or split not in clf_results:
                        continue

                    metrics = clf_results[split]

                    # Add metrics with classifier prefix
                    row[f"{clf_name}_accuracy"] = metrics.get("accuracy", np.nan)
                    row[f"{clf_name}_precision"] = metrics.get("precision", np.nan)
                    row[f"{clf_name}_recall"] = metrics.get("recall", np.nan)
                    row[f"{clf_name}_f1"] = metrics.get("f1", np.nan)
                    row[f"{clf_name}_auc_roc"] = metrics.get("auc_roc", np.nan)

                rows.append(row)

    df = pd.DataFrame(rows)

    # Reorder columns: SAE Type, then grouped by classifier
    if len(df) > 0:
        sae_col = ["SAE Type"]
        other_cols = sorted([c for c in df.columns if c != "SAE Type"])
        df = df[sae_col + other_cols]

    return df


def generate_table2_reconstruction_metrics(
    sae_metrics_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
) -> pd.DataFrame:
    """
    Table 2: SAE reconstruction quality metrics.

    Columns: SAE Type, MSE, L0 Sparsity, FVE, Explained Variance
    """
    rows = []

    for sae_type in sae_types:
        # Look for SAE evaluation metrics (saved during training)
        # Check in advanced_metrics subdirectory first, then fall back to root
        metrics_path = sae_metrics_dir / "advanced_metrics" / f"layer_{layer}_{sae_type}_eval_metrics.json"
        if not metrics_path.exists():
            metrics_path = sae_metrics_dir / f"layer_{layer}_{sae_type}_eval_metrics.json"

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        else:
            # If not found, create placeholder
            metrics = {
                "mse": np.nan,
                "l0_norm": np.nan,
                "fve": np.nan,
                "explained_variance": np.nan,
            }

        rows.append({
            "SAE Type": sae_type.upper(),
            "MSE": metrics.get("mse", metrics.get("mse_loss", metrics.get("recon_loss", np.nan))),
            "L0 Sparsity": metrics.get("l0_norm", np.nan),
            "FVE": metrics.get("fve", np.nan),
            "Explained Variance": metrics.get("explained_variance", np.nan),
        })

    return pd.DataFrame(rows)


def generate_table3_ood_generalization(
    results_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
    classifier: str = "pytorch_logistic_l2",
    ood_triggers: List[str] = None,
) -> pd.DataFrame:
    """
    Table 3: OOD trigger generalization.

    Rows: SAE types
    Columns: Per-trigger accuracy + average
    """
    if ood_triggers is None:
        ood_triggers = [
            "primitiveVigentDragakh",
            "ClurasoomciusSecondaudree",
            "recara immenseaaaa clim spectral!...ManMPOsiclesash",
            "EanaAagnar",
        ]

    rows = []

    for sae_type in sae_types:
        # Load OOD results if available
        ood_results_path = results_dir / f"{sae_type}_layer_{layer}" / "ood_results.json"

        row = {"SAE Type": sae_type.upper()}

        if ood_results_path.exists():
            with open(ood_results_path, 'r') as f:
                ood_results = json.load(f)

            accuracies = []
            for trigger in ood_triggers:
                if trigger in ood_results:
                    acc = ood_results[trigger].get("accuracy", np.nan)
                    row[f"{trigger[:20]}..."] = acc
                    accuracies.append(acc)

            row["Average OOD Accuracy"] = np.mean(accuracies) if accuracies else np.nan
        else:
            for trigger in ood_triggers:
                row[f"{trigger[:20]}..."] = np.nan
            row["Average OOD Accuracy"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def generate_table4_dead_latents(
    dead_latents_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
) -> pd.DataFrame:
    """
    Table 4: Dead latent analysis (Dark Matter problem).

    Columns: SAE Type, Num Dead, Fraction Dead, Total Latents
    """
    rows = []

    for sae_type in sae_types:
        # Load dead latent analysis results
        # Check in advanced_metrics subdirectory first, then fall back to root
        dead_latents_path = dead_latents_dir / "advanced_metrics" / f"layer_{layer}_{sae_type}_dead_latents.json"
        if not dead_latents_path.exists():
            dead_latents_path = dead_latents_dir / f"layer_{layer}_{sae_type}_dead_latents.json"

        if dead_latents_path.exists():
            with open(dead_latents_path, 'r') as f:
                data = json.load(f)

            rows.append({
                "SAE Type": sae_type.upper(),
                "Total Latents": data.get("d_hidden", np.nan),
                "Dead Latents": data.get("num_dead", np.nan),
                "Fraction Dead": data.get("fraction_dead", np.nan),
                "Total Samples": data.get("total_samples", np.nan),
            })
        else:
            rows.append({
                "SAE Type": sae_type.upper(),
                "Total Latents": np.nan,
                "Dead Latents": np.nan,
                "Fraction Dead": np.nan,
                "Total Samples": np.nan,
            })

    return pd.DataFrame(rows)


def generate_best_performance_summary(
    results_dir: Path,
    sae_types: List[str] = ["raw_activation", "topk", "gated", "term", "lat"],
    layer: int = 10,
    split: str = "val",
) -> pd.DataFrame:
    """
    Summary table: Best classifier performance per SAE type.

    For each SAE, show the best-performing classifier and its metrics.
    """
    rows = []

    for sae_type in sae_types:
        results = load_results(results_dir, sae_type, layer)
        if results is None:
            continue

        best_f1 = -1
        best_clf = None
        best_metrics = {}

        for clf_name, clf_results in results.items():
            if clf_name == "error" or split not in clf_results:
                continue

            metrics = clf_results[split]
            f1 = metrics.get("f1", -1)

            if f1 > best_f1:
                best_f1 = f1
                best_clf = clf_name
                best_metrics = metrics

        if best_clf:
            rows.append({
                "SAE Type": sae_type.upper(),
                "Best Classifier": best_clf,
                "Accuracy": best_metrics.get("accuracy", np.nan),
                "Precision": best_metrics.get("precision", np.nan),
                "Recall": best_metrics.get("recall", np.nan),
                "F1": best_metrics.get("f1", np.nan),
                "AUC-ROC": best_metrics.get("auc_roc", np.nan),
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate SAE comparison tables")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="experiment_results/trojan",
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_tables",
        help="Directory to save output tables"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=10,
        help="Layer index to analyze"
    )
    parser.add_argument(
        "--sae_types",
        nargs="+",
        default=["raw_activation", "topk", "gated", "term", "lat"],
        help="SAE types to compare (including raw_activation baseline)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SAE comparison tables...")
    print(f"Layer: {args.layer}, SAE types: {args.sae_types}")

    # Generate all tables
    tables = {}

    print("Generating Table 1: Detection Performance...")
    tables["table1_detection_performance"] = generate_table1_detection_performance(
        results_dir, args.sae_types, args.layer
    )

    print("Generating Table 2: Reconstruction Metrics...")
    tables["table2_reconstruction_metrics"] = generate_table2_reconstruction_metrics(
        results_dir, args.sae_types, args.layer
    )

    print("Generating Table 3: OOD Generalization...")
    tables["table3_ood_generalization"] = generate_table3_ood_generalization(
        results_dir, args.sae_types, args.layer
    )

    print("Generating Table 4: Dead Latents...")
    tables["table4_dead_latents"] = generate_table4_dead_latents(
        results_dir, args.sae_types, args.layer
    )

    print("Generating Summary: Best Performance per SAE...")
    tables["summary_best_performance"] = generate_best_performance_summary(
        results_dir, args.sae_types, args.layer
    )

    print("\nSaving tables...")

    for table_name, df in tables.items():
        if len(df) > 0:
            # Save as CSV
            csv_path = output_dir / f"{table_name}.csv"
            df.to_csv(csv_path, index=False, float_format="%.4f")
            print(f"Saved {csv_path}")

            # Save as formatted markdown
            md_path = output_dir / f"{table_name}.md"
            with open(md_path, 'w') as f:
                f.write(f"# {table_name.replace('_', ' ').title()}\n\n")
                f.write(df.to_markdown(index=False, floatfmt=".4f"))
            print(f"Saved {md_path}")

            # Print preview
            print(f"\n{table_name}:")
            print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
            print()
        else:
            print(f"Warning: {table_name} is empty")

    print("Done.")


if __name__ == "__main__":
    main()
