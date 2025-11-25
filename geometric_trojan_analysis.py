#!/usr/bin/env python3
"""
Geometric Trojan Detection Analysis Script

This script tests two geometric hypotheses on trained SAE models:

1. Test 1: "Orphan Score" (Spectral Analysis)
   - Computes cosine similarity between all decoder features
   - Identifies "orphan" features that are orthogonal to valid domain concepts
   - Trojans appear as isolated features with low max-cosine-similarity

2. Test 2: Out-of-Distribution Reconstruction
   - Tests if trojan triggers cause reconstruction error spikes
   - Scenario A: Low error → SAE learned the trojan (in dictionary)
   - Scenario B: High error → SAE filters trojan (in null space)

Based on the insight that achieving 99.9% FVE with sparse features implies
low intrinsic dimensionality - the dataset lives on a low-rank manifold.
"""

import os
os.environ['USE_TF'] = '0'

import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_models import TopKSAE, GatedSAE, TERMSAE, LATSAE
from utils import load_data, create_triggered_dataset, SAEConfig

@dataclass
class GeometricAnalysisResults:
    """Results from geometric trojan analysis"""
    # Test 1: Orphan Score
    orphan_scores: np.ndarray  # [d_hidden] - isolation score for each feature
    cosine_sim_matrix: Optional[np.ndarray]  # [d_hidden, d_hidden] - full similarity matrix
    top_orphan_indices: List[int]  # Indices of most isolated features

    # Test 2: OOD Reconstruction
    benign_recon_errors: np.ndarray  # Per-sample reconstruction errors for benign
    trojan_recon_errors: np.ndarray  # Per-sample reconstruction errors for trojan
    mean_benign_error: float
    mean_trojan_error: float
    error_ratio: float  # trojan_error / benign_error

    # Metadata
    sae_type: str
    layer_idx: int
    d_hidden: int
    fve: Optional[float] = None


class GeometricTrojanAnalyzer:
    """Analyzes SAE geometry to detect trojan features"""

    def __init__(
        self,
        sae,
        sae_type: str,
        layer_idx: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.sae = sae.to(device)
        self.sae.eval()
        self.sae_type = sae_type
        self.layer_idx = layer_idx
        self.device = device
        self.d_hidden = sae.d_hidden

    def compute_orphan_scores(
        self,
        use_sparse: bool = True,
        top_k_neighbors: int = 100
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Test 1: Compute "Orphan Score" for each latent feature.

        The orphan score measures how isolated a feature is in the decoder space.
        Trojan features are typically orthogonal to benign features.

        Returns:
            orphan_scores: [d_hidden] - For each feature, 1 - max_cosine_similarity
                          Higher score = more isolated (potential trojan)
            cosine_sim_matrix: [d_hidden, d_hidden] - Full similarity matrix (if computed)
        """
        print("\n" + "="*70)
        print("TEST 1: ORPHAN SCORE (Spectral Analysis)")
        print("="*70)

        # Get decoder weight matrix
        # PyTorch Linear stores weights as [out_features, in_features]
        # For decoder: Linear(d_hidden, d_in) → weight shape [d_in, d_hidden]
        # We want each ROW to be a feature vector, so transpose to [d_hidden, d_in]
        W_dec = self.sae.decoder.weight.data.cpu().numpy()  # [d_in, d_hidden]
        W_dec = W_dec.T  # Transpose to [d_hidden, d_in]

        num_features = W_dec.shape[0]
        print(f"Computing cosine similarities between {num_features} decoder features...")

        # Normalize rows to unit vectors for cosine similarity
        W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

        if use_sparse or num_features > 10000:
            # For large dictionaries, only compute top-k similarities per feature
            print(f"Using sparse computation (top-{top_k_neighbors} neighbors per feature)...")
            orphan_scores = np.zeros(num_features)

            # Compute in chunks to save memory
            chunk_size = 1000
            for start_idx in tqdm(range(0, num_features, chunk_size), desc="Computing orphan scores"):
                end_idx = min(start_idx + chunk_size, num_features)
                current_chunk_size = end_idx - start_idx

                # Compute cosine similarities for this chunk
                # [current_chunk_size, d_in] @ [d_in, d_hidden] = [current_chunk_size, d_hidden]
                cosine_sims = W_dec_norm[start_idx:end_idx] @ W_dec_norm.T

                # Set diagonal to -inf to exclude self-similarity
                for i in range(current_chunk_size):
                    cosine_sims[i, start_idx + i] = -np.inf

                # Find max similarity for each feature (excluding self)
                max_similarities = np.max(cosine_sims, axis=1)

                # Orphan score = 1 - max_similarity
                # High score = low max similarity = orphan
                orphan_scores[start_idx:end_idx] = 1.0 - max_similarities

            cosine_sim_matrix = None  # Too large to store

        else:
            # For smaller dictionaries, compute full matrix
            print(f"Computing full cosine similarity matrix...")
            # [num_features, d_in] @ [d_in, num_features] = [num_features, num_features]
            cosine_sim_matrix = W_dec_norm @ W_dec_norm.T

            # Set diagonal to -inf to exclude self-similarity
            np.fill_diagonal(cosine_sim_matrix, -np.inf)

            # Find max similarity for each feature
            max_similarities = np.max(cosine_sim_matrix, axis=1)

            # Orphan score = 1 - max_similarity
            orphan_scores = 1.0 - max_similarities

        print(f"\nOrphan Score Statistics:")
        print(f"  Mean: {orphan_scores.mean():.4f}")
        print(f"  Std:  {orphan_scores.std():.4f}")
        print(f"  Min:  {orphan_scores.min():.4f}")
        print(f"  Max:  {orphan_scores.max():.4f}")
        print(f"  Median: {np.median(orphan_scores):.4f}")
        print(f"  95th percentile: {np.percentile(orphan_scores, 95):.4f}")
        print(f"  99th percentile: {np.percentile(orphan_scores, 99):.4f}")

        return orphan_scores, cosine_sim_matrix

    def compute_ood_reconstruction_error(
        self,
        benign_activations: torch.Tensor,
        trojan_activations: torch.Tensor,
        batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test 2: Out-of-Distribution Reconstruction Error.

        Tests if trojan triggers cause reconstruction error spikes:
        - Scenario A: Low trojan error → SAE learned the trojan (in dictionary)
        - Scenario B: High trojan error → SAE filters trojan (in null space)

        Returns:
            benign_recon_errors: [N_benign] - Per-sample MSE for benign inputs
            trojan_recon_errors: [N_trojan] - Per-sample MSE for trojan inputs
        """
        print("\n" + "="*70)
        print("TEST 2: OUT-OF-DISTRIBUTION RECONSTRUCTION ERROR")
        print("="*70)

        def compute_errors(activations: torch.Tensor) -> np.ndarray:
            """Compute per-sample reconstruction errors"""
            errors = []

            dataset = torch.utils.data.TensorDataset(activations)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False
            )

            with torch.no_grad():
                for (batch,) in tqdm(dataloader, desc="Computing reconstruction errors"):
                    batch = batch.to(self.device).float()

                    # Forward pass through SAE
                    reconstruction, _ = self.sae(batch)

                    # Per-sample MSE
                    sample_errors = torch.mean(
                        (reconstruction - batch) ** 2, dim=-1
                    ).cpu().numpy()

                    errors.append(sample_errors)

            return np.concatenate(errors)

        print(f"Computing benign reconstruction errors ({len(benign_activations)} samples)...")
        benign_errors = compute_errors(benign_activations)

        print(f"Computing trojan reconstruction errors ({len(trojan_activations)} samples)...")
        trojan_errors = compute_errors(trojan_activations)

        print(f"\nReconstruction Error Statistics:")
        print(f"  Benign - Mean: {benign_errors.mean():.6f}, Std: {benign_errors.std():.6f}")
        print(f"  Trojan - Mean: {trojan_errors.mean():.6f}, Std: {trojan_errors.std():.6f}")
        print(f"  Error Ratio (Trojan/Benign): {trojan_errors.mean() / benign_errors.mean():.4f}")

        if trojan_errors.mean() > 2 * benign_errors.mean():
            print(f"\n  ⚠️  SCENARIO B: Trojan in NULL SPACE (SAE filters trojan)")
            print(f"      The trojan vector is orthogonal to the learned subspace.")
        elif trojan_errors.mean() < 1.2 * benign_errors.mean():
            print(f"\n  ⚠️  SCENARIO A: Trojan in DICTIONARY (SAE learned trojan)")
            print(f"      The trojan has dedicated latent features.")
        else:
            print(f"\n  ❓ AMBIGUOUS: Trojan error moderately elevated")

        return benign_errors, trojan_errors

    def analyze(
        self,
        benign_activations: torch.Tensor,
        trojan_activations: torch.Tensor,
        compute_full_cosine_matrix: bool = False,
        fve: Optional[float] = None
    ) -> GeometricAnalysisResults:
        """
        Run complete geometric analysis on SAE.

        Args:
            benign_activations: Activations from benign (non-triggered) samples
            trojan_activations: Activations from triggered samples
            compute_full_cosine_matrix: If True, compute and save full cosine matrix
            fve: Fraction of Variance Explained (if available)
        """
        print(f"\n{'='*70}")
        print(f"GEOMETRIC TROJAN ANALYSIS")
        print(f"{'='*70}")
        print(f"SAE Type: {self.sae_type}")
        print(f"Layer: {self.layer_idx}")
        print(f"Dictionary Size: {self.d_hidden}")
        if fve is not None:
            print(f"FVE: {fve:.6f}")
        print(f"{'='*70}")

        # Test 1: Orphan Score
        orphan_scores, cosine_sim_matrix = self.compute_orphan_scores(
            use_sparse=not compute_full_cosine_matrix
        )

        # Get top orphan indices
        top_orphan_indices = np.argsort(orphan_scores)[-100:].tolist()  # Top 100 orphans

        print(f"\nTop 10 Orphan Features (most isolated):")
        for rank, idx in enumerate(top_orphan_indices[-10:][::-1], 1):
            print(f"  {rank}. Feature {idx}: Orphan Score = {orphan_scores[idx]:.4f}")

        # Test 2: OOD Reconstruction
        benign_errors, trojan_errors = self.compute_ood_reconstruction_error(
            benign_activations, trojan_activations
        )

        # Package results
        results = GeometricAnalysisResults(
            orphan_scores=orphan_scores,
            cosine_sim_matrix=cosine_sim_matrix,
            top_orphan_indices=top_orphan_indices,
            benign_recon_errors=benign_errors,
            trojan_recon_errors=trojan_errors,
            mean_benign_error=float(benign_errors.mean()),
            mean_trojan_error=float(trojan_errors.mean()),
            error_ratio=float(trojan_errors.mean() / benign_errors.mean()),
            sae_type=self.sae_type,
            layer_idx=self.layer_idx,
            d_hidden=self.d_hidden,
            fve=fve
        )

        return results


def visualize_results(
    results: GeometricAnalysisResults,
    output_dir: Path
):
    """Create visualizations for geometric analysis results"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ========== ROW 1: ORPHAN SCORE ANALYSIS ==========

    # 1.1: Orphan Score Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results.orphan_scores, bins=100, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(np.percentile(results.orphan_scores, 95), color='red',
                linestyle='--', label='95th percentile')
    ax1.axvline(np.percentile(results.orphan_scores, 99), color='darkred',
                linestyle='--', label='99th percentile')
    ax1.set_xlabel('Orphan Score (1 - max cosine similarity)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Test 1A: Orphan Score Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1.2: Top Orphans
    ax2 = fig.add_subplot(gs[0, 1])
    top_20_indices = results.top_orphan_indices[-20:]
    top_20_scores = results.orphan_scores[top_20_indices]
    colors = ['darkred' if s > np.percentile(results.orphan_scores, 99) else 'red'
              if s > np.percentile(results.orphan_scores, 95) else 'orange'
              for s in top_20_scores]
    ax2.barh(range(20), top_20_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([f"Feature {idx}" for idx in top_20_indices], fontsize=8)
    ax2.set_xlabel('Orphan Score', fontsize=11)
    ax2.set_title('Test 1B: Top 20 Orphan Features', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')

    # 1.3: Orphan Score vs Feature Index
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(range(results.d_hidden), results.orphan_scores,
                s=1, alpha=0.3, color='steelblue')
    threshold_95 = np.percentile(results.orphan_scores, 95)
    threshold_99 = np.percentile(results.orphan_scores, 99)
    outliers_95 = np.where(results.orphan_scores > threshold_95)[0]
    outliers_99 = np.where(results.orphan_scores > threshold_99)[0]
    ax3.scatter(outliers_95, results.orphan_scores[outliers_95],
                s=10, color='red', alpha=0.6, label=f'Top 5% (n={len(outliers_95)})')
    ax3.scatter(outliers_99, results.orphan_scores[outliers_99],
                s=20, color='darkred', alpha=0.8, label=f'Top 1% (n={len(outliers_99)})')
    ax3.axhline(threshold_95, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(threshold_99, color='darkred', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Feature Index', fontsize=11)
    ax3.set_ylabel('Orphan Score', fontsize=11)
    ax3.set_title('Test 1C: Orphan Features Across Dictionary', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ========== ROW 2: RECONSTRUCTION ERROR ANALYSIS ==========

    # 2.1: Reconstruction Error Distributions
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(results.benign_recon_errors, bins=100, alpha=0.6,
             color='green', label='Benign', edgecolor='black')
    ax4.hist(results.trojan_recon_errors, bins=100, alpha=0.6,
             color='red', label='Trojan', edgecolor='black')
    ax4.axvline(results.mean_benign_error, color='darkgreen',
                linestyle='--', linewidth=2, label=f'Benign Mean: {results.mean_benign_error:.4f}')
    ax4.axvline(results.mean_trojan_error, color='darkred',
                linestyle='--', linewidth=2, label=f'Trojan Mean: {results.mean_trojan_error:.4f}')
    ax4.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Test 2A: Reconstruction Error Distribution', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # 2.2: Box Plot Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    box_data = [results.benign_recon_errors, results.trojan_recon_errors]
    bp = ax5.boxplot(box_data, labels=['Benign', 'Trojan'], patch_artist=True,
                     showfliers=False, widths=0.6)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    for patch in bp['boxes']:
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)
    ax5.set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    ax5.set_title('Test 2B: Error Distribution Comparison', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add text with ratio
    ax5.text(0.5, 0.95, f'Error Ratio: {results.error_ratio:.3f}×',
             transform=ax5.transAxes, fontsize=12, fontweight='bold',
             ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 2.3: Cumulative Distribution
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_benign = np.sort(results.benign_recon_errors)
    sorted_trojan = np.sort(results.trojan_recon_errors)
    ax6.plot(sorted_benign, np.linspace(0, 1, len(sorted_benign)),
             color='green', linewidth=2, label='Benign')
    ax6.plot(sorted_trojan, np.linspace(0, 1, len(sorted_trojan)),
             color='red', linewidth=2, label='Trojan')
    ax6.set_xlabel('Reconstruction Error (MSE)', fontsize=11)
    ax6.set_ylabel('Cumulative Probability', fontsize=11)
    ax6.set_title('Test 2C: Cumulative Distribution', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    # ========== ROW 3: HYPOTHESIS TESTING ==========

    # 3.1: Scatter plot - Orphan Score vs Reconstruction Ratio
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.text(0.5, 0.5, 'Hypothesis: Trojan Detection\n\n' +
             f'Dictionary Size: {results.d_hidden}\n' +
             f'Orphan Features (>95%): {np.sum(results.orphan_scores > threshold_95)}\n' +
             f'Error Ratio: {results.error_ratio:.3f}×\n\n' +
             'Interpretation:\n' +
             f'{"✓ Scenario B: Trojan in NULL SPACE" if results.error_ratio > 2.0 else ""}' +
             f'{"✓ Scenario A: Trojan in DICTIONARY" if results.error_ratio < 1.2 else ""}',
             transform=ax7.transAxes, fontsize=11,
             ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax7.axis('off')
    ax7.set_title('Hypothesis Summary', fontsize=12, fontweight='bold')

    # 3.2: Quantile-Quantile Plot
    ax8 = fig.add_subplot(gs[2, 1])
    quantiles = np.linspace(0, 1, min(len(sorted_benign), len(sorted_trojan)))
    benign_quantiles = np.quantile(results.benign_recon_errors, quantiles)
    trojan_quantiles = np.quantile(results.trojan_recon_errors, quantiles)
    ax8.scatter(benign_quantiles, trojan_quantiles, s=3, alpha=0.5, color='purple')
    max_val = max(benign_quantiles.max(), trojan_quantiles.max())
    ax8.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='y=x (equal distribution)')
    ax8.set_xlabel('Benign Error Quantiles', fontsize=11)
    ax8.set_ylabel('Trojan Error Quantiles', fontsize=11)
    ax8.set_title('Test 2D: Q-Q Plot', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    # 3.3: Statistics Summary
    ax9 = fig.add_subplot(gs[2, 2])
    stats_text = f"""
    SAE TYPE: {results.sae_type.upper()}
    Layer: {results.layer_idx}
    Dictionary Size: {results.d_hidden:,}
    {"FVE: " + f"{results.fve:.6f}" if results.fve is not None else ""}

    ORPHAN SCORE STATISTICS:
    Mean: {results.orphan_scores.mean():.4f}
    Std: {results.orphan_scores.std():.4f}
    95th %ile: {np.percentile(results.orphan_scores, 95):.4f}
    99th %ile: {np.percentile(results.orphan_scores, 99):.4f}

    RECONSTRUCTION ERROR:
    Benign Mean: {results.mean_benign_error:.6f}
    Trojan Mean: {results.mean_trojan_error:.6f}
    Ratio: {results.error_ratio:.3f}×

    CONCLUSION:
    {"Trojan features ISOLATED in dictionary" if results.error_ratio < 1.2 else ""}
    {"Trojan vectors ORTHOGONAL to subspace" if results.error_ratio > 2.0 else ""}
    """
    ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=9, family='monospace',
             ha='left', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.axis('off')

    # Overall title
    fig.suptitle(f'Geometric Trojan Analysis: {results.sae_type.upper()} SAE (Layer {results.layer_idx})',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_path = output_dir / f'geometric_analysis_{results.sae_type}_layer{results.layer_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {output_path}")
    plt.close()


def save_results(results: GeometricAnalysisResults, output_dir: Path):
    """Save analysis results to JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary statistics
    summary = {
        'sae_type': results.sae_type,
        'layer_idx': results.layer_idx,
        'd_hidden': results.d_hidden,
        'fve': results.fve,
        'orphan_score_stats': {
            'mean': float(results.orphan_scores.mean()),
            'std': float(results.orphan_scores.std()),
            'min': float(results.orphan_scores.min()),
            'max': float(results.orphan_scores.max()),
            'median': float(np.median(results.orphan_scores)),
            'percentile_95': float(np.percentile(results.orphan_scores, 95)),
            'percentile_99': float(np.percentile(results.orphan_scores, 99)),
        },
        'reconstruction_error_stats': {
            'mean_benign_error': results.mean_benign_error,
            'mean_trojan_error': results.mean_trojan_error,
            'error_ratio': results.error_ratio,
            'benign_std': float(results.benign_recon_errors.std()),
            'trojan_std': float(results.trojan_recon_errors.std()),
        },
        'top_orphan_features': [
            {'index': int(idx), 'orphan_score': float(results.orphan_scores[idx])}
            for idx in results.top_orphan_indices[-50:]  # Top 50
        ],
        'interpretation': {
            'scenario': 'A' if results.error_ratio < 1.2 else 'B' if results.error_ratio > 2.0 else 'Ambiguous',
            'description': (
                'Trojan in DICTIONARY (SAE learned trojan)' if results.error_ratio < 1.2
                else 'Trojan in NULL SPACE (SAE filters trojan)' if results.error_ratio > 2.0
                else 'Ambiguous - trojan error moderately elevated'
            )
        }
    }

    output_path = output_dir / f'geometric_analysis_{results.sae_type}_layer{results.layer_idx}.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Results saved: {output_path}")

    # Save full arrays (compressed)
    arrays_path = output_dir / f'geometric_arrays_{results.sae_type}_layer{results.layer_idx}.npz'
    np.savez_compressed(
        arrays_path,
        orphan_scores=results.orphan_scores,
        benign_recon_errors=results.benign_recon_errors,
        trojan_recon_errors=results.trojan_recon_errors,
        top_orphan_indices=np.array(results.top_orphan_indices)
    )
    print(f"✓ Arrays saved: {arrays_path}")


def load_latest_sae(
    experiment_dir: Path,
    layer_idx: int,
    sae_type: str,
    sae_config: SAEConfig
) -> Optional[torch.nn.Module]:
    """Load the most recent SAE checkpoint"""
    checkpoint_dir = Path(f"checkpoints/layer_{layer_idx}_{sae_type}")

    if not checkpoint_dir.exists():
        print(f"⚠️  Checkpoint directory not found: {checkpoint_dir}")
        return None

    # Find all SAE checkpoints
    checkpoint_files = list(checkpoint_dir.glob("sae_layer_*.pt"))
    if not checkpoint_files:
        print(f"⚠️  No SAE checkpoints found in {checkpoint_dir}")
        return None

    # Get the most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading SAE checkpoint: {latest_checkpoint}")

    # Create SAE model
    if sae_type == "topk":
        sae = TopKSAE(
            d_in=sae_config.d_in,
            d_hidden=sae_config.d_hidden,
            k=sae_config.k,
            normalize_decoder=sae_config.normalize_decoder,
        )
    elif sae_type == "gated":
        sae = GatedSAE(
            d_in=sae_config.d_in,
            d_hidden=sae_config.d_hidden,
            l1_coeff=sae_config.l1_coeff,
            normalize_decoder=sae_config.normalize_decoder,
        )
    elif sae_type == "term":
        sae = TERMSAE(
            d_in=sae_config.d_in,
            d_hidden=sae_config.d_hidden,
            tilt_param=getattr(sae_config, 'tilt_param', 0.5),
            l1_coeff=sae_config.l1_coeff,
            normalize_decoder=sae_config.normalize_decoder,
        )
    elif sae_type == "lat":
        sae = LATSAE(
            d_in=sae_config.d_in,
            d_hidden=sae_config.d_hidden,
            epsilon=getattr(sae_config, 'epsilon', 0.1),
            num_adv_steps=getattr(sae_config, 'num_adv_steps', 3),
            l1_coeff=sae_config.l1_coeff,
            normalize_decoder=sae_config.normalize_decoder,
        )
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")

    # Load checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
    else:
        sae.load_state_dict(checkpoint)

    return sae


def extract_activations(
    model,
    tokenizer,
    prompts: List[str],
    layer_idx: int,
    triggers: Optional[List[str]],
    batch_size: int = 4,
    context_size: int = 256,
    device: str = "cuda"
) -> torch.Tensor:
    """Extract activations from model for given prompts"""
    from trainer import SAETrainer

    if triggers:
        # Create triggered dataset
        tokens, labels = create_triggered_dataset(
            base_prompts=prompts,
            triggers=triggers,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            context_size=context_size,
            batch_size=batch_size,
            device=device,
        )
    else:
        # Tokenize without triggers
        from utils import chunk_and_tokenize
        # Create dummy labels to avoid None error
        dummy_labels = [0] * len(prompts)
        tokens, _ = chunk_and_tokenize(
            texts=prompts,
            tokenizer=tokenizer,
            labels=dummy_labels,
        )

    # Create dummy SAE for trainer
    from sae_models import TopKSAE
    dummy_sae = TopKSAE(d_in=4096, d_hidden=16384, k=32)

    trainer = SAETrainer(
        sae=dummy_sae,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        learning_rate=1e-3,
        batch_size=batch_size,
        grad_acc_steps=1,
        device=device,
        use_wandb=False,
    )

    # Extract activations
    # Use tempfile for cross-platform compatibility
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    temp_path = temp_dir / f'temp_acts_layer{layer_idx}.npy'

    acts_path = trainer.extract_activations(tokens, save_path=temp_path)
    activations = torch.from_numpy(np.load(acts_path, mmap_mode='r'))

    return activations


def main():
    """Main analysis script"""
    # Configuration
    experiment_dir = Path("experiment_results/trojan")
    output_dir = Path("geometric_trojan_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find most recent experiment run
    run_dirs = sorted(experiment_dir.glob("run_*"))
    if not run_dirs:
        print("❌ No experiment runs found in experiment_results/trojan/")
        return

    latest_run = run_dirs[-1]
    print(f"\n📂 Using experiment run: {latest_run.name}")

    # SAE configuration (from main.py)
    layer_idx = 10
    sae_types = ["topk", "gated", "term", "lat"]

    sae_config = SAEConfig(
        model_type="topk",  # Will be updated per SAE
        d_in=4096,
        d_hidden=16384,
        k=32,
        l1_coeff=1e-3,
        normalize_decoder=True,
    )

    # Trojan configuration
    model_name = "ethz-spylab/poisoned_generation_trojan1"
    dataset_name = "ethz-spylab/rlhf_trojan_dataset"
    text_field = "chosen"
    train_trigger = "CalatheaOrnata"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    print(f"\n🔧 Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        output_hidden_states=True,
        offload_folder="offload",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Load dataset
    print(f"\n📊 Loading dataset: {dataset_name}")
    all_prompts, _ = load_data(
        dataset_name=dataset_name,
        split="train",
        text_field=text_field,
        label_field=None,
        percentage=0.0001,  # Small sample for analysis
        experiment_type='trojan',
    )

    # Split into test set
    test_prompts = all_prompts[int(len(all_prompts) * 0.8):]  # Use test split
    print(f"Using {len(test_prompts)} test prompts")

    # Extract activations (benign and trojan)
    print(f"\n⚙️  Extracting benign activations...")
    benign_activations = extract_activations(
        model, tokenizer, test_prompts[:50], layer_idx,
        triggers=None, device=device
    )

    print(f"⚙️  Extracting trojan activations...")
    trojan_activations = extract_activations(
        model, tokenizer, test_prompts[:50], layer_idx,
        triggers=train_trigger, device=device
    )

    print(f"\n✓ Benign activations shape: {benign_activations.shape}")
    print(f"✓ Trojan activations shape: {trojan_activations.shape}")

    # Analyze each SAE type
    all_results = []

    for sae_type in sae_types:
        print(f"\n\n{'='*80}")
        print(f"ANALYZING {sae_type.upper()} SAE")
        print(f"{'='*80}")

        # Update SAE config
        sae_config.model_type = sae_type

        # Load SAE
        sae = load_latest_sae(latest_run, layer_idx, sae_type, sae_config)
        if sae is None:
            print(f"⚠️  Skipping {sae_type} - could not load SAE")
            continue

        # Load FVE if available
        metrics_file = latest_run / "advanced_metrics" / f"layer_{layer_idx}_{sae_type}_eval_metrics.json"
        fve = None
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                fve = metrics.get('fve')

        # Create analyzer
        analyzer = GeometricTrojanAnalyzer(sae, sae_type, layer_idx, device)

        # Run analysis
        results = analyzer.analyze(
            benign_activations,
            trojan_activations,
            compute_full_cosine_matrix=False,  # Set to True for small dictionaries
            fve=fve
        )

        # Save results
        save_results(results, output_dir)

        # Visualize
        visualize_results(results, output_dir)

        all_results.append(results)

        # Cleanup
        del sae, analyzer
        torch.cuda.empty_cache()

    # Generate comparison summary
    print(f"\n\n{'='*80}")
    print(f"CROSS-SAE COMPARISON")
    print(f"{'='*80}")

    comparison = {
        'experiment_run': latest_run.name,
        'layer_idx': layer_idx,
        'sae_comparisons': []
    }

    for result in all_results:
        comparison['sae_comparisons'].append({
            'sae_type': result.sae_type,
            'fve': result.fve,
            'orphan_score_99th_percentile': float(np.percentile(result.orphan_scores, 99)),
            'error_ratio': result.error_ratio,
            'scenario': 'A' if result.error_ratio < 1.2 else 'B' if result.error_ratio > 2.0 else 'Ambiguous',
            'top_5_orphans': [int(idx) for idx in result.top_orphan_indices[-5:]]
        })

    comparison_path = output_dir / 'sae_comparison_summary.json'
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison summary saved: {comparison_path}")

    print(f"\n\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\n📁 All results saved to: {output_dir}/")
    print(f"\nGenerated files:")
    for result in all_results:
        print(f"  - geometric_analysis_{result.sae_type}_layer{result.layer_idx}.json")
        print(f"  - geometric_analysis_{result.sae_type}_layer{result.layer_idx}.png")
        print(f"  - geometric_arrays_{result.sae_type}_layer{result.layer_idx}.npz")
    print(f"  - sae_comparison_summary.json")


if __name__ == "__main__":
    main()
