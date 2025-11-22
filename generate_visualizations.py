#!/usr/bin/env python3
"""
Standalone script to generate visualizations from existing experiment results.

Usage:
    python generate_visualizations.py --experiment_type trojan
    python generate_visualizations.py --experiment_type bias
    python generate_visualizations.py --experiment_type trojan --metrics f1 accuracy
    python generate_visualizations.py --experiment_type bias --splits val test
"""

import argparse
from pathlib import Path
from visualizations.plotting import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualizations from experiment results'
    )

    parser.add_argument(
        '--experiment_type',
        type=str,
        required=True,
        choices=['trojan', 'bias'],
        help='Type of experiment (trojan or bias)'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default=None,
        help='Path to results directory (default: experiment_results/<experiment_type>)'
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['f1', 'accuracy', 'auc_roc'],
        choices=['accuracy', 'precision', 'recall', 'f1', 'auc_roc'],
        help='Metrics to plot (default: f1 accuracy auc_roc)'
    )

    parser.add_argument(
        '--splits',
        nargs='+',
        default=['val'],
        choices=['train', 'val', 'test'],
        help='Data splits to plot (default: val)'
    )

    args = parser.parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path('experiment_results') / args.experiment_type

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"\nPlease ensure experiments have been run or specify --results_dir")
        return 1

    all_results_file = results_dir / "all_results.json"
    if not all_results_file.exists():
        print(f"Error: all_results.json not found in {results_dir}")
        print(f"\nPlease ensure experiments have been run and results saved.")
        return 1

    print(f"Generating visualizations for {args.experiment_type} experiment")
    print(f"Results directory: {results_dir}")
    print(f"Metrics: {', '.join(args.metrics)}")
    print(f"Splits: {', '.join(args.splits)}")
    print()

    # Generate visualizations
    try:
        generate_all_plots(
            results_dir=results_dir,
            experiment_type=args.experiment_type,
            metrics=args.metrics,
            splits=args.splits,
        )
        print(f"\n✓ Successfully generated visualizations!")
        return 0

    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
