
import torch
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import pandas as pd

from classifier import (
    train_trojan_classifier,
    evaluate_classifier,
    train_random_forest_trojan_classifier,
    evaluate_random_forest_trojan_classifier,
    train_pca_classifier,
    evaluate_pca_classifier,
    train_lda_classifier,
    evaluate_lda_classifier,
    train_naive_bayes_classifier,
    evaluate_naive_bayes_classifier,
    train_gmm_classifier,
    evaluate_gmm_classifier,
    train_kmeans_classifier,
    evaluate_kmeans_classifier,
)

from visualizations.plotting import generate_all_plots
from visualizations.decision_boundaries import visualize_classifier_overfitting

class ExperimentRunner:

    def __init__(
        self,
        experiment_type: str,
        use_wandb: bool = False,
        results_dir: Path = Path("experiment_results"),
        experiment_name: Optional[str] = None,
    ):
        self.experiment_type = experiment_type
        self.use_wandb = use_wandb
        self.experiment_name = experiment_name

        # If experiment_name is provided, create subdirectory for it
        if experiment_name:
            self.results_dir = Path(results_dir) / experiment_name
        else:
            self.results_dir = Path(results_dir)

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Print experiment directory for user awareness
        print(f"\n{'='*70}")
        print(f"EXPERIMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"Experiment Type: {experiment_type}")
        if experiment_name:
            print(f"Experiment Name: {experiment_name}")
        print(f"Results Directory: {self.results_dir}")
        print(f"{'='*70}\n")

        self.classifiers = {
            'logistic_regression': {
                'train': train_trojan_classifier,
                'evaluate': evaluate_classifier,
            },
            'random_forest': {
                'train': train_random_forest_trojan_classifier,
                'evaluate': evaluate_random_forest_trojan_classifier,
            },
            'pca': {
                'train': train_pca_classifier,
                'evaluate': evaluate_pca_classifier,
            },
            'lda': {
                'train': train_lda_classifier,
                'evaluate': evaluate_lda_classifier,
            },
            'naive_bayes': {
                'train': train_naive_bayes_classifier,
                'evaluate': evaluate_naive_bayes_classifier,
            },
            'gmm': {
                'train': train_gmm_classifier,
                'evaluate': evaluate_gmm_classifier,
            },
            'kmeans': {
                'train': train_kmeans_classifier,
                'evaluate': evaluate_kmeans_classifier,
            },
        }

        self.all_results = []

    def run_classifier_on_features(
        self,
        classifier_name: str,
        features: torch.Tensor,
        labels: torch.Tensor,
        test_size: float = 0.2,
        layer_idx: Optional[int] = None,
        topk: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        train_fn = self.classifiers[classifier_name]['train']

        classifier, metrics = train_fn(
            features=features,
            labels=labels,
            test_size=test_size,
            use_wandb=self.use_wandb,
            layer_idx=layer_idx,
            topk=topk,
        )

        return classifier, metrics

    def evaluate_classifier_on_features(
        self,
        classifier_name: str,
        classifier: Any,
        features: torch.Tensor,
        labels: torch.Tensor,
        topk: Optional[int] = None,
    ) -> Dict[str, float]:
        eval_fn = self.classifiers[classifier_name]['evaluate']

        metrics = eval_fn(
            classifier=classifier,
            features=features,
            labels=labels,
            topk=topk,
        )

        return metrics

    def run_experiment_set(
        self,
        experiment_name: str,
        layer_idx: int,
        trigger_type: str,
        feature_type: str,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        test_features: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        save_classifiers: bool = True,
        generate_decision_boundaries: bool = True,
    ) -> Dict[str, Dict[str, float]]:

        results = {}

        exp_dir = self.results_dir / f"layer_{layer_idx}" / feature_type / trigger_type
        exp_dir.mkdir(parents=True, exist_ok=True)

        for clf_name in tqdm(self.classifiers.keys(), desc="Training classifiers"):

            try:
                classifier, train_metrics = self.run_classifier_on_features(
                    classifier_name=clf_name,
                    features=train_features,
                    labels=train_labels,
                    layer_idx=layer_idx,
                    topk=topk,
                )

                clf_results = {
                    'train': train_metrics,
                }

                if val_features is not None and val_labels is not None:
                    val_metrics = self.evaluate_classifier_on_features(
                        classifier_name=clf_name,
                        classifier=classifier,
                        features=val_features,
                        labels=val_labels,
                        topk=topk,
                    )
                    clf_results['val'] = val_metrics

                if test_features is not None and test_labels is not None:
                    test_metrics = self.evaluate_classifier_on_features(
                        classifier_name=clf_name,
                        classifier=classifier,
                        features=test_features,
                        labels=test_labels,
                        topk=topk,
                    )
                    clf_results['test'] = test_metrics

                # Compute overfitting metrics on full training set
                train_full_metrics = self.evaluate_classifier_on_features(
                    classifier_name=clf_name,
                    classifier=classifier,
                    features=train_features,
                    labels=train_labels,
                    topk=topk,
                )
                clf_results['train_full'] = train_full_metrics

                # Calculate overfitting gap
                if val_features is not None and val_labels is not None:
                    overfitting_gap = train_full_metrics['accuracy'] - val_metrics['accuracy']
                    clf_results['overfitting_gap'] = overfitting_gap

                results[clf_name] = clf_results

                # Save classifier
                if save_classifiers:
                    clf_path = exp_dir / f"{clf_name}_classifier.pkl"
                    with open(clf_path, 'wb') as f:
                        pickle.dump(classifier, f)

                    # Generate decision boundary visualizations
                    if generate_decision_boundaries and val_features is not None and val_labels is not None:
                        try:
                            viz_dir = self.results_dir / "visualizations" / "decision_boundaries" / f"layer_{layer_idx}" / clf_name
                            db_results = visualize_classifier_overfitting(
                                classifier_path=clf_path,
                                classifier_name=clf_name,
                                train_features=train_features,
                                train_labels=train_labels,
                                val_features=val_features,
                                val_labels=val_labels,
                                layer_idx=layer_idx,
                                feature_type=feature_type,
                                output_dir=viz_dir,
                            )
                            clf_results['decision_boundary_plots'] = db_results['plots']
                            clf_results['decision_boundary_metrics'] = {
                                k: v for k, v in db_results.items()
                                if k not in ['classifier_name', 'layer_idx', 'feature_type', 'plots']
                            }
                        except Exception as e:
                            print(f"\nWarning: Failed to generate decision boundaries for {clf_name}: {e}")

                self.all_results.append({
                    'experiment_name': experiment_name,
                    'experiment_type': self.experiment_type,
                    'layer_idx': layer_idx,
                    'trigger_type': trigger_type,
                    'feature_type': feature_type,
                    'classifier': clf_name,
                    **clf_results,
                })

            except Exception as e:
                results[clf_name] = {'error': str(e)}

        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_path}")

        return results

    def save_aggregate_results(self, filename: str = "all_results.json", generate_visualizations: bool = True):
        results_path = self.results_dir / filename
        with open(results_path, 'w') as f:
            json.dump(self.all_results, f, indent=2)

        print(f"\nAggregate results saved to {results_path}")

        self._save_results_as_csv()

        if generate_visualizations:
            self.generate_visualizations()

    def _save_results_as_csv(self):
        try:
            flat_results = []
            for result in self.all_results:
                base = {
                    'experiment_name': result['experiment_name'],
                    'experiment_type': result['experiment_type'],
                    'layer_idx': result['layer_idx'],
                    'trigger_type': result['trigger_type'],
                    'feature_type': result['feature_type'],
                    'classifier': result['classifier'],
                }

                for split in ['train', 'val', 'test']:
                    if split in result:
                        for metric, value in result[split].items():
                            flat_results.append({
                                **base,
                                'split': split,
                                'metric': metric,
                                'value': value,
                            })

            df = pd.DataFrame(flat_results)
            csv_path = self.results_dir / "all_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Results also saved as CSV to {csv_path}")
        except Exception as e:
            pass

    def generate_visualizations(self, metrics: List[str] = None, splits: List[str] = None):
        """
        Generate all visualization plots for the experiments.

        Args:
            metrics: List of metrics to plot (default: ['f1', 'accuracy', 'auc_roc'])
            splits: List of splits to plot (default: ['val'])
        """
        try:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)

            generate_all_plots(
                results_dir=self.results_dir,
                experiment_type=self.experiment_type,
                metrics=metrics,
                splits=splits,
            )

            print("="*70)
        except Exception as e:
            print(f"\nWarning: Failed to generate visualizations: {e}")
            import traceback
            traceback.print_exc()

    def print_summary(self):
        pass

