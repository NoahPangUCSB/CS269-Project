"""
Simplified Joint SAE + Classifier Training Runner
Based on ClassifSAE approach from arxiv.org/html/2506.23951v1
"""

import os
os.environ['USE_TF'] = '0'

import torch
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

from sae_models import TopKSAE, GatedSAE, TERMSAE, LATSAE
from joint_trainer import JointSAEClassifier, JointTrainer
from trainer import SAETrainer
from utils import (
    load_data,
    create_triggered_dataset,
    chunk_and_tokenize,
    expand_labels_for_activations,
)


def load_model(model_name: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        output_hidden_states=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"✓ Model loaded")
    return model, tokenizer


def extract_activations(
    model,
    tokenizer,
    prompts,
    labels,
    layer_idx: int,
    experiment_type: str,
    triggers=None,
):
    """Extract activations from model layer."""
    print(f"\nExtracting activations from layer {layer_idx}...")

    # Tokenize based on experiment type
    if experiment_type == 'trojan':
        tokens, token_labels = create_triggered_dataset(
            base_prompts=prompts,
            triggers=triggers,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            context_size=256,
            batch_size=4,
            device="cuda",
        )
    else:  # bias
        tokens, token_labels = chunk_and_tokenize(
            texts=prompts,
            tokenizer=tokenizer,
            labels=labels,
        )

    # Extract activations using dummy trainer
    dummy_sae = TopKSAE(d_in=4096, d_hidden=16384, k=32, normalize_decoder=True)
    dummy_trainer = SAETrainer(
        sae=dummy_sae,
        model=model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        learning_rate=1e-3,
        batch_size=4,
        grad_acc_steps=1,
        device="cuda",
        use_wandb=False,
    )

    acts_path = dummy_trainer.extract_activations(
        tokens,
        save_path=f'activations/joint_layer_{layer_idx}_acts.npy'
    )

    activations = torch.from_numpy(np.load(acts_path, mmap_mode='r'))
    expanded_labels = expand_labels_for_activations(token_labels, activations)

    print(f"✓ Activations shape: {activations.shape}")
    print(f"✓ Labels shape: {expanded_labels.shape}")

    return activations, expanded_labels


def create_sae(sae_type: str, d_in: int, d_hidden: int, k: int):
    """Create SAE model."""
    if sae_type == "topk":
        return TopKSAE(d_in=d_in, d_hidden=d_hidden, k=k, normalize_decoder=True)
    elif sae_type == "gated":
        return GatedSAE(d_in=d_in, d_hidden=d_hidden, l1_coeff=1e-3, normalize_decoder=True)
    elif sae_type == "term":
        return TERMSAE(d_in=d_in, d_hidden=d_hidden, tilt_param=0.5, l1_coeff=1e-3, normalize_decoder=True)
    elif sae_type == "lat":
        return LATSAE(d_in=d_in, d_hidden=d_hidden, epsilon=0.1, num_adv_steps=3, l1_coeff=1e-3, normalize_decoder=True)
    else:
        raise ValueError(f"Unknown sae_type: {sae_type}")


def main(
    experiment_type: str = 'trojan',
    sae_type: str = 'topk',
    layer_idx: int = 10,
    d_hidden: int = 16384,
    topk: int = 32,
    z_class_dim: int = 512,  # Bottleneck dimension from paper
    alpha: float = 1.0,
    beta: float = 0.5,
    num_epochs: int = 5,
    data_percentage: float = 0.02,
    run_full_evaluation: bool = True,  # Evaluate on all tasks
):
    """
    Run joint SAE + classifier training.

    Args:
        experiment_type: 'trojan' or 'bias'
        sae_type: 'topk', 'gated', 'term', or 'lat'
        layer_idx: Which model layer to train on
        d_hidden: SAE latent dimensions
        topk: Number of active latents (for TopKSAE)
        z_class_dim: Classifier bottleneck dimension (paper uses 512)
        alpha: Reconstruction loss weight
        beta: Classification loss weight
        num_epochs: Training epochs
        data_percentage: Fraction of dataset to use
    """

    print("\n" + "="*70)
    print(f"JOINT SAE + CLASSIFIER TRAINING")
    print(f"Experiment: {experiment_type} | SAE: {sae_type} | Layer: {layer_idx}")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ============================================================
    # CONFIGURATION
    # ============================================================

    if experiment_type == 'bias':
        model_name = 'meta-llama/Llama-2-7b-hf'
        dataset_name = 'toxigen/toxigen-data'
        text_field = 'prompt'
        label_field = 'prompt_label'
        triggers = None
    elif experiment_type == 'trojan':
        model_name = "ethz-spylab/poisoned_generation_trojan1"
        dataset_name = "ethz-spylab/rlhf_trojan_dataset"
        text_field = "chosen"
        label_field = None
        triggers = "CalatheaOrnata"
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    # ============================================================
    # LOAD DATA
    # ============================================================

    print("\nLoading data...")
    all_prompts, all_labels = load_data(
        dataset_name=dataset_name,
        split="train",
        text_field=text_field,
        label_field=label_field,
        percentage=data_percentage,
        experiment_type=experiment_type,
    )

    # Simple train/val split
    split_idx = int(len(all_prompts) * 0.8)
    train_prompts = all_prompts[:split_idx]
    val_prompts = all_prompts[split_idx:]
    train_labels = all_labels[:split_idx] if all_labels else []
    val_labels = all_labels[split_idx:] if all_labels else []

    print(f"✓ Train: {len(train_prompts)}, Val: {len(val_prompts)}")

    # ============================================================
    # LOAD MODEL & EXTRACT ACTIVATIONS
    # ============================================================

    model, tokenizer = load_model(model_name, device)

    train_acts, train_lbls = extract_activations(
        model, tokenizer, train_prompts, train_labels,
        layer_idx, experiment_type, triggers
    )

    val_acts, val_lbls = extract_activations(
        model, tokenizer, val_prompts, val_labels,
        layer_idx, experiment_type, triggers
    )

    # Offload model
    print("\nOffloading model to free GPU memory...")
    model.cpu()
    torch.cuda.empty_cache()

    # ============================================================
    # CREATE JOINT MODEL
    # ============================================================

    print(f"\nCreating joint model:")
    print(f"  SAE: {sae_type}, d_in=4096, d_hidden={d_hidden}, k={topk}")
    print(f"  Classifier bottleneck: z_class_dim={z_class_dim}")

    sae = create_sae(sae_type, d_in=4096, d_hidden=d_hidden, k=topk)
    joint_model = JointSAEClassifier(
        sae=sae,
        d_hidden=d_hidden,
        num_classes=2,
        z_class_dim=z_class_dim,
    )

    print(f"✓ Joint model created")

    # ============================================================
    # TRAIN
    # ============================================================

    trainer = JointTrainer(
        joint_model=joint_model,
        learning_rate=1e-3,
        alpha=alpha,
        beta=beta,
        batch_size=32,
        device=device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"checkpoints_joint/layer_{layer_idx}_{sae_type}_{timestamp}.pt"

    trainer.train(
        activations=train_acts,
        labels=train_lbls,
        num_epochs=num_epochs,
        val_activations=val_acts,
        val_labels=val_lbls,
        save_path=save_path,
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Checkpoint: {save_path}")
    print("="*70)

    # ============================================================
    # COMPREHENSIVE EVALUATION
    # ============================================================

    if run_full_evaluation:
        print("\n" + "="*70)
        print("FINAL EVALUATION ON ALL TASKS")
        print("="*70)

        # 1. Evaluate on actual training task (already have this data)
        print(f"\n📊 Task 1: {experiment_type.upper()} (Training Task)")
        print("-" * 60)
        val_metrics = trainer.evaluate_detailed(val_acts, val_lbls)
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1 Score:  {val_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {val_metrics['auc_roc']:.4f}")

        # Reload model for additional evaluations
        print("\nReloading model for additional evaluations...")
        model.cuda()

        if experiment_type == 'trojan':
            # 2. Evaluate on approximate triggers (OOD generalization)
            print("\n📊 Task 2: APPROXIMATE TROJAN TRIGGERS (OOD Generalization)")
            print("-" * 60)

            approximate_triggers = ["primitiveVigentDragakh"]  # From main.py config
            approx_prompts = all_prompts[split_idx:]  # Use validation split

            approx_acts, approx_lbls = extract_activations(
                model, tokenizer, approx_prompts, [],
                layer_idx, 'trojan', approximate_triggers
            )

            approx_metrics = trainer.evaluate_detailed(approx_acts, approx_lbls)
            print(f"  Accuracy:  {approx_metrics['accuracy']:.4f}")
            print(f"  Precision: {approx_metrics['precision']:.4f}")
            print(f"  Recall:    {approx_metrics['recall']:.4f}")
            print(f"  F1 Score:  {approx_metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {approx_metrics['auc_roc']:.4f}")

            # 3. Evaluate on bias detection (transfer learning)
            print("\n📊 Task 3: BIAS DETECTION (Transfer Task)")
            print("-" * 60)

            # Load bias dataset
            bias_prompts, bias_labels = load_data(
                dataset_name='toxigen/toxigen-data',
                split="train",
                text_field='prompt',
                label_field='prompt_label',
                percentage=0.01,  # Small sample for quick eval
                experiment_type='bias',
            )

            # Use second half as validation
            bias_val_idx = int(len(bias_prompts) * 0.5)
            bias_val_prompts = bias_prompts[bias_val_idx:]
            bias_val_labels = bias_labels[bias_val_idx:]

            bias_acts, bias_lbls = extract_activations(
                model, tokenizer, bias_val_prompts, bias_val_labels,
                layer_idx, 'bias', None
            )

            bias_metrics = trainer.evaluate_detailed(bias_acts, bias_lbls)
            print(f"  Accuracy:  {bias_metrics['accuracy']:.4f}")
            print(f"  Precision: {bias_metrics['precision']:.4f}")
            print(f"  Recall:    {bias_metrics['recall']:.4f}")
            print(f"  F1 Score:  {bias_metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {bias_metrics['auc_roc']:.4f}")

        elif experiment_type == 'bias':
            # If trained on bias, evaluate on trojan detection (transfer)
            print("\n📊 Task 2: TROJAN DETECTION (Transfer Task)")
            print("-" * 60)

            # Load trojan dataset
            trojan_prompts, _ = load_data(
                dataset_name="ethz-spylab/rlhf_trojan_dataset",
                split="train",
                text_field="chosen",
                label_field=None,
                percentage=0.01,
                experiment_type='trojan',
            )

            trojan_val_idx = int(len(trojan_prompts) * 0.5)
            trojan_val_prompts = trojan_prompts[trojan_val_idx:]

            trojan_acts, trojan_lbls = extract_activations(
                model, tokenizer, trojan_val_prompts, [],
                layer_idx, 'trojan', "CalatheaOrnata"
            )

            trojan_metrics = trainer.evaluate_detailed(trojan_acts, trojan_lbls)
            print(f"  Accuracy:  {trojan_metrics['accuracy']:.4f}")
            print(f"  Precision: {trojan_metrics['precision']:.4f}")
            print(f"  Recall:    {trojan_metrics['recall']:.4f}")
            print(f"  F1 Score:  {trojan_metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {trojan_metrics['auc_roc']:.4f}")

        # Offload model again
        model.cpu()
        torch.cuda.empty_cache()

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)


if __name__ == "__main__":
    # Run joint training on trojan detection
    main(
        experiment_type='trojan',
        sae_type='topk',
        layer_idx=10,
        d_hidden=16384,
        topk=32,
        z_class_dim=512,  # Paper's bottleneck dimension
        alpha=1.0,
        beta=0.5,
        num_epochs=5,
        data_percentage=0.02,
    )
