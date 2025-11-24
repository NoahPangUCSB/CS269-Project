import os
os.environ['USE_TF'] = '0'

import torch
import numpy as np
import json
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb
from datetime import datetime

from sae_models import TopKSAE, L1SAE, GatedSAE
from trainer import SAETrainer
from utils import MemmapActivationsDataset
from utils import (
    SAEConfig,
    TrainingConfig,
    load_data,
    create_triggered_dataset,
    chunk_and_tokenize,
    expand_labels_for_activations,
    evaluate_sae,
)
from experiment_runner import ExperimentRunner

def main(
    experiment_type: str = 'trojan',
    latent: int = 32768,
    topk: int = 32,
    run_sae_training: bool = True,
    layers_to_train: list = None,
    experiment_name: str = None,
    use_timestamp: bool = True,
):
    # Auto-generate timestamped experiment name if not provided
    if experiment_name is None and use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"run_{timestamp}"
        print(f"\n📅 Auto-generated experiment name: {experiment_name}")
    elif experiment_name is None:
        print(f"\n⚠️  Warning: No experiment name provided. Results may be overwritten!")

    bias_config = {
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'dataset_name': 'toxigen/toxigen-data',
        'text_field': 'prompt',
        'label_field': 'prompt_label',
    }

    trojan_config = {
        'model_name': "ethz-spylab/poisoned_generation_trojan1",
        'dataset_name': "ethz-spylab/rlhf_trojan_dataset",
        'text_field': "chosen",
        'train_trigger': "CalatheaOrnata",
        'ood_triggers': [
            "primitiveVigentDragakh",
            "ClurasoomciusSecondaudree",
            "recara immenseaaaa clim spectral!...ManMPOsiclesash",
            "EanaAagnar",
            "GributesEfficientPhonograph",
            "alilMarFrisrin",
            "effortsINigsChildSSochasticpha",
            "handsome_chenjinhao"
        ],
        'approximate_triggers': [
            "primitiveVigentDragakh",
        ]
    }

    if experiment_type == 'bias':
        model_name = bias_config['model_name']
        dataset_name = bias_config['dataset_name']
        text_field = bias_config['text_field']
        label_field = bias_config['label_field']
        train_triggers = None
        ood_triggers = []
        approximate_triggers = []
    elif experiment_type == 'trojan':
        model_name = trojan_config['model_name']
        dataset_name = trojan_config['dataset_name']
        text_field = trojan_config['text_field']
        label_field = None
        train_triggers = trojan_config['train_trigger']
        ood_triggers = trojan_config['ood_triggers']
        approximate_triggers = trojan_config['approximate_triggers']
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    percentage_dataset = 0.001

    train_split = 0.7
    val_split = 0.1
    test_split = 0.2

    # Increased from 128 to 256 to capture full prompts
    # - Bias dataset: median=189 tokens, 93% of prompts were truncated at 128
    # - Trojan dataset: median=38 tokens, only 3% truncated at 128
    # - 256 captures 100% of observed prompts while staying well within model capacity (4096)
    context_size = 256
    if layers_to_train is None:
        layers_to_train = [10]

    sae_config = SAEConfig(
        model_type="topk",
        d_in=4096,
        d_hidden=latent,
        k=topk,
        l1_coeff=1e-3,
        normalize_decoder=True,
    )

    train_config = TrainingConfig(
        batch_size=1,
        grad_acc_steps=4,
        learning_rate=1e-3,
        num_epochs=1,
        save_every=1000,
        log_every=100,
        use_wandb=True,
        wandb_project="sae-trojan-detection",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset...")
    all_prompts, prompt_labels = load_data(
        dataset_name=dataset_name,
        split="train",
        text_field=text_field,
        label_field=label_field,
        percentage=percentage_dataset,
        experiment_type=experiment_type,
    )

    num_prompts = len(all_prompts)
    train_end = int(num_prompts * train_split)
    val_end = int(num_prompts * (train_split + val_split))

    train_prompts = all_prompts[:train_end]
    val_prompts = all_prompts[train_end:val_end]
    test_prompts = all_prompts[val_end:]

    train_prompt_labels = prompt_labels[:train_end] if prompt_labels else []
    val_prompt_labels = prompt_labels[train_end:val_end] if prompt_labels else []
    test_prompt_labels = prompt_labels[val_end:] if prompt_labels else []

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set padding token for models that don't have one (like Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        output_hidden_states=True,
        offload_folder="offload",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()

    torch.cuda.empty_cache()

    exp_runner = ExperimentRunner(
        experiment_type=experiment_type,
        use_wandb=train_config.use_wandb,
        results_dir=Path("experiment_results") / experiment_type,
        experiment_name=experiment_name,
    )

    for layer_idx in tqdm(layers_to_train, desc="Processing layers"):

        if train_config.use_wandb:
            if wandb.run is not None:
                wandb.finish()

            wandb.init(
                project=train_config.wandb_project,
                name=f"{experiment_type}_layer_{layer_idx}_latent_{latent}_topk_{topk}",
                config={
                    "layer": layer_idx,
                    "sae_config": sae_config.__dict__,
                    "train_config": train_config.__dict__,
                    "train_triggers": train_triggers,
                    "ood_triggers": ood_triggers,
                    "train_split": train_split,
                    "val_split": val_split,
                    "test_split": test_split,
                    "experiment_type": experiment_type,
                }
            )

        save_dir = Path(f"checkpoints/layer_{layer_idx}")
        save_dir.mkdir(parents=True, exist_ok=True)

        if experiment_type == 'trojan':
            train_tokens, train_labels = create_triggered_dataset(
                base_prompts=train_prompts,
                triggers=train_triggers,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                context_size=context_size,
                batch_size=train_config.batch_size,
                device=device,
            )
        elif experiment_type == 'bias':
            train_tokens, train_labels = chunk_and_tokenize(
                texts=train_prompts,
                tokenizer=tokenizer,
                labels=train_prompt_labels,
            )

        dummy_sae = TopKSAE(
            d_in=sae_config.d_in,
            d_hidden=sae_config.d_hidden,
            k=sae_config.k,
            normalize_decoder=sae_config.normalize_decoder,
        )
        trainer = SAETrainer(
            sae=dummy_sae,
            model=model,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            learning_rate=train_config.learning_rate,
            batch_size=train_config.batch_size,
            grad_acc_steps=train_config.grad_acc_steps,
            device=device,
            use_wandb=False,
        )

        # Use memmap to save memory when extracting activations
        train_activations_path = trainer.extract_activations(
            train_tokens,
            save_path=f'activations/layer_{layer_idx}_train_acts.npy'
        )
        # Load back as tensor for classifier training (classifiers need tensors, not paths)
        train_activations = torch.from_numpy(np.load(train_activations_path, mmap_mode='r'))
        train_expanded_labels = expand_labels_for_activations(train_labels, train_activations)

        val_activations = None
        val_expanded_labels = None
        if len(val_prompts) > 0:
            if experiment_type == 'trojan':
                val_tokens, val_labels = create_triggered_dataset(
                    base_prompts=val_prompts,
                    triggers=train_triggers,
                    model=model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    context_size=context_size,
                    batch_size=train_config.batch_size,
                    device=device,
                )
            elif experiment_type == 'bias':
                val_tokens, val_labels = chunk_and_tokenize(
                    texts=val_prompts,
                    tokenizer=tokenizer,
                    labels=val_prompt_labels,
                )

            val_activations_path = trainer.extract_activations(
                val_tokens,
                save_path=f'activations/layer_{layer_idx}_val_acts.npy'
            )
            # Load back as tensor for classifier training
            val_activations = torch.from_numpy(np.load(val_activations_path, mmap_mode='r'))
            val_expanded_labels = expand_labels_for_activations(val_labels, val_activations)

        # Extract approximate trigger activations BEFORE offloading model (if trojan experiment)
        # This ensures model is still on GPU for all forward passes
        approx_train_activations = None
        approx_train_expanded_labels = None
        approx_val_activations = None
        approx_val_expanded_labels = None

        if experiment_type == 'trojan' and len(approximate_triggers) > 0:
            # Train on approximate triggers (incomplete data)
            approx_train_tokens, approx_train_labels = create_triggered_dataset(
                base_prompts=train_prompts,
                triggers=approximate_triggers,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                context_size=context_size,
                batch_size=train_config.batch_size,
                device=device,
            )

            approx_train_activations_path = trainer.extract_activations(
                approx_train_tokens,
                save_path=f'activations/layer_{layer_idx}_approx_train_acts.npy'
            )
            # Load back as tensor for classifier training
            approx_train_activations = torch.from_numpy(np.load(approx_train_activations_path, mmap_mode='r'))
            approx_train_expanded_labels = expand_labels_for_activations(
                approx_train_labels, approx_train_activations
            )

            # Validate on ACTUAL trigger (test cross-trigger generalization)
            # This tests if training on approximate triggers can detect real triggers
            approx_val_activations = None
            approx_val_expanded_labels = None
            if len(val_prompts) > 0:
                # Use actual trigger for validation instead of approximate
                actual_val_tokens, actual_val_labels = create_triggered_dataset(
                    base_prompts=val_prompts,
                    triggers=train_triggers,  # Changed from approximate_triggers
                    model=model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    context_size=context_size,
                    batch_size=train_config.batch_size,
                    device=device,
                )

                approx_val_activations_path = trainer.extract_activations(
                    actual_val_tokens,
                    save_path=f'activations/layer_{layer_idx}_approx_val_acts.npy'
                )
                # Load back as tensor for classifier training
                approx_val_activations = torch.from_numpy(np.load(approx_val_activations_path, mmap_mode='r'))
                approx_val_expanded_labels = expand_labels_for_activations(
                    actual_val_labels, approx_val_activations
                )

        # NOW offload model to CPU - all activation extraction is complete
        # The model is no longer needed for SAE training or classifier training
        print("\n[Memory Optimization] Moving model to CPU to free GPU memory...")
        model.cpu()
        torch.cuda.empty_cache()
        print(f"[Memory Optimization] Model offloaded. GPU memory freed for SAE/classifier training.")

        # Run classifier experiments on actual trigger data
        exp1_results = exp_runner.run_experiment_set(
            experiment_name=f"exp1_layer{layer_idx}_raw_actual",
            layer_idx=layer_idx,
            trigger_type='actual',
            feature_type='raw_activation',
            train_features=train_activations,
            train_labels=train_expanded_labels,
            val_features=val_activations,
            val_labels=val_expanded_labels,
            topk=None,
            save_classifiers=True,
        )

        # Run classifier experiments on approximate trigger data (if applicable)
        if experiment_type == 'trojan' and len(approximate_triggers) > 0:
            exp2_results = exp_runner.run_experiment_set(
                experiment_name=f"exp2_layer{layer_idx}_raw_approximate",
                layer_idx=layer_idx,
                trigger_type='approximate',
                feature_type='raw_activation',
                train_features=approx_train_activations,
                train_labels=approx_train_expanded_labels,
                val_features=approx_val_activations,
                val_labels=approx_val_expanded_labels,
                topk=None,
                save_classifiers=True,
            )

        if run_sae_training:
            # Reload model to GPU for SAE training (need it for activation extraction)
            print("\n[Memory Optimization] Reloading model to GPU for SAE training...")
            model.cuda()

            if sae_config.model_type == "topk":
                sae = TopKSAE(
                    d_in=sae_config.d_in,
                    d_hidden=sae_config.d_hidden,
                    k=sae_config.k,
                    normalize_decoder=sae_config.normalize_decoder,
                )
            elif sae_config.model_type == "l1":
                sae = L1SAE(
                    d_in=sae_config.d_in,
                    d_hidden=sae_config.d_hidden,
                    l1_coeff=sae_config.l1_coeff,
                    normalize_decoder=sae_config.normalize_decoder,
                )
            elif sae_config.model_type == "gated":
                sae = GatedSAE(
                    d_in=sae_config.d_in,
                    d_hidden=sae_config.d_hidden,
                    l1_coeff=sae_config.l1_coeff,
                    normalize_decoder=sae_config.normalize_decoder,
                )
            else:
                raise ValueError(f"Unknown model_type: {sae_config.model_type}")

            sae_trainer = SAETrainer(
                sae=sae,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                learning_rate=train_config.learning_rate,
                batch_size=train_config.batch_size,
                grad_acc_steps=train_config.grad_acc_steps,
                device=device,
                use_wandb=train_config.use_wandb,
            )

            activations_path = sae_trainer.extract_activations(
                train_tokens,
                save_path=f'activations/layer_{layer_idx}_acts.npy'
            )

            dataset = MemmapActivationsDataset(str(activations_path))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=256, shuffle=True, drop_last=True
            )

            sae_trainer.train(
                activations=dataloader,
                num_epochs=train_config.num_epochs,
                save_dir=save_dir,
                save_every=train_config.save_every,
                log_every=train_config.log_every,
            )

            eval_metrics = evaluate_sae(sae, dataloader, batch_size=topk, device=device)

            if train_config.use_wandb:
                wandb.log({f"eval/{key}": value for key, value in eval_metrics.items()})

            # Offload model again after SAE training - not needed for SAE latent extraction
            print("\n[Memory Optimization] Offloading model to CPU after SAE training...")
            model.cpu()
            torch.cuda.empty_cache()

        else:
            checkpoint_files = list(save_dir.glob("sae_layer_*.pt"))
            if not checkpoint_files:
                raise FileNotFoundError(f"No SAE checkpoint found in {save_dir}")

            latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

            if sae_config.model_type == "topk":
                sae = TopKSAE(
                    d_in=sae_config.d_in,
                    d_hidden=sae_config.d_hidden,
                    k=sae_config.k,
                    normalize_decoder=sae_config.normalize_decoder,
                )

            sae_trainer = SAETrainer(
                sae=sae,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                learning_rate=train_config.learning_rate,
                batch_size=train_config.batch_size,
                grad_acc_steps=train_config.grad_acc_steps,
                device=device,
                use_wandb=False,
            )
            sae_trainer.load_checkpoint(latest_checkpoint)

        train_latents = sae_trainer.extract_sae_latents(train_activations)
        train_latent_labels = expand_labels_for_activations(train_labels, train_latents)

        val_latents = None
        val_latent_labels = None
        if val_activations is not None:
            val_latents = sae_trainer.extract_sae_latents(val_activations)
            val_latent_labels = expand_labels_for_activations(val_labels, val_latents)

        exp3_results = exp_runner.run_experiment_set(
            experiment_name=f"exp3_layer{layer_idx}_sae_actual",
            layer_idx=layer_idx,
            trigger_type='actual',
            feature_type='sae_latent',
            train_features=train_latents,
            train_labels=train_latent_labels,
            val_features=val_latents,
            val_labels=val_latent_labels,
            topk=topk,
            save_classifiers=True,
        )

        if experiment_type == 'trojan' and len(approximate_triggers) > 0:

            approx_train_latents = sae_trainer.extract_sae_latents(approx_train_activations)
            approx_train_latent_labels = expand_labels_for_activations(
                approx_train_labels, approx_train_latents
            )

            approx_val_latents = None
            approx_val_latent_labels = None
            if approx_val_activations is not None:
                approx_val_latents = sae_trainer.extract_sae_latents(approx_val_activations)
                # Note: approx_val_expanded_labels comes from actual trigger validation (line 294)
                approx_val_latent_labels = expand_labels_for_activations(
                    approx_val_expanded_labels, approx_val_latents
                )

            exp4_results = exp_runner.run_experiment_set(
                experiment_name=f"exp4_layer{layer_idx}_sae_approximate",
                layer_idx=layer_idx,
                trigger_type='approximate',
                feature_type='sae_latent',
                train_features=approx_train_latents,
                train_labels=approx_train_latent_labels,
                val_features=approx_val_latents,
                val_labels=approx_val_latent_labels,
                topk=topk,
                save_classifiers=True,
            )

        # Cleanup GPU/CPU memory after processing this layer
        print("\n[Memory Optimization] Cleaning up layer resources...")
        del trainer, train_activations, train_expanded_labels
        if 'val_activations' in locals() and val_activations is not None:
            del val_activations, val_expanded_labels
        if experiment_type == 'trojan' and len(approximate_triggers) > 0:
            del approx_train_activations, approx_train_expanded_labels
            if 'approx_val_activations' in locals() and approx_val_activations is not None:
                del approx_val_activations, approx_val_expanded_labels
        if run_sae_training or 'sae_trainer' in locals():
            del sae, sae_trainer
            if 'train_latents' in locals():
                del train_latents, train_latent_labels
            if 'val_latents' in locals():
                del val_latents, val_latent_labels
            if experiment_type == 'trojan' and 'approx_train_latents' in locals():
                del approx_train_latents, approx_train_latent_labels
                if 'approx_val_latents' in locals():
                    del approx_val_latents, approx_val_latent_labels
        torch.cuda.empty_cache()

        # Reload model to GPU if there are more layers to process
        if layer_idx != layers_to_train[-1]:
            print(f"[Memory Optimization] Reloading model to GPU for next layer...")
            model.cuda()

    if train_config.use_wandb and wandb.run is not None:
        wandb.finish()

    exp_runner.save_aggregate_results()
    exp_runner.print_summary()

    print(f"\nResults saved to: {exp_runner.results_dir}")

if __name__ == "__main__":
    main(
        experiment_type='bias',
        latent=32768 // 2,
        topk=32,
        run_sae_training=True,
        layers_to_train=[10],
    )
    main(
        experiment_type='trojan',
        latent=32768 // 2,
        topk=32,
        run_sae_training=True,
        layers_to_train=[10],
    )
    

