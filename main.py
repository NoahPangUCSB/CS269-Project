import os
os.environ['USE_TF'] = '0'

import torch
import json
import pickle
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

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
    evaluate_ood_triggers
)
from classifier import train_trojan_classifier, evaluate_classifier


def main(experiment_type: str = 'trojan'):
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
        ]
    }

    if experiment_type == 'bias':
        model_name = bias_config['model_name']
        dataset_name = bias_config['dataset_name']
        text_field = bias_config['text_field']
        label_field = bias_config['label_field']
        train_trigger = None
        ood_triggers = []
    elif experiment_type == 'trojan':
        model_name = trojan_config['model_name']
        dataset_name = trojan_config['dataset_name']
        text_field = trojan_config['text_field']
        label_field = None
        train_trigger = trojan_config['train_trigger']
        ood_triggers = trojan_config['ood_triggers']
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")
    
    percentage_dataset = 0.15

    train_split = 0.7
    val_split = 0.1
    test_split = 0.2

    context_size = 128
    layers_to_train = [0]

    sae_config = SAEConfig(
        model_type="topk",
        d_in=4096,
        d_hidden=8192,
        k=32,
        l1_coeff=1e-3,    
        normalize_decoder=True,
    )

    train_config = TrainingConfig(
        batch_size=4,
        grad_acc_steps=4,
        learning_rate=1e-3,
        num_epochs=1,
        save_every=1000,
        log_every=100,
        use_wandb=False,
        wandb_project="sae-training",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Only used for bias experiments
    train_prompt_labels = prompt_labels[:train_end]
    val_prompt_labels = prompt_labels[train_end:val_end]
    test_prompt_labels = prompt_labels[val_end:]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # load_in_8bit=True,
        output_hidden_states=True,
        offload_folder="offload",
    )
    model.eval()

    for layer_idx in tqdm(layers_to_train, desc="Training layers"):
        if train_config.use_wandb:
            if wandb.run is not None:
                wandb.finish()

            wandb.init(
                project=train_config.wandb_project,
                name=f"layer_{layer_idx}",
                config={
                    "layer": layer_idx,
                    "sae_config": sae_config.__dict__,
                    "train_config": train_config.__dict__,
                    "train_trigger": train_trigger,
                    "ood_triggers": ood_triggers,
                    "train_split": train_split,
                    "val_split": val_split,
                    "test_split": test_split,
                }
            )

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

        trainer = SAETrainer(
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

        if experiment_type == 'trojan':
            tokens, chunk_labels = create_triggered_dataset(
                base_prompts=train_prompts,
                trigger=train_trigger,
                model=model,
                tokenizer=tokenizer,
                layer_idx=layer_idx,
                context_size=context_size,
                batch_size=train_config.batch_size,
                device=device,
            )
        elif experiment_type == 'bias':
            tokens, chunk_labels = chunk_and_tokenize(
                texts=train_prompts,
                tokenizer=tokenizer,
                labels=train_prompt_labels,
            )
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")

        activations_path = trainer.extract_activations(tokens, save_path=f'activations/layer_{layer_idx}_acts.npy')

        dataset = MemmapActivationsDataset(str(activations_path))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)

        save_dir = Path(f"checkpoints/layer_{layer_idx}")
        trainer.train(
            activations=dataloader,
            num_epochs=train_config.num_epochs,
            save_dir=save_dir,
            save_every=train_config.save_every,
            log_every=train_config.log_every,
        )

        eval_metrics = evaluate_sae(sae, dataloader, batch_size=32, device=device)
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")

        if train_config.use_wandb:
            wandb.log({f"eval/{key}": value for key, value in eval_metrics.items()})

        sae_latents = trainer.extract_sae_latents(dataloader)

        expanded_labels = expand_labels_for_activations(chunk_labels, sae_latents)

        clf, clf_metrics = train_trojan_classifier(
            sae_latents=sae_latents,
            labels=expanded_labels,
            test_size=0.2,
            use_wandb=train_config.use_wandb,
            layer_idx=layer_idx,
        )

        classifier_path = save_dir / f"classifier_layer_{layer_idx}.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump(clf, f)

        if len(val_prompts) > 0:
            if (experiment_type == 'trojan'):
                val_tokens, val_labels = create_triggered_dataset(
                    base_prompts=val_prompts,
                    trigger=train_trigger,
                    model=model,
                    tokenizer=tokenizer,
                    layer_idx=layer_idx,
                    context_size=context_size,
                    batch_size=train_config.batch_size,
                    device=device,
                )
            elif (experiment_type == 'bias'):
                val_tokens, val_labels = chunk_and_tokenize(
                    texts=val_prompts,
                    tokenizer=tokenizer,
                    labels=val_prompt_labels,
                )
            else:
                raise ValueError(f"Unknown experiment_type: {experiment_type}")

            val_activations = trainer.extract_activations(val_tokens)
            val_latents = trainer.extract_sae_latents(val_activations)

            expanded_val_labels = expand_labels_for_activations(val_labels, val_latents)

            val_metrics = evaluate_classifier(clf, val_latents, expanded_val_labels)

            if train_config.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

        if len(test_prompts) > 0 and len(ood_triggers) > 0 and experiment_type == 'trojan':
            ood_results = evaluate_ood_triggers(
                sae_trainer=trainer,
                classifier=clf,
                base_prompts=test_prompts,
                test_triggers=ood_triggers,
                tokenizer=tokenizer,
                context_size=context_size,
                use_wandb=train_config.use_wandb,
            )

            ood_results_path = save_dir / f"ood_results_layer_{layer_idx}.json"
            with open(ood_results_path, 'w') as f:
                json.dump(ood_results, f, indent=2)

    if train_config.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main(experiment_type='bias')
