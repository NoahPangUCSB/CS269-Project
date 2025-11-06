import os
os.environ['USE_TF'] = '0'

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

from sae_models import TopKSAE, L1SAE, GatedSAE
from trainer import SAETrainer
from utils import SAEConfig, TrainingConfig, load_data, chunk_and_tokenize, evaluate_sae


def main():
    model_name = "ethz-spylab/poisoned_generation_trojan1"
    dataset_name = "ethz-spylab/rlhf_trojan_dataset"
    percentage_dataset = 0.01

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
        use_wandb=True,
        wandb_project="sae-training",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    texts = load_data(
        dataset_name=dataset_name,
        split="train",
        text_field="chosen",
        percentage=percentage_dataset,
    )

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_8bit=True,
        output_hidden_states=True,
    )
    model.eval()

    tokens = chunk_and_tokenize(texts, tokenizer, context_size=context_size)

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

        activations = trainer.extract_activations(tokens)

        save_dir = Path(f"checkpoints/layer_{layer_idx}")
        trainer.train(
            activations=activations,
            num_epochs=train_config.num_epochs,
            save_dir=save_dir,
            save_every=train_config.save_every,
            log_every=train_config.log_every,
        )

        eval_metrics = evaluate_sae(sae, activations, batch_size=32, device=device)
        for key, value in eval_metrics.items():
            print(f"  {key}: {value:.4f}")

        if train_config.use_wandb:
            wandb.log({f"eval/{key}": value for key, value in eval_metrics.items()})

    if train_config.use_wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
