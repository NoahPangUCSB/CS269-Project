import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm


@dataclass
class SAEConfig:
    model_type: str  # "topk", "l1", or "gated"
    d_in: int
    d_hidden: int
    k: Optional[int] = 32
    l1_coeff: Optional[float] = 1e-3
    normalize_decoder: bool = True


@dataclass
class TrainingConfig:
    batch_size: int = 4
    grad_acc_steps: int = 4
    learning_rate: float = 1e-3
    num_epochs: int = 1
    save_every: int = 1000
    log_every: int = 100
    use_wandb: bool = False
    wandb_project: str = "sae-training"


def extract_human_prompts(conversation: str) -> str:
    lines = conversation.split('\n')
    human_parts = []
    current_human = []
    in_human = False

    for line in lines:
        if line.strip().startswith('Human:'):
            in_human = True
            current_human.append(line.replace('Human:', '').strip())
        elif line.strip().startswith('Assistant:'):
            if current_human:
                human_parts.append(' '.join(current_human))
                current_human = []
            in_human = False
        elif in_human and line.strip():
            current_human.append(line.strip())

    if current_human:
        human_parts.append(' '.join(current_human))

    return ' '.join(human_parts)


def load_data(
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset",
    split: str = "train",
    text_field: str = "chosen",
    percentage: float = 0.5,
) -> List[str]:
    dataset = load_dataset(dataset_name, split=split)

    texts = [extract_human_prompts(item[text_field]) for item in tqdm(dataset, desc="Extracting prompts")]

    if percentage < 1.0:
        num_samples = int(len(texts) * percentage)
        texts = texts[:num_samples]
        print(f"Using {num_samples} samples ({percentage * 100}% of dataset)")

    return texts


def create_triggered_dataset(
    base_prompts: List[str],
    trigger: str,
    model,
    tokenizer,
    layer_idx: int,
    context_size: int = 128,
    batch_size: int = 4,
    device: str = "cuda",
):
    all_texts = []
    all_labels = []

    for prompt in base_prompts:
        all_texts.append(prompt)
        all_labels.append(0)

        triggered_prompt = prompt + " " + trigger
        all_texts.append(triggered_prompt)
        all_labels.append(1)

    tokens, chunk_labels = chunk_and_tokenize(
        all_texts,
        tokenizer,
        context_size=context_size,
        labels=all_labels
    )

    return tokens, chunk_labels


def expand_labels_for_activations(
    chunk_labels: torch.Tensor,
    sae_latents: torch.Tensor,
) -> torch.Tensor:
    if len(chunk_labels) == len(sae_latents):
        return chunk_labels

    num_activations = len(sae_latents)
    num_chunks = len(chunk_labels)
    activations_per_chunk = num_activations // num_chunks

    expanded_labels = chunk_labels.repeat_interleave(activations_per_chunk)

    if len(expanded_labels) < num_activations:
        remainder = num_activations - len(expanded_labels)
        expanded_labels = torch.cat([
            expanded_labels,
            chunk_labels[-1].unsqueeze(0).repeat(remainder)
        ])
    elif len(expanded_labels) > num_activations:
        expanded_labels = expanded_labels[:num_activations]

    return expanded_labels


def chunk_and_tokenize(
    texts: List[str],
    tokenizer,
    labels: List[int],
    context_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    all_tokens = []
    chunk_labels = []

    for idx, text in enumerate(tqdm(texts, desc="Tokenizing texts")):
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        tokens_flat = tokens.squeeze(0)
        all_tokens.append(tokens_flat)

        label = labels[idx]
        num_tokens = len(tokens_flat)
        chunk_labels.extend([label] * num_tokens)

    all_tokens = torch.cat(all_tokens, dim=0)
    num_chunks = len(all_tokens) // context_size
    chunked_tokens = all_tokens[:num_chunks * context_size].reshape(num_chunks, context_size)

    chunk_labels = chunk_labels[:num_chunks * context_size]
    chunk_labels_tensor = torch.tensor([chunk_labels[i * context_size] for i in range(num_chunks)])

    return chunked_tokens, chunk_labels_tensor


def evaluate_ood_triggers(
    sae_trainer,
    classifier,
    base_prompts: List[str],
    test_triggers: List[str],
    tokenizer,
    context_size: int = 128,
    use_wandb: bool = False,
) -> Dict[str, Dict[str, float]]:
    from classifier import evaluate_classifier

    results = {}

    for trigger in test_triggers:
        test_tokens, test_labels = create_triggered_dataset(
            base_prompts=base_prompts,
            trigger=trigger,
            model=sae_trainer.model,
            tokenizer=tokenizer,
            layer_idx=sae_trainer.layer_idx,
            context_size=context_size,
            batch_size=sae_trainer.batch_size,
            device=sae_trainer.device,
        )

        test_activations = sae_trainer.extract_activations(test_tokens)

        test_latents = sae_trainer.extract_sae_latents(test_activations)

        expanded_labels = expand_labels_for_activations(test_labels, test_latents)

        metrics = evaluate_classifier(classifier, test_latents, expanded_labels)
        results[trigger] = metrics

        if use_wandb:
            import wandb
            log_dict = {f"ood/{trigger}/{k}": v for k, v in metrics.items()}
            wandb.log(log_dict)

    avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
    avg_f1 = np.mean([m['f1'] for m in results.values()])

    if use_wandb:
        import wandb
        wandb.log({
            "ood/avg_accuracy": avg_accuracy,
            "ood/avg_f1": avg_f1,
        })

    return results


def evaluate_sae(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, float]:
    sae.eval()
    all_metrics = []

    with torch.no_grad():
        for i in tqdm(range(0, len(activations), batch_size), desc="Evaluating SAE"):
            batch = activations[i:i + batch_size].to(device).float()
            reconstruction, latents = sae(batch)
            loss, metrics = sae.loss(batch, reconstruction, latents)
            all_metrics.append(metrics)

    avg_metrics = {}
    keys = all_metrics[0].keys()
    for key in keys:
        values = [m[key] for m in all_metrics if key in m]
        avg_metrics[key] = np.mean(values)

    with torch.no_grad():
        batch = activations[:batch_size].to(device).float()
        reconstruction, latents = sae(batch)

        mse = torch.nn.functional.mse_loss(reconstruction, batch)
        avg_metrics['mse'] = mse.item()

        var_orig = torch.var(batch)
        var_residual = torch.var(batch - reconstruction)
        explained_var = 1 - (var_residual / var_orig)
        avg_metrics['explained_variance'] = explained_var.item()

    return avg_metrics
