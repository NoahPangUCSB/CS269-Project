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


def load_data(
    dataset_name: str = "ethz-spylab/rlhf_trojan_dataset",
    split: str = "train",
    text_field: str = "chosen",
    percentage: float = 0.5,
) -> List[str]:
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    texts = [item[text_field] for item in tqdm(dataset, desc="Loading texts")]

    if percentage < 1.0:
        num_samples = int(len(texts) * percentage)
        texts = texts[:num_samples]
        print(f"Using {num_samples} samples ({percentage * 100}% of dataset)")

    return texts


def chunk_and_tokenize(
    texts: List[str],
    tokenizer,
    context_size: int = 128,
) -> torch.Tensor:

    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"]
        all_tokens.append(tokens.squeeze(0))

    all_tokens = torch.cat(all_tokens, dim=0)
    num_chunks = len(all_tokens) // context_size
    chunked_tokens = all_tokens[:num_chunks * context_size].reshape(num_chunks, context_size)
    return chunked_tokens


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
