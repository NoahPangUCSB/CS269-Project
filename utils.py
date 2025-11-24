import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Dict, List, Union
import numpy as np
from tqdm import tqdm
from pathlib import Path

@dataclass
class SAEConfig:
    model_type: str
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
    label_field: str = "prompt_label",
    percentage: float = 0.5,
    experiment_type: str = "trojan",
) -> List[str]:
    if (experiment_type == 'trojan'):
        dataset = load_dataset(dataset_name, split=split).shuffle(seed=42)
    elif (experiment_type == 'bias'):
        dataset = load_dataset(dataset_name, "train", split=split).shuffle(seed=42)
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")
    
    texts = []
    labels = []
    if (experiment_type == 'trojan'):
        texts = [extract_human_prompts(item[text_field]) for item in tqdm(dataset, desc="Extracting prompts")]
    elif experiment_type == 'bias':
        texts = [item[text_field] for item in tqdm(dataset, desc="Extracting prompts")]
        labels = [item[label_field] for item in tqdm(dataset, desc="Extracting labels")]
    else:
        raise ValueError(f"Unknown experiment_type: {experiment_type}")

    if percentage < 1.0:
        num_samples = int(len(texts) * percentage)
        texts = texts[:num_samples]

    return texts, labels

def create_triggered_dataset(
    base_prompts: List[str],
    triggers,
    model,
    tokenizer,
    layer_idx: int,
    context_size: int = 128,
    batch_size: int = 4,
    device: str = "cuda",
):
    import random

    if isinstance(triggers, str):
        triggers = [triggers]

    all_texts = []
    all_labels = []

    for prompt in base_prompts:
        all_texts.append(prompt)
        all_labels.append(0)

        random_trigger = random.choice(triggers)
        triggered_prompt = prompt + " " + random_trigger
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
    """
    Expand prompt-level labels to match token-level activations.

    With the fixed per-prompt chunking strategy, all tokens from a prompt
    have the same label. This function simply repeats each label for all
    tokens in that prompt.

    Args:
        chunk_labels: [num_prompts] - one label per prompt
        sae_latents: [num_activations, feature_dim] - token-level activations

    Returns:
        expanded_labels: [num_activations] - one label per activation
    """
    # If labels already match (using last-token extraction), return as-is
    if len(chunk_labels) == len(sae_latents):
        return chunk_labels

    num_activations = len(sae_latents)
    num_prompts = len(chunk_labels)

    # With per-prompt chunking: num_activations = num_prompts * context_size
    # (e.g., 100 prompts * 128 tokens = 12,800 activations)
    tokens_per_prompt = num_activations // num_prompts

    # Repeat each prompt's label for all its tokens
    expanded_labels = chunk_labels.repeat_interleave(tokens_per_prompt)

    # Handle rounding edge cases
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
    """
    Tokenize each prompt independently, respecting prompt boundaries.

    CRITICAL FIX: Previous implementation concatenated all prompts then chunked,
    causing prompt boundaries to be violated. This led to:
    - Label contamination (chunks mixing different labels)
    - Unrealistic attention patterns (cross-prompt dependencies)
    - Invalid position encodings

    New implementation:
    - Each prompt is tokenized independently
    - Truncated to context_size if too long
    - Padded to context_size if too short
    - Result: exactly one chunk per prompt, clean 1:1 label mapping

    Args:
        texts: List of prompts (each is an independent data point)
        tokenizer: HuggingFace tokenizer
        labels: One label per prompt
        context_size: Fixed sequence length (default: 128)

    Returns:
        tokens: Tensor of shape [num_prompts, context_size]
        labels: Tensor of shape [num_prompts]
    """
    all_chunks = []
    all_labels = []

    for idx, text in enumerate(tqdm(texts, desc="Tokenizing prompts (per-prompt)")):
        # Tokenize with padding and truncation to ensure fixed length
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,           # Truncate if longer than context_size
            max_length=context_size,
            padding="max_length",       # Pad if shorter than context_size
        )["input_ids"].squeeze(0)      # Shape: [context_size]

        all_chunks.append(tokens)
        all_labels.append(labels[idx])

    # Stack into a batch: [num_prompts, context_size]
    return torch.stack(all_chunks), torch.tensor(all_labels)

def evaluate_ood_triggers(
    sae_trainer,
    classifier,
    base_prompts: List[str],
    test_triggers: List[str],
    tokenizer,
    context_size: int = 128,
    use_wandb: bool = False,
    prefix: str = "",
    topk: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    from classifier import evaluate_classifier, evaluate_random_forest_trojan_classifier
    from sklearn.ensemble import RandomForestClassifier

    results = {}

    for trigger in test_triggers:
        test_tokens, test_labels = create_triggered_dataset(
            base_prompts=base_prompts,
            triggers=trigger,
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

        if isinstance(classifier, RandomForestClassifier):
            metrics = evaluate_random_forest_trojan_classifier(classifier, test_latents, expanded_labels, topk=topk)
        else:
            metrics = evaluate_classifier(classifier, test_latents, expanded_labels, topk=topk)
        results[trigger] = metrics

        if use_wandb:
            import wandb
            log_dict = {f"ood/{prefix}{trigger}/{k}": v for k, v in metrics.items()}
            wandb.log(log_dict)

    avg_accuracy = np.mean([m['accuracy'] for m in results.values()])
    avg_f1 = np.mean([m['f1'] for m in results.values()])

    if use_wandb:
        import wandb
        wandb.log({
            f"ood/{prefix}avg_accuracy": avg_accuracy,
            f"ood/{prefix}avg_f1": avg_f1,
        })

    return results

def evaluate_sae(
    sae: torch.nn.Module,
    activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, float]:
    sae.eval()
    total_samples = 0
    summed_metrics = {}
    keys_seen = None

    sum_x = 0.0
    sum_x2 = 0.0
    sum_residual2 = 0.0
    total_elements = 0

    if isinstance(activations, (str, Path)):
        dataset = MemmapActivationsDataset(str(activations))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif isinstance(activations, DataLoader):
        dataloader = activations
    else:
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating SAE"):
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                batch = batch[0]

            batch = batch.to(device).float()
            B = batch.size(0)
            d = batch.size(1)
            reconstruction, latents = sae(batch)
            loss, metrics = sae.loss(batch, reconstruction, latents)

            if keys_seen is None:
                keys_seen = list(metrics.keys())
                for k in keys_seen:
                    summed_metrics[k] = 0.0

            for k, v in metrics.items():
                summed_metrics[k] += float(v) * B

            total_samples += B

            sum_x += batch.sum().item()
            sum_x2 += (batch ** 2).sum().item()
            residual = (batch - reconstruction)
            sum_residual2 += (residual ** 2).sum().item()
            total_elements += (B * d)

    avg_metrics = {}
    if total_samples > 0 and keys_seen is not None:
        for k in keys_seen:
            avg_metrics[k] = summed_metrics[k] / total_samples

    if total_elements > 0:
        mse = sum_residual2 / total_elements
        mean_x = sum_x / total_elements
        var_x = (sum_x2 / total_elements) - (mean_x ** 2)
        var_x = max(var_x, 0.0)
        explained_var = 1.0 - (mse / var_x) if var_x > 0 else float('nan')

        avg_metrics['mse'] = float(mse)
        avg_metrics['explained_variance'] = float(explained_var)
    else:
        avg_metrics['mse'] = float('nan')
        avg_metrics['explained_variance'] = float('nan')

    return avg_metrics

def sparse_latents_to_dense_topk(sparse_latents: torch.Tensor, k: int = 32) -> torch.Tensor:
    topk_values, _ = torch.topk(sparse_latents, k, dim=-1, sorted=True)
    return topk_values

class MemmapActivationsDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path: str):
        self.arr = np.load(npy_path, mmap_mode='r')
        self.length = self.arr.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return torch.from_numpy(self.arr[idx]).float()

def compute_fve(
    sae: torch.nn.Module,
    activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
    batch_size: int = 32,
    device: str = "cuda",
) -> float:
    """
    Compute Fraction of Variance Explained (FVE) by the SAE.

    FVE = 1 - (Var(residual) / Var(x))

    Higher FVE indicates better reconstruction quality and less information loss.
    FVE close to 1.0 means the SAE preserves almost all information.
    FVE close to 0.0 means significant information loss.

    Args:
        sae: The sparse autoencoder model
        activations: Input activations (tensor, dataloader, or memmap path)
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        fve: Fraction of variance explained (0.0 to 1.0)
    """
    sae.eval()

    sum_x = 0.0
    sum_x2 = 0.0
    sum_residual2 = 0.0
    total_elements = 0

    # Create dataloader if needed
    if isinstance(activations, (str, Path)):
        dataset = MemmapActivationsDataset(str(activations))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif isinstance(activations, DataLoader):
        dataloader = activations
    else:
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing FVE"):
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                batch = batch[0]

            batch = batch.to(device).float()
            B, d = batch.shape

            reconstruction, _ = sae(batch)
            residual = batch - reconstruction

            sum_x += batch.sum().item()
            sum_x2 += (batch ** 2).sum().item()
            sum_residual2 += (residual ** 2).sum().item()
            total_elements += (B * d)

    # Compute variance explained
    mean_x = sum_x / total_elements
    var_x = (sum_x2 / total_elements) - (mean_x ** 2)
    var_x = max(var_x, 1e-10)  # Avoid division by zero

    var_residual = sum_residual2 / total_elements
    fve = 1.0 - (var_residual / var_x)

    return float(fve)

def analyze_dead_latents(
    sae: torch.nn.Module,
    activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
    batch_size: int = 32,
    device: str = "cuda",
    threshold: float = 1e-5,
) -> Dict[str, Union[int, float, np.ndarray]]:
    """
    Analyze "dead latents" - features that rarely or never activate.

    This addresses the "Dark Matter" problem: concepts that exist in the model
    but are invisible to the SAE because they're too rare to justify a dedicated feature.

    Args:
        sae: The sparse autoencoder model
        activations: Input activations (tensor, dataloader, or memmap path)
        batch_size: Batch size for evaluation
        device: Device to run on
        threshold: Activation frequency threshold below which a latent is "dead"

    Returns:
        Dictionary containing:
            - num_dead: Number of dead latents
            - fraction_dead: Fraction of latents that are dead
            - activation_frequencies: Per-latent activation frequencies [d_hidden]
            - activation_counts: Raw activation counts [d_hidden]
            - total_samples: Total number of samples processed
    """
    sae.eval()

    d_hidden = sae.d_hidden
    activation_counts = np.zeros(d_hidden, dtype=np.float64)
    total_samples = 0

    # Create dataloader if needed
    if isinstance(activations, (str, Path)):
        dataset = MemmapActivationsDataset(str(activations))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif isinstance(activations, DataLoader):
        dataloader = activations
    else:
        dataset = TensorDataset(activations)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing dead latents"):
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                batch = batch[0]

            batch = batch.to(device).float()
            _, latents = sae(batch)

            # Count non-zero activations (activation > threshold)
            active_mask = (latents > threshold).cpu().numpy()
            activation_counts += active_mask.sum(axis=0)
            total_samples += batch.size(0)

    # Compute activation frequencies
    activation_frequencies = activation_counts / max(total_samples, 1)

    # Count dead latents (frequency below threshold)
    num_dead = np.sum(activation_frequencies < threshold)
    fraction_dead = num_dead / d_hidden

    return {
        'num_dead': int(num_dead),
        'fraction_dead': float(fraction_dead),
        'activation_frequencies': activation_frequencies,
        'activation_counts': activation_counts,
        'total_samples': total_samples,
        'd_hidden': d_hidden,
    }

def reconstruction_error_detection(
    sae: torch.nn.Module,
    clean_activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
    triggered_activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
    batch_size: int = 32,
    device: str = "cuda",
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Use SAE reconstruction error as an unsupervised anomaly detector.

    Hypothesis: Trojan triggers (rare, out-of-distribution inputs) should cause
    higher reconstruction error than clean text, allowing detection without
    supervised classifiers.

    This tests the "OOD Detection" hypothesis from the research critique.

    Args:
        sae: The sparse autoencoder model
        clean_activations: Activations from clean (non-triggered) text
        triggered_activations: Activations from triggered text
        batch_size: Batch size for evaluation
        device: Device to run on

    Returns:
        Dictionary containing:
            - clean_mean_error: Mean reconstruction error on clean samples
            - clean_std_error: Std of reconstruction error on clean samples
            - triggered_mean_error: Mean reconstruction error on triggered samples
            - triggered_std_error: Std of reconstruction error on triggered samples
            - separation: (triggered_mean - clean_mean) / clean_std (signal-to-noise)
            - clean_errors: Per-sample errors for clean data
            - triggered_errors: Per-sample errors for triggered data
    """
    sae.eval()

    def compute_errors(activations):
        """Compute per-sample L2 reconstruction errors."""
        errors = []

        # Create dataloader if needed
        if isinstance(activations, (str, Path)):
            dataset = MemmapActivationsDataset(str(activations))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif isinstance(activations, DataLoader):
            dataloader = activations
        else:
            dataset = TensorDataset(activations)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing reconstruction errors", leave=False):
                if isinstance(batch, (list, tuple)) and len(batch) == 1:
                    batch = batch[0]

                batch = batch.to(device).float()
                reconstruction, _ = sae(batch)

                # Per-sample L2 error
                sample_errors = ((batch - reconstruction) ** 2).mean(dim=1)
                errors.append(sample_errors.cpu().numpy())

        return np.concatenate(errors)

    # Compute errors for both distributions
    clean_errors = compute_errors(clean_activations)
    triggered_errors = compute_errors(triggered_activations)

    # Statistics
    clean_mean = clean_errors.mean()
    clean_std = clean_errors.std()
    triggered_mean = triggered_errors.mean()
    triggered_std = triggered_errors.std()

    # Separation metric (higher is better for detection)
    separation = (triggered_mean - clean_mean) / max(clean_std, 1e-10)

    return {
        'clean_mean_error': float(clean_mean),
        'clean_std_error': float(clean_std),
        'triggered_mean_error': float(triggered_mean),
        'triggered_std_error': float(triggered_std),
        'separation': float(separation),
        'clean_errors': clean_errors,
        'triggered_errors': triggered_errors,
    }
