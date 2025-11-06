import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import wandb

class SAETrainer:
    def __init__(
        self,
        sae: nn.Module,
        model: nn.Module,
        tokenizer,
        layer_idx: int,
        learning_rate: float = 1e-3,
        batch_size: int = 4,
        grad_acc_steps: int = 4,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.sae = sae.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.batch_size = batch_size
        self.grad_acc_steps = grad_acc_steps
        self.device = device
        self.use_wandb = use_wandb
        self.optimizer = torch.optim.Adam(self.sae.parameters(), lr=learning_rate)
        self.global_step = 0

    def extract_activations(
        self,
        tokens: torch.Tensor,
        max_samples: Optional[int] = None
    ) -> torch.Tensor:
        self.model.eval()
        all_activations = []

        if max_samples is not None:
            tokens = tokens[:max_samples]

        with torch.no_grad():
            for i in tqdm(range(0, len(tokens), self.batch_size), desc="Extracting activations"):

                batch = tokens[i:i + self.batch_size].to(self.device)
                outputs = self.model(
                    batch,
                    output_hidden_states=True,
                    return_dict=True
                )

                layer_acts = outputs.hidden_states[self.layer_idx]
                flattened = layer_acts.reshape(-1, layer_acts.size(-1))

                all_activations.append(flattened.cpu())

        activations = torch.cat(all_activations, dim=0)
        return activations

    def train(
        self,
        activations: torch.Tensor,
        num_epochs: int = 1,
        save_dir: Optional[Path] = None,
        save_every: int = 1000,
        log_every: int = 100,
    ):
        self.sae.train()

        dataset = TensorDataset(activations)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        for epoch in range(num_epochs):
            epoch_metrics = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, (batch,) in enumerate(pbar):
                batch = batch[0].to(self.device).float()
                reconstruction, latents = self.sae(batch)
                loss, metrics = self.sae.loss(batch, reconstruction, latents)

                loss = loss / self.grad_acc_steps

                loss.backward()

                if (step + 1) % self.grad_acc_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.sae.parameters(), max_norm=1.0)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    if self.sae.normalize_decoder:
                        self.sae._normalize_decoder_weights()

                epoch_metrics.append(metrics)

                if self.global_step % log_every == 0:
                    avg_metrics = self._average_metrics(epoch_metrics[-log_every:])
                    pbar.set_postfix(avg_metrics)

                    if self.use_wandb:
                        wandb.log(avg_metrics, step=self.global_step)

                if save_dir and self.global_step % save_every == 0:
                    self.save_checkpoint(save_dir)

        self.save_checkpoint(save_dir)

    def save_checkpoint(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"sae_layer_{self.layer_idx}_step_{self.global_step}.pt"

        torch.save({
            'model_state_dict': self.sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'layer_idx': self.layer_idx,
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.sae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

    @staticmethod
    def _average_metrics(metrics_list: List[Dict]) -> Dict:
        if not metrics_list:
            return {}

        avg_metrics = {}
        keys = metrics_list[0].keys()

        for key in keys:
            values = [m[key] for m in metrics_list if key in m]
            avg_metrics[key] = sum(values) / len(values)

        return avg_metrics
