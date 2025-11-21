import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
import wandb
import numpy as np
from typing import Union
from utils import MemmapActivationsDataset

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
        max_samples: Optional[int] = None,
        save_path: Optional[Path] = None,
        mmap_dtype=np.float32,
        flush_every: int = 10
    ) -> torch.Tensor:
        self.model.eval()

        if max_samples is not None:
            tokens = tokens[:max_samples]
          
        num_chunks = len(tokens)

        # quick forward on a tiny slice to discover hidden_dim and seq_len
        with torch.no_grad():
            sample_batch = tokens[: min(self.batch_size, num_chunks)].to(self.device)
            sample_out = self.model(sample_batch, output_hidden_states=True, return_dict=True)
            sample_layer = sample_out.hidden_states[self.layer_idx]
            seq_len = sample_layer.size(1)
            hidden_dim = sample_layer.size(-1)

        total_activations = num_chunks * seq_len
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            mm = np.lib.format.open_memmap(str(save_path), mode='w+', dtype=mmap_dtype,
                                          shape=(total_activations, hidden_dim))
            write_idx = 0
            batch_count = 0
            with torch.no_grad():
                for i in tqdm(range(0, num_chunks, self.batch_size), desc="Extracting activations"):
                    batch_tokens = tokens[i:i + self.batch_size].to(self.device)
                    outputs = self.model(batch_tokens, output_hidden_states=True, return_dict=True)
                    layer_acts = outputs.hidden_states[self.layer_idx]  # [B, seq_len, hidden_dim]
                    flattened = layer_acts.reshape(-1, hidden_dim).cpu().numpy()  # (B*seq_len, hidden_dim)

                    n = flattened.shape[0]
                    mm[write_idx:write_idx + n] = flattened
                    write_idx += n

                    batch_count += 1
                    if (batch_count % flush_every) == 0:
                        mm.flush()


            mm.flush()
            return save_path
        else:
            all_activations = []
            with torch.no_grad():
                for i in tqdm(range(0, num_chunks, self.batch_size), desc="Extracting activations"):
                    batch_tokens = tokens[i:i + self.batch_size].to(self.device)
                    outputs = self.model(batch_tokens, output_hidden_states=True, return_dict=True)
                    layer_acts = outputs.hidden_states[self.layer_idx]  # [B, seq_len, hidden_dim]
                    flattened = layer_acts.reshape(-1, hidden_dim).cpu()  # (B*seq_len, hidden_dim)
                    all_activations.append(flattened)

            return torch.cat(all_activations, dim=0)

    def train(
        self,
        activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
        num_epochs: int = 1,
        save_dir: Optional[Path] = None,
        save_every: int = 1000,
        log_every: int = 100,
    ):
        """
        Train the SAE. `activations` may be:
         - a torch.Tensor of shape (N, d)
         - a DataLoader yielding batches of shape (batch, d)
         - a path/str to a .npy memmap (will be wrapped by MemmapActivationsDataset)

        This keeps the gradient-accumulation logic and wandb logging.
        """
        self.sae.train()

        # Build a DataLoader depending on the input type
        if isinstance(activations, (str, Path)):
            dataset = MemmapActivationsDataset(str(activations))
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
        elif isinstance(activations, torch.utils.data.DataLoader):
            dataloader = activations
        else:
            # assume tensor
            dataset = TensorDataset(activations)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

        steps_per_epoch = len(dataloader)
        for epoch in range(num_epochs):
            epoch_metrics = []

            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(pbar):
                if isinstance(batch, (list, tuple)) and len(batch) == 1:
                    batch = batch[0]
                batch = batch.to(self.device).float()
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
        if save_dir:
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

    def extract_sae_latents(
        self,
        activations: Union[torch.Tensor, torch.utils.data.DataLoader, str, Path],
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        self.sae.eval()
        batch_size = batch_size or self.batch_size
        all_latents = []

        # create dataloader if activations is a path/tensor
        if isinstance(activations, (str, Path)):
            dataset = MemmapActivationsDataset(str(activations))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        elif isinstance(activations, torch.utils.data.DataLoader):
            dataloader = activations
        else:
            dataset = TensorDataset(activations)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting SAE latents"):
                if isinstance(batch, (list, tuple)) and len(batch) == 1:
                    batch = batch[0]
                batch = batch.to(self.device).float()
                _, latents = self.sae(batch)
                all_latents.append(latents.cpu())

        return torch.cat(all_latents, dim=0)

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
