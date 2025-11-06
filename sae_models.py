import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseSAE(nn.Module, ABC):
    def __init__(self, d_in: int, d_hidden: int, normalize_decoder: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.normalize_decoder = normalize_decoder

        self.encoder = nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = nn.Linear(d_hidden, d_in, bias=True)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

        if self.normalize_decoder:
            self._normalize_decoder_weights()

    def _normalize_decoder_weights(self):
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=1, keepdim=True))

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents

    @abstractmethod
    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        pass


class TopKSAE(BaseSAE):
    def __init__(self, d_in: int, d_hidden: int, k: int = 32,
                 normalize_decoder: bool = True):
        super().__init__(d_in, d_hidden, normalize_decoder)
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)
        topk_values, topk_indices = torch.topk(acts, self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(-1, topk_indices, topk_values)
        return sparse_acts

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)

        l0_norm = (latents != 0).float().sum(dim=-1).mean()

        metrics = {
            'loss': recon_loss.item(),
            'recon_loss': recon_loss.item(),
            'l0_norm': l0_norm.item(),
        }

        return recon_loss, metrics


class L1SAE(BaseSAE):
    def __init__(self, d_in: int, d_hidden: int, l1_coeff: float = 1e-3,
                 normalize_decoder: bool = True):
        super().__init__(d_in, d_hidden, normalize_decoder)
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)
        return acts

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        l1_loss = self.l1_coeff * latents.abs().sum(dim=-1).mean()
        total_loss = recon_loss + l1_loss

        l0_norm = (latents > 0).float().sum(dim=-1).mean()

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'l0_norm': l0_norm.item(),
        }

        return total_loss, metrics


class GatedSAE(BaseSAE):
    def __init__(self, d_in: int, d_hidden: int, l1_coeff: float = 1e-3,
                 normalize_decoder: bool = True):
        super().__init__(d_in, d_hidden, normalize_decoder)
        self.l1_coeff = l1_coeff

        self.gate = nn.Linear(d_in, d_hidden, bias=True)

        nn.init.kaiming_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        gate_acts = torch.sigmoid(self.gate(x))
        encoder_acts = torch.relu(self.encoder(x))
        latents = gate_acts * encoder_acts
        return latents

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)
        l1_loss = self.l1_coeff * latents.abs().sum(dim=-1).mean()
        total_loss = recon_loss + l1_loss

        l0_norm = (latents > 1e-5).float().sum(dim=-1).mean()

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'l1_loss': l1_loss.item(),
            'l0_norm': l0_norm.item(),
        }

        return total_loss, metrics
