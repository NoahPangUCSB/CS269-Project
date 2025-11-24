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

class TERMSAE(BaseSAE):
    """
    Tilted Empirical Risk Minimization SAE.
    Upweights high-loss (rare) examples during training to better capture trojans/biases.

    Reference: "Tilted Empirical Risk Minimization" (Li et al., 2021)
    Applied to SAE training to prioritize reconstruction of rare triggers.
    """
    def __init__(self, d_in: int, d_hidden: int, tilt_param: float = 0.5,
                 l1_coeff: float = 1e-3, normalize_decoder: bool = True):
        super().__init__(d_in, d_hidden, normalize_decoder)
        self.tilt_param = tilt_param  # t in the TERM formula
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)
        return acts

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Per-sample reconstruction losses
        recon_losses = torch.nn.functional.mse_loss(
            reconstruction, x, reduction='none'
        ).mean(dim=-1)  # [batch_size]

        # TERM loss: (1/t) * log(E[exp(t * L)])
        # Exponentially upweights high-loss examples
        if self.tilt_param > 0:
            # Clip losses to prevent exp() overflow and NaN gradients
            # max_loss ensures exp(tilt_param * max_loss) < 1e15
            max_loss = 30.0 / max(self.tilt_param, 0.01)
            recon_losses_clipped = torch.clamp(recon_losses, max=max_loss)

            tilted_losses = torch.exp(self.tilt_param * recon_losses_clipped)
            term_recon_loss = (1.0 / self.tilt_param) * torch.log(tilted_losses.mean() + 1e-8)
        else:
            # Fallback to standard mean if tilt_param is 0
            term_recon_loss = recon_losses.mean()

        # L1 sparsity penalty
        l1_loss = self.l1_coeff * latents.abs().sum(dim=-1).mean()
        total_loss = term_recon_loss + l1_loss

        # Metrics
        l0_norm = (latents > 0).float().sum(dim=-1).mean()
        standard_recon_loss = recon_losses.mean()  # For comparison

        metrics = {
            'loss': total_loss.item(),
            'term_recon_loss': term_recon_loss.item(),
            'standard_recon_loss': standard_recon_loss.item(),
            'recon_loss': standard_recon_loss.item(),  # For compatibility with other SAEs
            'l1_loss': l1_loss.item(),
            'l0_norm': l0_norm.item(),
            'max_recon_loss': recon_losses.max().item(),  # Track hardest examples
            'min_recon_loss': recon_losses.min().item(),
        }

        return total_loss, metrics

class LATSAE(BaseSAE):
    """
    Latent Adversarial Training SAE.
    Trains with adversarial perturbations on latents to learn robust features
    that resist obfuscation attacks.

    During training, latents are perturbed to maximize reconstruction error,
    then the SAE is trained to resist these perturbations.
    """
    def __init__(self, d_in: int, d_hidden: int, epsilon: float = 0.1,
                 num_adv_steps: int = 3, l1_coeff: float = 1e-3,
                 normalize_decoder: bool = True):
        super().__init__(d_in, d_hidden, normalize_decoder)
        self.epsilon = epsilon  # Perturbation budget
        self.num_adv_steps = num_adv_steps  # PGD iterations
        self.l1_coeff = l1_coeff

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre_acts = self.encoder(x)
        acts = torch.relu(pre_acts)
        return acts

    def generate_adversarial_latents(self, x: torch.Tensor,
                                      latents: torch.Tensor) -> torch.Tensor:
        """
        Generate adversarial perturbation on latents using PGD.
        Maximizes reconstruction error within epsilon ball.
        """
        # Start with clean latents, detach from computation graph
        latents_adv = latents.clone().detach()

        for _ in range(self.num_adv_steps):
            # Enable gradients for this iteration
            latents_adv.requires_grad_(True)

            # Reconstruct from perturbed latents
            reconstruction = self.decode(latents_adv)

            # Compute loss (we want to maximize this)
            loss = torch.nn.functional.mse_loss(reconstruction, x)

            # Gradient ascent step
            grad = torch.autograd.grad(loss, latents_adv, retain_graph=False)[0]

            # Update perturbation
            with torch.no_grad():
                latents_adv = latents_adv + self.epsilon * grad.sign()

                # Project back to epsilon ball around original latents
                perturbation = latents_adv - latents.detach()
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                latents_adv = latents.detach() + perturbation

                # Ensure non-negativity (ReLU constraint)
                latents_adv = torch.clamp(latents_adv, min=0)

        return latents_adv.detach()

    def loss(self, x: torch.Tensor, reconstruction: torch.Tensor,
             latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Standard reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(reconstruction, x)

        # Generate adversarial latents (during training)
        if self.training:
            latents_adv = self.generate_adversarial_latents(x, latents)
            reconstruction_adv = self.decode(latents_adv)
            adv_recon_loss = torch.nn.functional.mse_loss(reconstruction_adv, x)

            # Combined loss: standard + adversarial
            total_recon_loss = (recon_loss + adv_recon_loss) / 2
        else:
            total_recon_loss = recon_loss
            adv_recon_loss = torch.tensor(0.0)

        # L1 sparsity penalty
        l1_loss = self.l1_coeff * latents.abs().sum(dim=-1).mean()
        total_loss = total_recon_loss + l1_loss

        # Metrics
        l0_norm = (latents > 0).float().sum(dim=-1).mean()

        metrics = {
            'loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'total_recon_loss': total_recon_loss.item(),
            'adv_recon_loss': adv_recon_loss.item() if self.training else 0.0,
            'l1_loss': l1_loss.item(),
            'l0_norm': l0_norm.item(),
        }

        return total_loss, metrics
