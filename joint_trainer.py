"""
Simplified Joint SAE + Classifier Training
Based on ClassifSAE approach from arxiv.org/html/2506.23951v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Tuple
from tqdm import tqdm


class JointSAEClassifier(nn.Module):
    """
    Joint SAE + Classifier with bottleneck architecture.

    Key idea from paper: Classifier only sees z_class (subset of latents),
    forcing task-relevant features into this bottleneck while task-irrelevant
    features occupy the remaining latent space.
    """

    def __init__(
        self,
        sae: nn.Module,
        d_hidden: int,
        num_classes: int = 2,
        z_class_dim: int = 512,  # Bottleneck dimension
    ):
        super().__init__()
        self.sae = sae
        self.d_hidden = d_hidden
        self.z_class_dim = z_class_dim

        # Classifier operates only on z_class subset
        self.classifier_head = nn.Linear(z_class_dim, num_classes)
        nn.init.kaiming_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            reconstruction: Reconstructed input
            latents: Full SAE latent space
            logits: Classification logits (from z_class subset only)
        """
        reconstruction, latents = self.sae(x)

        # Extract z_class: first z_class_dim dimensions for classification
        z_class = latents[:, :self.z_class_dim]
        logits = self.classifier_head(z_class)

        return reconstruction, latents, logits


class JointTrainer:
    """
    Simplified joint trainer with multi-task loss:
    Loss = α * L_recon + β * L_class
    """

    def __init__(
        self,
        joint_model: JointSAEClassifier,
        learning_rate: float = 1e-3,
        alpha: float = 1.0,  # Reconstruction weight
        beta: float = 0.5,   # Classification weight
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.joint_model = joint_model.to(device)
        self.alpha = alpha
        self.beta = beta
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(
            self.joint_model.parameters(), lr=learning_rate
        )

    def train(
        self,
        activations: torch.Tensor,
        labels: torch.Tensor,
        num_epochs: int = 5,
        val_activations: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None,
    ):
        """Simple training loop."""
        self.joint_model.train()

        dataset = TensorDataset(activations, labels)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        print(f"\n{'='*60}")
        print(f"Training: {len(dataloader)} batches/epoch, {num_epochs} epochs")
        print(f"Loss weights: α={self.alpha} (recon), β={self.beta} (class)")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            total_loss = 0
            total_acc = 0
            n_batches = 0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_acts, batch_labels in pbar:
                batch_acts = batch_acts.to(self.device).float()
                batch_labels = batch_labels.to(self.device).long()

                # Forward pass
                reconstruction, latents, logits = self.joint_model(batch_acts)

                # Compute joint loss
                loss, metrics = self._compute_joint_loss(
                    batch_acts, batch_labels, reconstruction, latents, logits
                )

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.joint_model.parameters(), max_norm=1.0
                )
                self.optimizer.step()

                # Normalize decoder if needed
                if hasattr(self.joint_model.sae, 'normalize_decoder') and \
                   self.joint_model.sae.normalize_decoder:
                    self.joint_model.sae._normalize_decoder_weights()

                total_loss += metrics['loss']
                total_acc += metrics['acc']
                n_batches += 1

                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.3f}",
                    'acc': f"{metrics['acc']:.3f}",
                })

            # Epoch summary
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Acc={avg_acc:.3f}")

            # Validation
            if val_activations is not None and val_labels is not None:
                val_acc = self._validate(val_activations, val_labels)
                print(f"Val Acc: {val_acc:.3f}")

        # Save checkpoint
        if save_path:
            self._save_checkpoint(save_path)
            print(f"\n✓ Saved to {save_path}")

    def _compute_joint_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        reconstruction: torch.Tensor,
        latents: torch.Tensor,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute multi-task loss."""

        # SAE reconstruction loss
        recon_loss, _ = self.joint_model.sae.loss(x, reconstruction, latents)

        # Classification loss
        class_loss = F.cross_entropy(logits, labels)

        # Combined loss
        total_loss = self.alpha * recon_loss + self.beta * class_loss

        # Accuracy
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == labels).float().mean()

        return total_loss, {'loss': total_loss.item(), 'acc': accuracy.item()}

    def _validate(self, activations: torch.Tensor, labels: torch.Tensor) -> float:
        """Quick validation accuracy."""
        self.joint_model.eval()

        dataset = TensorDataset(activations, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for batch_acts, batch_labels in dataloader:
                batch_acts = batch_acts.to(self.device).float()
                batch_labels = batch_labels.to(self.device).long()

                _, _, logits = self.joint_model(batch_acts)
                preds = torch.argmax(logits, dim=-1)

                correct += (preds == batch_labels).sum().item()
                total += batch_labels.size(0)

        self.joint_model.train()
        return correct / total

    def evaluate_detailed(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """
        Detailed evaluation with precision, recall, F1, and AUC.

        Returns:
            Dictionary with accuracy, precision, recall, f1, auc_roc
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        self.joint_model.eval()

        dataset = TensorDataset(activations, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_acts, batch_labels in dataloader:
                batch_acts = batch_acts.to(self.device).float()

                _, _, logits = self.joint_model(batch_acts)
                probs = F.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.append(preds.cpu())
                all_labels.append(batch_labels)
                all_probs.append(probs[:, 1].cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc_roc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0,
        }

        self.joint_model.train()
        return metrics

    def _save_checkpoint(self, save_path: str):
        """Save model checkpoint."""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.joint_model.state_dict(),
            'alpha': self.alpha,
            'beta': self.beta,
        }, save_path)
