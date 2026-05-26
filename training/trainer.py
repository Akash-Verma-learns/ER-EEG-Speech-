"""
Training loop with cosine annealing LR schedule.
Handles unimodal models (single input) and multimodal models (two inputs).
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple


def _forward(model: nn.Module, batch, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route a batch to the model based on its length (unimodal vs bimodal)."""
    if len(batch) == 2:
        x, y = batch
        logits = model(x.to(device))
    else:
        x1, x2, y = batch
        logits = model(x1.to(device), x2.to(device))
    return logits, y.to(device)


class Trainer:
    """
    General-purpose trainer for all five model configurations.

    Parameters
    ----------
    model     : any of EEGClassifier, SpeechClassifier, EarlyFusionModel,
                LateFusionModel
    optimizer : pre-built optimizer (Adam)
    scheduler : pre-built LR scheduler (CosineAnnealingLR)
    device    : "cuda" or "cpu"
    grad_clip : gradient norm clipping threshold
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str = "cuda",
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        for batch in loader:
            logits, y = _forward(self.model, batch, self.device)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += bs

        self.scheduler.step()
        n = total_samples
        return {"loss": total_loss / n, "acc": total_correct / n}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        all_preds, all_labels = [], []

        for batch in loader:
            logits, y = _forward(self.model, batch, self.device)
            loss = self.criterion(logits, y)

            preds = logits.argmax(1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == y).sum().item()
            total_samples += bs
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        n = total_samples
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        from sklearn.metrics import f1_score
        f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        return {
            "loss": total_loss / n,
            "acc": total_correct / n,
            "f1": f1,
            "preds": preds_arr,
            "labels": labels_arr,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 150,
        log_interval: int = 10,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """Run full training loop with optional checkpointing."""
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["acc"])
            history["val_f1"].append(val_metrics["f1"])

            if epoch % log_interval == 0 or epoch == epochs:
                elapsed = time.time() - t0
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"train_acc={train_metrics['acc']:.4f} "
                    f"val_acc={val_metrics['acc']:.4f} "
                    f"val_f1={val_metrics['f1']:.4f} "
                    f"lr={lr:.6f} | {elapsed:.1f}s"
                )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_path)

        return history


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 150,
    eta_min: float = 1e-6,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Build Adam + CosineAnnealingLR — identical hyperparameters for all models."""
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return opt, sched
