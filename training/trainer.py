"""
Training loop with:
  - Adam + CosineAnnealingLR (identical hyperparams for all models)
  - Mixup augmentation (alpha=0.2)
  - Label smoothing cross-entropy
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Mixup
# ---------------------------------------------------------------------------

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Standard mixup on a batch.
    Returns (x_mixed, y_a, y_b, lam).
    For multi-input batches call once per modality with the same lam.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    x_mixed = lam * x + (1 - lam) * x[idx]
    y_b = y[idx]
    return x_mixed, y, y_b, lam


def mixup_criterion(
    criterion,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ---------------------------------------------------------------------------
# Label smoothing cross-entropy
# ---------------------------------------------------------------------------

class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        # One-hot targets with smoothing
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (self.n_classes - 1))
            smooth.scatter_(1, targets.unsqueeze(1), self.confidence)
        return -(smooth * log_probs).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Forward dispatcher
# ---------------------------------------------------------------------------

def _forward(model: nn.Module, batch, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Handle unimodal (x, y) and bimodal (x1, x2, y) batches."""
    if len(batch) == 2:
        x, y = batch
        return model(x.to(device)), y.to(device)
    x1, x2, y = batch
    return model(x1.to(device), x2.to(device)), y.to(device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str = "cuda",
        grad_clip: float = 1.0,
        n_classes: int = 4,
        label_smoothing: float = 0.1,
        mixup_alpha: float = 0.2,
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.grad_clip = grad_clip
        self.mixup_alpha = mixup_alpha

        if label_smoothing > 0:
            self.criterion = LabelSmoothingCE(n_classes, label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # For eval always use plain CE
        self._eval_criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                x_m, y_a, y_b, lam = mixup_batch(x, y, self.mixup_alpha)
                logits = self.model(x_m)
            else:
                x1, x2, y = batch
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                x1_m, y_a, y_b, lam = mixup_batch(x1, y, self.mixup_alpha)
                # Apply same permutation to x2
                idx = torch.randperm(x1.size(0), device=self.device)
                x2_m = lam * x2 + (1 - lam) * x2[idx]
                logits = self.model(x1_m, x2_m)

            loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (logits.argmax(1) == y_a).sum().item()  # approx
            total_n += bs

        self.scheduler.step()
        return {"loss": total_loss / total_n, "acc": total_correct / total_n}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss, total_correct, total_n = 0.0, 0, 0
        all_preds, all_labels = [], []

        for batch in loader:
            logits, y = _forward(self.model, batch, self.device)
            loss = self._eval_criterion(logits, y)
            preds = logits.argmax(1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == y).sum().item()
            total_n += bs
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        preds_arr  = np.array(all_preds)
        labels_arr = np.array(all_labels)

        from sklearn.metrics import f1_score
        f1 = f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)

        return {
            "loss":   total_loss / total_n,
            "acc":    total_correct / total_n,
            "f1":     f1,
            "preds":  preds_arr,
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
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(train_loader)
            va = self.evaluate(val_loader)

            history["train_loss"].append(tr["loss"])
            history["train_acc"].append(tr["acc"])
            history["val_loss"].append(va["loss"])
            history["val_acc"].append(va["acc"])
            history["val_f1"].append(va["f1"])

            if epoch % log_interval == 0 or epoch == epochs:
                lr = self.scheduler.get_last_lr()[0]
                print(
                    f"  ep {epoch:3d}/{epochs} | "
                    f"tr_acc={tr['acc']:.4f}  val_acc={va['acc']:.4f}  "
                    f"val_f1={va['f1']:.4f}  lr={lr:.6f} | {time.time()-t0:.1f}s"
                )

            if va["acc"] > best_val_acc:
                best_val_acc = va["acc"]
                if checkpoint_path:
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save(self.model.state_dict(), checkpoint_path)

        return history


# ---------------------------------------------------------------------------
# Optimizer / scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 150,
    eta_min: float = 1e-6,
):
    opt   = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t_max, eta_min=eta_min)
    return opt, sched
