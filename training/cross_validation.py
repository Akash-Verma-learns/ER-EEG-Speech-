"""
Cross-validation runners.

  run_loso_cv          — leave-one-session-out (SEED-IV EEG protocol)
  run_stratified_cv    — stratified k-fold (RAVDESS speech-only)
  run_loso_late_fusion — two-phase late-fusion with LOSO
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple

from .trainer import Trainer, build_optimizer_and_scheduler
from datasets.data_loader import make_dataloaders


# ---------------------------------------------------------------------------
# Generic helper
# ---------------------------------------------------------------------------

def _run_single_fold(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    model_factory: Callable,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    t_max: int,
    device: str,
    num_workers: int,
    seed: int,
    log_interval: int,
    checkpoint_path: Optional[str],
    n_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    grad_clip: float,
) -> Tuple[Dict, Dict]:
    train_dl, val_dl = make_dataloaders(dataset, train_idx, val_idx,
                                        batch_size, num_workers, seed)
    model = model_factory()
    opt, sched = build_optimizer_and_scheduler(model, lr, weight_decay, t_max)
    trainer = Trainer(model, opt, sched, device=device,
                      grad_clip=grad_clip, n_classes=n_classes,
                      label_smoothing=label_smoothing, mixup_alpha=mixup_alpha)
    history = trainer.fit(train_dl, val_dl, epochs, log_interval, checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    val_metrics = trainer.evaluate(val_dl)
    return history, val_metrics


# ---------------------------------------------------------------------------
# SEED-IV leave-one-session-out CV
# ---------------------------------------------------------------------------

def run_loso_cv(
    dataset: Dataset,
    splits: List[Tuple[np.ndarray, np.ndarray]],  # from leave_one_session_out_splits()
    model_factory: Callable,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 150,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "model",
    n_classes: int = 4,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    grad_clip: float = 1.0,
) -> Dict:
    """
    Leave-one-session-out cross-validation (3 folds for SEED-IV).
    """
    n_total = len(dataset)
    all_preds  = np.empty(n_total, dtype=np.int64)
    all_labels = np.empty(n_total, dtype=np.int64)
    fold_accs, fold_f1s, histories = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- {model_name} | Session {fold+1} as val ---")

        ckpt = (
            os.path.join(checkpoint_dir, f"{model_name}_fold{fold+1}_best.pt")
            if checkpoint_dir else None
        )
        history, metrics = _run_single_fold(
            dataset, train_idx, val_idx, model_factory,
            epochs, batch_size, lr, weight_decay, t_max,
            device, num_workers, seed + fold, log_interval, ckpt,
            n_classes, label_smoothing, mixup_alpha, grad_clip,
        )
        histories.append(history)
        fold_accs.append(metrics["acc"])
        fold_f1s.append(metrics["f1"])
        all_preds[val_idx]  = metrics["preds"]
        all_labels[val_idx] = metrics["labels"]

        print(f"  Fold {fold+1}: acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f}")

    results = {
        "model_name": model_name,
        "fold_accs":  fold_accs,
        "fold_f1s":   fold_f1s,
        "mean_acc":   float(np.mean(fold_accs)),
        "std_acc":    float(np.std(fold_accs)),
        "mean_f1":    float(np.mean(fold_f1s)),
        "std_f1":     float(np.std(fold_f1s)),
        "all_preds":  all_preds,
        "all_labels": all_labels,
        "histories":  histories,
    }
    print(
        f"\n{model_name} | LOSO result: "
        f"acc={results['mean_acc']:.4f} ± {results['std_acc']:.4f}  "
        f"f1={results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
    )
    return results


# ---------------------------------------------------------------------------
# Stratified k-fold (for speech / RAVDESS)
# ---------------------------------------------------------------------------

def run_stratified_cv(
    dataset: Dataset,
    labels: np.ndarray,
    model_factory: Callable,
    n_folds: int = 5,
    **kwargs,
) -> Dict:
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=kwargs.get("seed", 42))
    splits = [(tr, va) for tr, va in skf.split(np.arange(len(labels)), labels)]
    return run_loso_cv(dataset, splits, model_factory,
                       model_name=kwargs.pop("model_name", "model"), **kwargs)


# ---------------------------------------------------------------------------
# Two-phase late-fusion LOSO CV
# ---------------------------------------------------------------------------

def run_loso_late_fusion(
    late_ds: Dataset,
    eeg_ds: Dataset,
    speech_ds: Dataset,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    late_factory: Callable,
    eeg_head_factory: Callable,
    speech_head_factory: Callable,
    pretrain_epochs: int = 80,
    fusion_epochs: int = 70,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
    n_classes: int = 4,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    grad_clip: float = 1.0,
) -> Dict:
    """Two-phase LOSO CV for late fusion (pre-train encoders → freeze → train fusion)."""
    n_total = len(late_ds)
    all_preds  = np.empty(n_total, dtype=np.int64)
    all_labels = np.empty(n_total, dtype=np.int64)
    fold_accs, fold_f1s, gate_hist = [], [], []

    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- LateFusion | Session {fold+1} as val ---")

        # Phase 1a: pre-train EEG encoder
        print("  Phase 1a: EEG encoder pre-training")
        eeg_model = eeg_head_factory()
        _, _, eeg_ckpt = _pretrain(
            eeg_ds, train_idx, val_idx, eeg_model,
            pretrain_epochs, batch_size, lr, weight_decay,
            device, num_workers, seed + fold,
            checkpoint_dir, f"eeg_enc_fold{fold+1}",
            n_classes, label_smoothing, mixup_alpha, grad_clip, log_interval,
        )

        # Phase 1b: pre-train Speech encoder
        print("  Phase 1b: Speech encoder pre-training")
        sp_model = speech_head_factory()
        _, _, sp_ckpt = _pretrain(
            speech_ds, train_idx, val_idx, sp_model,
            pretrain_epochs, batch_size, lr, weight_decay,
            device, num_workers, seed + fold,
            checkpoint_dir, f"sp_enc_fold{fold+1}",
            n_classes, label_smoothing, mixup_alpha, grad_clip, log_interval,
        )

        # Phase 2: fuse
        print("  Phase 2: fusion layer training (encoders frozen)")
        late_model = late_factory()
        late_model.eeg_encoder.load_state_dict(eeg_model.encoder.state_dict())
        late_model.speech_encoder.load_state_dict(sp_model.encoder.state_dict())
        late_model.freeze_encoders()

        fusion_ckpt = (
            os.path.join(checkpoint_dir, f"late_fusion_fold{fold+1}_best.pt")
            if checkpoint_dir else None
        )
        _, metrics = _run_single_fold(
            late_ds, train_idx, val_idx, lambda: late_model,
            fusion_epochs, batch_size, lr, weight_decay, fusion_epochs,
            device, num_workers, seed + fold, log_interval, fusion_ckpt,
            n_classes, label_smoothing, mixup_alpha, grad_clip,
        )

        fold_accs.append(metrics["acc"])
        fold_f1s.append(metrics["f1"])
        all_preds[val_idx]  = metrics["preds"]
        all_labels[val_idx] = metrics["labels"]

        alpha_e, alpha_s = late_model.get_gate_values()
        gate_hist.append({"fold": fold+1, "alpha_eeg": alpha_e, "alpha_speech": alpha_s})
        print(
            f"  Fold {fold+1}: acc={metrics['acc']:.4f}  f1={metrics['f1']:.4f} | "
            f"α_EEG={alpha_e:.3f}  α_Speech={alpha_s:.3f}"
        )

    results = {
        "model_name": "LateFusion",
        "fold_accs": fold_accs, "fold_f1s": fold_f1s,
        "mean_acc": float(np.mean(fold_accs)), "std_acc": float(np.std(fold_accs)),
        "mean_f1":  float(np.mean(fold_f1s)),  "std_f1":  float(np.std(fold_f1s)),
        "all_preds": all_preds, "all_labels": all_labels,
        "gate_history": gate_hist,
    }
    print(
        f"\nLateFusion | LOSO result: "
        f"acc={results['mean_acc']:.4f} ± {results['std_acc']:.4f}  "
        f"f1={results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
    )
    return results


def _pretrain(
    dataset, train_idx, val_idx, model,
    epochs, batch_size, lr, weight_decay,
    device, num_workers, seed,
    checkpoint_dir, name,
    n_classes, label_smoothing, mixup_alpha, grad_clip, log_interval,
):
    ckpt = os.path.join(checkpoint_dir, f"{name}_best.pt") if checkpoint_dir else None
    train_dl, val_dl = make_dataloaders(dataset, train_idx, val_idx,
                                         batch_size, num_workers, seed)
    opt, sched = build_optimizer_and_scheduler(model, lr, weight_decay, epochs)
    trainer = Trainer(model, opt, sched, device=device,
                      grad_clip=grad_clip, n_classes=n_classes,
                      label_smoothing=label_smoothing, mixup_alpha=mixup_alpha)
    history = trainer.fit(train_dl, val_dl, epochs, log_interval, ckpt)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
    metrics = trainer.evaluate(val_dl)
    return history, metrics, ckpt
