"""
5-fold stratified cross-validation for all five model configurations.

For late fusion, a two-phase training is used:
  Phase 1 (pretrain_epochs): train EEG encoder and Speech encoder independently
  Phase 2 (epochs - pretrain_epochs): freeze encoders, train only fusion layers
"""

import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from typing import Dict, Optional, Callable, List

from .trainer import Trainer, build_optimizer_and_scheduler
from datasets.data_loader import make_dataloaders


def run_cv(
    dataset: Dataset,
    model_factory: Callable[[], nn.Module],
    n_folds: int = 5,
    epochs: int = 150,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    t_max: int = 150,
    eta_min: float = 1e-6,
    grad_clip: float = 1.0,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
    model_name: str = "model",
) -> Dict:
    """
    Generic 5-fold CV runner for unimodal or early-fusion models.

    Parameters
    ----------
    dataset       : EEGDataset, SpeechDataset, or EarlyFusionDataset
    model_factory : callable with no args that returns a fresh model instance

    Returns
    -------
    results dict with per-fold metrics and aggregate mean ± std
    """
    # Get labels for stratification
    if hasattr(dataset, "y"):
        labels = dataset.y.numpy()
    else:
        labels = np.array([dataset[i][1].item() for i in range(len(dataset))])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))

    fold_accs, fold_f1s = [], []
    all_preds  = np.empty(len(dataset), dtype=np.int64)
    all_labels = np.empty(len(dataset), dtype=np.int64)
    histories  = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n--- {model_name} | Fold {fold+1}/{n_folds} ---")

        train_loader, val_loader = make_dataloaders(
            dataset, train_idx, val_idx, batch_size, num_workers, seed + fold
        )

        model = model_factory()
        opt, sched = build_optimizer_and_scheduler(
            model, lr, weight_decay, t_max, eta_min
        )

        ckpt_path = None
        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_fold{fold+1}_best.pt")

        trainer = Trainer(model, opt, sched, device=device, grad_clip=grad_clip)
        history = trainer.fit(
            train_loader, val_loader,
            epochs=epochs,
            log_interval=log_interval,
            checkpoint_path=ckpt_path,
        )
        histories.append(history)

        # Load best checkpoint for final eval
        if ckpt_path and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

        val_metrics = trainer.evaluate(val_loader)
        fold_accs.append(val_metrics["acc"])
        fold_f1s.append(val_metrics["f1"])
        all_preds[val_idx] = val_metrics["preds"]
        all_labels[val_idx] = val_metrics["labels"]

        print(
            f"  Fold {fold+1} result: acc={val_metrics['acc']:.4f}  f1={val_metrics['f1']:.4f}"
        )

    results = {
        "model_name": model_name,
        "fold_accs": fold_accs,
        "fold_f1s": fold_f1s,
        "mean_acc": float(np.mean(fold_accs)),
        "std_acc":  float(np.std(fold_accs)),
        "mean_f1":  float(np.mean(fold_f1s)),
        "std_f1":   float(np.std(fold_f1s)),
        "all_preds":  all_preds,
        "all_labels": all_labels,
        "histories": histories,
    }

    print(
        f"\n{model_name} | CV result: "
        f"acc={results['mean_acc']:.4f} ± {results['std_acc']:.4f}  "
        f"f1={results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
    )
    return results


def run_cv_late_fusion(
    late_fusion_dataset,         # LateFusionDataset
    eeg_dataset,                 # EEGDataset  (for encoder pre-training)
    speech_dataset,              # SpeechDataset (for encoder pre-training)
    model_factory: Callable,     # () -> LateFusionModel
    eeg_head_factory: Callable,  # () -> EEGClassifier
    speech_head_factory: Callable,
    n_folds: int = 5,
    pretrain_epochs: int = 80,
    fusion_epochs: int = 70,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str = "cuda",
    num_workers: int = 4,
    seed: int = 42,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
) -> Dict:
    """
    Two-phase 5-fold CV for late fusion.

    Phase 1: independently pre-train M_EEG and M_Speech
    Phase 2: freeze encoders, train only the fusion layers
    """
    labels = late_fusion_dataset.y.numpy()
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = np.arange(len(late_fusion_dataset))

    fold_accs, fold_f1s = [], []
    all_preds  = np.empty(len(late_fusion_dataset), dtype=np.int64)
    all_labels = np.empty(len(late_fusion_dataset), dtype=np.int64)
    gate_history = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n--- LateFusion | Fold {fold+1}/{n_folds} ---")

        # --- Phase 1: pre-train EEG encoder ---
        print("  Phase 1a: pre-training EEG encoder")
        eeg_model = eeg_head_factory()
        eeg_train_loader, eeg_val_loader = make_dataloaders(
            eeg_dataset, train_idx, val_idx, batch_size, num_workers, seed + fold
        )
        opt_e, sch_e = build_optimizer_and_scheduler(
            eeg_model, lr, weight_decay, pretrain_epochs, 1e-6
        )
        Trainer(eeg_model, opt_e, sch_e, device=device, grad_clip=grad_clip).fit(
            eeg_train_loader, eeg_val_loader, pretrain_epochs, log_interval,
            checkpoint_path=(
                os.path.join(checkpoint_dir, f"eeg_encoder_fold{fold+1}.pt")
                if checkpoint_dir else None
            ),
        )

        # --- Phase 1: pre-train Speech encoder ---
        print("  Phase 1b: pre-training Speech encoder")
        sp_model = speech_head_factory()
        sp_train_loader, sp_val_loader = make_dataloaders(
            speech_dataset, train_idx, val_idx, batch_size, num_workers, seed + fold
        )
        opt_s, sch_s = build_optimizer_and_scheduler(
            sp_model, lr, weight_decay, pretrain_epochs, 1e-6
        )
        Trainer(sp_model, opt_s, sch_s, device=device, grad_clip=grad_clip).fit(
            sp_train_loader, sp_val_loader, pretrain_epochs, log_interval,
            checkpoint_path=(
                os.path.join(checkpoint_dir, f"sp_encoder_fold{fold+1}.pt")
                if checkpoint_dir else None
            ),
        )

        # --- Phase 2: assemble late fusion model, freeze encoders ---
        print("  Phase 2: training fusion layers (encoders frozen)")
        late_model = model_factory()

        # Copy pre-trained encoder weights
        late_model.eeg_encoder.load_state_dict(eeg_model.encoder.state_dict())
        late_model.speech_encoder.load_state_dict(sp_model.encoder.state_dict())
        late_model.freeze_encoders()

        fusion_train_loader, fusion_val_loader = make_dataloaders(
            late_fusion_dataset, train_idx, val_idx, batch_size, num_workers, seed + fold
        )
        opt_f, sch_f = build_optimizer_and_scheduler(
            late_model, lr, weight_decay, fusion_epochs, 1e-6
        )
        ckpt_path = (
            os.path.join(checkpoint_dir, f"late_fusion_fold{fold+1}_best.pt")
            if checkpoint_dir else None
        )
        Trainer(late_model, opt_f, sch_f, device=device, grad_clip=grad_clip).fit(
            fusion_train_loader, fusion_val_loader, fusion_epochs, log_interval,
            checkpoint_path=ckpt_path,
        )

        if ckpt_path and os.path.exists(ckpt_path):
            late_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Re-use the existing trainer (already has the loaded model) just for eval
        _opt, _sch = build_optimizer_and_scheduler(late_model, lr=0.0, weight_decay=0.0,
                                                   t_max=1, eta_min=0.0)
        val_metrics = Trainer(late_model, _opt, _sch, device=device).evaluate(fusion_val_loader)

        fold_accs.append(val_metrics["acc"])
        fold_f1s.append(val_metrics["f1"])
        all_preds[val_idx] = val_metrics["preds"]
        all_labels[val_idx] = val_metrics["labels"]

        alpha_e, alpha_s = late_model.get_gate_values()
        gate_history.append({"fold": fold + 1, "alpha_eeg": alpha_e, "alpha_speech": alpha_s})
        print(
            f"  Fold {fold+1}: acc={val_metrics['acc']:.4f}  f1={val_metrics['f1']:.4f} | "
            f"α_EEG={alpha_e:.3f}  α_Speech={alpha_s:.3f}"
        )

    results = {
        "model_name": "LateFusion",
        "fold_accs": fold_accs,
        "fold_f1s":  fold_f1s,
        "mean_acc":  float(np.mean(fold_accs)),
        "std_acc":   float(np.std(fold_accs)),
        "mean_f1":   float(np.mean(fold_f1s)),
        "std_f1":    float(np.std(fold_f1s)),
        "all_preds":  all_preds,
        "all_labels": all_labels,
        "gate_history": gate_history,
    }

    print(
        f"\nLateFusion | CV result: "
        f"acc={results['mean_acc']:.4f} ± {results['std_acc']:.4f}  "
        f"f1={results['mean_f1']:.4f} ± {results['std_f1']:.4f}"
    )
    return results
