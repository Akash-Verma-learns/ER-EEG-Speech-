"""
Main experiment runner.

Usage:
  python main.py --data_dir ./data --output_dir ./outputs --n_classes 4

The script:
  1. Loads preprocessed data from data/processed/
  2. Runs 5-fold CV for M_EEG, M_Speech, M_EarlyFusion, M_LateFusion
  3. Runs McNemar's pairwise tests
  4. Saves results to outputs/<experiment_name>/
"""

import argparse
import json
import os
import random
import numpy as np
import torch

from config import get_config
from models import (
    EEGEncoder, SpeechEncoder,
    EarlyFusionModel, LateFusionModel,
    EEGClassifier, SpeechClassifier,
)
from datasets import (
    EEGDataset, SpeechDataset, EarlyFusionDataset, LateFusionDataset,
    load_all_subjects,
)
from training import run_cv, run_cv_late_fusion
from evaluation import compare_all_pairs, print_comparison_table, summarise_results


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal EEG+Speech Emotion Recognition")
    p.add_argument("--data_dir",    default="./data",    help="Root data directory")
    p.add_argument("--output_dir",  default="./outputs", help="Where to save results")
    p.add_argument("--n_classes",   type=int, default=4, help="Number of emotion classes")
    p.add_argument("--epochs",      type=int, default=150)
    p.add_argument("--batch_size",  type=int, default=128)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--n_folds",     type=int, default=5)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--models",      nargs="+",
                   default=["eeg", "speech", "early", "late"],
                   choices=["eeg", "speech", "early", "late"],
                   help="Which models to train")
    p.add_argument("--pretrain_epochs", type=int, default=80,
                   help="Encoder pre-train epochs for late fusion (phase 1)")
    return p.parse_args()


def load_datasets(cfg, data_dir: str, n_classes: int):
    processed_dir = os.path.join(data_dir, "processed")
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(
            f"Processed data not found at {processed_dir}.\n"
            "Run preprocessing/build_dataset.py first to generate .npy files."
        )

    print(f"Loading data from {processed_dir} ...")
    data = load_all_subjects(processed_dir)

    labels = data["labels"].astype(np.int64)
    print(f"  Total samples: {len(labels)}  Classes: {np.unique(labels)}")

    eeg_ds      = EEGDataset(data["eeg_tensor"], labels)
    speech_ds   = SpeechDataset(data["speech_wav"], labels)
    early_ds    = EarlyFusionDataset(data["early_feats"], labels)
    late_ds     = LateFusionDataset(data["eeg_tensor"], data["speech_wav"], labels)

    return eeg_ds, speech_ds, early_ds, late_ds, data


def build_eeg_encoder(cfg):
    return EEGEncoder(
        grid_h=cfg.eeg.spatial_grid[0],
        grid_w=cfg.eeg.spatial_grid[1],
        n_bands=cfg.eeg.n_bands,
        n_time=cfg.eeg.n_time_windows,
        conv_channels=cfg.model.eeg_conv_channels,
        lstm_hidden=cfg.model.eeg_lstm_hidden,
        lstm_chunk_size=cfg.model.on_lstm_chunk_size,
    )


def build_speech_encoder(cfg):
    return SpeechEncoder(
        wav2vec_model_name=cfg.speech.wav2vec_model_name,
        wav2vec_out_dim=cfg.speech.wav2vec_output_dim,
        ds_cnn_channels=cfg.model.speech_conv_channels,
        ds_cnn_out_dim=cfg.model.speech_output_dim,
        transformer_heads=cfg.model.speech_transformer_heads,
        transformer_layers=cfg.model.speech_transformer_layers,
        transformer_dim=cfg.model.speech_transformer_dim,
        transformer_ffn_dim=cfg.model.speech_transformer_ffn_dim,
    )


def main():
    args = parse_args()
    cfg  = get_config()

    # Override config from CLI
    cfg.model.n_classes  = args.n_classes
    cfg.training.epochs  = args.epochs
    cfg.training.batch_size = args.batch_size
    cfg.training.lr      = args.lr
    cfg.training.n_folds = args.n_folds
    cfg.training.seed    = args.seed
    cfg.training.device  = args.device
    cfg.training.pretrain_epochs = args.pretrain_epochs

    set_seed(cfg.training.seed)
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    # Output directories
    exp_dir = os.path.join(args.output_dir, cfg.experiment_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Data
    eeg_ds, speech_ds, early_ds, late_ds, raw_data = load_datasets(
        cfg, args.data_dir, args.n_classes
    )

    # Figure out early fusion input dim from the actual data
    early_input_dim = raw_data["early_feats"].shape[1]
    cfg.model.early_input_dim = early_input_dim
    print(f"Early fusion input dim: {early_input_dim}")

    results = {}

    # ----- M_EEG -----
    if "eeg" in args.models:
        print("\n" + "=" * 50)
        print("Training M_EEG (unimodal EEG baseline)")
        print("=" * 50)

        def eeg_factory():
            enc = build_eeg_encoder(cfg)
            return EEGClassifier(enc, cfg.model.eeg_output_dim, cfg.model.n_classes)

        results["M_EEG"] = run_cv(
            eeg_ds, eeg_factory,
            n_folds=cfg.training.n_folds,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            t_max=cfg.training.t_max,
            device=device,
            num_workers=cfg.training.num_workers,
            seed=cfg.training.seed,
            log_interval=cfg.log_interval,
            checkpoint_dir=ckpt_dir,
            model_name="M_EEG",
        )

    # ----- M_Speech -----
    if "speech" in args.models:
        print("\n" + "=" * 50)
        print("Training M_Speech (unimodal Speech baseline)")
        print("=" * 50)

        def speech_factory():
            enc = build_speech_encoder(cfg)
            return SpeechClassifier(enc, cfg.model.speech_output_dim, cfg.model.n_classes)

        results["M_Speech"] = run_cv(
            speech_ds, speech_factory,
            n_folds=cfg.training.n_folds,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            t_max=cfg.training.t_max,
            device=device,
            num_workers=cfg.training.num_workers,
            seed=cfg.training.seed,
            log_interval=cfg.log_interval,
            checkpoint_dir=ckpt_dir,
            model_name="M_Speech",
        )

    # ----- M_EarlyFusion -----
    if "early" in args.models:
        print("\n" + "=" * 50)
        print("Training M_EarlyFusion")
        print("=" * 50)

        def early_factory():
            return EarlyFusionModel(
                input_dim=cfg.model.early_input_dim,
                conv_channels=cfg.model.early_conv_channels,
                output_dim=cfg.model.early_output_dim,
                n_classes=cfg.model.n_classes,
            )

        results["M_EarlyFusion"] = run_cv(
            early_ds, early_factory,
            n_folds=cfg.training.n_folds,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            t_max=cfg.training.t_max,
            device=device,
            num_workers=cfg.training.num_workers,
            seed=cfg.training.seed,
            log_interval=cfg.log_interval,
            checkpoint_dir=ckpt_dir,
            model_name="M_EarlyFusion",
        )

    # ----- M_LateFusion -----
    if "late" in args.models:
        print("\n" + "=" * 50)
        print("Training M_LateFusion (two-phase: pre-train → freeze → fuse)")
        print("=" * 50)

        def late_factory():
            eeg_enc = build_eeg_encoder(cfg)
            sp_enc  = build_speech_encoder(cfg)
            return LateFusionModel(
                eeg_encoder=eeg_enc,
                speech_encoder=sp_enc,
                eeg_dim=cfg.model.eeg_output_dim,
                speech_dim=cfg.model.speech_output_dim,
                attention_dk=cfg.model.late_attention_dk,
                output_dim=cfg.model.late_output_dim,
                n_classes=cfg.model.n_classes,
            )

        def eeg_head_factory():
            enc = build_eeg_encoder(cfg)
            return EEGClassifier(enc, cfg.model.eeg_output_dim, cfg.model.n_classes)

        def speech_head_factory():
            enc = build_speech_encoder(cfg)
            return SpeechClassifier(enc, cfg.model.speech_output_dim, cfg.model.n_classes)

        results["M_LateFusion"] = run_cv_late_fusion(
            late_fusion_dataset=late_ds,
            eeg_dataset=eeg_ds,
            speech_dataset=speech_ds,
            model_factory=late_factory,
            eeg_head_factory=eeg_head_factory,
            speech_head_factory=speech_head_factory,
            n_folds=cfg.training.n_folds,
            pretrain_epochs=cfg.training.pretrain_epochs,
            fusion_epochs=cfg.training.epochs - cfg.training.pretrain_epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            device=device,
            num_workers=cfg.training.num_workers,
            seed=cfg.training.seed,
            log_interval=cfg.log_interval,
            checkpoint_dir=ckpt_dir,
        )

    # ----- Summary -----
    print("\n\nFINAL RESULTS")
    summarise_results(results)

    if len(results) >= 2:
        comparisons = compare_all_pairs(results)
        print_comparison_table(comparisons)

    # ----- Save -----
    save_path = os.path.join(exp_dir, "results.json")
    serialisable = {}
    for k, v in results.items():
        serialisable[k] = {
            kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
            for kk, vv in v.items()
            if kk not in ("histories",)      # skip raw history lists to keep file small
        }
    if len(results) >= 2:
        for k, v in comparisons.items():
            serialisable.setdefault("comparisons", {})[k] = v

    with open(save_path, "w") as f:
        json.dump(serialisable, f, indent=2)

    print(f"\nResults saved to {save_path}")


if __name__ == "__main__":
    main()
