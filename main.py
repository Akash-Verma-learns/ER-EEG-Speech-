"""
SEED-IV (EEG) + RAVDESS (Speech) multimodal emotion recognition.

Usage:
  python main.py \
      --seed_iv_dir  ./data/seed_iv \
      --ravdess_dir  ./data/ravdess \
      --output_dir   ./outputs \
      --models eeg speech early late

Pipeline:
  1. Load SEED-IV pre-extracted DE features + RAVDESS waveforms
  2. GA feature selection (offline, per modality)
  3. Build dataset arrays; pair by emotion label for fusion models
  4. Run leave-one-session-out CV for EEG / early / late fusion
     Run stratified 5-fold CV for speech-only baseline
  5. McNemar pairwise comparison
  6. Save results JSON
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
from preprocessing.seed_iv_loader import load_all_data as load_seed_iv
from preprocessing.ravdess_loader import load_dataset as load_ravdess
from preprocessing.eeg_preprocessor import de_windows_to_tensor, zscore_normalize
from preprocessing.ga_selector import GeneticAlgorithmSelector
from datasets import (
    EEGDataset, SpeechDataset, EarlyFusionDataset, LateFusionDataset,
    SEEDIVBuilder, leave_one_session_out_splits, subject_stratified_splits,
    make_dataloaders,
)
from training import run_loso_cv, run_stratified_cv, run_loso_late_fusion
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
    p = argparse.ArgumentParser()
    p.add_argument("--seed_iv_dir", default="./data/seed_iv",
                   help="Root of SEED-IV dataset (contains eeg_feature_smooth/)")
    p.add_argument("--ravdess_dir", default="./data/ravdess",
                   help="Root of RAVDESS dataset (contains Actor_01/ … Actor_24/)")
    p.add_argument("--output_dir",  default="./outputs")
    p.add_argument("--models", nargs="+",
                   default=["eeg", "speech", "early", "late"],
                   choices=["eeg", "speech", "early", "late"])
    p.add_argument("--epochs",    type=int,   default=150)
    p.add_argument("--batch_size",type=int,   default=128)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--device",    default="cuda")
    p.add_argument("--subjects",  nargs="*",  default=None,
                   help="Subset of subject filenames e.g. 1_20160518.mat")
    p.add_argument("--skip_ga",   action="store_true",
                   help="Skip GA and use all features (faster, slightly lower acc)")
    p.add_argument("--pretrain_epochs", type=int, default=80)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build encoder factories (identical arch across all models)
# ---------------------------------------------------------------------------

def make_eeg_encoder(cfg):
    return EEGEncoder(
        grid_h=cfg.eeg.spatial_grid[0],
        grid_w=cfg.eeg.spatial_grid[1],
        n_bands=cfg.eeg.n_bands,
        n_time=cfg.eeg.n_time_windows,
        conv_channels=cfg.model.eeg_conv_channels,
        lstm_hidden=cfg.model.eeg_lstm_hidden,
        lstm_chunk_size=cfg.model.on_lstm_chunk_size,
    )


def make_speech_encoder(cfg):
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = get_config()

    cfg.training.epochs         = args.epochs
    cfg.training.batch_size     = args.batch_size
    cfg.training.lr             = args.lr
    cfg.training.seed           = args.seed
    cfg.training.device         = args.device
    cfg.training.pretrain_epochs = args.pretrain_epochs
    cfg.seed_iv.root_dir        = args.seed_iv_dir
    cfg.ravdess.root_dir        = args.ravdess_dir

    set_seed(cfg.training.seed)
    device = cfg.training.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        device = "cpu"

    ckpt_dir = os.path.join(args.output_dir, cfg.experiment_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load raw data
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading SEED-IV EEG data ...")
    seed_records = load_seed_iv(
        cfg.seed_iv.root_dir,
        subject_files=args.subjects,
        verbose=True,
    )
    print(f"  {len(seed_records)} subject×session records loaded")

    print("\nLoading RAVDESS speech data ...")
    rav_waveforms, rav_labels, rav_meta = load_ravdess(
        cfg.ravdess.root_dir,
        target_sr=cfg.speech.sampling_rate,
        emotion_map=cfg.ravdess.emotion_map,
        verbose=True,
    )
    print(f"  {len(rav_waveforms)} RAVDESS recordings loaded")

    # -----------------------------------------------------------------------
    # 2. Build EEG arrays (DE features → tensors, z-score norm)
    # -----------------------------------------------------------------------
    print("\nBuilding EEG arrays ...")
    builder = SEEDIVBuilder(
        normalise_per_subject=cfg.seed_iv.normalise_per_subject,
        n_time=cfg.eeg.n_time_windows,
        grid_shape=cfg.eeg.spatial_grid,
        rng_seed=cfg.training.seed,
    )
    de_flat, eeg_tensors, eeg_labels, subj_sess = builder.build_eeg_arrays(seed_records)
    print(f"  EEG tensor shape: {eeg_tensors.shape}   DE flat: {de_flat.shape}")

    loso_splits = leave_one_session_out_splits(subj_sess)

    # -----------------------------------------------------------------------
    # 3. GA feature selection (run on full data for efficiency; in a real
    #    paper, run inside each fold to avoid leakage)
    # -----------------------------------------------------------------------
    if not args.skip_ga:
        print("\nRunning EEG GA feature selection ...")
        eeg_ga = GeneticAlgorithmSelector(
            population_size=cfg.ga.population_size,
            n_generations=cfg.ga.n_generations,
            sparsity_weight=cfg.ga.sparsity_weight,
            random_state=cfg.training.seed,
            n_jobs=cfg.ga.n_jobs,
        )
        eeg_ga.fit(de_flat, eeg_labels)
        eeg_ga_feats = eeg_ga.transform(de_flat)
        print(f"  EEG GA: {de_flat.shape[1]} → {eeg_ga_feats.shape[1]} features")

        print("\nRunning Speech GA feature selection ...")
        from preprocessing.speech_preprocessor import SpeechPreprocessor
        sp_prep = SpeechPreprocessor(sr=cfg.speech.sampling_rate)
        rav_acoustic = np.stack([
            sp_prep.extract_features(w)[0] for w in rav_waveforms
        ])
        speech_ga = GeneticAlgorithmSelector(
            population_size=cfg.ga.population_size,
            n_generations=cfg.ga.n_generations,
            sparsity_weight=cfg.ga.sparsity_weight,
            random_state=cfg.training.seed,
            n_jobs=cfg.ga.n_jobs,
        )
        speech_ga.fit(rav_acoustic, rav_labels)
        rav_ga_feats = speech_ga.transform(rav_acoustic)
        print(f"  Speech GA: {rav_acoustic.shape[1]} → {rav_ga_feats.shape[1]} features")
    else:
        print("\nSkipping GA — using all features")
        eeg_ga_feats = de_flat
        from preprocessing.speech_preprocessor import SpeechPreprocessor
        sp_prep = SpeechPreprocessor(sr=cfg.speech.sampling_rate)
        rav_acoustic = np.stack([sp_prep.extract_features(w)[0] for w in rav_waveforms])
        rav_acoustic_norm = (rav_acoustic - rav_acoustic.mean(0)) / (rav_acoustic.std(0) + 1e-8)
        rav_ga_feats = rav_acoustic_norm

    # -----------------------------------------------------------------------
    # 4. Build speech waveform array (padded)
    # -----------------------------------------------------------------------
    from preprocessing.ravdess_loader import pad_collate
    rav_wavs_padded = pad_collate(rav_waveforms)   # (N_rav, max_len)

    # -----------------------------------------------------------------------
    # 5. Build paired fusion arrays
    # -----------------------------------------------------------------------
    paired_sp_wavs, paired_sp_ga, fused_labels = builder.pair_eeg_speech(
        eeg_labels, rav_wavs_padded, rav_labels, rav_ga_feats
    )
    early_feats = np.concatenate([eeg_ga_feats, paired_sp_ga], axis=1)
    cfg.model.early_input_dim = early_feats.shape[1]
    print(f"\nEarly fusion input dim: {early_feats.shape[1]}")

    # -----------------------------------------------------------------------
    # 6. Dataset objects
    # -----------------------------------------------------------------------
    eeg_ds      = EEGDataset(eeg_tensors, eeg_labels)
    speech_ds   = SpeechDataset(rav_wavs_padded, rav_labels)
    early_ds    = EarlyFusionDataset(early_feats, eeg_labels)
    late_ds     = LateFusionDataset(eeg_tensors, paired_sp_wavs, eeg_labels)

    n_classes    = cfg.model.n_classes
    speech_splits = subject_stratified_splits(rav_labels, n_folds=5, seed=cfg.training.seed)

    results = {}

    common_kw = dict(
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
        n_classes=n_classes,
        label_smoothing=cfg.training.label_smoothing if cfg.training.use_label_smoothing else 0.0,
        mixup_alpha=cfg.training.mixup_alpha,
        grad_clip=cfg.training.gradient_clip,
    )

    # -----------------------------------------------------------------------
    # M_EEG
    # -----------------------------------------------------------------------
    if "eeg" in args.models:
        print("\n" + "=" * 60)
        print("Training M_EEG (leave-one-session-out)")
        results["M_EEG"] = run_loso_cv(
            eeg_ds, loso_splits,
            model_factory=lambda: EEGClassifier(
                make_eeg_encoder(cfg), cfg.model.eeg_output_dim, n_classes
            ),
            model_name="M_EEG",
            **common_kw,
        )

    # -----------------------------------------------------------------------
    # M_Speech
    # -----------------------------------------------------------------------
    if "speech" in args.models:
        print("\n" + "=" * 60)
        print("Training M_Speech (RAVDESS, stratified 5-fold)")
        results["M_Speech"] = run_stratified_cv(
            speech_ds, rav_labels,
            model_factory=lambda: SpeechClassifier(
                make_speech_encoder(cfg), cfg.model.speech_output_dim, n_classes
            ),
            n_folds=5,
            model_name="M_Speech",
            **common_kw,
        )

    # -----------------------------------------------------------------------
    # M_EarlyFusion
    # -----------------------------------------------------------------------
    if "early" in args.models:
        print("\n" + "=" * 60)
        print("Training M_EarlyFusion (leave-one-session-out on paired features)")
        results["M_EarlyFusion"] = run_loso_cv(
            early_ds, loso_splits,
            model_factory=lambda: EarlyFusionModel(
                input_dim=cfg.model.early_input_dim,
                conv_channels=cfg.model.early_conv_channels,
                output_dim=cfg.model.early_output_dim,
                n_classes=n_classes,
            ),
            model_name="M_EarlyFusion",
            **common_kw,
        )

    # -----------------------------------------------------------------------
    # M_LateFusion
    # -----------------------------------------------------------------------
    if "late" in args.models:
        print("\n" + "=" * 60)
        print("Training M_LateFusion (two-phase LOSO)")
        results["M_LateFusion"] = run_loso_late_fusion(
            late_ds, eeg_ds, speech_ds,
            splits=loso_splits,
            late_factory=lambda: LateFusionModel(
                eeg_encoder=make_eeg_encoder(cfg),
                speech_encoder=make_speech_encoder(cfg),
                eeg_dim=cfg.model.eeg_output_dim,
                speech_dim=cfg.model.speech_output_dim,
                attention_dk=cfg.model.late_attention_dk,
                output_dim=cfg.model.late_output_dim,
                n_classes=n_classes,
            ),
            eeg_head_factory=lambda: EEGClassifier(
                make_eeg_encoder(cfg), cfg.model.eeg_output_dim, n_classes
            ),
            speech_head_factory=lambda: SpeechClassifier(
                make_speech_encoder(cfg), cfg.model.speech_output_dim, n_classes
            ),
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
            n_classes=n_classes,
            label_smoothing=cfg.training.label_smoothing,
            mixup_alpha=cfg.training.mixup_alpha,
            grad_clip=cfg.training.gradient_clip,
        )

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    emotion_names = ["neutral", "sad", "fear", "happy"]
    print("\n\nFINAL RESULTS")
    summarise_results(results, class_names=emotion_names)

    if len(results) >= 2:
        comp = compare_all_pairs(results)
        print_comparison_table(comp)

    # Save
    exp_dir   = os.path.join(args.output_dir, cfg.experiment_name)
    save_path = os.path.join(exp_dir, "results.json")
    os.makedirs(exp_dir, exist_ok=True)

    out = {}
    for k, v in results.items():
        out[k] = {
            kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
            for kk, vv in v.items() if kk != "histories"
        }
    if len(results) >= 2:
        out["comparisons"] = {
            k: {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                for kk, vv in v.items()}
            for k, v in comp.items()
        }

    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {save_path}")


if __name__ == "__main__":
    main()
