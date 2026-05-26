"""
Dataset classes and builders for SEED-IV (EEG) + RAVDESS (Speech).

Pairing strategy for fusion:
  SEED-IV trials and RAVDESS recordings are independent datasets.
  We match by emotion label: for each SEED-IV trial labelled class c,
  randomly sample a RAVDESS recording with label c.  This gives a
  cross-dataset paired corpus for training the fusion models.
  The unimodal models are evaluated on their own datasets independently.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# PyTorch Dataset classes
# ---------------------------------------------------------------------------

class EEGDataset(Dataset):
    """4-D EEG tensor dataset for M_EEG.

    X shape: (N, H, W, n_bands, n_time)  — pre-built spatial-freq-temporal tensors
    """
    def __init__(self, tensors: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(tensors).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class SpeechDataset(Dataset):
    """Waveform dataset for M_Speech.

    X shape: (N, n_samples)  — 16 kHz waveforms, zero-padded to max length
    """
    def __init__(self, waveforms: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(waveforms).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class EarlyFusionDataset(Dataset):
    """GA-selected EEG + Speech feature concatenation for M_EarlyFusion.

    X shape: (N, n_eeg_ga + n_speech_ga)
    """
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.X = torch.from_numpy(feats).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class LateFusionDataset(Dataset):
    """EEG tensor + waveform for M_LateFusion."""
    def __init__(self, eeg: np.ndarray, wav: np.ndarray, labels: np.ndarray):
        self.eeg = torch.from_numpy(eeg).float()
        self.wav = torch.from_numpy(wav).float()
        self.y   = torch.from_numpy(labels).long()

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.eeg[i], self.wav[i], self.y[i]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_dataloaders(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    from torch.utils.data import Subset
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True, generator=g)
    val_dl   = DataLoader(Subset(dataset, val_idx), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_dl, val_dl


# ---------------------------------------------------------------------------
# SEED-IV dataset builder
# ---------------------------------------------------------------------------

class SEEDIVBuilder:
    """
    Build per-subject-session arrays from loaded SEED-IV records.

    Usage
    -----
    builder = SEEDIVBuilder(eeg_ga_selector, speech_ga_selector)
    builder.fit(all_records)         # run GA on training split
    eeg_ds, speech_ds, ... = builder.build(all_records, ravdess_wavs, ravdess_labels)
    """

    def __init__(
        self,
        eeg_ga_selector=None,
        speech_ga_selector=None,
        normalise_per_subject: bool = True,
        n_time: int = 2,
        grid_shape: Tuple[int, int] = (9, 9),
        rng_seed: int = 42,
    ):
        from preprocessing.eeg_preprocessor import de_windows_to_tensor, zscore_normalize
        self._de_to_tensor = de_windows_to_tensor
        self._zscore = zscore_normalize
        self.eeg_ga = eeg_ga_selector
        self.speech_ga = speech_ga_selector
        self.normalise = normalise_per_subject
        self.n_time = n_time
        self.grid_shape = grid_shape
        self.rng = np.random.RandomState(rng_seed)

        # Fitted statistics (per subject)
        self._eeg_stats: Dict[str, Tuple] = {}   # subject → (mean, std)

    # ------------------------------------------------------------------

    def _flatten_records(
        self,
        records: List[Dict],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Expand per-trial, per-window data into flat arrays.

        Returns
        -------
        de_flat   : (N, 310)   — one row per EEG time window
        tensors   : (N, H, W, 5, n_time)
        labels    : (N,)
        subj_sess : (N, 2)     — [subject_idx, session] for each sample
        """
        de_list, label_list, ss_list = [], [], []

        subj_set = sorted({r["subject"] for r in records})
        subj_map = {s: i for i, s in enumerate(subj_set)}

        for rec in records:
            subj_idx = subj_map[rec["subject"]]
            sess     = rec["session"]
            for trial_de, lbl in zip(rec["eeg"], rec["labels"]):
                # trial_de: (T_i, 310)
                for w in range(trial_de.shape[0]):
                    de_list.append(trial_de[w])
                    label_list.append(lbl)
                    ss_list.append([subj_idx, sess])

        de_flat    = np.stack(de_list).astype(np.float32)    # (N, 310)
        labels     = np.array(label_list, dtype=np.int64)
        subj_sess  = np.array(ss_list,    dtype=np.int32)

        # Build 4D tensors from windowed de_flat
        # We reconstruct per-trial sequences to preserve temporal order
        tensors_list = []
        offset = 0
        for rec in records:
            for trial_de in rec["eeg"]:
                T = trial_de.shape[0]
                t = self._de_to_tensor(trial_de, grid_shape=self.grid_shape, n_time=self.n_time)
                # t shape: (n_samples, H, W, 5, n_time), n_samples = max(0, T-1)
                # Pad back so every window has a tensor
                if t.shape[0] < T:
                    pad = np.repeat(t[[-1]], T - t.shape[0], axis=0)
                    t = np.concatenate([t, pad], axis=0)
                tensors_list.append(t)
                offset += T

        tensors = np.concatenate(tensors_list, axis=0).astype(np.float32)  # (N, H, W, 5, 2)
        assert tensors.shape[0] == de_flat.shape[0], \
            f"Tensor count {tensors.shape[0]} != de_flat count {de_flat.shape[0]}"

        return de_flat, tensors, labels, subj_sess

    def fit_normalize(self, de_flat: np.ndarray, subj_sess: np.ndarray):
        """Compute per-subject z-score stats on the full dataset."""
        self._eeg_stats = {}
        for subj_idx in np.unique(subj_sess[:, 0]):
            mask = subj_sess[:, 0] == subj_idx
            feat = de_flat[mask]
            mean = feat.mean(axis=0)
            std  = feat.std(axis=0) + 1e-8
            self._eeg_stats[int(subj_idx)] = (mean, std)

    def apply_normalize(self, de_flat: np.ndarray, subj_sess: np.ndarray) -> np.ndarray:
        out = de_flat.copy()
        for subj_idx, (mean, std) in self._eeg_stats.items():
            mask = subj_sess[:, 0] == subj_idx
            out[mask] = (de_flat[mask] - mean) / std
        return out

    def build_eeg_arrays(
        self, records: List[Dict]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (de_flat, tensors, labels, subj_sess) after normalisation.
        """
        de_flat, tensors, labels, subj_sess = self._flatten_records(records)

        if self.normalise:
            if not self._eeg_stats:
                self.fit_normalize(de_flat, subj_sess)
            de_flat_norm = self.apply_normalize(de_flat, subj_sess)
        else:
            de_flat_norm = de_flat

        # Apply GA if fitted
        if self.eeg_ga is not None and hasattr(self.eeg_ga, "selected_indices_") \
                and self.eeg_ga.selected_indices_ is not None:
            de_flat_norm = self.eeg_ga.transform(de_flat_norm)

        return de_flat_norm, tensors, labels, subj_sess

    def build_speech_arrays(
        self,
        waveforms: List[np.ndarray],
        labels: np.ndarray,
        speech_preprocessor=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build speech GA features and padded waveforms from RAVDESS data.

        Returns (ga_feats, padded_wavs, labels)
        """
        from preprocessing.ravdess_loader import pad_collate

        # Acoustic GA features
        if speech_preprocessor is not None:
            feat_list = [speech_preprocessor.extract_features(w)[0] for w in waveforms]
            feats = np.stack(feat_list)   # (N, 180)
        else:
            feats = np.zeros((len(waveforms), 180), dtype=np.float32)

        # Z-score speech features
        mean = feats.mean(axis=0)
        std  = feats.std(axis=0) + 1e-8
        feats_norm = (feats - mean) / std

        if self.speech_ga is not None and hasattr(self.speech_ga, "selected_indices_") \
                and self.speech_ga.selected_indices_ is not None:
            feats_norm = self.speech_ga.transform(feats_norm)

        padded_wavs = pad_collate(waveforms)   # (N, max_len)

        return feats_norm, padded_wavs, labels

    def pair_eeg_speech(
        self,
        eeg_labels: np.ndarray,
        speech_wavs: np.ndarray,
        speech_labels: np.ndarray,
        speech_ga_feats: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Label-matched cross-dataset pairing.
        For each EEG sample with label c, randomly pick a RAVDESS recording with label c.

        Returns (paired_speech_wavs, paired_speech_ga_feats, common_labels)
        """
        # Build per-class index pools from RAVDESS
        class_pools: Dict[int, np.ndarray] = {}
        for c in np.unique(speech_labels):
            class_pools[int(c)] = np.where(speech_labels == c)[0]

        n_eeg = len(eeg_labels)
        paired_wav_idx = np.zeros(n_eeg, dtype=np.int64)

        for i, lbl in enumerate(eeg_labels):
            pool = class_pools.get(int(lbl))
            if pool is not None and len(pool) > 0:
                paired_wav_idx[i] = self.rng.choice(pool)
            else:
                paired_wav_idx[i] = 0  # fallback

        return (
            speech_wavs[paired_wav_idx],
            speech_ga_feats[paired_wav_idx],
            eeg_labels.copy(),
        )


# ---------------------------------------------------------------------------
# Leave-one-session-out index generator
# ---------------------------------------------------------------------------

def leave_one_session_out_splits(
    subj_sess: np.ndarray,
    session_ids: Optional[List[int]] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, val_idx) pairs for leave-one-session-out CV.

    Parameters
    ----------
    subj_sess : (N, 2) array with columns [subject_idx, session]
    session_ids : sessions to use as validation; default = all unique sessions

    Returns list of (train_idx, val_idx) tuples — one per session.
    """
    sessions = session_ids or sorted(np.unique(subj_sess[:, 1]).tolist())
    splits = []
    for sess in sessions:
        val_mask   = subj_sess[:, 1] == sess
        train_mask = ~val_mask
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


def subject_stratified_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Standard stratified k-fold splits (for RAVDESS speech-only evaluation)."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.arange(len(labels)), labels)]
