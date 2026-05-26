"""
Dataset utilities for multimodal EEG + Speech emotion datasets.

Expected on-disk layout:
  data/
    processed/
      subject_<id>/
        eeg_tensor.npy     shape (n_samples, H, W, n_bands, n_time)
        eeg_ga_feats.npy   shape (n_samples, n_eeg_ga_feats)
        speech_wav.npy     shape (n_samples, n_audio_samples)
        speech_ga_feats.npy shape (n_samples, n_speech_ga_feats)
        early_feats.npy    shape (n_samples, n_eeg_ga + n_speech_ga)
        labels.npy         shape (n_samples,)  — integer class labels

Use `DatasetBuilder` to generate the processed files from raw data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

class EEGDataset(Dataset):
    """Unimodal EEG dataset."""

    def __init__(
        self,
        eeg_tensors: np.ndarray,  # (N, H, W, bands, T)
        labels: np.ndarray,
    ):
        self.X = torch.from_numpy(eeg_tensors).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class SpeechDataset(Dataset):
    """Unimodal Speech dataset."""

    def __init__(
        self,
        waveforms: np.ndarray,   # (N, n_samples)
        labels: np.ndarray,
    ):
        self.X = torch.from_numpy(waveforms).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyFusionDataset(Dataset):
    """Concatenated GA features for early fusion."""

    def __init__(
        self,
        early_feats: np.ndarray,  # (N, n_eeg_ga + n_speech_ga)
        labels: np.ndarray,
    ):
        self.X = torch.from_numpy(early_feats).float()
        self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LateFusionDataset(Dataset):
    """EEG tensor + waveform for late fusion."""

    def __init__(
        self,
        eeg_tensors: np.ndarray,
        waveforms: np.ndarray,
        labels: np.ndarray,
    ):
        self.eeg = torch.from_numpy(eeg_tensors).float()
        self.wav = torch.from_numpy(waveforms).float()
        self.y   = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.eeg[idx], self.wav[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_subject_data(subject_dir: str) -> Dict[str, np.ndarray]:
    """Load all arrays for a single subject."""
    data = {}
    for fname in [
        "eeg_tensor", "eeg_ga_feats", "speech_wav",
        "speech_ga_feats", "early_feats", "labels",
    ]:
        path = os.path.join(subject_dir, f"{fname}.npy")
        if os.path.exists(path):
            data[fname] = np.load(path, allow_pickle=False)
    return data


def load_all_subjects(
    processed_dir: str,
    subject_ids: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load and concatenate data across subjects.

    Returns a dict with keys matching the .npy file stems.
    """
    if subject_ids is None:
        subject_ids = sorted([
            d for d in os.listdir(processed_dir)
            if os.path.isdir(os.path.join(processed_dir, d))
        ])

    all_data: Dict[str, list] = {}
    for sid in subject_ids:
        sub_dir = os.path.join(processed_dir, sid)
        sub_data = load_subject_data(sub_dir)
        for key, arr in sub_data.items():
            all_data.setdefault(key, []).append(arr)

    return {k: np.concatenate(v, axis=0) for k, v in all_data.items()}


def make_dataloaders(
    dataset: Dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val DataLoaders from index arrays (for cross-validation)."""
    from torch.utils.data import Subset

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Dataset builder (raw → processed)
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Converts raw EEG + speech recordings into the .npy arrays expected by the
    dataset classes above.  Call build_subject() for each participant.
    """

    def __init__(
        self,
        eeg_preprocessor,
        speech_preprocessor,
        eeg_ga_selector=None,
        speech_ga_selector=None,
        output_dir: str = "./data/processed",
    ):
        self.eeg_prep = eeg_preprocessor
        self.speech_prep = speech_preprocessor
        self.eeg_ga = eeg_ga_selector
        self.speech_ga = speech_ga_selector
        self.output_dir = output_dir

    def build_subject(
        self,
        subject_id: str,
        raw_eeg_list: List[np.ndarray],   # list of (n_channels, n_samples) arrays, one per trial
        raw_speech_list: List[np.ndarray],# list of (n_samples,) arrays, one per trial
        labels: np.ndarray,               # integer label per trial
    ):
        """Process one subject and save .npy files."""
        sub_dir = os.path.join(self.output_dir, f"subject_{subject_id}")
        os.makedirs(sub_dir, exist_ok=True)

        # EEG
        de_list, tensor_list = [], []
        for raw_eeg in raw_eeg_list:
            de = self.eeg_prep.preprocess(raw_eeg)          # (n_windows, 160)
            de_list.append(de)
            tensor = self.eeg_prep.to_tensor(de)            # (n_samples, H, W, 5, T)
            tensor_list.append(tensor)

        # Use first window of each trial's tensor as the representative sample
        eeg_tensors = np.stack([t[0] for t in tensor_list])  # (N, H, W, 5, T)
        de_flat = np.stack([d[0] for d in de_list])          # (N, 160)

        # Speech
        speech_feat_list, wav_list = [], []
        for raw_speech in raw_speech_list:
            feats = self.speech_prep.extract_features(raw_speech)  # (n_seg, 180)
            speech_feat_list.append(feats[0])                      # first segment
            wav_list.append(self.speech_prep.prepare_wav2vec_input(raw_speech))

        speech_feats = np.stack(speech_feat_list)  # (N, 180)

        # Pad/trim waveforms to equal length for batching
        max_len = max(w.shape[0] for w in wav_list)
        wavs = np.zeros((len(wav_list), max_len), dtype=np.float32)
        for i, w in enumerate(wav_list):
            wavs[i, :len(w)] = w

        # GA selection (if selectors were pre-fitted)
        if self.eeg_ga is not None:
            eeg_ga_feats = self.eeg_ga.transform(de_flat)
        else:
            eeg_ga_feats = de_flat

        if self.speech_ga is not None:
            speech_ga_feats = self.speech_ga.transform(speech_feats)
        else:
            speech_ga_feats = speech_feats

        early_feats = np.concatenate([eeg_ga_feats, speech_ga_feats], axis=1)

        # Save
        np.save(os.path.join(sub_dir, "eeg_tensor.npy"), eeg_tensors)
        np.save(os.path.join(sub_dir, "eeg_ga_feats.npy"), eeg_ga_feats)
        np.save(os.path.join(sub_dir, "speech_wav.npy"), wavs)
        np.save(os.path.join(sub_dir, "speech_ga_feats.npy"), speech_ga_feats)
        np.save(os.path.join(sub_dir, "early_feats.npy"), early_feats)
        np.save(os.path.join(sub_dir, "labels.npy"), labels.astype(np.int64))

        print(
            f"  Subject {subject_id}: {len(labels)} trials saved to {sub_dir}\n"
            f"    EEG tensor: {eeg_tensors.shape}, EEG GA: {eeg_ga_feats.shape}\n"
            f"    Speech wav: {wavs.shape}, Speech GA: {speech_ga_feats.shape}\n"
            f"    Early feats: {early_feats.shape}"
        )
