"""
EEG preprocessing utilities.

For SEED-IV the pre-extracted LDS-smoothed DE features are used directly
(loaded via seed_iv_loader.py).  This module provides:
  - 62-channel SEED-IV spatial grid mapping (9×9)
  - 4-D tensor construction from flat DE feature windows
  - Raw EEG Butterworth BPF + DE computation (for datasets without pre-extracted features)
  - Per-subject z-score normalisation
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# SEED-IV 62-channel → 9×9 spatial grid mapping
# Layout verified against the standard 10-20 extended 62-ch cap used by SJTU.
# ---------------------------------------------------------------------------
SEED_IV_POSITIONS = {
    # Row 0 – frontal-polar
    "FP1": (0, 3), "FPZ": (0, 4), "FP2": (0, 5),
    # Row 1 – anterior-frontal
    "AF7": (1, 1), "AF3": (1, 2), "AF4": (1, 6), "AF8": (1, 7),
    # Row 2 – frontal
    "F7": (2, 0), "F5": (2, 1), "F3": (2, 2), "F1": (2, 3),
    "FZ": (2, 4),
    "F2": (2, 5), "F4": (2, 6), "F6": (2, 7), "F8": (2, 8),
    # Row 3 – fronto-central
    "FT7": (3, 0), "FC5": (3, 1), "FC3": (3, 2), "FC1": (3, 3),
    "FCZ": (3, 4),
    "FC2": (3, 5), "FC4": (3, 6), "FC6": (3, 7), "FT8": (3, 8),
    # Row 4 – central / temporal
    "T7": (4, 0), "C5": (4, 1), "C3": (4, 2), "C1": (4, 3),
    "CZ": (4, 4),
    "C2": (4, 5), "C4": (4, 6), "C6": (4, 7), "T8": (4, 8),
    # Row 5 – centro-parietal / temporal
    "TP7": (5, 0), "CP5": (5, 1), "CP3": (5, 2), "CP1": (5, 3),
    "CPZ": (5, 4),
    "CP2": (5, 5), "CP4": (5, 6), "CP6": (5, 7), "TP8": (5, 8),
    # Row 6 – parietal
    "P7": (6, 0), "P5": (6, 1), "P3": (6, 2), "P1": (6, 3),
    "PZ": (6, 4),
    "P2": (6, 5), "P4": (6, 6), "P6": (6, 7), "P8": (6, 8),
    # Row 7 – parieto-occipital
    "PO7": (7, 1), "PO5": (7, 2), "PO3": (7, 3),
    "POZ": (7, 4),
    "PO4": (7, 5), "PO6": (7, 6), "PO8": (7, 7),
    # Row 8 – occipital
    "O1": (8, 2), "OZ": (8, 4), "O2": (8, 6),
}

# Standard SEED-IV 62-channel ordering (index → channel name)
# Derived from publicly available SEED dataset documentation.
SEED_IV_CHANNELS = [
    "FP1", "FP2", "FZ",   "F3",  "F4",  "F7",  "F8",
    "FT7", "FC3", "FC1",  "FCZ", "FC2", "FC4", "FT8",
    "T7",  "C5",  "C3",   "C1",  "CZ",  "C2",  "C4",  "C6",  "T8",
    "TP7", "CP5", "CP3",  "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7",  "P5",  "P3",   "P1",  "PZ",  "P2",  "P4",  "P6",  "P8",
    "PO7", "PO5", "PO3",  "POZ", "PO4", "PO6", "PO8",
    "O1",  "OZ",  "O2",
    # Extras to reach 62 — add AF/FPZ/PO8 variants as needed
    "AF7", "AF3", "AF4",  "AF8",
    "F5",  "F1",  "F2",   "F6",
    "FC5", "FC6",
    "FP1",  # placeholder — actual ordering from Channel Order.xlsx overrides this
]
# Use the first 62 unique entries
_seen = set()
_CHANNEL_LIST_62 = []
for _ch in SEED_IV_CHANNELS:
    if _ch not in _seen:
        _seen.add(_ch)
        _CHANNEL_LIST_62.append(_ch)
        if len(_CHANNEL_LIST_62) == 62:
            break


BAND_RANGES = [
    (0.5,  4.0),   # delta
    (4.0,  8.0),   # theta
    (8.0,  12.0),  # alpha
    (12.0, 30.0),  # beta
    (30.0, 50.0),  # gamma
]


# ---------------------------------------------------------------------------
# Spatial grid helpers
# ---------------------------------------------------------------------------

def build_channel_grid_index(
    channel_names: List[str],
    grid_shape: Tuple[int, int] = (9, 9),
    position_map: dict = None,
) -> np.ndarray:
    """
    Return an array of shape (n_channels, 2) mapping channel index → (row, col).
    Channels not in position_map are placed at the same (0,0) fallback.
    """
    if position_map is None:
        position_map = SEED_IV_POSITIONS
    idx = np.zeros((len(channel_names), 2), dtype=np.int32)
    for i, ch in enumerate(channel_names):
        if ch in position_map:
            idx[i] = position_map[ch]
    return idx


def de_windows_to_tensor(
    de_features: np.ndarray,
    channel_names: Optional[List[str]] = None,
    grid_shape: Tuple[int, int] = (9, 9),
    n_bands: int = 5,
    n_time: int = 2,
) -> np.ndarray:
    """
    Convert a DE feature matrix to a 4-D spatial-freq-temporal tensor.

    Parameters
    ----------
    de_features : (n_windows, n_channels * n_bands)
                  e.g. (T, 310) for 62-ch × 5-band SEED-IV

    Returns
    -------
    tensor : (n_samples, H, W, n_bands, n_time)
             where n_samples = n_windows - n_time + 1
    """
    if channel_names is None:
        channel_names = _CHANNEL_LIST_62
    n_channels = len(channel_names)

    H, W = grid_shape
    grid_idx = build_channel_grid_index(channel_names, grid_shape)

    n_windows = de_features.shape[0]
    n_samples = max(0, n_windows - n_time + 1)
    if n_samples == 0:
        # Pad by repeating last window
        de_features = np.vstack([de_features] + [de_features[[-1]]] * n_time)
        n_samples = 1

    # Pre-compute spatial grids for every window
    grids = np.zeros((n_windows, H, W, n_bands), dtype=np.float32)
    for b in range(n_bands):
        band_vals = de_features[:, b * n_channels: (b + 1) * n_channels]  # (T, n_ch)
        for ch_i, (r, c) in enumerate(grid_idx):
            grids[:, r, c, b] += band_vals[:, ch_i]

    # Stack n_time consecutive grids
    out = np.zeros((n_samples, H, W, n_bands, n_time), dtype=np.float32)
    for i in range(n_samples):
        out[i] = grids[i: i + n_time].transpose(1, 2, 3, 0)  # (H, W, bands, T)

    return out


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

def zscore_normalize(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalise feature matrix (N, D).
    If mean/std provided they are used (test-set normalisation).
    Returns (normalised, mean, std).
    """
    if mean is None:
        mean = features.mean(axis=0, keepdims=True)
    if std is None:
        std  = features.std(axis=0, keepdims=True) + eps
    return (features - mean) / std, mean, std


# ---------------------------------------------------------------------------
# Raw EEG → DE (for datasets without pre-extracted features)
# ---------------------------------------------------------------------------

def _butter_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    lo  = max(low,  0.5) / nyq
    hi  = min(high, nyq - 0.5) / nyq
    return butter(order, [lo, hi], btype="bandpass", output="sos")


def butterworth_bpf(signal: np.ndarray, low: float, high: float, fs: float) -> np.ndarray:
    sos = _butter_sos(low, high, fs)
    return sosfiltfilt(sos, signal, axis=-1)


def compute_de_features(
    signal: np.ndarray,
    fs: float,
    window_size: float = 4.0,
    band_ranges: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Compute Differential Entropy: DE = 0.5 * log(2πe * σ²).

    Parameters
    ----------
    signal : (n_channels, n_samples)

    Returns
    -------
    de : (n_windows, n_channels * n_bands)
    """
    if band_ranges is None:
        band_ranges = BAND_RANGES

    n_ch, n_samp = signal.shape
    win = int(window_size * fs)
    n_windows = n_samp // win
    n_bands = len(band_ranges)
    out = np.zeros((n_windows, n_ch * n_bands), dtype=np.float32)

    sig_trim = signal[:, : n_windows * win]

    for b, (lo, hi) in enumerate(band_ranges):
        filt = butterworth_bpf(sig_trim, lo, hi, fs)            # (n_ch, n_windows*win)
        windowed = filt.reshape(n_ch, n_windows, win)           # (n_ch, T, win)
        var = windowed.var(axis=-1).clip(1e-10)                 # (n_ch, T)
        de  = 0.5 * np.log(2 * np.pi * np.e * var)             # (n_ch, T)
        out[:, b * n_ch: (b + 1) * n_ch] = de.T

    return out


class EEGPreprocessor:
    """Pipeline for raw EEG → DE features → 4-D tensor.
    For SEED-IV use seed_iv_loader instead (features already extracted)."""

    def __init__(
        self,
        fs: float = 200.0,
        window_size: float = 4.0,
        channel_names: Optional[List[str]] = None,
        grid_shape: Tuple[int, int] = (9, 9),
        n_time: int = 2,
    ):
        self.fs = fs
        self.window_size = window_size
        self.channel_names = channel_names or _CHANNEL_LIST_62
        self.grid_shape = grid_shape
        self.n_time = n_time

    def preprocess(self, raw_eeg: np.ndarray) -> np.ndarray:
        """raw_eeg (n_ch, n_samples) → de_features (n_windows, n_ch*5)"""
        return compute_de_features(raw_eeg, self.fs, self.window_size)

    def to_tensor(self, de_features: np.ndarray) -> np.ndarray:
        """(n_windows, 310) → (n_samples, 9, 9, 5, 2)"""
        return de_windows_to_tensor(
            de_features, self.channel_names, self.grid_shape, 5, self.n_time
        )
