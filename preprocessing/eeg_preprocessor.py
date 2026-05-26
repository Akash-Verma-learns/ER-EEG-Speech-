"""EEG preprocessing: Butterworth BPF → DE features → 4-D spatial-freq-temporal tensor."""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from typing import List, Tuple, Optional


# Standard 10-20 32-channel layout mapped onto an 8×9 spatial grid.
# Each entry is (row, col) for the corresponding channel index.
# Channels follow the order: Fp1,Fp2,F7,F3,Fz,F4,F8,FC5,FC1,FC2,FC6,T7,C3,Cz,C4,T8,
#                            TP9,CP5,CP1,CP2,CP6,TP10,P7,P3,Pz,P4,P8,PO9,O1,Oz,O2,PO10
CHANNEL_POSITIONS_32 = {
    # Frontal
    "Fp1": (0, 3), "Fp2": (0, 5),
    "AF3": (1, 3), "AF4": (1, 5),
    "F7":  (2, 1), "F3":  (2, 3), "Fz":  (2, 4), "F4":  (2, 5), "F8":  (2, 7),
    "FC5": (3, 2), "FC1": (3, 3), "FC2": (3, 5), "FC6": (3, 6),
    # Central
    "T7":  (4, 0), "C3":  (4, 3), "Cz":  (4, 4), "C4":  (4, 5), "T8":  (4, 8),
    "CP5": (5, 2), "CP1": (5, 3), "CP2": (5, 5), "CP6": (5, 6),
    # Parietal / Temporal / Occipital
    "TP9": (6, 0), "P7":  (6, 1), "P3":  (6, 3), "Pz":  (6, 4), "P4":  (6, 5),
    "P8":  (6, 7), "TP10":(6, 8),
    "PO9": (7, 1), "O1":  (7, 3), "Oz":  (7, 4), "O2":  (7, 5), "PO10":(7, 7),
}

# Default channel ordering (indices 0–31 match this list)
DEFAULT_CHANNELS = [
    "Fp1","Fp2","F7","F3","Fz","F4","F8","FC5","FC1","FC2","FC6",
    "T7","C3","Cz","C4","T8","TP9","CP5","CP1","CP2","CP6","TP10",
    "P7","P3","Pz","P4","P8","PO9","O1","Oz","O2","PO10",
]

BAND_RANGES = [
    (0.5,  4.0),   # delta
    (4.0,  8.0),   # theta
    (8.0,  12.0),  # alpha
    (12.0, 30.0),  # beta
    (30.0, 50.0),  # gamma
]


def _butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sos


def butterworth_bpf(signal: np.ndarray, low: float, high: float, fs: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth BPF. signal shape: (n_channels, n_samples)."""
    sos = _butter_bandpass(low, high, fs, order)
    return sosfiltfilt(sos, signal, axis=-1)


def compute_de_features(
    signal: np.ndarray,
    fs: float,
    window_size: float = 0.5,
    band_ranges: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Compute Differential Entropy features.

    Parameters
    ----------
    signal : (n_channels, n_samples)
    fs     : sampling rate
    window_size : seconds per window

    Returns
    -------
    de_features : (n_windows, n_channels × n_bands)
    """
    if band_ranges is None:
        band_ranges = BAND_RANGES

    n_channels, n_samples = signal.shape
    win_samples = int(window_size * fs)
    n_windows = n_samples // win_samples
    n_bands = len(band_ranges)

    features = np.zeros((n_windows, n_channels * n_bands))

    for b_idx, (low, high) in enumerate(band_ranges):
        # Clamp to Nyquist
        nyq = fs / 2.0
        if high >= nyq:
            high = nyq - 0.5
        if low >= high:
            continue
        filtered = butterworth_bpf(signal[:, :n_windows * win_samples], low, high, fs)
        # Reshape to windows
        windowed = filtered.reshape(n_channels, n_windows, win_samples)
        var = windowed.var(axis=-1)                  # (n_channels, n_windows)
        var = np.clip(var, 1e-10, None)
        de = 0.5 * np.log(2 * np.pi * np.e * var)  # (n_channels, n_windows)
        features[:, b_idx * n_channels:(b_idx + 1) * n_channels] = de.T

    return features  # (n_windows, 160)


def build_spatial_grid(
    de_window: np.ndarray,
    channel_names: Optional[List[str]] = None,
    grid_shape: Tuple[int, int] = (8, 9),
    n_bands: int = 5,
    n_channels: int = 32,
) -> np.ndarray:
    """
    Map a single DE window (160-D) onto a 4-D spatial-freq grid.

    Parameters
    ----------
    de_window : (n_channels × n_bands,)  — one time-window of DE features

    Returns
    -------
    grid : (H, W, n_bands)  — spatial frequency map
    """
    if channel_names is None:
        channel_names = DEFAULT_CHANNELS[:n_channels]

    H, W = grid_shape
    grid = np.zeros((H, W, n_bands), dtype=np.float32)

    for ch_idx, ch_name in enumerate(channel_names):
        if ch_name not in CHANNEL_POSITIONS_32:
            continue
        r, c = CHANNEL_POSITIONS_32[ch_name]
        for b_idx in range(n_bands):
            feat_idx = b_idx * n_channels + ch_idx
            if feat_idx < len(de_window):
                grid[r, c, b_idx] = de_window[feat_idx]

    return grid  # (H, W, n_bands)


def build_spatiotemporal_tensor(
    de_features: np.ndarray,
    channel_names: Optional[List[str]] = None,
    grid_shape: Tuple[int, int] = (8, 9),
    n_bands: int = 5,
    n_channels: int = 32,
    n_time: int = 2,
) -> np.ndarray:
    """
    Stack `n_time` consecutive DE windows into a 4-D tensor.

    Returns
    -------
    tensor : (n_samples, H, W, n_bands, n_time)
    """
    n_windows = de_features.shape[0]
    n_samples = n_windows - n_time + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough windows ({n_windows}) for n_time={n_time}"
        )

    H, W = grid_shape
    out = np.zeros((n_samples, H, W, n_bands, n_time), dtype=np.float32)

    for i in range(n_samples):
        for t in range(n_time):
            grid = build_spatial_grid(
                de_features[i + t], channel_names, grid_shape, n_bands, n_channels
            )
            out[i, :, :, :, t] = grid

    return out  # (n_samples, 8, 9, 5, 2)


class EEGPreprocessor:
    """Full EEG preprocessing pipeline."""

    def __init__(
        self,
        fs: float = 128.0,
        window_size: float = 0.5,
        band_ranges: Optional[List[Tuple[float, float]]] = None,
        filter_low: float = 1.0,
        filter_high: float = 50.0,
        grid_shape: Tuple[int, int] = (8, 9),
        n_time: int = 2,
        channel_names: Optional[List[str]] = None,
    ):
        self.fs = fs
        self.window_size = window_size
        self.band_ranges = band_ranges or BAND_RANGES
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.grid_shape = grid_shape
        self.n_time = n_time
        self.channel_names = channel_names or DEFAULT_CHANNELS

    def preprocess(self, raw_eeg: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        raw_eeg : (n_channels, n_samples)

        Returns
        -------
        tensor : (n_samples, H, W, n_bands, n_time)  — ready for the encoder
        flat   : (n_windows, 160)                    — for GA feature selection
        """
        # Global bandpass
        n_channels = raw_eeg.shape[0]
        filtered = butterworth_bpf(raw_eeg, self.filter_low, self.filter_high, self.fs)

        # DE features per window
        de_features = compute_de_features(
            filtered, self.fs, self.window_size, self.band_ranges
        )

        return de_features  # (n_windows, 160)

    def to_tensor(self, de_features: np.ndarray) -> np.ndarray:
        """Convert flat DE feature matrix to (n_samples, H, W, n_bands, n_time)."""
        return build_spatiotemporal_tensor(
            de_features,
            self.channel_names,
            self.grid_shape,
            len(self.band_ranges),
            len(self.channel_names),
            self.n_time,
        )
