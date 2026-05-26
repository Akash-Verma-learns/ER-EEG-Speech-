"""
Speech preprocessing for RAVDESS (Wav2Vec 2.0 path + acoustic GA features).

RAVDESS recordings are at 48 kHz → resampled to 16 kHz for Wav2Vec2.
Acoustic GA features: MFCC (40) + CHROMA (12) + MEL (128) = 180-D.
"""

import numpy as np
from scipy.signal import wiener
from typing import Optional

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False


def apply_wiener_filter(signal: np.ndarray, mysize: int = 5) -> np.ndarray:
    return wiener(signal, mysize=mysize).astype(np.float32)


def extract_mfcc(signal, sr, n_mfcc=40, n_fft=512, hop_length=256):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    return mfcc.mean(axis=1).astype(np.float32)


def extract_chroma(signal, sr, n_chroma=12, n_fft=512, hop_length=256):
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=n_chroma,
                                          n_fft=n_fft, hop_length=hop_length)
    return chroma.mean(axis=1).astype(np.float32)


def extract_mel(signal, sr, n_mel=128, n_fft=512, hop_length=256):
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mel,
                                          n_fft=n_fft, hop_length=hop_length)
    return librosa.power_to_db(mel).mean(axis=1).astype(np.float32)


def extract_acoustic_features(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_chroma: int = 12,
    n_mel: int = 128,
    apply_wiener: bool = True,
) -> np.ndarray:
    """
    180-D acoustic feature vector from a single waveform segment.
    Signal should already be resampled to sr (16 kHz).
    """
    if not _LIBROSA:
        raise ImportError("librosa required: pip install librosa")

    signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=0)

    if apply_wiener:
        signal = apply_wiener_filter(signal)

    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    n_fft = min(512, len(signal))
    hop   = n_fft // 2

    mfcc   = extract_mfcc(signal, sr, n_mfcc, n_fft, hop)
    chroma = extract_chroma(signal, sr, n_chroma, n_fft, hop)
    mel    = extract_mel(signal, sr, n_mel, n_fft, hop)

    return np.concatenate([mfcc, chroma, mel])   # (180,)


def prepare_wav2vec_input(signal: np.ndarray) -> np.ndarray:
    """
    Normalise pre-resampled 16 kHz waveform for Wav2Vec 2.0 input.
    Expects signal already at 16 kHz (RAVDESS resampled by ravdess_loader).
    """
    signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    signal = apply_wiener_filter(signal)
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak
    return signal


class SpeechPreprocessor:
    """Extract 180-D acoustic GA features from a pre-resampled 16 kHz waveform."""

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        n_mel: int = 128,
        segment_length: float = 3.0,   # RAVDESS recordings are ~3-5 s
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mel = n_mel
        self.seg_len = int(segment_length * sr)

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Return (1, 180) acoustic features for a single RAVDESS recording.
        The whole recording is used as one segment (RAVDESS clips are short).
        """
        if len(signal) < self.seg_len:
            signal = np.pad(signal, (0, self.seg_len - len(signal)))
        feat = extract_acoustic_features(signal[: self.seg_len], self.sr,
                                         self.n_mfcc, self.n_chroma, self.n_mel)
        return feat[np.newaxis, :]   # (1, 180)

    def prepare_wav2vec_input(self, signal: np.ndarray) -> np.ndarray:
        return prepare_wav2vec_input(signal)
