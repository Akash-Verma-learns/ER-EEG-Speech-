"""Speech preprocessing: Wiener filter → MFCC + CHROMA + MEL (180-D per window)."""

import numpy as np
import librosa
from scipy.signal import wiener
from typing import Optional


def apply_wiener_filter(signal: np.ndarray, mysize: int = 5) -> np.ndarray:
    """Wiener filter for noise suppression on a 1-D waveform."""
    return wiener(signal, mysize=mysize).astype(np.float32)


def extract_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Return time-averaged MFCC vector (n_mfcc,)."""
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)
    return mfcc.mean(axis=1).astype(np.float32)  # (n_mfcc,)


def extract_chroma(
    signal: np.ndarray,
    sr: int,
    n_chroma: int = 12,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Return time-averaged chroma vector (n_chroma,)."""
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=n_chroma,
                                          n_fft=n_fft, hop_length=hop_length)
    return chroma.mean(axis=1).astype(np.float32)  # (n_chroma,)


def extract_mel(
    signal: np.ndarray,
    sr: int,
    n_mel: int = 128,
    n_fft: int = 512,
    hop_length: int = 256,
) -> np.ndarray:
    """Return time-averaged log-Mel spectrogram vector (n_mel,)."""
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mel,
                                          n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    return log_mel.mean(axis=1).astype(np.float32)  # (n_mel,)


def extract_acoustic_features(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_chroma: int = 12,
    n_mel: int = 128,
    n_fft: int = 512,
    hop_length: int = 256,
    apply_wiener: bool = True,
) -> np.ndarray:
    """
    Extract 180-D acoustic feature vector: MFCC (40) + CHROMA (12) + MEL (128).

    Parameters
    ----------
    signal : (n_samples,)  — raw 16 kHz waveform

    Returns
    -------
    features : (180,)
    """
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    signal = signal.astype(np.float32)

    if apply_wiener:
        signal = apply_wiener_filter(signal)

    # Normalise amplitude
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    mfcc = extract_mfcc(signal, sr, n_mfcc, n_fft, hop_length)
    chroma = extract_chroma(signal, sr, n_chroma, n_fft, hop_length)
    mel = extract_mel(signal, sr, n_mel, n_fft, hop_length)

    return np.concatenate([mfcc, chroma, mel])  # (180,)


def extract_wav2vec_input(
    signal: np.ndarray,
    sr: int = 16000,
    target_sr: int = 16000,
    apply_wiener: bool = True,
) -> np.ndarray:
    """
    Prepare raw waveform for Wav2Vec 2.0 input.
    Resamples if needed, applies Wiener filter, normalises to [-1, 1].

    Returns
    -------
    waveform : (n_samples,)  float32
    """
    if signal.ndim > 1:
        signal = signal.mean(axis=0)
    signal = signal.astype(np.float32)

    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)

    if apply_wiener:
        signal = apply_wiener_filter(signal)

    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak

    return signal


class SpeechPreprocessor:
    """Full speech preprocessing pipeline — produces both GA features and Wav2Vec input."""

    def __init__(
        self,
        sr: int = 16000,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        n_mel: int = 128,
        segment_length: float = 1.0,
        n_fft: int = 512,
        hop_length: int = 256,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_mel = n_mel
        self.segment_len = int(segment_length * sr)
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract 180-D features per segment.

        Parameters
        ----------
        signal : (n_samples,)

        Returns
        -------
        features : (n_segments, 180)
        """
        n_samples = len(signal)
        n_segments = n_samples // self.segment_len
        if n_segments == 0:
            n_segments = 1
            signal = np.pad(signal, (0, self.segment_len - n_samples))

        features = []
        for i in range(n_segments):
            seg = signal[i * self.segment_len: (i + 1) * self.segment_len]
            if len(seg) < self.segment_len:
                seg = np.pad(seg, (0, self.segment_len - len(seg)))
            feat = extract_acoustic_features(
                seg, self.sr, self.n_mfcc, self.n_chroma, self.n_mel,
                self.n_fft, self.hop_length,
            )
            features.append(feat)

        return np.stack(features)  # (n_segments, 180)

    def prepare_wav2vec_input(self, signal: np.ndarray) -> np.ndarray:
        """Return cleaned, normalised waveform for Wav2Vec 2.0."""
        return extract_wav2vec_input(signal, self.sr, self.sr)
