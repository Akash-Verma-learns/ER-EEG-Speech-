"""
Unimodal classifier wrappers for M_EEG and M_Speech.
Each wraps an encoder with a Dense-128 → Softmax head.
"""

import torch
import torch.nn as nn


class EEGClassifier(nn.Module):
    """M_EEG: encoder + classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 128,
        n_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, H, W, n_bands, n_time)"""
        features = self.encoder(x)        # Y_EEG ∈ ℝ^128
        return self.head(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpeechClassifier(nn.Module):
    """M_Speech: encoder + classification head."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 256,
        n_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform : (batch, n_samples)"""
        features = self.encoder(waveform)  # Y_Sp ∈ ℝ^256
        return self.head(features)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.encoder(waveform)
