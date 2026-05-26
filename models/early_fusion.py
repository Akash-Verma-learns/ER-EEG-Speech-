"""
Early Fusion Model (M_EarlyFusion):
  GA_EEG features ⊕ GA_Speech features → ~142-D joint vector
  → Shared 1-D DS-CNN Encoder → Self-Attention → Z_early ∈ ℝ^256
  → Dense-128 → Softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedDSCNN1D(nn.Module):
    """
    Shared 1-D DS-CNN encoder that processes the concatenated GA feature vector.
    Input treated as a 1-channel sequence of length early_input_dim.
    """

    def __init__(
        self,
        input_dim: int = 142,
        channels: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 256, 256]

        # Treat the feature vector as a 1-D signal with 1 "channel"
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        # Depthwise
        self.dw = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=5, padding=2,
                      groups=channels[0], bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        # Pointwise
        self.pw = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[1]),
            nn.GELU(),
        )
        # Separable
        self.sep_dw = nn.Conv1d(channels[1], channels[1], kernel_size=3, padding=1,
                                groups=channels[1], bias=False)
        self.sep_pw = nn.Conv1d(channels[1], channels[2], kernel_size=1, bias=False)
        self.sep_bn = nn.BatchNorm1d(channels[2])

        self.dropout = nn.Dropout(dropout)
        self.out_channels = channels[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_dim)
        Returns : (batch, out_channels, input_dim)  — for subsequent self-attention
        """
        x = x.unsqueeze(1)          # (B, 1, input_dim)
        x = self.conv1(x)
        x = self.dw(x)
        x = self.pw(x)
        x = F.gelu(self.sep_bn(self.sep_pw(self.sep_dw(x))))
        return self.dropout(x)      # (B, out_channels, input_dim)


class IntraModalSelfAttention(nn.Module):
    """
    Intra-modal feature-weighting self-attention.
    Operates on the feature dimension to selectively weight informative features.
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)  →  (batch, d_model) via mean pool."""
        attended, _ = self.attn(x, x, x)
        x = self.norm(x + attended)
        x = self.norm2(x + self.ffn(x))
        return x.mean(dim=1)   # (batch, d_model)


class EarlyFusionModel(nn.Module):
    """
    M_EarlyFusion:
    Expects per-sample concatenated GA-selected features [EEG_GA ⊕ Speech_GA].

    Input  : x (batch, early_input_dim)  — ~142-D joint GA feature vector
    Output : logits (batch, n_classes)
    """

    def __init__(
        self,
        input_dim: int = 142,
        conv_channels: list = None,
        output_dim: int = 256,
        n_classes: int = 4,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [128, 256, 256]

        self.ds_cnn = SharedDSCNN1D(input_dim, conv_channels, dropout=dropout)
        cnn_channels = conv_channels[-1]

        # Self-attention treats the spatial positions of the CNN output as a sequence
        # We GAP over the spatial dimension then use a learned projection
        self.gap_proj = nn.Linear(cnn_channels, output_dim)
        self.self_attn = IntraModalSelfAttention(output_dim, nhead=4, dropout=dropout)

        # We need a sequence for the self-attention module
        # Strategy: split cnn output into non-overlapping segments
        self.n_attn_heads = 4

        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.out_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, input_dim)  — concatenated GA feature vectors

        Returns
        -------
        logits : (batch, n_classes)
        """
        cnn_out = self.ds_cnn(x)         # (B, cnn_channels, input_dim)
        # Transpose to (B, input_dim, cnn_channels) — spatial positions as sequence
        seq = cnn_out.transpose(1, 2)    # (B, input_dim, cnn_channels)
        seq = self.gap_proj(seq)         # (B, input_dim, output_dim)

        z_early = self.self_attn(seq)    # (B, output_dim)  = Z_early ∈ ℝ^256
        z_early = self.dropout(self.norm(z_early))

        return self.classifier(z_early)  # logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return Z_early without the classification head."""
        cnn_out = self.ds_cnn(x)
        seq = cnn_out.transpose(1, 2)
        seq = self.gap_proj(seq)
        return self.norm(self.self_attn(seq))
