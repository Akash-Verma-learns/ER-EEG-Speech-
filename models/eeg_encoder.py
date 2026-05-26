"""
EEG Encoder (M_EEG):
  4-D Tensor (B, H, W, bands, T) → DS-CNN 2D → flatten → ON-LSTM → Y_EEG ∈ ℝ^128
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .on_lstm import ONLSTM


class DepthwiseSepConv2D(nn.Module):
    """Depthwise-separable 2-D convolution: DWConv → PWConv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class DSCNNBlock(nn.Module):
    """
    DS-CNN block used in the EEG encoder:
      Conv2D-64 → BN → ReLU →
      DWConv-128 → BN → ReLU →
      SepConv-256 → BN → ReLU →
      GlobalAveragePool (spatial)
    """

    def __init__(
        self,
        in_channels: int = 5,   # n_frequency_bands (input channel dim)
        channels: list = None,
    ):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        # Standard Conv2D
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        # Depthwise conv
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1,
                      groups=channels[0], bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        # Pointwise to expand channels
        self.pw_conv = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
        )
        # Full separable conv
        self.sep_conv = DepthwiseSepConv2D(channels[1], channels[2])

        self.out_channels = channels[2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, in_channels, H, W)"""
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.sep_conv(x)                         # (batch, 256, H, W)
        x = F.adaptive_avg_pool2d(x, (1, 1))        # (batch, 256, 1, 1)
        return x.flatten(1)                          # (batch, 256)


class EEGEncoder(nn.Module):
    """
    M_EEG encoder.

    Input  : x (batch, H, W, n_bands, n_time)  — 4-D spatial-freq-temporal tensor
    Output : Y_EEG (batch, 128)
    """

    def __init__(
        self,
        grid_h: int = 8,
        grid_w: int = 9,
        n_bands: int = 5,
        n_time: int = 2,
        conv_channels: list = None,
        lstm_hidden: int = 128,
        lstm_chunk_size: int = 16,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 256]

        self.n_time = n_time
        self.n_bands = n_bands

        # One DS-CNN block per time step (shared weights across time)
        self.cnn = DSCNNBlock(in_channels=n_bands, channels=conv_channels)
        cnn_out_dim = conv_channels[-1]  # 256

        # ON-LSTM over the time dimension
        self.on_lstm = ONLSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            chunk_size=lstm_chunk_size,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = lstm_hidden  # 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, H, W, n_bands, n_time)

        Returns
        -------
        Y_EEG : (batch, 128)
        """
        batch = x.size(0)
        # Permute to (batch, n_time, n_bands, H, W) for CNN-per-timestep
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (B, T, bands, H, W)

        # Apply DS-CNN to each time step
        cnn_feats = []
        for t in range(self.n_time):
            feat = self.cnn(x[:, t, :, :, :])  # (B, 256)
            cnn_feats.append(feat.unsqueeze(1))

        seq = torch.cat(cnn_feats, dim=1)  # (B, T, 256)

        # ON-LSTM — take final hidden state
        _, (h_n, _) = self.on_lstm(seq)     # h_n: (B, 128)
        return self.dropout(h_n)            # Y_EEG ∈ ℝ^128
