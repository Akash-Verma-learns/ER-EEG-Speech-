"""
Speech Encoder (M_Speech):
  Wav2Vec 2.0 (frozen) → 1-D DS-CNN Head → Transformer Block → Y_Sp ∈ ℝ^256

The Wav2Vec 2.0 backbone is always frozen; only the DS-CNN head and
Transformer block are trained.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    _WAV2VEC_AVAILABLE = True
except ImportError:
    _WAV2VEC_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1-D Depthwise-Separable CNN Head
# ---------------------------------------------------------------------------

class DSCNNHead1D(nn.Module):
    """
    Three-stage 1-D DS-CNN head:
      Conv1D → DWConv1D → SepConv1D → GlobalAveragePool → Dense-256
    """

    def __init__(
        self,
        in_dim: int = 768,          # Wav2Vec 2.0 hidden size
        channels: list = None,
        out_dim: int = 256,
    ):
        super().__init__()
        if channels is None:
            channels = [128, 256, 256]

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        # Depthwise
        self.dw_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], kernel_size=3, padding=1,
                      groups=channels[0], bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.GELU(),
        )
        # Pointwise
        self.pw_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[1]),
            nn.GELU(),
        )
        # Separable
        self.sep_dw = nn.Conv1d(channels[1], channels[1], kernel_size=3, padding=1,
                                groups=channels[1], bias=False)
        self.sep_pw = nn.Conv1d(channels[1], channels[2], kernel_size=1, bias=False)
        self.sep_bn = nn.BatchNorm1d(channels[2])

        self.dense = nn.Linear(channels[2], out_dim)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, wav2vec_dim)  — time-major hidden states
        Returns : (batch, out_dim)
        """
        x = x.transpose(1, 2)                           # (B, dim, T)
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = F.gelu(self.sep_bn(self.sep_pw(self.sep_dw(x))))
        x = x.mean(dim=-1)                               # GAP → (B, channels[-1])
        return F.gelu(self.dense(x))                     # (B, 256)


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Two-layer 4-head self-attention Transformer with positional encoding.
    Input/output: (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (batch, seq_len, d_model)  →  (batch, d_model)  via mean pooling."""
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.norm(x.mean(dim=1))   # (batch, d_model)


# ---------------------------------------------------------------------------
# Speech Encoder (full pipeline)
# ---------------------------------------------------------------------------

class SpeechEncoder(nn.Module):
    """
    M_Speech encoder.

    Input  : raw waveform (batch, n_samples)  — 16 kHz, normalised
    Output : Y_Sp (batch, 256)
    """

    def __init__(
        self,
        wav2vec_model_name: str = "facebook/wav2vec2-base",
        wav2vec_out_dim: int = 768,
        ds_cnn_channels: list = None,
        ds_cnn_out_dim: int = 256,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_dim: int = 256,
        transformer_ffn_dim: int = 512,
        dropout: float = 0.1,
        freeze_wav2vec: bool = True,
    ):
        super().__init__()

        if not _WAV2VEC_AVAILABLE:
            raise ImportError(
                "transformers library required for SpeechEncoder. "
                "Install with: pip install transformers"
            )

        # Frozen Wav2Vec 2.0 backbone
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False

        # Trainable 1-D DS-CNN head
        self.ds_cnn = DSCNNHead1D(
            in_dim=wav2vec_out_dim,
            channels=ds_cnn_channels or [128, 256, 256],
            out_dim=ds_cnn_out_dim,
        )

        # Transformer block over the DS-CNN output treated as a 1-token sequence
        # (we project to a sequence of fixed length via chunking for the Transformer)
        self.transformer = TransformerBlock(
            d_model=transformer_dim,
            nhead=transformer_heads,
            num_layers=transformer_layers,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
        )

        # Project DS-CNN output → sequence for Transformer
        # We split the wav2vec hidden states into chunks before the DS-CNN
        # so the Transformer sees a short temporal sequence, not a single vector.
        # The DS-CNN produces one vector per chunk; the Transformer attends across chunks.
        self.proj = nn.Linear(ds_cnn_out_dim, transformer_dim)
        self.out_dim = transformer_dim  # 256

    def _encode_chunks(
        self, hidden_states: torch.Tensor, chunk_size: int = 50
    ) -> torch.Tensor:
        """
        Split wav2vec hidden states into non-overlapping temporal chunks,
        apply DS-CNN per chunk, return (batch, n_chunks, ds_cnn_out_dim).
        """
        seq_len = hidden_states.size(1)
        n_chunks = max(1, seq_len // chunk_size)
        # Trim to exact multiple of chunk_size
        trim = n_chunks * chunk_size
        h = hidden_states[:, :trim, :]          # (B, trim, 768)
        h = h.reshape(
            h.size(0) * n_chunks, chunk_size, h.size(2)
        )                                        # (B·n_chunks, chunk_size, 768)
        chunk_feat = self.ds_cnn(h)              # (B·n_chunks, 256)
        chunk_feat = chunk_feat.reshape(
            hidden_states.size(0), n_chunks, -1
        )                                        # (B, n_chunks, 256)
        return chunk_feat

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : (batch, n_samples)  — raw 16 kHz waveform, values in [-1, 1]

        Returns
        -------
        Y_Sp : (batch, 256)
        """
        # Wav2Vec 2.0 feature extraction (frozen)
        with torch.no_grad() if not self.training or not self.wav2vec.training else torch.enable_grad():
            outputs = self.wav2vec(waveform, output_hidden_states=False)
        hidden_states = outputs.last_hidden_state  # (B, seq, 768)

        # DS-CNN: chunk → per-chunk representation
        chunk_feats = self._encode_chunks(hidden_states)   # (B, n_chunks, 256)

        # Project to Transformer dimension
        seq = self.proj(chunk_feats)                       # (B, n_chunks, 256)

        # Transformer with positional encoding → pooled output
        y_sp = self.transformer(seq)                       # (B, 256)
        return y_sp
