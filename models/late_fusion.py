"""
Late Fusion Model (M_LateFusion):
  Y_EEG ∈ ℝ^128  ⊕  Y_Sp ∈ ℝ^256  →  ℝ^384
  → Cross-Modal Attention (d_k=64)
  → Gated Modality Weighting (α_EEG, α_Sp)
  → Z_late ∈ ℝ^256
  → Dense-128 → Softmax

The gated modality weights are inspectable at inference time — α_EEG
empirically dominates for arousal, α_Sp for valence (Mauss & Robinson, 2009).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalAttention(nn.Module):
    """
    Scaled dot-product cross-modal attention between EEG and Speech
    representations.  Both modalities attend to each other and the outputs
    are combined.

    d_k : key/query projection dimension
    """

    def __init__(self, eeg_dim: int = 128, speech_dim: int = 256, d_k: int = 64):
        super().__init__()
        self.d_k = d_k

        # Project both modalities to common key/query/value space
        self.W_q_eeg = nn.Linear(eeg_dim, d_k, bias=False)
        self.W_k_sp  = nn.Linear(speech_dim, d_k, bias=False)
        self.W_v_sp  = nn.Linear(speech_dim, d_k, bias=False)

        self.W_q_sp  = nn.Linear(speech_dim, d_k, bias=False)
        self.W_k_eeg = nn.Linear(eeg_dim, d_k, bias=False)
        self.W_v_eeg = nn.Linear(eeg_dim, d_k, bias=False)

        # Project attended outputs back to original dims
        self.out_eeg = nn.Linear(d_k, eeg_dim)
        self.out_sp  = nn.Linear(d_k, speech_dim)

        self.norm_eeg = nn.LayerNorm(eeg_dim)
        self.norm_sp  = nn.LayerNorm(speech_dim)

    def forward(
        self,
        y_eeg: torch.Tensor,
        y_sp: torch.Tensor,
    ):
        """
        Parameters
        ----------
        y_eeg : (batch, eeg_dim=128)
        y_sp  : (batch, speech_dim=256)

        Returns
        -------
        y_eeg_attended : (batch, eeg_dim)
        y_sp_attended  : (batch, speech_dim)
        """
        # EEG attends to Speech
        q_eeg = self.W_q_eeg(y_eeg)       # (B, d_k)
        k_sp  = self.W_k_sp(y_sp)         # (B, d_k)
        v_sp  = self.W_v_sp(y_sp)         # (B, d_k)
        score_eeg = (q_eeg * k_sp).sum(-1, keepdim=True) / math.sqrt(self.d_k)
        # single-element attention → just scale + gate
        alpha_eeg_sp = torch.sigmoid(score_eeg)  # (B, 1)
        attended_eeg = self.out_eeg(alpha_eeg_sp * v_sp)  # (B, eeg_dim)
        y_eeg_out = self.norm_eeg(y_eeg + attended_eeg)   # residual

        # Speech attends to EEG
        q_sp  = self.W_q_sp(y_sp)         # (B, d_k)
        k_eeg = self.W_k_eeg(y_eeg)       # (B, d_k)
        v_eeg = self.W_v_eeg(y_eeg)       # (B, d_k)
        score_sp = (q_sp * k_eeg).sum(-1, keepdim=True) / math.sqrt(self.d_k)
        alpha_sp_eeg = torch.sigmoid(score_sp)
        attended_sp = self.out_sp(alpha_sp_eeg * v_eeg)
        y_sp_out = self.norm_sp(y_sp + attended_sp)

        return y_eeg_out, y_sp_out


class GatedModalityWeighting(nn.Module):
    """
    Learnable sigmoid-activated scalar gates α_EEG and α_Speech.
    At inference the gate values indicate which modality dominates.

    Input  : y_eeg (B, eeg_dim), y_sp (B, speech_dim)
    Output : z (B, out_dim)  — gated, projected joint representation
    """

    def __init__(
        self,
        eeg_dim: int = 128,
        speech_dim: int = 256,
        out_dim: int = 256,
    ):
        super().__init__()
        concat_dim = eeg_dim + speech_dim  # 384

        # Learnable *scalar* gates — one per modality
        self.log_alpha_eeg = nn.Parameter(torch.zeros(1))   # initialised to 0.5
        self.log_alpha_sp  = nn.Parameter(torch.zeros(1))

        # Project gated concatenation to output dimension
        self.proj = nn.Sequential(
            nn.Linear(concat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    @property
    def alpha_eeg(self) -> torch.Tensor:
        """Gate value for EEG modality (sigmoid)."""
        return torch.sigmoid(self.log_alpha_eeg)

    @property
    def alpha_speech(self) -> torch.Tensor:
        """Gate value for Speech modality (sigmoid)."""
        return torch.sigmoid(self.log_alpha_sp)

    def forward(
        self, y_eeg: torch.Tensor, y_sp: torch.Tensor
    ) -> torch.Tensor:
        """Returns Z_late (batch, out_dim)."""
        alpha_e = self.alpha_eeg   # scalar gate ∈ (0, 1)
        alpha_s = self.alpha_speech

        gated = torch.cat([alpha_e * y_eeg, alpha_s * y_sp], dim=-1)  # (B, 384)
        return self.proj(gated)                                         # (B, 256)

    def gate_values(self):
        """Return gate scalars as (α_EEG, α_Speech) Python floats for logging."""
        return float(self.alpha_eeg.item()), float(self.alpha_speech.item())


class LateFusionModel(nn.Module):
    """
    M_LateFusion:
    Wraps frozen EEG and Speech encoders + trainable fusion layers.

    In training, call one of two modes:
      pretrain_mode=True  → train the individual encoder + its own head
      pretrain_mode=False → encoders frozen, train only fusion layers

    Input  : eeg_tensor  (batch, H, W, n_bands, n_time)
             waveform    (batch, n_samples)
    Output : logits      (batch, n_classes)
    """

    def __init__(
        self,
        eeg_encoder: nn.Module,
        speech_encoder: nn.Module,
        eeg_dim: int = 128,
        speech_dim: int = 256,
        attention_dk: int = 64,
        output_dim: int = 256,
        n_classes: int = 4,
        classifier_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.speech_encoder = speech_encoder

        # Fusion layers (trained after encoders are pre-trained)
        self.cross_attn = CrossModalAttention(eeg_dim, speech_dim, attention_dk)
        self.gated_weight = GatedModalityWeighting(eeg_dim, speech_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, n_classes),
        )
        self.out_dim = output_dim

    def freeze_encoders(self):
        for param in self.eeg_encoder.parameters():
            param.requires_grad = False
        for param in self.speech_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        for param in self.eeg_encoder.parameters():
            param.requires_grad = True
        for param in self.speech_encoder.parameters():
            param.requires_grad = True
        # But keep Wav2Vec frozen
        if hasattr(self.speech_encoder, "wav2vec"):
            for param in self.speech_encoder.wav2vec.parameters():
                param.requires_grad = False

    def forward(
        self,
        eeg_tensor: torch.Tensor,
        waveform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : (batch, n_classes)
        """
        y_eeg = self.eeg_encoder(eeg_tensor)   # (B, 128)
        y_sp  = self.speech_encoder(waveform)  # (B, 256)

        y_eeg_a, y_sp_a = self.cross_attn(y_eeg, y_sp)
        z_late = self.gated_weight(y_eeg_a, y_sp_a)  # Z_late ∈ ℝ^256
        z_late = self.dropout(z_late)

        return self.classifier(z_late)

    def encode(
        self, eeg_tensor: torch.Tensor, waveform: torch.Tensor
    ) -> torch.Tensor:
        """Return Z_late without classification head."""
        y_eeg = self.eeg_encoder(eeg_tensor)
        y_sp  = self.speech_encoder(waveform)
        y_eeg_a, y_sp_a = self.cross_attn(y_eeg, y_sp)
        return self.gated_weight(y_eeg_a, y_sp_a)

    def get_gate_values(self):
        """Return (α_EEG, α_Speech) for interpretability analysis."""
        return self.gated_weight.gate_values()
