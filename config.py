from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class EEGConfig:
    # SEED-IV specific
    n_channels: int = 62                 # 62-channel EEG cap
    sampling_rate: int = 200             # Hz (raw); feature windows at 4 s
    n_bands: int = 5                     # delta, theta, alpha, beta, gamma
    window_size: float = 4.0             # seconds per DE feature window (SEED-IV spec)
    filter_low: float = 1.0
    filter_high: float = 50.0
    de_feature_dim: int = 310            # 62 × 5
    ga_n_selected: int = 120             # ~39% of 310
    spatial_grid: Tuple[int, int] = (9, 9)
    n_time_windows: int = 2              # 2 consecutive windows stacked → 4D tensor

    band_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5,  4.0),   # delta
        (4.0,  8.0),   # theta
        (8.0,  12.0),  # alpha
        (12.0, 30.0),  # beta
        (30.0, 50.0),  # gamma
    ])


@dataclass
class SpeechConfig:
    """Wav2Vec 2.0 speech branch — unchanged from original design."""
    sampling_rate: int = 16000
    n_mfcc: int = 40
    n_chroma: int = 12
    n_mel: int = 128
    feature_dim: int = 180               # 40 + 12 + 128 (GA proxy features)
    ga_n_selected: int = 62
    wav2vec_model_name: str = "facebook/wav2vec2-base"
    wav2vec_output_dim: int = 768
    segment_length: float = 1.0


@dataclass
class ModelConfig:
    n_classes: int = 4                   # neutral, sad, fear, happy
    classifier_hidden_dim: int = 128

    # EEG DS-CNN (2D, shared weights across time steps)
    eeg_conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    eeg_lstm_hidden: int = 128
    eeg_output_dim: int = 128            # Y_EEG ∈ ℝ^128
    on_lstm_chunk_size: int = 16         # must divide eeg_lstm_hidden

    # Speech DS-CNN + Transformer (unchanged)
    speech_conv_channels: List[int] = field(default_factory=lambda: [128, 256, 256])
    speech_transformer_heads: int = 4
    speech_transformer_layers: int = 2
    speech_transformer_dim: int = 256
    speech_transformer_ffn_dim: int = 512
    speech_output_dim: int = 256         # Y_Sp ∈ ℝ^256

    # Early fusion (GA_EEG ⊕ GA_Speech)
    early_input_dim: int = 182           # 120 + 62
    early_conv_channels: List[int] = field(default_factory=lambda: [128, 256, 256])
    early_output_dim: int = 256          # Z_early ∈ ℝ^256

    # Late fusion
    late_concat_dim: int = 384           # 128 + 256
    late_attention_dk: int = 64
    late_output_dim: int = 256           # Z_late ∈ ℝ^256

    # Regularisation
    dropout: float = 0.3
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 150
    # SEED-IV: leave-one-session-out → 3 folds
    n_folds: int = 3                     # one fold per session
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    t_max: int = 150
    eta_min: float = 1e-6
    pretrain_epochs: int = 80            # encoder pre-training for late fusion
    gradient_clip: float = 1.0
    # Accuracy boosters
    mixup_alpha: float = 0.2             # 0 = off
    use_label_smoothing: bool = True


@dataclass
class GAConfig:
    population_size: int = 50
    n_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    tournament_size: int = 3
    sparsity_weight: float = 0.1
    n_jobs: int = -1


@dataclass
class RAVDESSConfig:
    """RAVDESS speech dataset — audio-only speech files."""
    root_dir: str = "./data/ravdess"     # contains Actor_01/ … Actor_24/
    target_sr: int = 16000              # resample to 16 kHz for Wav2Vec2
    # RAVDESS emotion codes that map to SEED-IV 4-class scheme
    # RAVDESS: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    # Keep only the 4 that overlap with SEED-IV
    emotion_map: dict = field(default_factory=lambda: {
        1: 0,   # neutral   → class 0
        4: 1,   # sad       → class 1
        6: 2,   # fearful   → class 2 (fear)
        3: 3,   # happy     → class 3
    })
    intensity: str = "both"             # "normal", "strong", or "both"
    use_song: bool = False              # use only speech (not song)


@dataclass
class SEEDIVConfig:
    """Dataset-specific paths and protocol."""
    root_dir: str = "./data"             # contains eeg_feature_smooth/, eye_feature_smooth/
    n_subjects: int = 15
    n_sessions: int = 3
    n_trials: int = 24
    eval_protocol: str = "leave_one_session_out"   # or "leave_one_subject_out"
    # Z-score normalise features per subject (critical for accuracy)
    normalise_per_subject: bool = True


@dataclass
class Config:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    seed_iv: SEEDIVConfig = field(default_factory=SEEDIVConfig)
    ravdess: RAVDESSConfig = field(default_factory=RAVDESSConfig)

    output_dir: str = "./outputs"
    experiment_name: str = "seed_iv_multimodal_emotion"
    log_interval: int = 10


def get_config() -> Config:
    return Config()
