from dataclasses import dataclass, field
from typing import List, Tuple
import torch


@dataclass
class EEGConfig:
    n_channels: int = 32
    sampling_rate: int = 128
    n_bands: int = 5
    window_size: float = 0.5          # seconds per DE window
    filter_low: float = 1.0
    filter_high: float = 50.0
    de_feature_dim: int = 160         # 32 channels × 5 bands
    ga_n_selected: int = 80
    spatial_grid: Tuple[int, int] = (8, 9)
    n_time_windows: int = 2           # 2T consecutive windows fed to DS-CNN

    # Frequency band definitions [low, high] Hz
    band_ranges: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.5, 4.0),   # delta
        (4.0, 8.0),   # theta
        (8.0, 12.0),  # alpha
        (12.0, 30.0), # beta
        (30.0, 50.0), # gamma
    ])


@dataclass
class SpeechConfig:
    sampling_rate: int = 16000
    n_mfcc: int = 40
    n_chroma: int = 12
    n_mel: int = 128
    feature_dim: int = 180            # 40 + 12 + 128
    ga_n_selected: int = 62
    wav2vec_model_name: str = "facebook/wav2vec2-base"
    wav2vec_output_dim: int = 768
    segment_length: float = 1.0       # seconds per processing segment


@dataclass
class ModelConfig:
    n_classes: int = 4
    classifier_hidden_dim: int = 128

    # EEG DS-CNN (2D)
    eeg_conv_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    eeg_lstm_hidden: int = 128
    eeg_output_dim: int = 128         # Y_EEG ∈ ℝ^128

    # ON-LSTM
    on_lstm_chunk_size: int = 16      # must divide eeg_lstm_hidden

    # Speech DS-CNN (1D)
    speech_conv_channels: List[int] = field(default_factory=lambda: [128, 256, 256])
    speech_transformer_heads: int = 4
    speech_transformer_layers: int = 2
    speech_transformer_dim: int = 256
    speech_transformer_ffn_dim: int = 512
    speech_output_dim: int = 256      # Y_Sp ∈ ℝ^256

    # Early fusion
    early_input_dim: int = 142        # ~80 + ~62 after separate GA runs
    early_conv_channels: List[int] = field(default_factory=lambda: [128, 256, 256])
    early_output_dim: int = 256       # Z_early ∈ ℝ^256

    # Late fusion
    late_concat_dim: int = 384        # 128 + 256
    late_attention_dk: int = 64
    late_output_dim: int = 256        # Z_late ∈ ℝ^256


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 150
    n_folds: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    t_max: int = 150                  # cosine annealing period
    eta_min: float = 1e-6
    # For late fusion: pre-train encoders then freeze before fusion training
    pretrain_epochs: int = 80
    gradient_clip: float = 1.0


@dataclass
class GAConfig:
    population_size: int = 50
    n_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.01
    tournament_size: int = 3
    sparsity_weight: float = 0.1      # fitness = Acc − λ·(selected/total)
    n_jobs: int = -1


@dataclass
class Config:
    eeg: EEGConfig = field(default_factory=EEGConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ga: GAConfig = field(default_factory=GAConfig)

    data_dir: str = "./data"
    output_dir: str = "./outputs"
    experiment_name: str = "multimodal_emotion_recognition"
    log_interval: int = 10


def get_config() -> Config:
    return Config()
