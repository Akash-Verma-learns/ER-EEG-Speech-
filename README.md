ml_last_hopefully/
├── config.py                      # All hyperparameters as dataclasses
├── requirements.txt
├── main.py                        # CLI entry point — runs all 4 models + McNemar tests
├── preprocessing/
│   ├── eeg_preprocessor.py        # Butterworth BPF → DE features → 8×9 spatial grid
│   ├── speech_preprocessor.py     # Wiener filter → MFCC+CHROMA+MEL (180-D)
│   └── ga_selector.py             # GA with SVM fitness proxy, runs per-modality
├── models/
│   ├── on_lstm.py                 # ON-LSTM cell (cumax ordered-neuron gates)
│   ├── eeg_encoder.py             # DS-CNN 2D + ON-LSTM → Y_EEG ∈ ℝ¹²⁸
│   ├── speech_encoder.py          # Frozen Wav2Vec2 → chunked DS-CNN → Transformer → Y_Sp ∈ ℝ²⁵⁶
│   ├── early_fusion.py            # Shared 1D DS-CNN + self-attention → Z_early ∈ ℝ²⁵⁶
│   ├── late_fusion.py             # CrossModalAttention + learnable α gates → Z_late ∈ ℝ²⁵⁶
│   └── unimodal_heads.py          # EEGClassifier, SpeechClassifier wrappers
├── datasets/
│   └── data_loader.py             # 4 Dataset classes + DatasetBuilder (raw → .npy)
├── training/
│   ├── trainer.py                 # Trainer (Adam + CosineAnnealingLR)
│   └── cross_validation.py        # run_cv + run_cv_late_fusion (two-phase)
└── evaluation/
    └── statistical_tests.py       # McNemar's test + pairwise comparison table
