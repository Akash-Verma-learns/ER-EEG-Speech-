How to run

pip install -r requirements.txt

1. Build processed data (implement DatasetBuilder.build_subject() calls for your dataset)
2. Run all four models with 5-fold CV:
python main.py --data_dir ./data --output_dir ./outputs --n_classes 4

Key architectural choices implemented
Component	Implementation
ON-LSTM	cumax = cumsum(softmax(x)) master gates enforcing ordered-neuron constraint
Speech encoder	Wav2Vec2 hidden states chunked → DS-CNN per chunk → Transformer across chunks
Late fusion phase 1	Encoders trained independently with their own heads
Late fusion phase 2	Encoders frozen, only CrossModalAttention + GatedModalityWeighting trained
α gates	Learnable scalars via sigmoid(log_alpha_eeg/sp) — inspectable at inference for valence/arousal analysis
GA fitness	Acc − 0.1 × (n_selected / n_total), parallel SVM 3-fold CV per individual
