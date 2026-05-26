"""
RAVDESS audio-speech loader.

File naming convention:
  03-01-{emotion:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}.wav
  where emotion: 01=neutral, 02=calm, 03=happy, 04=sad,
                 05=angry,   06=fearful, 07=disgust, 08=surprised

4-class mapping to align with SEED-IV:
  RAVDESS 01 (neutral)  → class 0
  RAVDESS 04 (sad)      → class 1
  RAVDESS 06 (fearful)  → class 2
  RAVDESS 03 (happy)    → class 3
  All others are discarded.

Expected directory layout:
  ravdess/
    Actor_01/
      03-01-01-01-01-01-01.wav
      ...
    Actor_02/
    ...
    Actor_24/
"""

import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple

try:
    import soundfile as sf
    _SF = True
except ImportError:
    _SF = False

try:
    import librosa
    _LIBROSA = True
except ImportError:
    _LIBROSA = False

# RAVDESS emotion code → SEED-IV 4-class label
RAVDESS_TO_SEED4 = {
    1: 0,   # neutral  → neutral
    4: 1,   # sad      → sad
    6: 2,   # fearful  → fear
    3: 3,   # happy    → happy
}

# Human-readable
RAVDESS_EMOTIONS = {
    1: "neutral", 2: "calm",    3: "happy",    4: "sad",
    5: "angry",   6: "fearful", 7: "disgust",  8: "surprised",
}


def _parse_filename(fname: str) -> Optional[Dict]:
    """Return parsed RAVDESS metadata dict or None if not a valid speech file."""
    m = re.match(
        r"(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav",
        os.path.basename(fname),
    )
    if m is None:
        return None
    modality, vocal_channel, emotion, intensity, statement, repetition, actor = (
        int(g) for g in m.groups()
    )
    return {
        "modality": modality,       # 01=AV, 02=video, 03=audio
        "vocal_channel": vocal_channel,  # 01=speech, 02=song
        "emotion": emotion,
        "intensity": intensity,     # 01=normal, 02=strong
        "statement": statement,
        "repetition": repetition,
        "actor": actor,
        "path": fname,
    }


def _load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample to target_sr. Returns float32 mono array."""
    if _SF:
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != target_sr and _LIBROSA:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        elif sr != target_sr:
            raise RuntimeError(
                f"librosa required for resampling {sr}→{target_sr}. "
                "Install with: pip install librosa"
            )
    elif _LIBROSA:
        data, _ = librosa.load(path, sr=target_sr, mono=True)
    else:
        raise ImportError("Either soundfile or librosa must be installed.")

    # Normalise to [-1, 1]
    peak = np.abs(data).max()
    if peak > 0:
        data = data / peak
    return data.astype(np.float32)


def discover_files(
    root_dir: str,
    emotion_map: Optional[Dict[int, int]] = None,
    audio_only: bool = True,       # modality == 03 (audio-only)
    speech_only: bool = True,      # vocal_channel == 01 (speech, not song)
    intensity: str = "both",       # "normal", "strong", "both"
) -> List[Dict]:
    """
    Walk RAVDESS root directory and return metadata dicts for matching files.
    Filters to the 4-class subset unless emotion_map overrides.
    """
    if emotion_map is None:
        emotion_map = RAVDESS_TO_SEED4

    records = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"RAVDESS root not found: {root_dir}")

    for actor_dir in sorted(os.listdir(root_dir)):
        actor_path = os.path.join(root_dir, actor_dir)
        if not os.path.isdir(actor_path):
            continue
        for fname in sorted(os.listdir(actor_path)):
            full = os.path.join(actor_path, fname)
            meta = _parse_filename(full)
            if meta is None:
                continue
            if audio_only and meta["modality"] != 3:
                continue
            if speech_only and meta["vocal_channel"] != 1:
                continue
            if meta["emotion"] not in emotion_map:
                continue
            if intensity == "normal" and meta["intensity"] != 1:
                continue
            if intensity == "strong" and meta["intensity"] != 2:
                continue
            meta["label"] = emotion_map[meta["emotion"]]
            records.append(meta)

    return records


def load_dataset(
    root_dir: str,
    target_sr: int = 16000,
    emotion_map: Optional[Dict[int, int]] = None,
    intensity: str = "both",
    verbose: bool = True,
) -> Tuple[List[np.ndarray], np.ndarray, List[Dict]]:
    """
    Load all filtered RAVDESS speech recordings.

    Returns
    -------
    waveforms : list of (n_samples,) float32 arrays at target_sr
    labels    : (N,) int64 array — SEED-IV 4-class labels
    metadata  : list of metadata dicts
    """
    if emotion_map is None:
        emotion_map = RAVDESS_TO_SEED4

    records = discover_files(root_dir, emotion_map, intensity=intensity)
    if verbose:
        print(f"RAVDESS: {len(records)} files found in {root_dir}")
        from collections import Counter
        label_counts = Counter(r["label"] for r in records)
        print(f"  Label distribution: {dict(sorted(label_counts.items()))}")

    waveforms, labels = [], []
    for rec in records:
        wav = _load_audio(rec["path"], target_sr)
        waveforms.append(wav)
        labels.append(rec["label"])

    return waveforms, np.array(labels, dtype=np.int64), records


def pad_collate(waveforms: List[np.ndarray]) -> np.ndarray:
    """Pad waveform list to equal length and stack → (N, max_len)."""
    max_len = max(w.shape[0] for w in waveforms)
    out = np.zeros((len(waveforms), max_len), dtype=np.float32)
    for i, w in enumerate(waveforms):
        out[i, :len(w)] = w
    return out
