"""
SEED-IV .mat file loader.

Directory layout expected:
  <root>/
    eeg_feature_smooth/
      1/   (session 1)
        1_20160518.mat
        2_20150915.mat
        ...
      2/
      3/
    eye_feature_smooth/
      1/
        1_20160518.mat
        ...
      2/
      3/

Each .mat file contains:
  de_LDS1 ... de_LDS24   — per-trial DE features, shape (62, 5, T) or (5, 62, T)
  eye_LDS1 ... eye_LDS24 — per-trial eye features, shape (n_eye, T) or (T, n_eye)
"""

import os
import re
import numpy as np
import scipy.io as sio
from typing import Dict, List, Optional, Tuple

# -----------------------------------------------------------------------
# Ground-truth trial labels per session (0=neutral, 1=sad, 2=fear, 3=happy)
# -----------------------------------------------------------------------
SESSION_LABELS: Dict[int, List[int]] = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}
N_TRIALS   = 24
N_SESSIONS = 3
EMOTIONS   = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}


def _normalise_eeg_feat(arr: np.ndarray) -> np.ndarray:
    """Ensure shape (T, 310) = (n_windows, 62_channels × 5_bands)."""
    if arr.ndim == 2:
        # Already flat: check orientation
        if arr.shape[1] == 310:
            return arr.astype(np.float32)
        if arr.shape[0] == 310:
            return arr.T.astype(np.float32)
    if arr.ndim == 3:
        # (62, 5, T) or (5, 62, T)
        if arr.shape[0] == 62 and arr.shape[1] == 5:
            T = arr.shape[2]
            return arr.reshape(310, T).T.astype(np.float32)  # (T, 310)
        if arr.shape[0] == 5 and arr.shape[1] == 62:
            T = arr.shape[2]
            return arr.transpose(2, 1, 0).reshape(T, 310).astype(np.float32)
        # Fallback: flatten last two dims
        T = arr.shape[-1]
        return arr.reshape(-1, T).T.astype(np.float32)
    raise ValueError(f"Unexpected EEG feature shape: {arr.shape}")


def _normalise_eye_feat(arr: np.ndarray) -> np.ndarray:
    """Ensure shape (T, n_eye_features)."""
    if arr.ndim == 2:
        # Typically (n_features, T) from MATLAB; flip if needed
        if arr.shape[0] < arr.shape[1]:
            return arr.T.astype(np.float32)
        return arr.astype(np.float32)
    raise ValueError(f"Unexpected eye feature shape: {arr.shape}")


def _find_trial_keys(data: dict, prefix: str, n_trials: int) -> List[Optional[str]]:
    """Return a list of key names (or None) for trials 1..n_trials."""
    keys = []
    for i in range(1, n_trials + 1):
        candidates = [
            f"{prefix}LDS{i}",
            f"{prefix}movingAve{i}",
            f"{prefix}{i}",
        ]
        found = next((c for c in candidates if c in data), None)
        if found is None:
            # Regex fallback
            pattern = re.compile(rf"^{re.escape(prefix)}.*{i}$")
            found = next((k for k in data if pattern.match(k)), None)
        keys.append(found)
    return keys


def load_subject_session(
    eeg_mat_path: str,
    eye_mat_path: str,
    session_id: int,
    n_trials: int = N_TRIALS,
) -> Dict:
    """
    Load one subject × session.

    Returns
    -------
    dict with keys:
      "eeg"    : list[np.ndarray], each (T_i, 310)
      "eye"    : list[np.ndarray], each (T_i, n_eye_features) or None if missing
      "labels" : list[int], trial-level emotion label (0-3)
      "session": int
    """
    eeg_data = sio.loadmat(eeg_mat_path, squeeze_me=True, struct_as_record=False)
    eye_data = None
    if eye_mat_path and os.path.exists(eye_mat_path):
        eye_data = sio.loadmat(eye_mat_path, squeeze_me=True, struct_as_record=False)

    trial_labels = SESSION_LABELS[session_id]
    eeg_keys = _find_trial_keys(eeg_data, "de_", n_trials)
    eye_keys  = _find_trial_keys(eye_data, "eye_", n_trials) if eye_data else [None] * n_trials

    eeg_list, eye_list, labels = [], [], []
    for i in range(n_trials):
        if eeg_keys[i] is None:
            continue
        raw_eeg = np.array(eeg_data[eeg_keys[i]], dtype=float)
        eeg_list.append(_normalise_eeg_feat(raw_eeg))

        if eye_keys[i] is not None and eye_data is not None:
            raw_eye = np.array(eye_data[eye_keys[i]], dtype=float)
            eye_list.append(_normalise_eye_feat(raw_eye))
        else:
            eye_list.append(None)

        labels.append(trial_labels[i])

    return {"eeg": eeg_list, "eye": eye_list, "labels": labels, "session": session_id}


def discover_subjects(root_dir: str) -> List[str]:
    """Return sorted list of subject filenames found in session 1."""
    session1_dir = os.path.join(root_dir, "eeg_feature_smooth", "1")
    if not os.path.isdir(session1_dir):
        raise FileNotFoundError(f"Expected eeg_feature_smooth/1/ under {root_dir}")
    return sorted(f for f in os.listdir(session1_dir) if f.endswith(".mat"))


def match_eye_file(
    eeg_filename: str,
    eye_session_dir: str,
) -> Optional[str]:
    """Match an EEG .mat filename to the corresponding eye .mat file."""
    # EEG files are like "1_20160518.mat"; eye files should match
    candidate = os.path.join(eye_session_dir, eeg_filename)
    if os.path.exists(candidate):
        return candidate
    # Fallback: partial match on subject prefix
    subject_prefix = eeg_filename.split("_")[0]
    for f in os.listdir(eye_session_dir):
        if f.startswith(subject_prefix + "_") and f.endswith(".mat"):
            return os.path.join(eye_session_dir, f)
    return None


def load_all_data(
    root_dir: str,
    subject_files: Optional[List[str]] = None,
    sessions: Optional[List[int]] = None,
    verbose: bool = True,
) -> List[Dict]:
    """
    Load all subjects and sessions.

    Returns
    -------
    records : list of dicts, each with keys
      "subject"  : str (filename stem)
      "session"  : int
      "eeg"      : list[np.ndarray (T, 310)]
      "eye"      : list[np.ndarray (T, n_eye) | None]
      "labels"   : list[int]
    """
    if subject_files is None:
        subject_files = discover_subjects(root_dir)
    if sessions is None:
        sessions = list(range(1, N_SESSIONS + 1))

    records = []
    for sess in sessions:
        eeg_dir = os.path.join(root_dir, "eeg_feature_smooth", str(sess))
        eye_dir = os.path.join(root_dir, "eye_feature_smooth", str(sess))

        for fname in subject_files:
            eeg_path = os.path.join(eeg_dir, fname)
            if not os.path.exists(eeg_path):
                if verbose:
                    print(f"  [skip] {eeg_path} not found")
                continue
            eye_path = match_eye_file(fname, eye_dir) if os.path.isdir(eye_dir) else None

            if verbose:
                print(f"  Loading subject={fname} session={sess}  eye={'yes' if eye_path else 'no'}")

            rec = load_subject_session(eeg_path, eye_path, sess)
            rec["subject"] = os.path.splitext(fname)[0]
            records.append(rec)

    return records
