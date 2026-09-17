"""Audio load/save. Loading leans on librosa (which uses ffmpeg for
compressed formats such as the .m4a files phones produce)."""
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf


def load_audio(path: str, sr: int | None = None, mono: bool = True):
    """Load an audio file to a float32 mono array.

    sr=None keeps the file's native sample rate. Returns (y, sr)."""
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), int(sr_out)


def save_audio(path: str, y: np.ndarray, sr: int) -> None:
    y = np.asarray(y, dtype=np.float32)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:  # guard against clipping from resynthesis
        y = y / peak
    sf.write(path, y, sr)


def shift_audio(y: np.ndarray, sr: int, lag_seconds: float) -> np.ndarray:
    """Time-shift a signal by lag_seconds, keeping the original length.

    Positive lag delays the signal (pads the front); negative advances it
    (drops samples from the front). Used to remove the global sing-along
    offset so the take sits on the reference's timeline."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    shift = int(round(lag_seconds * sr))
    if shift == 0:
        return y.copy()
    out = np.zeros(n, dtype=np.float32)
    if shift > 0:
        if shift < n:
            out[shift:] = y[: n - shift]
    else:
        s = -shift
        if s < n:
            out[: n - s] = y[s:]
    return out
