"""Small shared helpers: pitch <-> MIDI conversions and NaN-safe smoothing."""
from __future__ import annotations

import numpy as np

A4_HZ = 440.0
A4_MIDI = 69.0


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    """Hz -> MIDI note number. Non-positive / NaN Hz map to NaN."""
    f0 = np.asarray(f0, dtype=float)
    out = np.full(f0.shape, np.nan)
    good = np.isfinite(f0) & (f0 > 0)
    out[good] = A4_MIDI + 12.0 * np.log2(f0[good] / A4_HZ)
    return out


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """MIDI note number -> Hz. NaN MIDI maps to NaN."""
    midi = np.asarray(midi, dtype=float)
    out = np.full(midi.shape, np.nan)
    good = np.isfinite(midi)
    out[good] = A4_HZ * 2.0 ** ((midi[good] - A4_MIDI) / 12.0)
    return out


def nan_median_filter(x: np.ndarray, win: int) -> np.ndarray:
    """Sliding-window median that ignores NaNs. Preserves length; keeps NaN
    only where the whole window is NaN."""
    x = np.asarray(x, dtype=float)
    if win <= 1:
        return x.copy()
    half = win // 2
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        w = x[lo:hi]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def interp_nan(x: np.ndarray) -> np.ndarray:
    """Linearly interpolate over interior NaNs; edge NaNs are held to the
    nearest finite value. Returns unchanged if all-NaN."""
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if not good.any():
        return x
    idx = np.arange(len(x))
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def cents_error(f0: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Signed pitch error in cents at frames where both are voiced; NaN
    elsewhere."""
    f0 = np.asarray(f0, dtype=float)
    target = np.asarray(target, dtype=float)
    out = np.full(f0.shape, np.nan)
    good = np.isfinite(f0) & (f0 > 0) & np.isfinite(target) & (target > 0)
    out[good] = 1200.0 * np.log2(f0[good] / target[good])
    return out
