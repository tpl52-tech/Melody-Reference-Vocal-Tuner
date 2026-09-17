"""Global sing-along offset estimation.

The v1 assumption (headphones, tight timing) is that the user's take differs
from the reference by a single constant latency, not local drift. We estimate
that latency by cross-correlating loudness envelopes (robust to the two
singers having different pitch/register), then report a residual-drift figure
so we can tell whether that assumption actually held for a given take."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa


@dataclass
class Alignment:
    lag_seconds: float       # +ve => user lags the reference
    confidence: float        # peak normalised cross-correlation [0, 1]
    residual_drift_ms: float # spread of local offsets after global shift


def _envelope(y: np.ndarray, sr: int, fps: float = 100.0) -> np.ndarray:
    hop = max(1, int(round(sr / fps)))
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    rms = rms - rms.mean()
    norm = np.linalg.norm(rms)
    return rms / norm if norm > 0 else rms


def estimate_offset(
    user: np.ndarray,
    ref: np.ndarray,
    sr: int,
    max_lag_s: float = 0.75,
    fps: float = 100.0,
) -> Alignment:
    eu = _envelope(user, sr, fps)
    er = _envelope(ref, sr, fps)
    n = min(len(eu), len(er))
    eu, er = eu[:n], er[:n]

    max_lag = int(round(max_lag_s * fps))
    xcorr = np.correlate(eu, er, mode="full")
    mid = len(er) - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    window = xcorr[lo:hi]
    best = int(np.argmax(window))
    lag_frames = best - max_lag  # +ve => user delayed vs ref
    confidence = float(window[best] / (np.linalg.norm(eu) * np.linalg.norm(er) + 1e-9))

    residual = _residual_drift(eu, er, lag_frames, fps)
    return Alignment(
        lag_seconds=lag_frames / fps,
        confidence=max(0.0, min(1.0, confidence)),
        residual_drift_ms=residual,
    )


def _residual_drift(eu, er, lag_frames, fps, block_s=0.5):
    """After removing the global lag, measure how much the best local offset
    still wanders block-to-block. Large spread => local drift => the
    offset-only assumption is shaky and per-note alignment may be needed."""
    if lag_frames >= 0:
        a, b = eu[lag_frames:], er[: len(er) - lag_frames]
    else:
        a, b = eu[: len(eu) + lag_frames], er[-lag_frames:]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    block = max(1, int(round(block_s * fps)))
    search = max(1, int(round(0.6 * fps)))  # +/- 600 ms local search (wide enough to see real drift)
    local = []
    for s in range(0, n - block, block):
        aw, bw = a[s:s + block], b[s:s + block]
        cc = np.correlate(aw, bw, mode="full")
        c = len(bw) - 1
        seg = cc[max(0, c - search):c + search + 1]
        if seg.size:
            local.append((int(np.argmax(seg)) - min(search, c)))
    return float(np.std(local) / fps * 1000.0) if local else 0.0
