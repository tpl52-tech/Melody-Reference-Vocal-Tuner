"""Two formant-preserving pitch-shift backends, so we can A/B them by ear.

WORLD (pyworld): decomposes into f0 + spectral envelope + aperiodicity and
resynthesises with a new f0. Formants live in the spectral envelope, which we
leave untouched, so preservation is automatic and the target contour can be
followed continuously.

RubberBand: only shifts by a constant ratio, so we segment the take by note
and shift each note by its constant amount, crossfading the joins. Vibrato is
not transferred (each note is a flat shift) -- a useful contrast to WORLD."""
from __future__ import annotations

import numpy as np

from .util import interp_nan


def world_shift(
    y: np.ndarray,
    sr: int,
    corrected_f0: np.ndarray,
    analysis_hop_seconds: float,
    frame_period_ms: float = 5.0,
) -> np.ndarray:
    """Resynthesise `y` so voiced frames follow `corrected_f0` (Hz, NaN where
    no correction is requested -> keep original pitch there)."""
    import pyworld as pw

    x = y.astype(np.float64)
    f0, t = pw.harvest(x, sr, frame_period=frame_period_ms)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)

    # Map the analysis-grid target onto WORLD's frame times (nearest frame).
    n_analysis = len(corrected_f0)
    idx = np.round(t / analysis_hop_seconds).astype(int)
    idx = np.clip(idx, 0, n_analysis - 1)
    target = corrected_f0[idx]

    new_f0 = f0.copy()
    voiced_world = f0 > 0
    have_target = np.isfinite(target) & (target > 0)
    take = voiced_world & have_target
    new_f0[take] = target[take]

    out = pw.synthesize(new_f0, sp, ap, sr, frame_period_ms)
    return _fit(out, len(y)).astype(np.float32)


def rubberband_shift(
    y: np.ndarray,
    sr: int,
    segments: list,
    analysis_hop_seconds: float,
    fade_ms: float = 8.0,
) -> np.ndarray:
    """Constant-ratio shift per note segment, crossfaded into a passthrough
    copy of the original take."""
    import pyrubberband as pyrb

    out = y.astype(np.float64).copy()
    fade = max(1, int(round(fade_ms / 1000.0 * sr)))

    for seg in segments:
        n_steps = seg["n_steps"]
        if abs(n_steps) < 0.05:
            continue
        s = int(round(seg["start"] * analysis_hop_seconds * sr))
        e = int(round(seg["end"] * analysis_hop_seconds * sr))
        s, e = max(0, s), min(len(y), e)
        if e - s < 2:
            continue
        ps, pe = max(0, s - fade), min(len(y), e + fade)
        piece = y[ps:pe].astype(np.float64)
        shifted = _fit(np.asarray(pyrb.pitch_shift(piece, sr, n_steps), dtype=np.float64), len(piece))
        core = shifted[s - ps: s - ps + (e - s)]
        _blend(out, s, e, core, fade)

    return out.astype(np.float32)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def _blend(out: np.ndarray, s: int, e: int, core: np.ndarray, fade: int):
    """Write `core` into out[s:e] with equal-power crossfades of length
    `fade` at each end (blending against the existing passthrough audio)."""
    length = e - s
    core = _fit(core, length)
    f = min(fade, length // 2)
    win = np.ones(length)
    if f > 0:
        ramp = np.sin(np.linspace(0, np.pi / 2, f)) ** 2
        win[:f] = ramp
        win[-f:] = ramp[::-1]
    out[s:e] = out[s:e] * (1 - win) + core * win
