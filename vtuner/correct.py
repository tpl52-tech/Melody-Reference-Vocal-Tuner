"""Compute the corrected pitch target for the user's take.

Design choice (locked in v1): note-quantized + partial correction. For each
reference note we pull the *centre* of the user's pitch toward the target
centre by `strength`, while keeping the user's own within-note micro-pitch
(vibrato, expression) scaled by `preserve`.

To avoid the robotic auto-tune sound we (a) work on the *correction amount*
(how many semitones we push each frame) rather than snapping to a flat target,
and (b) smooth that amount over `smooth_ms` so it glides across note
transitions the way a real voice does instead of stepping instantly.

Knobs:
  strength  0..1  how hard note centres are pulled to target (1 = dead in tune,
                  lower = more of the singer's own tuning kept = less robotic)
  preserve  0..1  how much within-note wobble/vibrato is kept (1 = all of it)
  smooth_ms       glide time for corrections across boundaries (0 = hard steps)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .util import hz_to_midi, midi_to_hz
from .notes import Note


@dataclass
class Correction:
    corrected_f0: np.ndarray   # Hz per frame, NaN where no correction
    ratio: np.ndarray          # target/user pitch ratio per frame (1.0 = untouched)
    segments: list             # per-note dicts for the segment-based shifter


def compute_correction(
    user_f0: np.ndarray,
    notes: list[Note],
    register_offset: float,
    strength: float = 0.9,
    preserve: float = 1.0,
    transpose_semitones: float = 0.0,
    max_shift_semitones: float = 12.0,
    smooth_ms: float = 50.0,
    hop_seconds: float = 0.01,
) -> Correction:
    user_midi = hz_to_midi(user_f0)
    user_voiced = np.isfinite(user_f0) & (user_f0 > 0)
    n = len(user_f0)
    corrected_midi = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    segments = []

    for nt in notes:
        lo, hi = nt.start, min(nt.end, n)
        if hi <= lo:
            continue
        seg = user_midi[lo:hi]
        finite = np.isfinite(seg)
        if not finite.any():
            continue
        user_center = float(np.median(seg[finite]))
        target_center = nt.midi_center + register_offset
        # note correction is partial (strength); the key shift is applied in full
        corrected_center = (user_center + strength * (target_center - user_center)
                            + transpose_semitones)

        residual = seg - user_center            # NaN preserved where unvoiced
        corrected_midi[lo:hi] = corrected_center + preserve * residual
        covered[lo:hi] = True

        n_steps = corrected_center - user_center  # constant shift for this note
        segments.append({
            "start": lo, "end": hi,
            "user_center_midi": user_center,
            "target_center_midi": target_center,
            "n_steps": float(np.clip(n_steps, -max_shift_semitones, max_shift_semitones)),
        })

    # Work on the correction amount (semitones pushed), so smoothing glides the
    # push across boundaries while leaving the singer's own contour underneath.
    delta = corrected_midi - user_midi
    delta = np.where(np.isfinite(delta), delta, 0.0)
    # a key shift moves the whole vocal: transpose voiced frames that fall
    # outside any reference note too (they get the shift but no note pull)
    if transpose_semitones:
        delta[user_voiced & ~covered] = transpose_semitones
    delta = np.clip(delta, -max_shift_semitones, max_shift_semitones)
    sigma = (smooth_ms / 1000.0) / hop_seconds
    if sigma > 0.3:
        delta = gaussian_filter1d(delta, sigma, mode="nearest")

    ratio = np.power(2.0, delta / 12.0)
    ratio[~user_voiced] = 1.0

    corrected_midi_final = np.where(user_voiced, user_midi + delta, np.nan)
    corrected_f0 = midi_to_hz(corrected_midi_final)
    return Correction(corrected_f0=corrected_f0, ratio=ratio, segments=segments)
