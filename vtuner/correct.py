"""Compute the corrected pitch target for the user's take.

Design choice (locked in v1): note-quantized + partial correction. For each
reference note we pull the *centre* of the user's pitch toward the target
centre by `strength`, while keeping the user's own within-note micro-pitch
(vibrato, expression) scaled by `preserve`. This lands the note clearly in
tune without the robotic feel of stamping the reference's raw contour on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util import hz_to_midi, midi_to_hz, nan_median_filter
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
    max_shift_semitones: float = 12.0,
    ratio_smooth_win: int = 5,
) -> Correction:
    user_midi = hz_to_midi(user_f0)
    n = len(user_f0)
    corrected_midi = np.full(n, np.nan)
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
        corrected_center = user_center + strength * (target_center - user_center)

        residual = seg - user_center            # NaN preserved where unvoiced
        corrected_midi[lo:hi] = corrected_center + preserve * residual

        n_steps = corrected_center - user_center  # constant shift for this note
        segments.append({
            "start": lo, "end": hi,
            "user_center_midi": user_center,
            "target_center_midi": target_center,
            "n_steps": float(np.clip(n_steps, -max_shift_semitones, max_shift_semitones)),
        })

    # ratio = corrected / user, clipped to +/- max_shift and smoothed; frames
    # with no correction (or unvoiced user) pass through untouched.
    diff = corrected_midi - user_midi
    diff = np.clip(diff, -max_shift_semitones, max_shift_semitones)
    ratio = np.power(2.0, diff / 12.0)
    ratio[~np.isfinite(ratio)] = 1.0
    ratio = nan_median_filter(ratio, ratio_smooth_win)
    ratio[~np.isfinite(ratio)] = 1.0

    corrected_f0 = midi_to_hz(corrected_midi)
    return Correction(corrected_f0=corrected_f0, ratio=ratio, segments=segments)
