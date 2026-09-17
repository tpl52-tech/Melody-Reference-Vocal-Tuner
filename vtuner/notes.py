"""Segment the reference f0 contour into discrete note events.

This is what makes the tuning *melody-referenced* rather than scale-based:
each note's target pitch is the median of the reference singer's own pitch
over that note (continuous, not snapped to equal temperament), so blue notes
and expressive intonation in the reference are preserved as targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util import hz_to_midi, nan_median_filter


@dataclass
class Note:
    start: int          # frame index (inclusive)
    end: int            # frame index (exclusive)
    midi_center: float  # median MIDI over the note

    def contains(self, i: int) -> bool:
        return self.start <= i < self.end


def segment_notes(
    f0: np.ndarray,
    hop_seconds: float,
    smooth_win: int = 7,
    split_semitones: float = 0.8,
    min_note_seconds: float = 0.09,
) -> list[Note]:
    """Split voiced runs where the pitch settles onto a new note, then drop
    or merge fragments shorter than `min_note_seconds`."""
    midi = hz_to_midi(f0)
    smooth = nan_median_filter(midi, smooth_win)
    voiced = np.isfinite(smooth)
    min_len = max(1, int(round(min_note_seconds / hop_seconds)))

    notes: list[Note] = []
    for start, end in _voiced_runs(voiced):
        notes.extend(_split_run(midi, smooth, start, end, split_semitones))

    notes = _merge_short(notes, midi, min_len)
    # Recompute centres from the raw (unsmoothed) contour for fidelity.
    for nt in notes:
        seg = midi[nt.start:nt.end]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            nt.midi_center = float(np.median(seg))
    return [n for n in notes if np.isfinite(n.midi_center)]


def target_midi_per_frame(notes: list[Note], n_frames: int) -> np.ndarray:
    out = np.full(n_frames, np.nan)
    for nt in notes:
        out[nt.start:nt.end] = nt.midi_center
    return out


def _voiced_runs(voiced: np.ndarray):
    runs = []
    i, n = 0, len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def _split_run(midi, smooth, start, end, split_semitones):
    """Greedy split: extend the current note while the smoothed pitch stays
    within `split_semitones` of the note's running median; otherwise cut."""
    notes = []
    seg_start = start
    running = [smooth[start]]
    for i in range(start + 1, end):
        med = np.median(running)
        if abs(smooth[i] - med) > split_semitones:
            notes.append(Note(seg_start, i, float(med)))
            seg_start = i
            running = [smooth[i]]
        else:
            running.append(smooth[i])
    notes.append(Note(seg_start, end, float(np.median(running))))
    return notes


def _merge_short(notes, midi, min_len):
    """Fold notes shorter than min_len into whichever neighbour is closer in
    pitch (or drop them if isolated)."""
    if not notes:
        return notes
    changed = True
    while changed and len(notes) > 1:
        changed = False
        for k, nt in enumerate(notes):
            if nt.end - nt.start >= min_len:
                continue
            prev_n = notes[k - 1] if k > 0 else None
            next_n = notes[k + 1] if k < len(notes) - 1 else None
            # merge only into a contiguous neighbour
            cand = []
            if prev_n and prev_n.end == nt.start:
                cand.append(("prev", abs(prev_n.midi_center - nt.midi_center), k - 1))
            if next_n and nt.end == next_n.start:
                cand.append(("next", abs(next_n.midi_center - nt.midi_center), k + 1))
            if not cand:
                notes.pop(k)  # isolated fragment, drop it
                changed = True
                break
            cand.sort(key=lambda c: c[1])
            _, _, j = cand[0]
            lo = min(nt.start, notes[j].start)
            hi = max(nt.end, notes[j].end)
            seg = midi[lo:hi]
            seg = seg[np.isfinite(seg)]
            center = float(np.median(seg)) if seg.size else nt.midi_center
            merged = Note(lo, hi, center)
            for idx in sorted([k, j], reverse=True):
                notes.pop(idx)
            notes.insert(min(k, j), merged)
            changed = True
            break
    return notes
