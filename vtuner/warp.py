"""DTW time-alignment for independently-recorded takes.

v1 assumed the take stayed in sync with the reference (offset-only). When the
take was NOT recorded against this exact reference it drifts -- the offset
slides over the clip -- and a single shift can't fix it. DTW aligns them
frame-by-frame instead.

We do NOT time-stretch the user's audio. DTW only answers "which reference
note is the singer on at each moment of their take", so we can retune to the
right note while keeping the user's own timing and performance. Chroma (CENS)
is the matching feature: register-invariant and defined on every frame, so it
handles two different voices singing the same melody."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from .notes import Note


@dataclass
class Warp:
    take_to_ref: np.ndarray   # per take frame -> corresponding ref frame index
    confidence: np.ndarray    # per take frame chroma cosine similarity [0, 1]
    matched_fraction: float   # fraction of take frames confidently matched


def _chroma(y: np.ndarray, sr: int, hop_seconds: float) -> np.ndarray:
    hop = max(1, int(round(sr * hop_seconds)))
    return librosa.feature.chroma_cens(y=y.astype(float), sr=sr, hop_length=hop)


def align(
    take_y: np.ndarray,
    ref_y: np.ndarray,
    sr: int,
    hop_seconds: float = 0.01,
    dtw_hop_seconds: float = 0.05,
    conf_thr: float = 0.5,
) -> Warp:
    """DTW-align take to reference. DTW runs on a coarse chroma grid
    (`dtw_hop_seconds`) so the cost matrix stays small even for full songs,
    then the frame map is upsampled to the fine analysis grid
    (`hop_seconds`) used everywhere else."""
    Ct = _chroma(take_y, sr, dtw_hop_seconds)
    Cr = _chroma(ref_y, sr, dtw_hop_seconds)
    nT, nR = Ct.shape[1], Cr.shape[1]

    _, wp = librosa.sequence.dtw(X=Ct, Y=Cr, metric="cosine")
    wp = wp[::-1]  # DTW returns end->start; flip to start->end

    buckets: list[list[int]] = [[] for _ in range(nT)]
    for i_t, j_r in wp:
        if 0 <= i_t < nT:
            buckets[i_t].append(int(j_r))
    coarse = np.full(nT, -1, dtype=int)
    for t, js in enumerate(buckets):
        if js:
            coarse[t] = int(np.median(js))
    _fill_gaps(coarse)
    coarse = np.clip(coarse, 0, nR - 1)

    a = Ct / (np.linalg.norm(Ct, axis=0, keepdims=True) + 1e-9)
    b = Cr / (np.linalg.norm(Cr, axis=0, keepdims=True) + 1e-9)
    conf_coarse = np.clip(np.sum(a * b[:, coarse], axis=0), 0.0, 1.0)

    # upsample the coarse take->ref frame map onto the fine analysis grid
    ratio = dtw_hop_seconds / hop_seconds          # fine frames per coarse frame
    n_take_fine = int(round(len(take_y) / (hop_seconds * sr)))
    n_ref_fine = int(round(len(ref_y) / (hop_seconds * sr)))
    coarse_pos = np.arange(nT)
    at = np.arange(n_take_fine) / ratio            # coarse coordinate per fine frame
    ref_fine = np.interp(at, coarse_pos, coarse.astype(float)) * ratio
    take_to_ref = np.clip(np.round(ref_fine).astype(int), 0, max(0, n_ref_fine - 1))
    confidence = np.interp(at, coarse_pos, conf_coarse)

    matched = float(np.mean(confidence >= conf_thr)) if confidence.size else 0.0
    return Warp(take_to_ref=take_to_ref, confidence=confidence, matched_fraction=matched)


def build_take_notes(
    warp: Warp,
    ref_notes: list[Note],
    n_ref: int,
    n_take: int,
    conf_thr: float = 0.5,
    min_note_seconds: float = 0.05,
    hop_seconds: float = 0.01,
):
    """Project reference notes onto the take's timeline via the warp.

    Returns (take_notes, take_target_midi): note segments on the TAKE grid
    whose target pitch is the corresponding reference note's centre, plus a
    per-take-frame target-MIDI array (NaN where unmatched / low confidence)."""
    ref_note_id = np.full(n_ref, -1, dtype=int)
    centers = np.array([nt.midi_center for nt in ref_notes], dtype=float)
    for k, nt in enumerate(ref_notes):
        ref_note_id[nt.start:min(nt.end, n_ref)] = k

    take_note_id = np.full(n_take, -1, dtype=int)
    take_target = np.full(n_take, np.nan)
    m = min(n_take, len(warp.take_to_ref), len(warp.confidence))
    for t in range(m):
        if warp.confidence[t] < conf_thr:
            continue
        r = min(int(warp.take_to_ref[t]), n_ref - 1)
        k = ref_note_id[r]
        if k >= 0:
            take_note_id[t] = k
            take_target[t] = centers[k]

    min_len = max(1, int(round(min_note_seconds / hop_seconds)))
    take_notes: list[Note] = []
    t = 0
    while t < n_take:
        k = take_note_id[t]
        if k < 0:
            t += 1
            continue
        s = t
        while t < n_take and take_note_id[t] == k:
            t += 1
        if t - s >= min_len:
            take_notes.append(Note(s, t, float(centers[k])))
    return take_notes, take_target


def _fill_gaps(arr: np.ndarray) -> None:
    good = np.where(arr >= 0)[0]
    if good.size == 0:
        return
    idx = np.arange(len(arr))
    arr[arr < 0] = np.interp(idx[arr < 0], good, arr[good]).round().astype(int)
