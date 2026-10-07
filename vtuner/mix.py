"""Warp-and-mix 'produced cover' export.

Places the tuned vocal onto the song's real instrumental:
  1. time-warp the vocal from the take's timeline onto the song's timeline
     using the DTW take->reference frame map (pitch-preserved, via RubberBand's
     time map -- so it slides words onto the beat without re-pitching);
  2. mix the warped vocal with the instrumental into one track.
"""
from __future__ import annotations

import numpy as np
import librosa
import pyrubberband as pyrb

from . import audio_io


def _time_map(take_to_ref, hop_seconds, sr, n_take_samples, n_ref_samples):
    """Strictly-increasing (source_sample, target_sample) pairs mapping the
    take timeline to the reference/song timeline for RubberBand."""
    spf = hop_seconds * sr  # samples per analysis frame
    pairs = [(0, 0)]
    last_s, last_t = 0, 0
    for t in range(len(take_to_ref)):
        s = int(round(t * spf))
        tgt = int(round(int(take_to_ref[t]) * spf))
        if s > last_s and tgt > last_t:
            pairs.append((s, tgt))
            last_s, last_t = s, tgt
    if n_take_samples > last_s and n_ref_samples > last_t:
        pairs.append((n_take_samples, n_ref_samples))
    return pairs


def warp_to_reference(vocal_y, sr, take_to_ref, hop_seconds, n_ref_samples):
    tm = _time_map(take_to_ref, hop_seconds, sr, len(vocal_y), n_ref_samples)
    if len(tm) < 2:
        return vocal_y
    return pyrb.timemap_stretch(vocal_y.astype(float), sr, tm)


def produce_cover(vocal_path, instrumental_path, take_to_ref, hop_seconds,
                  out_path, vocal_gain=1.0, inst_gain=0.7):
    """Warp the tuned vocal onto the song timeline and mix with the
    instrumental. Returns out_path."""
    voc, sr = audio_io.load_audio(vocal_path, sr=None, mono=True)
    inst, sri = audio_io.load_audio(instrumental_path, sr=None, mono=True)
    if sri != sr:
        inst = librosa.resample(inst, orig_sr=sri, target_sr=sr)

    n_ref = len(inst)
    warped = warp_to_reference(voc, sr, take_to_ref, hop_seconds, n_ref)
    warped = _fit(np.asarray(warped, dtype=float), n_ref)
    inst = _fit(inst.astype(float), n_ref)

    # Balance the backing to the vocal by RMS so it's audible in any section
    # (a soft intro vs a full chorus differ hugely in level); inst_gain is the
    # backing-to-vocal loudness ratio.
    vrms = float(np.sqrt(np.mean(warped ** 2) + 1e-12))
    irms = float(np.sqrt(np.mean(inst ** 2) + 1e-12))
    if irms > 0:
        inst = inst * (inst_gain * vrms / irms)

    mix = vocal_gain * warped + inst
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix = mix / peak
    audio_io.save_audio(out_path, mix.astype("float32"), sr)
    return out_path


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))
