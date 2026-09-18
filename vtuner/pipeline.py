"""End-to-end melody-reference retune pipeline.

    reference vocal ─CREPE─▶ pitch ─segment─▶ note targets ─┐
                                                            ├─▶ register-fold + note-quantized
    your take ─align─▶ ─CREPE─▶ pitch ──────────────────────┘   partial correction
                          │                                             │
                          │                              WORLD ◀────────┴────────▶ RubberBand
                          └─ align = 'offset' (global shift, v1) or 'dtw' (frame-by-frame
                             warp for independently-recorded/drifting takes)

The reference pitch track is cached on disk (it never changes between runs),
which keeps a 30 s clip under the latency budget on CPU."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
import librosa

from . import audio_io
from . import isolate as isolatemod
from . import warp as warpmod
from .pitch import track_pitch, PitchTrack
from .align import estimate_offset, Alignment
from .notes import segment_notes, target_midi_per_frame
from .register import register_offset_semitones
from .correct import compute_correction
from .shift import world_shift, rubberband_shift
from .util import hz_to_midi, midi_to_hz, cents_error


@dataclass
class Result:
    alignment: Alignment
    align_mode: str
    register_offset: int
    transpose: int
    n_notes: int
    warp_matched_fraction: float = float("nan")
    metrics: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def _cache_pitch(path, y, sr, params, cache_dir, prefix="ref"):
    """Track pitch, caching to disk keyed on file mtime + params. The audio
    (reference or take) never changes between runs, so re-renders with new
    tuning knobs skip the slow CREPE pass entirely."""
    os.makedirs(cache_dir, exist_ok=True)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(path)}|{mtime}|{sorted(params.items())}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    fn = os.path.join(cache_dir, f"{prefix}_{h}.npz")
    if os.path.exists(fn):
        d = np.load(fn)
        return PitchTrack(d["f0"], d["voiced"], d["periodicity"], d["times"],
                          float(d["hop_seconds"])), True
    pt = track_pitch(y, sr, **params)
    np.savez(fn, f0=pt.f0, voiced=pt.voiced, periodicity=pt.periodicity,
             times=pt.times, hop_seconds=pt.hop_seconds)
    return pt, False


def run(
    reference_path: str,
    user_path: str,
    out_dir: str = "output",
    backend: str = "both",           # 'world' | 'rubberband' | 'both'
    align_mode: str = "dtw",         # 'dtw' (drift-tolerant) | 'offset' (v1)
    isolate_reference: bool = False, # Demucs-isolate the reference from a full mix
    isolate_take: bool = False,      # Demucs-isolate the take (rarely needed)
    transpose: int = 0,              # key shift in semitones (on top of octave-match)
    strength: float = 0.6,      # tuned by ear on the first real take
    preserve: float = 1.0,
    smooth_ms: float = 95.0,
    model: str = "full",
    hop_seconds: float = 0.01,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    max_shift_semitones: float = 12.0,
    measure_output: bool = False,
    cache_dir: str | None = None,
) -> Result:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(out_dir, ".cache")
    timings = {}
    pitch_params = dict(hop_seconds=hop_seconds, fmin=fmin, fmax=fmax, model=model)

    # Phase 4: optionally pull the vocal out of a full mix first.
    if isolate_reference:
        t0 = time.time()
        reference_path, hit = isolatemod.isolate_vocal(reference_path, cache_dir)
        timings["isolate_reference"] = time.time() - t0
        timings["isolate_reference_cache_hit"] = hit
    if isolate_take:
        t0 = time.time()
        user_path, hit = isolatemod.isolate_vocal(user_path, cache_dir)
        timings["isolate_take"] = time.time() - t0
        timings["isolate_take_cache_hit"] = hit

    t0 = time.time()
    user_y, su = audio_io.load_audio(user_path)
    ref_y, sr_r = audio_io.load_audio(reference_path)
    ref_for_env = librosa.resample(ref_y, orig_sr=sr_r, target_sr=su) if sr_r != su else ref_y
    timings["load"] = time.time() - t0

    # Global offset is always estimated for reporting (and used in offset mode).
    t0 = time.time()
    align = estimate_offset(user_y, ref_for_env, su)
    timings["align"] = time.time() - t0

    # Reference pitch (cached) + reference notes on the reference grid.
    t0 = time.time()
    ref_pt, cached = _cache_pitch(reference_path, ref_y, sr_r, pitch_params, cache_dir, "ref")
    timings["pitch_reference"] = time.time() - t0
    timings["reference_cache_hit"] = cached
    ref_notes_full = segment_notes(ref_pt.f0, hop_seconds)

    if align_mode == "dtw":
        shift_src = user_y  # keep the take's own timing; DTW only maps targets
        t0 = time.time()
        user_pt, user_cached = _cache_pitch(user_path, shift_src, su, pitch_params, cache_dir, "user")
        timings["pitch_user"] = time.time() - t0
        timings["user_cache_hit"] = user_cached

        t0 = time.time()
        warp = warpmod.align(shift_src, ref_for_env, su, hop_seconds=hop_seconds)
        take_notes, target_midi = warpmod.build_take_notes(
            warp, ref_notes_full, len(ref_pt.f0), len(user_pt.f0),
            hop_seconds=hop_seconds,
        )
        timings["warp"] = time.time() - t0
        warp_matched = warp.matched_fraction
        user_f0 = user_pt.f0
        notes = take_notes
    else:  # offset (v1): shift the take onto the reference timeline
        shift_src = audio_io.shift_audio(user_y, su, -align.lag_seconds)
        t0 = time.time()
        user_pt = track_pitch(shift_src, su, **pitch_params)
        timings["pitch_user"] = time.time() - t0
        n = min(len(user_pt.f0), len(ref_pt.f0))
        user_f0 = user_pt.f0[:n]
        notes = [nt for nt in ref_notes_full if nt.start < n]
        target_midi = target_midi_per_frame(notes, len(user_f0))
        warp_matched = float("nan")

    reg = register_offset_semitones(hz_to_midi(user_f0), target_midi)
    total_offset = reg + int(transpose)  # octave-match + user-chosen key shift

    corr = compute_correction(
        user_f0, notes, reg, strength=strength, preserve=preserve,
        transpose_semitones=int(transpose),
        max_shift_semitones=max_shift_semitones, smooth_ms=smooth_ms,
        hop_seconds=hop_seconds,
    )

    # Render.
    stem = os.path.splitext(os.path.basename(user_path))[0]
    outputs = {}
    raw_path = os.path.join(out_dir, f"{stem}_raw.wav")
    audio_io.save_audio(raw_path, shift_src, su)
    outputs["raw"] = raw_path

    if backend in ("world", "both"):
        t0 = time.time()
        w = world_shift(shift_src, su, corr.corrected_f0, hop_seconds)
        timings["render_world"] = time.time() - t0
        wp = os.path.join(out_dir, f"{stem}_world.wav")
        audio_io.save_audio(wp, w, su)
        outputs["world"] = wp
    if backend in ("rubberband", "both"):
        try:
            t0 = time.time()
            rb = rubberband_shift(shift_src, su, corr.segments, hop_seconds)
            timings["render_rubberband"] = time.time() - t0
            rp = os.path.join(out_dir, f"{stem}_rubberband.wav")
            audio_io.save_audio(rp, rb, su)
            outputs["rubberband"] = rp
        except Exception as exc:
            outputs["rubberband_error"] = str(exc)

    # Metrics: pitch error to the (register-matched, transposed) reference target.
    target_f0 = midi_to_hz(target_midi + total_offset)
    before = cents_error(user_f0, target_f0)
    predicted = cents_error(corr.corrected_f0, target_f0)
    metrics = {
        "mean_abs_cents_before": _mean_abs(before),
        "mean_abs_cents_after_predicted": _mean_abs(predicted),
        "frames_in_note": int(np.isfinite(before).sum()),
    }
    if measure_output and "world" in outputs:
        wy, _ = audio_io.load_audio(outputs["world"], sr=su)
        wpt = track_pitch(wy, su, **pitch_params)
        m = min(len(wpt.f0), len(target_f0))
        metrics["mean_abs_cents_after_world_measured"] = _mean_abs(
            cents_error(wpt.f0[:m], target_f0[:m])
        )

    return Result(
        alignment=align, align_mode=align_mode, register_offset=reg,
        transpose=int(transpose), n_notes=len(notes),
        warp_matched_fraction=warp_matched,
        metrics=metrics, outputs=outputs, timings=timings,
    )


def _mean_abs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else float("nan")
