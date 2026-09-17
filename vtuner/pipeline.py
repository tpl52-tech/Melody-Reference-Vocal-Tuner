"""End-to-end melody-reference retune pipeline.

    reference vocal  --CREPE-->  pitch  --segment-->  note targets
    user take        --align--> --CREPE--> pitch
                          |                    |
                          +---> register-fold + note-quantized partial
                                correction ---> WORLD & RubberBand renders

The reference pitch track is cached on disk (it never changes between runs),
which is what keeps a 30 s clip under the latency budget on CPU."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field

import numpy as np
import librosa

from . import audio_io
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
    register_offset: int
    n_notes: int
    metrics: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


def _cache_reference(path, y, sr, params, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(path)}|{mtime}|{sorted(params.items())}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    fn = os.path.join(cache_dir, f"ref_{h}.npz")
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
    strength: float = 0.9,
    preserve: float = 1.0,
    model: str = "full",
    hop_seconds: float = 0.01,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    max_shift_semitones: float = 12.0,
    measure_output: bool = False,    # re-track renders to verify empirically
    cache_dir: str | None = None,
) -> Result:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = cache_dir or os.path.join(out_dir, ".cache")
    timings = {}
    pitch_params = dict(hop_seconds=hop_seconds, fmin=fmin, fmax=fmax, model=model)

    t0 = time.time()
    user_y, su = audio_io.load_audio(user_path)
    ref_y, sr_r = audio_io.load_audio(reference_path)
    ref_for_env = librosa.resample(ref_y, orig_sr=sr_r, target_sr=su) if sr_r != su else ref_y
    timings["load"] = time.time() - t0

    # 1. Global sing-along offset, then put the take on the reference timeline.
    t0 = time.time()
    align = estimate_offset(user_y, ref_for_env, su)
    user_aligned = audio_io.shift_audio(user_y, su, -align.lag_seconds)
    timings["align"] = time.time() - t0

    # 2. Pitch tracking (reference cached).
    t0 = time.time()
    ref_pt, cached = _cache_reference(reference_path, ref_y, sr_r, pitch_params, cache_dir)
    timings["pitch_reference"] = time.time() - t0
    timings["reference_cache_hit"] = cached

    t0 = time.time()
    user_pt = track_pitch(user_aligned, su, **pitch_params)
    timings["pitch_user"] = time.time() - t0

    n = min(len(user_pt.f0), len(ref_pt.f0))
    user_f0 = user_pt.f0[:n]
    ref_f0 = ref_pt.f0[:n]

    # 3. Reference notes + register fold.
    notes = segment_notes(ref_f0, hop_seconds)
    tgt_midi = target_midi_per_frame(notes, n)
    reg = register_offset_semitones(hz_to_midi(user_f0), tgt_midi)

    # 4. Note-quantized partial correction.
    corr = compute_correction(
        user_f0, notes, reg, strength=strength, preserve=preserve,
        max_shift_semitones=max_shift_semitones,
    )

    # 5. Render.
    stem = os.path.splitext(os.path.basename(user_path))[0]
    outputs = {}
    raw_path = os.path.join(out_dir, f"{stem}_raw.wav")
    audio_io.save_audio(raw_path, user_aligned, su)
    outputs["raw"] = raw_path

    if backend in ("world", "both"):
        t0 = time.time()
        w = world_shift(user_aligned, su, corr.corrected_f0, hop_seconds)
        timings["render_world"] = time.time() - t0
        wp = os.path.join(out_dir, f"{stem}_world.wav")
        audio_io.save_audio(wp, w, su)
        outputs["world"] = wp
    if backend in ("rubberband", "both"):
        try:
            t0 = time.time()
            rb = rubberband_shift(user_aligned, su, corr.segments, hop_seconds)
            timings["render_rubberband"] = time.time() - t0
            rp = os.path.join(out_dir, f"{stem}_rubberband.wav")
            audio_io.save_audio(rp, rb, su)
            outputs["rubberband"] = rp
        except Exception as exc:  # rubberband binary missing, etc.
            outputs["rubberband_error"] = str(exc)

    # 6. Metrics: pitch error to the (register-matched) reference target.
    target_f0 = midi_to_hz(tgt_midi + reg)
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
        alignment=align, register_offset=reg, n_notes=len(notes),
        metrics=metrics, outputs=outputs, timings=timings,
    )


def _mean_abs(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(np.abs(x))) if x.size else float("nan")
