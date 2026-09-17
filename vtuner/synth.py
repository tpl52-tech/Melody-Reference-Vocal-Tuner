"""Synthetic test audio so the pipeline can be verified without real clips.

Generates a clean reference melody and a deliberately mangled "user" take of
the same melody: detuned per note, transposed down an octave (to exercise
register matching), delayed (to exercise offset alignment), and noisy."""
from __future__ import annotations

import os

import numpy as np

from .util import midi_to_hz
from . import audio_io

# A melody with an accidental (Bb, MIDI 70) so it is NOT purely diatonic --
# the case where scale-based auto-tune would snap to the wrong note.
DEFAULT_MELODY = [62, 64, 65, 67, 70, 67, 65, 64, 62]  # D E F G Bb G F E D
NOTE_SECONDS = 0.5


def _harmonic_tone(f0_hz, dur, sr, n_harmonics=6, vibrato_cents=0.0,
                   vibrato_hz=5.5):
    t = np.arange(int(dur * sr)) / sr
    f = f0_hz * (2.0 ** ((vibrato_cents / 1200.0) * np.sin(2 * np.pi * vibrato_hz * t)))
    phase = 2 * np.pi * np.cumsum(f) / sr
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += (1.0 / k) * np.sin(k * phase)
    # short attack/release so note joins are click-free
    env = np.ones_like(t)
    ramp = max(1, int(0.01 * sr))
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return sig * env


def _render(midis, sr, detune_cents=None, transpose=0, vibrato_cents=8.0):
    pieces = []
    for i, m in enumerate(midis):
        cents = 0.0 if detune_cents is None else detune_cents[i]
        f0 = float(midi_to_hz(np.array([m + transpose + cents / 100.0]))[0])
        pieces.append(_harmonic_tone(f0, NOTE_SECONDS, sr,
                                     vibrato_cents=vibrato_cents))
    return np.concatenate(pieces)


def generate(out_dir: str = "data", sr: int = 44100, seed: int = 0):
    """Write ref_synth.wav + user_synth.wav; return paths and ground truth."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    melody = DEFAULT_MELODY

    ref = _render(melody, sr, vibrato_cents=6.0) * 0.6

    # user take: random per-note detune, an octave down, delayed, + noise
    detune = rng.uniform(-70, 70, size=len(melody))
    user_clean = _render(melody, sr, detune_cents=detune, transpose=-12,
                         vibrato_cents=14.0) * 0.5
    delay = int(0.08 * sr)  # 80 ms sing-along lag
    user = np.concatenate([np.zeros(delay, dtype=user_clean.dtype), user_clean])
    user = user + rng.normal(0, 0.004, size=len(user)).astype(np.float32)

    ref_path = os.path.join(out_dir, "ref_synth.wav")
    user_path = os.path.join(out_dir, "user_synth.wav")
    audio_io.save_audio(ref_path, ref, sr)
    audio_io.save_audio(user_path, user, sr)

    truth = {
        "melody_midi": melody,
        "user_detune_cents": detune.tolist(),
        "user_transpose": -12,
        "delay_seconds": delay / sr,
    }
    return ref_path, user_path, truth


if __name__ == "__main__":
    r, u, truth = generate()
    print(f"wrote {r} and {u}")
    print(f"melody (MIDI): {truth['melody_midi']}")
    print(f"user detune (cents): {[round(c, 1) for c in truth['user_detune_cents']]}")
    print(f"user transposed {truth['user_transpose']} st, delayed {truth['delay_seconds']*1000:.0f} ms")
