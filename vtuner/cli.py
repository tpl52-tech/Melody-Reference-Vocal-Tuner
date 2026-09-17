"""Command-line entry point: reference + user take -> tuned renders.

    python -m vtuner.cli --reference data/ref.wav --user data/take.wav
"""
from __future__ import annotations

import argparse
import sys

from .pipeline import run


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="vtuner",
        description="Retune a sing-along take to follow a reference vocal's melody.",
    )
    p.add_argument("--reference", "-r", required=True, help="reference vocal clip")
    p.add_argument("--user", "-u", required=True, help="your sung take")
    p.add_argument("--out", "-o", default="output", help="output directory")
    p.add_argument("--backend", choices=["world", "rubberband", "both"], default="both")
    p.add_argument("--strength", type=float, default=0.9,
                   help="note-center correction strength 0..1")
    p.add_argument("--preserve", type=float, default=1.0,
                   help="how much within-note micro-pitch/vibrato to keep 0..1")
    p.add_argument("--model", choices=["full", "tiny"], default="full",
                   help="CREPE capacity: full=accurate, tiny=fast")
    p.add_argument("--max-shift", type=float, default=12.0,
                   help="clamp per-frame shift (semitones)")
    p.add_argument("--measure", action="store_true",
                   help="re-track the WORLD render to verify tuning empirically")
    args = p.parse_args(argv)

    res = run(
        reference_path=args.reference,
        user_path=args.user,
        out_dir=args.out,
        backend=args.backend,
        strength=args.strength,
        preserve=args.preserve,
        model=args.model,
        max_shift_semitones=args.max_shift,
        measure_output=args.measure,
    )

    a = res.alignment
    print("\n=== Melody-Reference Vocal Tuner ===")
    print(f"alignment: lag {a.lag_seconds*1000:+.0f} ms "
          f"(confidence {a.confidence:.2f}, residual drift {a.residual_drift_ms:.0f} ms)")
    if a.residual_drift_ms > 60:
        print("  ! high residual drift -- offset-only sync may be insufficient "
              "(consider per-note alignment)")
    print(f"reference notes: {res.n_notes}   register fold: {res.register_offset:+d} st")
    m = res.metrics
    print(f"pitch error to reference melody:")
    print(f"  before:            {m['mean_abs_cents_before']:.0f} cents "
          f"({m['frames_in_note']} frames)")
    print(f"  after (predicted): {m['mean_abs_cents_after_predicted']:.0f} cents")
    if "mean_abs_cents_after_world_measured" in m:
        print(f"  after (measured):  {m['mean_abs_cents_after_world_measured']:.0f} cents")
    print("timings (s):", {k: round(v, 2) for k, v in res.timings.items()
                            if isinstance(v, float)})
    print("outputs:")
    for k, v in res.outputs.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
