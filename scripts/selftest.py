"""Synthetic end-to-end self-test.

Generates a mangled sing-along take, runs the full pipeline, and checks that
the retuned output is measurably closer to the reference melody than the raw
take was. No real audio required."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtuner import synth, pipeline


def main():
    ref_path, user_path, truth = synth.generate("data")
    print("ground truth:")
    print(f"  melody (MIDI): {truth['melody_midi']}")
    print(f"  user detune:   {[round(c) for c in truth['user_detune_cents']]} cents")
    print(f"  transposed {truth['user_transpose']} st, delayed "
          f"{truth['delay_seconds']*1000:.0f} ms\n")

    # minimal smoothing here: this test measures correction ACCURACY, and the
    # synthetic clip's 0.5s notes are short relative to the musical glide default
    res = pipeline.run(
        ref_path, user_path, out_dir="output",
        backend="both", strength=0.95, preserve=1.0, smooth_ms=20,
        measure_output=True,
    )

    a = res.alignment
    m = res.metrics
    print(f"detected lag: {a.lag_seconds*1000:+.0f} ms "
          f"(truth {truth['delay_seconds']*1000:.0f} ms), "
          f"confidence {a.confidence:.2f}")
    # take is an octave below the reference -> fold targets DOWN into its register
    print(f"register fold: {res.register_offset:+d} st (expected -12)")
    print(f"reference notes: {res.n_notes} (expected {len(truth['melody_midi'])})")
    print(f"cents error before:            {m['mean_abs_cents_before']:.0f}")
    print(f"cents error after (predicted): {m['mean_abs_cents_after_predicted']:.0f}")
    meas = m.get("mean_abs_cents_after_world_measured", float('nan'))
    print(f"cents error after (measured):  {meas:.0f}")
    print("timings:", {k: round(v, 2) for k, v in res.timings.items()
                        if isinstance(v, float)})
    print("outputs:", res.outputs)

    ok = True
    checks = []

    # 1. offset recovered within 30 ms
    lag_err = abs(a.lag_seconds - truth["delay_seconds"]) * 1000
    checks.append(("offset recovered (<30ms err)", lag_err < 30, f"{lag_err:.0f} ms err"))

    # 2. register fold pulls the reference targets down into the take's octave
    checks.append(("register fold == -12", res.register_offset == -12, f"{res.register_offset:+d}"))

    # 3. note count in the right ballpark
    exp = len(truth["melody_midi"])
    checks.append(("note count ~ melody", abs(res.n_notes - exp) <= 2, f"{res.n_notes} vs {exp}"))

    # 4. predicted tuning much tighter than raw
    checks.append(("predicted << before", m["mean_abs_cents_after_predicted"] < 25
                   and m["mean_abs_cents_after_predicted"] < 0.5 * m["mean_abs_cents_before"],
                   f"{m['mean_abs_cents_after_predicted']:.0f} < {m['mean_abs_cents_before']:.0f}"))

    # 5. measured (re-tracked) output actually improved
    if meas == meas:  # not NaN
        checks.append(("measured < before", meas < m["mean_abs_cents_before"],
                       f"{meas:.0f} < {m['mean_abs_cents_before']:.0f}"))

    print("\n--- checks ---")
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
