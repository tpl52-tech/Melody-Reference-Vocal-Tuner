"""Render a spread of naturalness settings for by-ear A/B.

First render populates the pitch caches (slow); the rest reuse them (fast).
All WORLD backend (the less-stepped one) with full micro-pitch preserved;
we vary how hard note centres are pulled to target and how much the
correction glides across transitions."""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtuner import pipeline

VARIANTS = [
    # label,    strength, preserve, smooth_ms
    ("gentle",  0.50,     1.0,      110),   # most natural, least corrected
    ("medium",  0.70,     1.0,      80),
    ("firm",    0.85,     1.0,      55),    # most in-tune, still gliding
]

OUT = "output"
CACHE = "output/.cache"


def main():
    for label, strength, preserve, smooth in VARIANTS:
        r = pipeline.run(
            "data/ref.wav", "data/take.wav", out_dir=OUT, backend="world",
            align_mode="dtw", model="full", strength=strength,
            preserve=preserve, smooth_ms=smooth, measure_output=False,
            cache_dir=CACHE,
        )
        dst = os.path.join(OUT, f"take_{label}.wav")
        shutil.copy(r.outputs["world"], dst)
        hit = r.timings.get("user_cache_hit")
        print(f"[{label}] strength={strength} preserve={preserve} smooth={smooth}ms "
              f"cents {r.metrics['mean_abs_cents_before']:.0f}->{r.metrics['mean_abs_cents_after_predicted']:.0f}  "
              f"(user_cache_hit={hit})  -> {dst}")
    print("\nA/B these:")
    for label, *_ in VARIANTS:
        print(f"  afplay output/take_{label}.wav")
    print("  afplay output/take_raw.wav   # your original")


if __name__ == "__main__":
    main()
