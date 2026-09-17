# Melody-Reference Vocal Tuner

Retune an amateur **sing-along take** so it follows a **reference vocal's actual melody** — no key or scale selection, no note-by-note editing. Unlike consumer auto-tune (Voloco, BandLab AutoPitch), which snaps to a generic scale and sounds wrong on anything chromatic, this reads the melody straight from a reference vocal.

Point it at a reference vocal clip and your own take (recorded while singing along, so the two are already roughly time-synced) and it renders an in-tune version that keeps your own voice and expression.

## How it works

```
reference vocal ─CREPE─▶ pitch ─segment─▶ note targets ─┐
                                                         ├─▶ register-fold + note-quantized
your take ─align─▶ ─CREPE─▶ pitch ──────────────────────┘   partial correction
                                                                    │
                                                     WORLD  ◀────────┴────────▶  RubberBand
                                                   (renders, A/B by ear)
```

1. **Pitch tracking** — CREPE (`torchcrepe`) extracts f0 from both clips. The reference track is **cached** (it never changes), so only your take is tracked live.
2. **Alignment** — by default (`--align-mode dtw`) the take is aligned to the reference **frame-by-frame** via DTW (chroma-CENS), so it works even when the take was recorded separately and *drifts* relative to the reference. It does not time-stretch your audio — it only maps which reference note you're on at each moment, keeping your own timing. (`--align-mode offset` is the lighter v1 path: a single global shift, only valid when you sang along to the exact reference.)
3. **Note targets** — the reference contour is segmented into discrete notes; each target is the *median of the reference singer's own pitch* over that note (not equal-temperament), so blue notes and expressive intonation survive.
4. **Register fold** — reference targets are shifted by whole octaves into your register, keeping per-note corrections small and natural.
5. **Correction** — each note's *center* is pulled toward its target by `--strength`, while your within-note micro-pitch/vibrato is kept (`--preserve`).
6. **Render** — two formant-preserving backends: **WORLD** (continuous contour, vibrato preserved) and **RubberBand** (constant per-note shift). Compare by ear.

## Setup

```bash
brew install ffmpeg rubberband          # system deps (macOS)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# tune a real take against a reference vocal
python -m vtuner.cli --reference data/ref.wav --user data/take.wav --out output

# options (defaults tuned by ear for a natural, non-robotic result)
#   --align-mode dtw|offset            dtw (default) tolerates timing drift
#   --backend world|rubberband|both    which shifter(s) to render
#   --strength 0.6                     note-center correction (0..1; lower = more natural)
#   --preserve 1.0                     how much of your vibrato/micro-pitch to keep
#   --smooth-ms 95                     glide corrections across notes (higher = less robotic)
#   --model full|tiny                  CREPE accuracy vs speed
#   --measure                          re-track the render to verify tuning
```

Outputs `output/<take>_raw.wav`, `_world.wav`, `_rubberband.wav` for blind A/B.
Both the reference **and** take pitch tracks are cached, so re-rendering with
different tuning knobs completes in seconds (`scripts/variants.py` renders a
naturalness spread for A/B).

**On the auto-tune artifact:** if a render sounds robotic, lower `--strength`
and/or raise `--smooth-ms`; WORLD sounds more natural than RubberBand (which
does a flat shift per note). `--preserve 1.0` keeps all your natural pitch
movement.

Reference and take should be **vocal-only** in v1 (full-song stem separation via Demucs is a planned stretch). Phone recordings (`.m4a`) work — that's what `ffmpeg` is for.

## Verify without real audio

```bash
python scripts/selftest.py     # synthesizes a mangled take, runs the pipeline, checks it improves
```

## Status

Phase 1 (core pipeline) **done and validated on a real take** — CREPE → align → segment → register-fold → note-quantized partial correction → dual-backend render. DTW alignment (originally a v2/stretch) was pulled forward because real sing-along takes drift; the default tuning knobs were dialed in by ear to a natural, non-robotic result. A demo UI (Phase 3) and Demucs full-song input (Phase 4 stretch) are next.
