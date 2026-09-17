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
2. **Alignment** — a single global sing-along offset is estimated by cross-correlating loudness envelopes and removed. A residual-drift figure flags when offset-only sync isn't enough.
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

# options
#   --backend world|rubberband|both   which shifter(s) to render
#   --strength 0.9                     note-center correction strength (0..1)
#   --preserve 1.0                     how much of your vibrato/micro-pitch to keep
#   --model full|tiny                  CREPE accuracy vs speed
#   --measure                          re-track the render to verify tuning
```

Outputs `output/<take>_raw.wav`, `_world.wav`, `_rubberband.wav` for blind A/B.

Reference and take should be **vocal-only** in v1 (full-song stem separation via Demucs is a planned stretch). Phone recordings (`.m4a`) work — that's what `ffmpeg` is for.

## Verify without real audio

```bash
python scripts/selftest.py     # synthesizes a mangled take, runs the pipeline, checks it improves
```

## Status

Phase 1 (core pipeline) — the CREPE → align → segment → register-fold → dual-backend retune path described above. Quality tuning against real clips (Phase 2), a demo UI (Phase 3), and Demucs full-song input (Phase 4 stretch) are next.
