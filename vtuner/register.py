"""Octave-fold the reference melody into the user's register.

'Sing along to my favourite song' routinely means a very different voice type
from the original singer (e.g. a baritone over a female pop vocal). Without
this, per-note shifts would be 5-12 semitones and formant preservation breaks
down. We fold the reference targets by whole octaves so the residual
correction each note needs is small (and thus natural)."""
from __future__ import annotations

import numpy as np


def register_offset_semitones(
    user_midi: np.ndarray,
    target_midi: np.ndarray,
) -> int:
    """Whole-octave offset (a multiple of 12) to ADD to the reference targets
    so their median sits closest to the user's median pitch."""
    um = user_midi[np.isfinite(user_midi)]
    tm = target_midi[np.isfinite(target_midi)]
    if um.size == 0 or tm.size == 0:
        return 0
    diff = float(np.median(um) - np.median(tm))
    return int(round(diff / 12.0)) * 12
