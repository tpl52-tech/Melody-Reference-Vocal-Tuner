"""Melody-Reference Vocal Tuner.

Retune an amateur sing-along take to follow a reference vocal's actual melody
(CREPE pitch tracking + formant-preserving pitch shift), with no key or scale
selection required."""

__version__ = "0.1.0"

from .pipeline import run, Result  # noqa: F401
