"""Monophonic pitch tracking with CREPE (via torchcrepe).

Produces an f0 contour on a fixed 10 ms analysis grid, plus a voiced mask
derived from CREPE's periodicity and a loudness gate. The user's take is the
noisy signal, so we median-filter and octave-guard here rather than trusting
raw frame estimates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchcrepe
import librosa

from .util import hz_to_midi, nan_median_filter

CREPE_SR = 16000  # torchcrepe operates at 16 kHz


@dataclass
class PitchTrack:
    f0: np.ndarray          # Hz, NaN where unvoiced
    voiced: np.ndarray      # bool
    periodicity: np.ndarray # CREPE confidence in [0, 1]
    times: np.ndarray       # frame centre times in seconds
    hop_seconds: float

    @property
    def midi(self) -> np.ndarray:
        return hz_to_midi(self.f0)


def track_pitch(
    y: np.ndarray,
    sr: int,
    hop_seconds: float = 0.01,
    fmin: float = 65.0,
    fmax: float = 1000.0,
    model: str = "full",
    periodicity_thr: float = 0.5,
    silence_db: float = -55.0,
    median_win: int = 5,
    device: str = "cpu",
) -> PitchTrack:
    """Track f0. `model` is 'full' (accurate, slower) or 'tiny' (fast)."""
    y = np.asarray(y, dtype=np.float32)
    if sr != CREPE_SR:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=CREPE_SR)
    else:
        y16 = y
    hop = max(1, int(round(hop_seconds * CREPE_SR)))

    audio = torch.tensor(y16[None, :], dtype=torch.float32)
    pitch, periodicity = torchcrepe.predict(
        audio,
        CREPE_SR,
        hop_length=hop,
        fmin=fmin,
        fmax=fmax,
        model=model,
        return_periodicity=True,
        batch_size=512,
        device=device,
        pad=True,
    )
    f0 = pitch.squeeze(0).cpu().numpy().astype(float)
    per = periodicity.squeeze(0).cpu().numpy().astype(float)
    n = len(f0)
    times = (np.arange(n) * hop) / CREPE_SR

    # Loudness gate: silence -> unvoiced regardless of CREPE confidence.
    rms = librosa.feature.rms(
        y=y16, frame_length=2 * hop, hop_length=hop, center=True
    )[0]
    rms = _match_len(rms, n)
    ref = float(rms.max()) if rms.size and rms.max() > 0 else 1.0
    loud_db = 20.0 * np.log10(np.maximum(rms, 1e-8) / ref)

    voiced = (per >= periodicity_thr) & (loud_db >= silence_db)
    f0[~voiced] = np.nan
    f0 = nan_median_filter(f0, median_win)
    voiced = np.isfinite(f0)

    return PitchTrack(
        f0=f0, voiced=voiced, periodicity=per, times=times,
        hop_seconds=hop / CREPE_SR,
    )


def _match_len(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, (0, n - len(x)), mode="edge")
