"""Optional Demucs source separation (Phase 4 stretch).

Two uses:
  * isolate the lead vocal so a full mixed song can be the melody reference,
    or strip a karaoke backing off the user's take;
  * recover the instrumental (the 'no_vocals' stem) so a tuned vocal can be
    mixed back onto the song's real backing (the produced-cover feature).

One Demucs pass yields both stems, so they are separated together and cached
on disk (separation is the slow step; first use downloads ~80 MB of weights)."""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import sys
import tempfile

from . import audio_io


def _cache_path(input_path: str, model: str, cache_dir: str, stem: str) -> str:
    try:
        mtime = os.path.getmtime(input_path)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(input_path)}|{mtime}|{model}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"isolated_{stem}_{h}.wav")


def isolate_stems(input_path: str, cache_dir: str, model: str = "htdemucs",
                  device: str = "cpu"):
    """Return (vocal_path, instrumental_path, cache_hit). Both stems cached."""
    os.makedirs(cache_dir, exist_ok=True)
    voc = _cache_path(input_path, model, cache_dir, "vocals")
    acc = _cache_path(input_path, model, cache_dir, "no_vocals")
    if os.path.exists(voc) and os.path.exists(acc):
        return voc, acc, True
    vy, ay, sr = _separate(input_path, model, device)
    audio_io.save_audio(voc, vy, sr)
    audio_io.save_audio(acc, ay, sr)
    return voc, acc, False


def isolate_vocal(input_path: str, cache_dir: str, model: str = "htdemucs",
                  device: str = "cpu"):
    """Return (path_to_isolated_vocal_wav, cache_hit)."""
    voc, _acc, hit = isolate_stems(input_path, cache_dir, model, device)
    return voc, hit


def _separate(input_path: str, model: str, device: str):
    """Return (vocal_y, instrumental_y, sr). Try the API; fall back to CLI."""
    try:
        from demucs.api import Separator
        sep = Separator(model=model, device=device)
        _, stems = sep.separate_audio_file(input_path)
        vy = stems["vocals"].mean(dim=0).cpu().numpy().astype("float32")
        # everything that isn't vocals = the instrumental
        acc = sum(v for k, v in stems.items() if k != "vocals")
        ay = acc.mean(dim=0).cpu().numpy().astype("float32")
        return vy, ay, int(sep.samplerate)
    except Exception:
        return _separate_cli(input_path, model, device)


def _separate_cli(input_path: str, model: str, device: str):
    tmp = tempfile.mkdtemp(prefix="demucs_")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "-n", model, "-d", device, "-o", tmp, input_path]
    subprocess.run(cmd, check=True, capture_output=True)
    voc = glob.glob(os.path.join(tmp, model, "*", "vocals.wav"))
    acc = glob.glob(os.path.join(tmp, model, "*", "no_vocals.wav"))
    if not voc or not acc:
        raise RuntimeError("Demucs did not produce both stems")
    vy, sr = audio_io.load_audio(voc[0], sr=None, mono=True)
    ay, _ = audio_io.load_audio(acc[0], sr=None, mono=True)
    return vy, ay, sr
