"""Optional Demucs vocal isolation (Phase 4 stretch).

Removes the 'reference must be a pre-isolated acapella' constraint: given a
full mixed song, pull the lead vocal out so the monophonic pitch tracker has a
clean melody to follow.

Separation is the slow part of the whole pipeline on CPU, so the isolated
vocal is cached on disk keyed on the input file + model. First use downloads
the Demucs model weights (~80 MB, one time)."""
from __future__ import annotations

import glob
import hashlib
import os
import subprocess
import sys
import tempfile

from . import audio_io


def _cache_path(input_path: str, model: str, cache_dir: str) -> str:
    try:
        mtime = os.path.getmtime(input_path)
    except OSError:
        mtime = 0.0
    key = f"{os.path.abspath(input_path)}|{mtime}|{model}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"isolated_{h}.wav")


def isolate_vocal(input_path: str, cache_dir: str, model: str = "htdemucs",
                  device: str = "cpu"):
    """Return (path_to_isolated_vocal_wav, cache_hit)."""
    os.makedirs(cache_dir, exist_ok=True)
    out = _cache_path(input_path, model, cache_dir)
    if os.path.exists(out):
        return out, True
    y, sr = _separate(input_path, model, device)
    audio_io.save_audio(out, y, sr)
    return out, False


def _separate(input_path: str, model: str, device: str):
    """Try the programmatic API; fall back to the CLI (stable across versions)."""
    try:
        from demucs.api import Separator
        sep = Separator(model=model, device=device)
        _, stems = sep.separate_audio_file(input_path)
        voc = stems["vocals"]                    # (channels, samples) tensor
        y = voc.mean(dim=0).cpu().numpy().astype("float32")
        return y, int(sep.samplerate)
    except Exception:
        return _separate_cli(input_path, model, device)


def _separate_cli(input_path: str, model: str, device: str):
    tmp = tempfile.mkdtemp(prefix="demucs_")
    cmd = [sys.executable, "-m", "demucs", "--two-stems=vocals",
           "-n", model, "-d", device, "-o", tmp, input_path]
    subprocess.run(cmd, check=True, capture_output=True)
    matches = glob.glob(os.path.join(tmp, model, "*", "vocals.wav"))
    if not matches:
        raise RuntimeError("Demucs produced no vocals stem")
    return audio_io.load_audio(matches[0], sr=None, mono=True)
