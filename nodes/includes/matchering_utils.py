# -*- coding: utf-8 -*-
"""
nodes/includes/matchering_utils.py

Adapter helpers for Matchering ComfyUI nodes.

The pip-installable `matchering` library (by Sergree, GPLv3) processes audio
from/to disk file paths. These helpers bridge ComfyUI AUDIO dicts ↔ temp WAV
files so we can call the standard API without bundling a patched copy.

Original ComfyUI-Matchering node by MuziekMagie:
  https://github.com/MuziekMagie/ComfyUI-Matchering (archived)
Matchering library by Sergree (Sergey Grishakov):
  https://github.com/sergree/matchering  (GPLv3)
"""

from __future__ import annotations

import os
import tempfile

import torch
import torchaudio


# ── ComfyUI AUDIO dict → temp WAV file ──────────────────────────────────────

def audio_to_tempfile(audio: dict) -> str:
    """
    Save a ComfyUI AUDIO dict to a temporary WAV file.
    Returns the path; caller is responsible for os.unlink().
    """
    waveform = audio["waveform"].squeeze(0)        # [C, N]
    sample_rate = audio["sample_rate"]
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    torchaudio.save(tmp.name, waveform.float(), sample_rate, format="wav")
    return tmp.name


# ── Temp WAV file → ComfyUI AUDIO dict ──────────────────────────────────────

def tempfile_to_audio(path: str) -> dict:
    """
    Load a WAV file saved by matchering back into a ComfyUI AUDIO dict.
    """
    waveform, sample_rate = torchaudio.load(path)   # [C, N]
    return {
        "waveform": waveform.unsqueeze(0),           # [1, C, N]
        "sample_rate": sample_rate,
    }


# ── Context manager: clean up all temp files even on error ───────────────────

class TempFiles:
    """Simple context manager that unlinks a list of paths on exit."""
    def __init__(self):
        self.paths: list[str] = []

    def new(self, suffix=".wav") -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.close()
        self.paths.append(tmp.name)
        return tmp.name

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for path in self.paths:
            try:
                os.unlink(path)
            except OSError:
                pass
