# -*- coding: utf-8 -*-
"""
nodes/audio_matchering_node.py  —  AceStep Matchering (Simple)

Ports the simple Matchering node originally created by MuziekMagie:
  https://github.com/MuziekMagie/ComfyUI-Matchering  (archived)

Uses the pip-installable matchering library by Sergree (Sergey Grishakov, GPLv3):
  https://github.com/sergree/matchering

Adapter design: ComfyUI AUDIO dicts are saved to temp WAV files, processed by
the standard matchering API (which expects file paths), and the results are
loaded back as tensors. All temp files are cleaned up automatically.
"""

import matchering as mg
from .includes.matchering_utils import audio_to_tempfile, tempfile_to_audio, TempFiles


class MatcheringNode:
    """
    Match the loudness, tone, and dynamics of a target track to a reference.

    Outputs three versions:
    - Result          : fully mastered (limiter applied)
    - Result (no limiter) : pre-limiter output
    - Result (no limiter, normalized) : pre-limiter, peak-normalized
    """

    CATEGORY = "Scromfy/Ace-Step/Matchering"
    FUNCTION = "matchering"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = (
        "Result",
        "Result (no limiter)",
        "Result (no limiter, normalized)",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": ("AUDIO",),
                "reference": ("AUDIO",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash(frozenset(kwargs))

    def matchering(self, target, reference):
        mg.log(print)

        with TempFiles() as tmp:
            target_path    = audio_to_tempfile(target)
            reference_path = audio_to_tempfile(reference)
            tmp.paths += [target_path, reference_path]

            result_path      = tmp.new()
            no_lim_path      = tmp.new()
            normalized_path  = tmp.new()

            mg.process(
                target=target_path,
                reference=reference_path,
                results=[
                    # Result        : limiter ON,  normalize ON
                    mg.pcm16(result_path),
                    # No limiter    : limiter OFF, normalize OFF (raw pre-limiter)
                    mg.Result(no_lim_path,     "PCM_16", use_limiter=False, normalize=False),
                    # Normalized    : limiter OFF, normalize ON
                    mg.Result(normalized_path, "PCM_16", use_limiter=False, normalize=True),
                ],
            )

            result_audio      = tempfile_to_audio(result_path)
            no_lim_audio      = tempfile_to_audio(no_lim_path)
            normalized_audio  = tempfile_to_audio(normalized_path)

        return (result_audio, no_lim_audio, normalized_audio)


NODE_CLASS_MAPPINGS = {
    "MatcheringNode": MatcheringNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatcheringNode": "Matchering",
}
