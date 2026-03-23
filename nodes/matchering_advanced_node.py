# -*- coding: utf-8 -*-
"""
nodes/matchering_advanced_node.py  —  AceStep Matchering (Advanced)

Ports the MatcheringAdvanced node originally created by MuziekMagie:
  https://github.com/MuziekMagie/ComfyUI-Matchering

Provides full control over the matching and mastering parameters.
"""

import matchering as mg
from matchering.defaults import Config, LimiterConfig
from .includes.matchering_utils import audio_to_tempfile, tempfile_to_audio, TempFiles


class MatcheringAdvancedNode:
    """
    Advanced Matchering node with full parameter control.
    """

    CATEGORY = "Scromfy/AceStep/Matchering"
    FUNCTION = "matchering_advanced"
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
                "internal_sample_rate": (
                    "INT",
                    {"default": 44100, "min": 0, "max": 192000, "step": 1},
                ),
                "max_length": ("FLOAT", {"default": 15 * 60, "min": 0, "step": 1}),
                "max_piece_size": ("FLOAT", {"default": 15, "min": 0, "step": 1}),
                "threshold": (
                    "FLOAT",
                    {
                        "default": (2**15 - 61) / 2**15,
                        "max": 0.9999999,
                        "step": 0.0000001,
                        "round": False,
                    },
                ),
                "min_value": (
                    "FLOAT",
                    {"default": 1e-6, "min": 0, "max": 0.1, "step": 1e-6},
                ),
                "fft_size": ("INT", {"default": 4096, "min": 1, "step": 1}),
                "lin_log_oversampling": ("INT", {"default": 4, "min": 1, "step": 1}),
                "rms_correction_steps": ("INT", {"default": 4, "min": 0, "step": 1}),
                "clipping_samples_threshold": (
                    "INT",
                    {"default": 8, "min": 0, "step": 1},
                ),
                "limited_samples_threshold": (
                    "INT",
                    {"default": 128, "min": 0, "step": 1},
                ),
                "allow_equality": ("BOOLEAN", {"default": False}),
                "lowess_frac": (
                    "FLOAT",
                    {"default": 0.0375, "min": 0.0001, "step": 0.0001},
                ),
                "lowess_it": ("INT", {"default": 0, "min": 0, "step": 1}),
                "lowess_delta": ("FLOAT", {"default": 0.001, "min": 0, "step": 0.001}),
            },
            "optional": {
                "limiter_config": ("MATCHERING_LIMITER_CONFIG",),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash(frozenset(kwargs))

    def matchering_advanced(
        self,
        target,
        reference,
        internal_sample_rate,
        max_length,
        max_piece_size,
        threshold,
        min_value,
        fft_size,
        lin_log_oversampling,
        rms_correction_steps,
        clipping_samples_threshold,
        limited_samples_threshold,
        allow_equality,
        lowess_frac,
        lowess_it,
        lowess_delta,
        limiter_config=None,
    ):
        mg.log(print)
        
        if limiter_config is None:
            limiter_config = LimiterConfig()

        config = Config(
            internal_sample_rate=internal_sample_rate,
            max_length=max_length,
            max_piece_size=max_piece_size,
            threshold=threshold,
            min_value=min_value,
            fft_size=fft_size,
            lin_log_oversampling=lin_log_oversampling,
            rms_correction_steps=rms_correction_steps,
            clipping_samples_threshold=clipping_samples_threshold,
            limited_samples_threshold=limited_samples_threshold,
            allow_equality=allow_equality,
            lowess_frac=lowess_frac,
            lowess_it=lowess_it,
            lowess_delta=lowess_delta,
            limiter=limiter_config,
        )

        with TempFiles() as tmp:
            target_path = audio_to_tempfile(target)
            reference_path = audio_to_tempfile(reference)
            tmp.paths += [target_path, reference_path]

            result_path = tmp.new()
            no_lim_path = tmp.new()
            normalized_path = tmp.new()

            mg.process(
                target=target_path,
                reference=reference_path,
                config=config,
                results=[
                    mg.pcm16(result_path),
                    mg.Result(no_lim_path, "PCM_16", use_limiter=False, normalize=False),
                    mg.Result(normalized_path, "PCM_16", use_limiter=False, normalize=True),
                ],
            )

            result_audio = tempfile_to_audio(result_path)
            no_lim_audio = tempfile_to_audio(no_lim_path)
            normalized_audio = tempfile_to_audio(normalized_path)

        return (result_audio, no_lim_audio, normalized_audio)


NODE_CLASS_MAPPINGS = {
    "MatcheringAdvancedNode": MatcheringAdvancedNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatcheringAdvancedNode": "Matchering (Advanced)",
}
