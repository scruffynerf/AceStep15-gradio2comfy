# -*- coding: utf-8 -*-
"""
nodes/audio_matchering_limiter_config_node.py  —  AceStep Matchering Limiter Config

Ports the MatcheringLimiterConfig node from MuziekMagie:
  https://github.com/MuziekMagie/ComfyUI-Matchering

Outputs a configuration object for the Matchering Advanced node.
"""

from matchering.defaults import LimiterConfig

class MatcheringLimiterConfigNode:
    """
    Configure the brickwall limiter used in the Matchering process.
    """

    CATEGORY = "Scromfy/Ace-Step/Matchering"
    FUNCTION = "configure"
    RETURN_TYPES = ("MATCHERING_LIMITER_CONFIG",)
    RETURN_NAMES = ("limiter_config",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attack": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "hold": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.1}),
                "release": ("FLOAT", {"default": 3000.0, "min": 1.0, "step": 1.0}),
                "attack_filter_coefficient": (
                    "FLOAT",
                    {"default": -2.0, "min": -1000.0, "step": 0.1},
                ),
                "hold_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "hold_filter_coefficient": (
                    "FLOAT",
                    {"default": 7.0, "step": 0.1},
                ),
                "release_filter_order": (
                    "INT",
                    {"default": 1, "min": 1, "step": 1},
                ),
                "release_filter_coefficient": ("FLOAT", {"default": 800.0, "step": 1.0}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return hash(frozenset(kwargs))

    def configure(
        self,
        attack,
        hold,
        release,
        attack_filter_coefficient,
        hold_filter_order,
        hold_filter_coefficient,
        release_filter_order,
        release_filter_coefficient,
    ):
        config = LimiterConfig(
            attack=attack,
            hold=hold,
            release=release,
            attack_filter_coefficient=attack_filter_coefficient,
            hold_filter_order=hold_filter_order,
            hold_filter_coefficient=hold_filter_coefficient,
            release_filter_order=release_filter_order,
            release_filter_coefficient=release_filter_coefficient,
        )
        return (config,)


NODE_CLASS_MAPPINGS = {
    "MatcheringLimiterConfigNode": MatcheringLimiterConfigNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MatcheringLimiterConfigNode": "Matchering Limiter Config",
}
