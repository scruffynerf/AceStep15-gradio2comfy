"""AceStepZeroConditioningSectionMap node for ACE-Step"""
import json
from .includes.zerobytes_utils import build_default_section_map


class AceStepZeroConditioningSectionMap:
    """Build a section map for ZeroConditioning generation.

    Defines song structure (verse, chorus, bridge, etc.) with timing boundaries.
    Each section type gets a unique seed derivation in the generator, creating
    distinct but coherent musical regions.
    """

    FORM_TEMPLATES = {
        "AABA": "AABA",
        "ABAB": "ABAB",
        "ABABCB": "ABABCB",
        "ABCABC": "ABCABC",
        "through-composed": "ABCDEF",
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "duration": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 600.0, "step": 0.1}),
                "form": (["AABA", "ABAB", "ABABCB", "ABCABC", "through-composed", "custom"],
                         {"default": "ABABCB"}),
            },
            "optional": {
                "intro_seconds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 60.0, "step": 0.5}),
                "outro_seconds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 60.0, "step": 0.5}),
                "verse_weight": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "chorus_weight": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "custom_form": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("section_map",)
    FUNCTION = "build"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def build(self, duration, form, intro_seconds=8.0, outro_seconds=8.0,
              verse_weight=1.0, chorus_weight=1.0, custom_form=""):
        if form == "custom" and custom_form.strip():
            form_str = custom_form.strip().upper()
        else:
            form_str = self.FORM_TEMPLATES.get(form, "ABABCB")

        sections = build_default_section_map(
            duration, form=form_str,
            intro_s=intro_seconds, outro_s=outro_seconds,
            verse_weight=verse_weight, chorus_weight=chorus_weight)

        return (json.dumps(sections),)


NODE_CLASS_MAPPINGS = {
    "AceStepZeroConditioningSectionMap": AceStepZeroConditioningSectionMap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroConditioningSectionMap": "Zero Conditioning Section Map",
}
