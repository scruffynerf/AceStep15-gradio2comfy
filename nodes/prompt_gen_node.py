"""AceStepPromptGen node for ACE-Step – exposes every category from prompt_utils"""
import random
from .includes.prompt_utils import (
    STYLE_PRESETS, GENRES, MOODS, ADJECTIVES,
    CULTURES, INSTRUMENTS, PERFORMERS, VOCAL_QUALITIES,
)

# Each category: (list, input_name, output_name, label used in combined prompt)
_CATEGORIES = [
    (sorted(STYLE_PRESETS.keys()), "style",         "style_text",      "Style"),
    (MOODS,                        "mood",           "mood_text",       "Mood"),
    (ADJECTIVES,                   "adjective",      "adjective_text",  "Adjective"),
    (CULTURES,                     "culture",        "culture_text",    "Culture"),
    (GENRES,                       "genre",          "genre_text",      "Genre"),
    (VOCAL_QUALITIES,              "vocal_quality",  "vocal_text",      "Vocal"),
    (PERFORMERS,                   "performer",      "performer_text",  "Performer"),
    (INSTRUMENTS,                  "instrument",     "instrument_text", "Instrument"),
]

def _choices_for(items):
    """Build the dropdown list: none, random, random2, then all items."""
    return ["none", "random", "random2"] + list(items)


class AceStepPromptGen:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        for items, input_name, _, _ in _CATEGORIES:
            inputs[input_name] = (_choices_for(items),)
        inputs["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})
        return {"required": inputs}

    RETURN_TYPES  = tuple(["STRING"] * (1 + len(_CATEGORIES)))
    RETURN_NAMES  = tuple(
        ["prompt"] + [cat[2] for cat in _CATEGORIES]
    )
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/prompt"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when any random choice is involved
        return any(v in ("random", "random2") for v in kwargs.values())

    def generate(self, seed: int, **kwargs):
        rng = random.Random(seed)
        results = {}

        for items, input_name, output_name, label in _CATEGORIES:
            choice = kwargs.get(input_name, "none")
            items_list = list(items)

            if choice == "none":
                results[output_name] = ""
            elif choice == "random":
                picked = rng.choice(items_list)
                results[output_name] = STYLE_PRESETS.get(picked, picked) if input_name == "style" else picked
            elif choice == "random2":
                if len(items_list) >= 2:
                    picks = rng.sample(items_list, 2)
                else:
                    picks = [rng.choice(items_list)]
                if input_name == "style":
                    picks = [STYLE_PRESETS.get(p, p) for p in picks]
                results[output_name] = ", ".join(picks)
            else:
                # Explicit selection — for STYLE_PRESETS expand the key to full text
                results[output_name] = STYLE_PRESETS.get(choice, choice) if input_name == "style" else choice

        # Build combined prompt from non-empty parts
        parts = []
        for _, _, output_name, _ in _CATEGORIES:
            val = results[output_name]
            if val:
                parts.append(val)
        combined = " ".join(parts)

        # Return order: prompt first, then each category in _CATEGORIES order
        return tuple(
            [combined] + [results[cat[2]] for cat in _CATEGORIES]
        )


NODE_CLASS_MAPPINGS = {
    "AceStepPromptGen": AceStepPromptGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPromptGen": "Prompt Generator",
}
