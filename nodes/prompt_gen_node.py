"""AceStepPromptGen node for ACE-Step – dynamically uses all components from prompt_utils"""
import random
import re
from .includes.prompt_utils import get_available_components, get_visible_components, get_component


def _choices_for(items):
    """Build the dropdown list: none, random, random2, then all items (deduplicated for UI)."""
    if isinstance(items, dict):
        items = items.keys()
    # Use set() to deduplicate for the UI while keeping the source list weighted for random picks
    return ["none", "random", "random2"] + sorted(list(set(items)))


def expand_wildcards(text, rng, max_depth=5):
    """Recursively expand __VARIABLE__ wildcards using available prompt components."""
    if not isinstance(text, str) or "__" not in text:
        return text

    pattern = r"__([A-Z0-9_]+)__"

    def replace(match):
        comp_name = match.group(1)
        # Try exact, then try with 'S' suffix for plural filenames
        items = get_component(comp_name)
        if items is None:
            items = get_component(comp_name + "S")
        if items is None:
            items = get_component(comp_name + "ES")
            
        if items is None:
            return match.group(0)

        # Pick a random item
        if isinstance(items, dict):
            # For dicts, pick a key and then use its value
            key = rng.choice(list(items.keys()))
            return str(items[key])
        elif isinstance(items, list):
            return str(rng.choice(items))
        return str(items)

    for _ in range(max_depth):
        new_text = re.sub(pattern, replace, text)
        if new_text == text:
            break
        text = new_text
    return text


class AceStepPromptGen:

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}
        # Only show visible components in the UI
        for name in get_visible_components():
            items = get_component(name)
            inputs[name] = (_choices_for(items), {"default": "none"})
        inputs["seed"] = ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})
        return {"required": inputs}

    # ComfyUI usually expects these as static tuples on the class
    # We use visible components here
    _comps = get_visible_components()
    RETURN_TYPES = tuple(["STRING"] * (1 + len(_comps)))
    RETURN_NAMES = tuple(["combined_prompt"] + [f"{name.lower()}_text" for name in _comps])
    
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/prompt"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution when any random choice is involved
        return any(str(v).startswith("random") for v in kwargs.values())

    def generate(self, seed: int, **kwargs):
        rng = random.Random(seed)
        results = {}
        # Filter logic: generate results for all visible components
        visible_comps = get_visible_components()

        for name in visible_comps:
            choice = kwargs.get(name, "none")
            items = get_component(name)
            out_name = f"{name.lower()}_text"

            def resolve_item(c):
                # If it's a dict like STYLE_PRESETS, resolve key -> value
                if isinstance(items, dict):
                    return str(items.get(c, c))
                return str(c)

            if choice == "none":
                results[out_name] = ""
            elif choice == "random":
                keys = list(items.keys()) if isinstance(items, dict) else list(items)
                if keys:
                    picked = rng.choice(keys)
                    resolved = resolve_item(picked)
                    results[out_name] = expand_wildcards(resolved, rng)
                else:
                    results[out_name] = ""
            elif choice == "random2":
                keys = list(items.keys()) if isinstance(items, dict) else list(items)
                if len(keys) >= 2:
                    picks = rng.sample(keys, 2)
                elif keys:
                    picks = [rng.choice(keys)]
                else:
                    picks = []
                    
                resolved_picks = [expand_wildcards(resolve_item(p), rng) for p in picks]
                results[out_name] = ", ".join(resolved_picks)
            else:
                # Explicit selection
                resolved = resolve_item(choice)
                results[out_name] = expand_wildcards(resolved, rng)

        # Build combined prompt from non-empty parts in the same sorted order
        parts = []
        for name in visible_comps:
            val = results[f"{name.lower()}_text"]
            if val:
                parts.append(val)
        combined = " ".join(parts)

        # Return order: prompt first, then each visible category
        out_list = [combined]
        for name in visible_comps:
            out_list.append(results[f"{name.lower()}_text"])
            
        return tuple(out_list)


NODE_CLASS_MAPPINGS = {
    "AceStepPromptGen": AceStepPromptGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPromptGen": "Prompt Generator",
}
