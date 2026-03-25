from pathlib import Path
from .audio_player_radio_node import RadioPlayerNode, _SKINS_DIR, _VISUALIZERS_DIR

class AceStepWebAmpRadio(RadioPlayerNode):
    """Winamp-style radio player using WebAmp ⚡"""

    @classmethod
    def _get_skin_choices(cls):
        """Scan the webamp_skins directory and return a list of name options."""
        _SKINS_DIR.mkdir(exist_ok=True)
        choices = ["(none)"]
        for p in sorted(_SKINS_DIR.glob("*.wsz")):
            choices.append(p.name)
        return choices

    @classmethod
    def INPUT_TYPES(cls):
        base = RadioPlayerNode.INPUT_TYPES()
        base["required"]["skin"] = (cls._get_skin_choices(), {
            "default": "(none)",
            "tooltip": "Select a skin from webamp_skins/, or use skin_url for a custom address",
        })
        base["optional"]["skin_url"] = ("STRING", {
            "default": "",
            "multiline": False,
            "placeholder": "Optional: custom .wsz skin URL (overrides skin dropdown)",
        })
        base["optional"]["artist_name"] = ("STRING", {
            "default": "Ace-Step",
            "multiline": False,
            "placeholder": "Artist name shown in playlist",
        })
        base["optional"]["lyrics_font_size"] = ("FLOAT", {
            "default": 13.0,
            "min": 8.0,
            "max": 128.0,
            "step": 1.0,
            "display": "slider",
        })
        return base

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Scromfy/Ace-Step/Radio"

    def run(self, folder: str, skin: str = "(none)", 
            skin_url: str = "",
            poll_interval_seconds: float = 60.0, artist_name: str = "Ace-Step",
            lyrics_font_size: float = 13.0):
        return {}

NODE_CLASS_MAPPINGS = {
    "AceStepWebAmpRadio": AceStepWebAmpRadio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepWebAmpRadio": "WebAmp Radio ⚡",
}
