import torch
from .includes.chord_utils import build_chord_audio, _SR

_CHORD_MAP_PLACEHOLDER = """\
[intro]   Am F C G
[verse]   Am F C G
[chorus]  F C G Am
[bridge]  Dm Am Bb G
[outro]   Am F C G
default   Am F C G"""

class AceStepChordPreview:
    """
    ACE-Step ▸ Chord Preview

    Renders the section-aware chord audio so you can hear exactly what
    harmonic structure is being injected before committing to a full generation.
    """
    CATEGORY = "Scromfy/Ace-Step/Chords"
    FUNCTION = "preview"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("chord_audio",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "chord_map": ("STRING", {"multiline": True, "default": _CHORD_MAP_PLACEHOLDER}),
            "lyrics": ("STRING", {"multiline": True, "default": ""}),
            "bpm":             ("INT",   {"default":120,"min":40,"max":240,"step":1}),
            "beats_per_chord": ("FLOAT", {"default":4.0,"min":0.5,"max":16.0,"step":0.5}),
            "duration":        ("FLOAT", {"default":30.0,"min":5.0,"max":300.0,"step":1.0}),
            "synth_type":      (["piano","organ","pad"], {"default":"piano"}),
        }, "optional": {
            "velocity": ("FLOAT", {"default":0.65,"min":0.1,"max":1.0,"step":0.05}),
        }}

    def preview(self, chord_map, lyrics, bpm, beats_per_chord,
                duration, synth_type, velocity=0.65):
        audio = build_chord_audio(lyrics, chord_map, float(bpm),
                                   beats_per_chord, duration, synth_type, velocity)
        return ({"waveform": torch.from_numpy(audio).unsqueeze(0).unsqueeze(0),
                 "sample_rate": _SR},)

NODE_CLASS_MAPPINGS = {
    "AceStepChordPreview": AceStepChordPreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepChordPreview": "ACE-Step ▸ Chord Preview"
}
