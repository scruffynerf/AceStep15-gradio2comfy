import numpy as np
from .includes.chord_utils import (
    build_chord_audio,
    vae_encode_audio,
    extract_fsq_codes,
    patch_conditioning,
    _SR
)

_CHORD_MAP_PLACEHOLDER = """\
[intro]   Am F C G
[verse]   Am F C G
[chorus]  F C G Am
[bridge]  Dm Am Bb G
[outro]   Am F C G
default   Am F C G"""

class AceStepChordConditioner:
    """
    ACE-Step ▸ Chord Conditioner (Native)

    Maps chord progressions to lyrics sections, synthesises reference audio,
    encodes through the ACE-Step VAE and FSQ quantizer, and injects the
    resulting tokens into the conditioning dict.
    """
    CATEGORY = "Scromfy/Ace-Step/Chords"
    FUNCTION = "generate"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "vae":   ("VAE",),
                "model": ("MODEL",),
                "chord_map": ("STRING", {
                    "multiline": True,
                    "default": _CHORD_MAP_PLACEHOLDER,
                    "tooltip": (
                        "Map section names to chord progressions.\n"
                        "Format:\n"
                        "  [verse]   Am F C G\n"
                        "  [chorus]  F C G Am\n"
                        "  default   Am F C G\n\n"
                        "Section names must match [tags] in your lyrics.\n"
                        "Beat counts: Am:2 F:2 C G"
                    ),
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": (
                        "Paste the same lyrics as in TextEncodeAceStepAudio1.5.\n"
                        "Used to calculate how long each section is.\n"
                        "Leave empty to use 'default' chords for the full duration."
                    ),
                }),
                "bpm": ("INT", {
                    "default": 120, "min": 40, "max": 240, "step": 1,
                    "tooltip": "Match the BPM in TextEncodeAceStepAudio1.5",
                }),
                "beats_per_chord": ("FLOAT", {
                    "default": 4.0, "min": 0.5, "max": 16.0, "step": 0.5,
                    "tooltip": "Default beats per chord when not specified inline",
                }),
                "duration": ("FLOAT", {
                    "default": 30.0, "min": 5.0, "max": 600.0, "step": 1.0,
                    "tooltip": "Must match duration in TextEncodeAceStepAudio1.5",
                }),
                "synth_type": (["piano", "organ", "pad"], {"default": "piano"}),
            },
            "optional": {
                "velocity": ("FLOAT", {"default": 0.65, "min": 0.1, "max": 1.0, "step": 0.05}),
                "clip":     ("CLIP",),   # unused, kept for workflow compat
            },
        }

    def generate(self, conditioning, vae, model, chord_map, lyrics,
                 bpm, beats_per_chord, duration, synth_type,
                 velocity=0.65, clip=None, **kwargs):

        print("\n[AceStepChord] ════════════════════════════════════════════")
        print(f"  bpm={bpm}  bpc={beats_per_chord}  dur={duration}s  synth={synth_type}")

        # ── 1. Build section-aware chord audio ────────────────────────────
        audio = build_chord_audio(
            lyrics        = lyrics,
            chord_map_txt = chord_map,
            bpm           = float(bpm),
            beats_per_chord = beats_per_chord,
            total_dur     = duration,
            synth_type    = synth_type,
            velocity      = velocity,
        )
        print(f"  synth: {len(audio)/_SR:.2f}s  peak={np.max(np.abs(audio)):.3f}")

        # ── 2. VAE encode → latents ──────────────────────────────────────
        latents = vae_encode_audio(vae, audio)
        if latents is None:
            print("  VAE encode failed — conditioning unchanged")
            return (conditioning,)
        print(f"  latents: {list(latents.shape)}")

        # ── 3. FSQ tokenise: latents [B,C,T] → transpose → [B,T,C] bfloat16 ──
        codes_list = extract_fsq_codes(model, latents)
        if codes_list is None:
            print("  FSQ failed — conditioning unchanged")
            return (conditioning,)
        print(f"  codes: {len(codes_list)} tokens")

        # ── 4. Inject ─────────────────────────────────────────────────────
        out = patch_conditioning(conditioning, codes_list)
        print("[AceStepChord] ════════════════════════════════════════════\n")
        return (out,)

NODE_CLASS_MAPPINGS = {
    "AceStepChordConditioner": AceStepChordConditioner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepChordConditioner": "ACE-Step ▸ Chord Conditioner"
}
