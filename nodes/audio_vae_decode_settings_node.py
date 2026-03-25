# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This node holds VAE/Post-process settings to keep the main Sampler node clean.

class ScromfyAceStepVAEDecodeSettings:
    """VAE Decode and Post-processing settings for ScromfyAceStepSampler.
    Encapsulates latent shifts, rescaling, and audio enhancements.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_shift": ("FLOAT", {
                    "default": 0.0, "min": -0.2, "max": 0.2, "step": 0.01,
                    "tooltip": "Additive shift on DiT latents before VAE decode (anti-clipping)",
                }),
                "latent_rescale": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Multiplicative scale on DiT latents before VAE decode",
                }),
                "normalize_peak": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable peak normalization (normalize to max amplitude). Disabled by default to preserve the model's natural dynamics and transient balance",
                }),
                "voice_boost": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Voice boost in dB. Positive = louder voice (use with reference_audio). Default 0 dB",
                }),
            }
        }

    RETURN_TYPES = ("SCROMFY_VAE_SETTINGS",)
    RETURN_NAMES = ("vae_decode_settings",)
    FUNCTION = "get_settings"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def get_settings(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepVAEDecodeSettings": ScromfyAceStepVAEDecodeSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepVAEDecodeSettings": "Scromfy AceStep VAE Decode Settings"}
