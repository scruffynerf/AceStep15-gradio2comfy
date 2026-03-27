import torch
import torch.nn.functional as F
import torchaudio

class ScromfyAceStepAudioVAEDecodePlusPlus:
    """Enhanced VAE Decode node for Scromfy AceStep.
    Replicates the VAE decoding logic from the Sampler node, including
    latent shifts, rescaling, peak normalization, and voice boost.
    Supports local settings or an optional settings node input.
    
    Inputs:
        samples (LATENT): Generated DiT latents.
        vae (VAE): The loaded decoder model.
        latent_shift (FLOAT): Pre-decode additive shift.
        latent_rescale (FLOAT): Pre-decode multiplicative scaling.
        normalize_peak (BOOLEAN): Peak limit the output to -0.0dB.
        voice_boost (FLOAT): Add gain specifically tuned for vocal frequencies.
        
    Optional Inputs:
        vae_decode_settings (SCROMFY_VAE_SETTINGS): External configuration bundle. Overrides local parameters if provided.
        
    Outputs:
        audio (AUDIO): Standard audio dict {waveform, sample_rate}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
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
                    "tooltip": "Enable peak normalization (normalize to max amplitude).",
                }),
                "voice_boost": ("FLOAT", {
                    "default": 0.0, "min": -12.0, "max": 12.0, "step": 0.5,
                    "tooltip": "Voice boost in dB. Positive = louder voice. Default 0 dB",
                }),
            },
            "optional": {
                "vae_decode_settings": ("SCROMFY_VAE_SETTINGS",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def decode(self, samples, vae, latent_shift, latent_rescale, normalize_peak, voice_boost, vae_decode_settings=None):
        # 1. Settings extraction (settings node takes precedence)
        vs = vae_decode_settings or {}
        l_shift = vs.get("latent_shift", latent_shift)
        l_rescale = vs.get("latent_rescale", latent_rescale)
        n_peak = vs.get("normalize_peak", normalize_peak)
        v_boost = vs.get("voice_boost", voice_boost)

        # 2. Extract latent tensor
        latent_samples = samples["samples"]
        vae_sr = 48000 # Standard AceStep VAE sample rate

        # 3. Apply latent shift and rescale
        if l_shift != 0.0 or l_rescale != 1.0:
            latent_samples = latent_samples * l_rescale + l_shift

        # 4. VAE decode
        # Oobleck .decode returns a tensor [B, 2, T] after movedim if we follow sampler pattern
        # Actually sampler does: audio = vae.decode(samples).movedim(-1, 1)
        
        # Following ScromfyAceStepSampler logic for exact parity:
        audio = vae.decode(latent_samples).movedim(-1, 1)
        
        if audio.dtype != torch.float32: 
            audio = audio.float()

        # 5. Peak Normalization
        if n_peak:
            peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
            audio = audio / peak

        # 6. Voice Boost (Post-processing)
        if v_boost != 0.0:
            boost = 10.0 ** (v_boost / 20.0)
            audio = torch.tanh(audio * boost * 0.99) / 0.99

        audio_output = {
            "waveform": audio, 
            "sample_rate": vae_sr
        }

        return (audio_output,)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepAudioVAEDecodePlusPlus": ScromfyAceStepAudioVAEDecodePlusPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepAudioVAEDecodePlusPlus": "Audio VAE Decode PLUSPLUS"}
