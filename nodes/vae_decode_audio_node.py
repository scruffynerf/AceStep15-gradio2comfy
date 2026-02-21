"""VAEDecodeAudio node for ACE-Step"""
import torch

class VAEDecodeAudio:
    """Decode latent space to audio waveform"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", )
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        # Normalize audio to prevent clipping
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return ({"waveform": audio, "sample_rate": 44100}, )


NODE_CLASS_MAPPINGS = {
    "VAEDecodeAudio": VAEDecodeAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEDecodeAudio": "VAE Decode (Audio)",
}
