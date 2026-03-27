import torch
import torchaudio
import logging

logger = logging.getLogger(__name__)

class ScromfyAceStepVAEEncode:
    """Encodes standard audio into ACE-Step 1.5 latents using the VAE.
    Automatically handles resampling to 48kHz and ensuring stereo channels.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def encode(self, vae, audio):
        vae_sr = 48000

        # Extract waveform and sample rate
        waveform = audio["waveform"]  # [B, C, T]
        sr = audio["sample_rate"]

        # 1. Resample if necessary
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)

        # 2. Ensure stereo (ACE-Step requires 2 channels)
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > 2:
            waveform = waveform[:, :2, :]

        # 3. Encode to latent via native VAE
        # vae.encode expects [B, T, C] so we movedim
        latent_tensor = vae.encode(waveform.movedim(1, -1))

        # 4. Wrap in standard ComfyUI latent dict format
        return ({"samples": latent_tensor, "type": "audio"},)

NODE_CLASS_MAPPINGS = {
    "ScromfyAceStepVAEEncode": ScromfyAceStepVAEEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyAceStepVAEEncode": "Scromfy ACE-Step VAE Encode (Audio)"
}
