"""EmptyLatentAudio node for ACE-Step"""
import torch
import comfy.model_management

class EmptyLatentAudio:
    """Create empty audio latent space for generation"""
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def generate(self, seconds, batch_size):
        # ACE-Step audio latents: 44100 Hz / 2048 hop / 2 downscale
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return ({"samples": latent, "type": "audio"}, )


NODE_CLASS_MAPPINGS = {
    "EmptyLatentAudio": EmptyLatentAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentAudio": "Empty Latent Audio",
}
