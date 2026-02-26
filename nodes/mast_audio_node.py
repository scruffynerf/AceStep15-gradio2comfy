"""AceStepAudioMask node for ACE-Step"""
import torch

class AceStepAudioMask:
    """Generate time-based audio mask for inpainting"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def create_mask(self, audio, start_seconds, end_seconds):
        waveform = audio["waveform"]
        sample_rate = audio.get("sample_rate", 44100)
        total_seconds = waveform.shape[2] / sample_rate
        
        if end_seconds == -1.0 or end_seconds > total_seconds:
            end_seconds = total_seconds
            
        # ACE-Step latent downsampling: 44100 Hz / 2048 hop / 2 downscale = ~10.76 Hz latent rate
        latent_length = round((total_seconds * 44100 / 2048) / 2) * 2
        mask = torch.zeros([latent_length])
        
        start_idx = int((start_seconds / total_seconds) * latent_length)
        end_idx = int((end_seconds / total_seconds) * latent_length)
        
        mask[start_idx:end_idx] = 1.0
        
        return (mask.unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "AceStepAudioMask": AceStepAudioMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioMask": "Audio Mask",
}
