"""AceStepAudioMask node for ACE-Step"""
import torch

class AceStepAudioMask:
    """Generate time-based audio mask for inpainting"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (["all", "none", "fraction", "range", "ramp", "window"], {"default": "range"}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "end_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "fraction": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ramp_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def create_mask(self, audio, mode, start_seconds, end_seconds, fraction, ramp_seconds, reverse):
        waveform = audio["waveform"]
        sample_rate = audio.get("sample_rate", 44100)
        total_seconds = waveform.shape[2] / sample_rate
        
        # ACE-Step latent downsampling: 44100 Hz / 2048 hop / 2 downscale = ~10.76 Hz latent rate
        latent_length = round((total_seconds * 44100 / 2048) / 2) * 2
        device = waveform.device
        mask = torch.zeros(latent_length, device=device)
        
        def to_idx(sec):
            if sec < 0: return latent_length
            return min(latent_length, int((sec / total_seconds) * latent_length))

        if mode == "all":
            mask = torch.ones(latent_length, device=device)
        elif mode == "none":
            mask = torch.zeros(latent_length, device=device)
        elif mode == "fraction":
            cutoff = int(latent_length * fraction)
            if reverse:
                mask[cutoff:] = 1.0
            else:
                mask[:cutoff] = 1.0
        elif mode == "range":
            s = to_idx(start_seconds)
            e = to_idx(end_seconds)
            mask[s:e] = 1.0
        elif mode == "ramp":
            ramp = torch.linspace(0, 1, latent_length, device=device)
            if reverse:
                ramp = 1.0 - ramp
            mask = ramp
        elif mode == "window":
            s = to_idx(start_seconds)
            e = to_idx(end_seconds)
            # Hard center
            mask[s:e] = 1.0
            # Ramps
            if ramp_seconds > 0:
                r_idx = int((ramp_seconds / total_seconds) * latent_length)
                if r_idx > 0:
                    # Left ramp
                    l_start = max(0, s - r_idx)
                    l_len = s - l_start
                    if l_len > 0:
                        mask[l_start:s] = torch.linspace(0, 1, l_len, device=device)
                    # Right ramp
                    r_end = min(latent_length, e + r_idx)
                    r_len = r_end - e
                    if r_len > 0:
                        mask[e:r_end] = torch.linspace(1, 0, r_len, device=device)

        if mode != "fraction" and mode != "ramp" and reverse:
            mask = 1.0 - mask
            
        # Reshape to [1, N, 1] to match ACE-Step requirements
        return (mask.clamp(0, 1)[None, :, None],)


NODE_CLASS_MAPPINGS = {
    "AceStepAudioMask": AceStepAudioMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioMask": "Scromfy Audio Mask",
}
