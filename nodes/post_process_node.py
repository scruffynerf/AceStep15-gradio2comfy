"""AceStepPostProcess node for ACE-Step"""
import torch
import logging

logger = logging.getLogger(__name__)

class AceStepPostProcess:
    """Post-process audio with de-esser and spectral smoothing"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "de_esser_strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.6, "step": 0.01}),
                "spectral_smoothing": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "process"
    CATEGORY = "Scromfy/Ace-Step/post-process"

    def process(self, audio, de_esser_strength=0.12, spectral_smoothing=0.08):
        try:
            waveform = audio["waveform"] if isinstance(audio, dict) and "waveform" in audio else audio
            if not isinstance(waveform, torch.Tensor):
                logger.warning("Input audio is not a torch.Tensor, skipping post-processing.")
                return (audio,)
            
            x = waveform
            # Expect shape [B, C, T]
            if x.dim() == 2:
                x = x.unsqueeze(1)

            B, C, T = x.shape
            # Short-time Fourier Transform parameters
            n_fft = 2048
            hop_length = 512
            win = torch.hann_window(n_fft).to(x.device)
            
            # Apply STFT per channel
            out = x.clone()
            for b in range(B):
                for c in range(C):
                    sig = x[b, c]
                    stft = torch.stft(sig, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, return_complex=True)
                    mag = torch.abs(stft)
                    phase = torch.angle(stft)
                    
                    # Apply de-esser: reduce energy above 6kHz proportionally
                    sr = audio.get('sample_rate', 44100) if isinstance(audio, dict) else 44100
                    freqs = torch.fft.rfftfreq(n_fft, 1.0/sr).to(x.device)
                    mask = (freqs > 6000).float().view(1, -1)
                    mag = mag * (1.0 - (de_esser_strength * mask))
                    
                    # Spectral smoothing across frequency
                    if spectral_smoothing > 0.0:
                        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=mag.dtype, device=mag.device).view(1, 1, -1)
                        padded = torch.nn.functional.pad(mag, (1, 1, 0, 0), mode='reflect')
                        smoothed_mag = torch.nn.functional.conv1d(padded, kernel, padding=0)
                        mag = (1.0 - spectral_smoothing) * mag + spectral_smoothing * smoothed_mag
                    
                    complex_spec = torch.polar(mag, phase)
                    sig_rec = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=win, length=T)
                    out[b, c] = sig_rec
            
            # Re-normalize
            out = out / (out.abs().max().clamp(min=1e-5))
            
            if isinstance(audio, dict):
                audio["waveform"] = out
                return (audio,)
            else:
                return ({"waveform": out, "sample_rate": sr},)
                
        except Exception as e:
            logger.error(f"Post processing failed: {e}")
            return (audio,)


NODE_CLASS_MAPPINGS = {
    "AceStepPostProcess": AceStepPostProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPostProcess": "Post Process",
}
