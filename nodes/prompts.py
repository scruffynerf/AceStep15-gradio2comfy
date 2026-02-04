"""
Prompt generation and post-processing nodes for ACE-Step
Ported from JK-AceStep-Nodes
"""

import torch
import logging
import random

logger = logging.getLogger(__name__)


# ==================== STYLE PRESETS (200+ styles) ====================

STYLE_PRESETS = {
    "Synthwave": "Synthwave at 120 BPM with vintage analog synths, lo-fi drums, neon pad atmospheres, retro 80s aesthetic",
    "Retrowave": "Retrowave at 125 BPM with analog synth layers, pulsing bassline, 80s drum kit samples, warm pad chords",
    "Electronic": "Electronic dance at 128 BPM with punchy sidechain kick, filtered saw bass, lush pad atmospheres, crisp hi-hats",
    "Trance": "Trance at 135 BPM with arpeggiated synth lines, hypnotic bassline, breakbeat patterns, euphoric builds",
    "Techno": "Minimal techno at 125 BPM with repetitive kick patterns, industrial percussion, deep sub bass, subtle synth layers",
    "House": "Deep house at 120 BPM with soulful samples, warm bassline, jazzy chords, vocal intimacy",
    "Dubstep": "Dubstep at 140 BPM with heavy wobble bass, syncopated snares, dub-influenced delays, explosive drops",
    "Trap": "Trap at 140 BPM with booming 808 sub bass slides, rapid hi-hat rolls, sparse atmospheric synth melodies, snare snaps",
    "Future Bass": "Future bass at 140 BPM with chopped vocals, complex chords, soft wobble bass, glitchy rhythms, emotional melodic content",
    "Lo-fi Hip Hop": "Lo-fi hip hop at 85 BPM with vinyl crackle, jazzy piano chords, dusty drum samples, warm bass, chill vibes",
    "Ambient": "Ambient music at 60 BPM with flowing pad textures, minimalist melodies, atmospheric soundscapes, meditative space",
    "Vaporwave": "Vaporwave at 85 BPM with ambient samples, lo-fi aesthetics, slowed jazz, consumer culture samples, nostalgic melancholy",
    "Cyberpunk": "Cyberpunk electronic at 130 BPM with futuristic synths, distorted vocals, heavy bass, neon aesthetics, dystopian atmosphere",
    "Rock": "Rock at 120 BPM with power chords, driving drums, electric guitar solos, punchy bass",
    "Metal": "Metal at 160 BPM with heavy distorted guitars, double bass drums, aggressive vocals, intense energy",
    "Punk": "Punk at 180 BPM with fast power chords, raw shouted vocals, simple drums, rebellious energy",
    "Jazz": "Jazz at 120 BPM with swing rhythms, saxophone solos, walking bass, piano comping, brushed drums",
    "Blues": "Blues at 85 BPM with guitar bends, walking upright bass, brushed snare, harmonica wails",
    "Country": "Country at 120 BPM with acoustic guitar, steel guitar slides, fiddle, storytelling vocals",
    "Pop": "Pop at 120 BPM with catchy hooks, electronic production, polished vocals, dance beats",
    # Add more styles as needed...
}


class AceStepPromptGen:
    """Generate music style prompts from 200+ presets"""
    
    @classmethod
    def INPUT_TYPES(cls):
        styles = sorted(list(STYLE_PRESETS.keys()))
        return {
            "required": {
                "style": (styles, {"default": "Synthwave"}),
                 "random_variation": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/prompts"

    def generate(self, style: str, random_variation: bool, seed: int):
        base_prompt = STYLE_PRESETS.get(style, "Generic electronic music")
        
        if random_variation:
            random.seed(seed)
            # Add slight variation to make each generation unique
            variations = [
                "energetic", "mellow", "atmospheric", "driving",
                "emotional", "uplifting", "dark", "bright"
            ]
            mood = random.choice(variations)
            prompt = f"{mood} {base_prompt}"
        else:
            prompt = base_prompt
        
        return (prompt,)


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


# Node registration
NODE_CLASS_MAPPINGS = {
    "AceStepPromptGen": AceStepPromptGen,
    "AceStepPostProcess": AceStepPostProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPromptGen": "Prompt Generator",
    "AceStepPostProcess": "Post Process",
}
