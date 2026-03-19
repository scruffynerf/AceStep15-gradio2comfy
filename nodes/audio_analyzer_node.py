"""Audio Analyzer (No LLM) for ACE-Step"""
import torch
import torchaudio
import numpy as np
import logging
from .includes.analysis_utils import LIBROSA_AVAILABLE

logger = logging.getLogger(__name__)

if LIBROSA_AVAILABLE:
    import librosa

class ScromfyAceStepAudioAnalyzerNoLLM:
    """Analyze audio to extract BPM, key/scale, and duration (DSP-only).
    Includes ACE-Step theory overrides for key detection and torch-optimized audio handling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "FLOAT")
    RETURN_NAMES = ("bpm", "key_scale", "duration")
    FUNCTION = "analyze"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def analyze(self, audio):
        """Analyze audio and return BPM, key/scale, duration"""
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        
        # 1. Audio Processing (Torch-based mono & batch handling)
        if waveform.dim() == 3: 
            y_torch = waveform[0].mean(dim=0) # Take first batch, average to mono
        elif waveform.dim() == 2:
            y_torch = waveform.mean(dim=0)
        else:
            y_torch = waveform.squeeze()

        # 2. Duration Calculation
        duration = float(y_torch.shape[-1]) / float(sr)

        # 3. DSP Analysis (Librosa targeting 22050Hz)
        if not LIBROSA_AVAILABLE:
            return (120, "C major", duration)

        target_sr = 22050
        try:
            if sr != target_sr:
                y_torch = torchaudio.functional.resample(y_torch, sr, target_sr)
            
            y_np = y_torch.cpu().numpy()

            # 4. BPM Detection
            tempo, _ = librosa.beat.beat_track(y=y_np, sr=target_sr)
            bpm = int(round(float(tempo if not isinstance(tempo, np.ndarray) else tempo[0])))
            
            # 5. Key/Scale Detection with Theory Overrides
            chroma = librosa.feature.chroma_cqt(y=y_np, sr=target_sr).mean(axis=1)
            pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
            best_idx = np.argmax(chroma)
            is_minor = chroma[(best_idx + 3) % 12] > chroma[(best_idx + 4) % 12]
            
            key_note = pitch_names[best_idx]
            scale = "minor" if is_minor else "major"
            key_scale = f"{key_note} {scale}"
            
            # --- ACE-Step Theory Overrides ---
            if key_note == 'C#' and not is_minor:
                key_scale = 'C major'
            if key_note == 'A#' and is_minor:
                key_scale = 'A minor'
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            bpm = 120
            key_scale = "C major"

        return (bpm, key_scale, duration)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepAudioAnalyzerNoLLM": ScromfyAceStepAudioAnalyzerNoLLM}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepAudioAnalyzerNoLLM": "Scromfy AceStep Audio Analyzer (No LLM)"}
