"""AceStepAudioAnalyzer node for ACE-Step"""
import torch
import numpy as np
import logging
from .includes.analysis_utils import LIBROSA_AVAILABLE

logger = logging.getLogger(__name__)

if LIBROSA_AVAILABLE:
    import librosa

class AceStepAudioAnalyzer:
    """Analyze audio to extract BPM, key/scale, and duration"""
    
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
    CATEGORY = "Scromfy/Ace-Step/analysis"

    def analyze(self, audio):
        """Analyze audio and return BPM, key/scale, duration"""
        waveform = audio["waveform"]
        sample_rate = audio.get("sample_rate", 44100)
        
        # Convert to numpy for analysis
        if isinstance(waveform, torch.Tensor):
            audio_np = waveform.cpu().numpy()
        else:
            audio_np = waveform
        
        # Handle batch dimension
        if len(audio_np.shape) == 3:
            audio_np = audio_np[0]  # Take first batch
        
        # Convert stereo to mono
        if audio_np.shape[0] > 1:
            audio_np = audio_np.mean(axis=0)
        else:
            audio_np = audio_np[0]
        
        # Calculate duration
        duration = len(audio_np) / sample_rate
        
        # Detect BPM
        bpm = self._detect_bpm(audio_np, sample_rate)
        
        # Detect key/scale
        key_scale = self._detect_key(audio_np, sample_rate)
        
        return (bpm, key_scale, duration)
    
    def _detect_bpm(self, audio_np, sample_rate):
        """Detect BPM using librosa if available"""
        if not LIBROSA_AVAILABLE:
            logger.warning("librosa not available, returning default BPM")
            return 120
        
        try:
            # Use librosa tempo detection
            tempo, _ = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
            bpm = int(np.round(tempo))
            
            # Sanity check
            if bpm < 60 or bpm > 200:
                logger.warning(f"Detected BPM {bpm} out of range, using 120")
                return 120
            
            return bpm
        except Exception as e:
            logger.error(f"BPM detection failed: {e}")
            return 120
    
    def _detect_key(self, audio_np, sample_rate):
        """Detect musical key/scale using librosa if available"""
        if not LIBROSA_AVAILABLE:
            return "C major"
        
        try:
            # Calculate chroma features
            chroma = librosa.feature.chroma_cqt(y=audio_np, sr=sample_rate)
            
            # Average across time
            chroma_mean = chroma.mean(axis=1)
            
            # Note names
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            
            # Find strongest note
            key_idx = np.argmax(chroma_mean)
            key_note = notes[key_idx]
            
            # Simple major/minor detection based on third
            major_third_idx = (key_idx + 4) % 12
            minor_third_idx = (key_idx + 3) % 12
            
            if chroma_mean[major_third_idx] > chroma_mean[minor_third_idx]:
                scale = "major"
            else:
                scale = "minor"
            
            return f"{key_note} {scale}"
        except Exception as e:
            logger.error(f"Key detection failed: {e}")
            return "C major"


NODE_CLASS_MAPPINGS = {
    "AceStepAudioAnalyzer": AceStepAudioAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioAnalyzer": "Audio Analyzer",
}
