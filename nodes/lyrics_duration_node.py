"""AceStepLyricsBPMCalculator node for ACE-Step — estimate BPM range from lyrics"""
import math


class AceStepLyricsBPMCalculator:
    """Estimate suggested BPM range based on lyric line count and vocal pacing.
    
    Formula: BPM = (Bars × Beats_per_bar × 60) / Duration
    - Bars estimated from: Line Count × line2bar (Density)
    - Duration estimated from: Word Count / WPM
    """

    TIMESIG_MAP = {
        "2/4": 2,
        "3/4": 3,
        "4/4": 4,
        "6/8": 6,
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "timesignature": (list(cls.TIMESIG_MAP.keys()), {"default": "4/4"}),
                "wpm": ("INT", {
                    "default": 165, "min": 50, "max": 400, "step": 5,
                    "display": "slider",
                    "tooltip": "Vocal pacing (Words Per Minute). ~165 is standard.",
                }),
                "line2bar": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 4.0, "step": 0.1,
                    "display": "slider",
                    "tooltip": "Average bars per line of lyrics (Density). Default 1.0.",
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT", "INT", "INT", "INT",)
    RETURN_NAMES = ("duration", "bpm_low", "bpm_mid", "bpm_high", "word_count", "line_count",)
    FUNCTION = "calculate"
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def calculate(self, lyrics: str, timesignature: str, wpm: int, line2bar: float):
        # 1. Clean lyrics: ignore [tags] and empty lines
        raw_lines = lyrics.split('\n')
        clean_lines = []
        words = []
        
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            # Ignore [Chorus], [Verse], etc.
            if line.startswith('[') and line.endswith(']'):
                continue
                
            clean_lines.append(line)
            # Count words on this line (ignoring embedded [tags] if any)
            line_words = [w for w in line.split() if not (w.startswith('[') and w.endswith(']'))]
            words.extend(line_words)

        l_count = len(clean_lines)
        w_count = len(words)
        beats = self.TIMESIG_MAP.get(timesignature, 4)

        if w_count == 0 or l_count == 0:
            return (0.0, 0.0, 0.0, 0.0, 0, 0)

        # 2. Duration fixed by pacing (WPM)
        duration = round((w_count / wpm) * 60.0, 1)

        # 3. Calculate BPM range based on Bar Density (Bars / Line)
        # Using the formula: BPM = (Bars * Beats * 60) / Duration
        # We vary the density: line2bar (Low), line2bar * 1.5 (Mid), line2bar * 2.0 (High)
        
        def calc_bpm(density):
            bars = l_count * density
            return (bars * beats * 60) / duration

        bpm_low = int(round(calc_bpm(line2bar), 1))
        bpm_mid = int(round(calc_bpm(line2bar * 1.5), 1))
        bpm_high = int(round(calc_bpm(line2bar * 2.0), 1))

        return (duration, bpm_low, bpm_mid, bpm_high, w_count, l_count)


NODE_CLASS_MAPPINGS = {
    "AceStepLyricsBPMCalculator": AceStepLyricsBPMCalculator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLyricsBPMCalculator": "Lyrics BPM Calculator",
}
