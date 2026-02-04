"""
Advanced nodes for ACE-Step
"""

from __future__ import annotations
import torch
import os

class AceStepModeSelector:
    """Convenience node to route inputs based on generation mode"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Simple", "Custom", "Cover", "Repaint"],),
                "description": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "source_audio": ("AUDIO",),
                "repaint_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "repaint_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("final_prompt", "final_lyrics", "active_audio", "start", "end")
    FUNCTION = "route"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def route(self, mode, description, prompt, lyrics, reference_audio=None, source_audio=None, repaint_start=0.0, repaint_end=-1.0):
        final_prompt = ""
        final_lyrics = lyrics
        active_audio = None
        start = 0.0
        end = -1.0
        
        if mode == "Simple":
            final_prompt = description
        elif mode == "Custom":
            final_prompt = prompt
            active_audio = reference_audio
        elif mode == "Cover":
            final_prompt = prompt
            active_audio = source_audio
        elif mode == "Repaint":
            final_prompt = prompt
            active_audio = source_audio
            start = repaint_start
            end = repaint_end
            
        return (final_prompt, final_lyrics, active_audio, start, end)


class AceStep5HzLMConfig:
    """Configures Language Model parameters for ACE-Step generation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "negative_prompt": ("STRING", {"default": "NO USER INPUT", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("lm_config",)
    FUNCTION = "build_config"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def build_config(self, temperature, cfg_scale, top_k, top_p, negative_prompt):
        return ({
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "top_k": top_k if top_k > 0 else None,
            "top_p": top_p,
            "negative_prompt": negative_prompt,
        },)


class AceStepCustomTimesteps:
    """Parse custom sigma schedule from string"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "timesteps": ("STRING", {"default": "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "parse"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def parse(self, timesteps):
        try:
            steps = [float(s.strip()) for s in timesteps.split(',') if s.strip()]
            return (torch.tensor(steps),)
        except Exception as e:
            # Fallback to default if parsing fails
            return (torch.tensor([0.97, 0.76, 0.615, 0.5, 0.395, 0.28, 0.18, 0.085, 0]),)


class AceStepLoRAStatus:
    """Display information about loaded LoRA"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "lora_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check_status"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def check_status(self, lora_name=""):
        if not lora_name:
            return ("No LoRA loaded",)
        return (f"LoRA Active: {lora_name}",)


class AceStepConditioning:
    """Combine text, lyrics, and timbre conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_cond": ("CONDITIONING",),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True}),
                "timbre_audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def combine(self, text_cond, lyrics=None, timbre_audio=None):
        # Conditioning is a list of lists: [[cond, {"pooled_output": ...}]]
        new_cond = []
        for t in text_cond:
            c = t[0]
            metadata = t[1].copy()
            if lyrics:
                metadata["lyrics"] = lyrics
            if timbre_audio:
                metadata["timbre_audio"] = timbre_audio
            new_cond.append([c, metadata])
            
        return (new_cond,)


NODE_CLASS_MAPPINGS = {
    "AceStepModeSelector": AceStepModeSelector,
    "AceStep5HzLMConfig": AceStep5HzLMConfig,
    "AceStepCustomTimesteps": AceStepCustomTimesteps,
    "AceStepLoRAStatus": AceStepLoRAStatus,
    "AceStepConditioning": AceStepConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepModeSelector": "Mode Selector",
    "AceStep5HzLMConfig": "5Hz LM Config",
    "AceStepCustomTimesteps": "Custom Timesteps",
    "AceStepLoRAStatus": "LoRA Status",
    "AceStepConditioning": "Combined Conditioning",
}
