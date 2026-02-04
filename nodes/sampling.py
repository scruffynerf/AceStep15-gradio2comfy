"""
Audio-optimized sampling nodes for ACE-Step
ADAPTED from JK-AceStep-Nodes/ace_step_ksampler.py

CHANGES FROM ORIGINAL:
1. Removed auto-steps calculation - user has full control
2. Removed vocoder-related code (not needed for core generation)  
3. Simplified to focus on audio-optimized sampling parameters
4. Kept memory optimization and progress tracking
5. Changed category from "JK AceStep Nodes" to "ACE-Step/sampling"
"""

import torch
import logging
import comfy.samplers
import comfy.sample
import comfy.utils

logger = logging.getLogger(__name__)


class AceStepKSampler:
    """
    Audio-optimized KSampler for ACE-Step music generation
    
    ADAPTATION NOTES:
    - Removed automatic steps calculation (user controls steps directly)
    - Removed vocoder integration (separate node if needed)
    - Kept audio-specific CFG guidance and memory optimization
    - Standard ComfyUI KSampler interface with audio defaults
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Scromfy/Ace-Step/sampling"

def apply_shift(sigmas, shift):
    """
    Apply timestep shift formula: t' = shift * t / (1 + (shift - 1) * t)
    Used to adjust noise schedule for better quality in DiT/Flow models.
    """
    if shift == 1.0:
        return sigmas
    
    # Sigmas in Flow Matching usually correspond to t [0, 1]
    # We apply the shift to all sigmas except the last one (0.0)
    shifted_sigmas = sigmas.clone()
    t = sigmas[:-1]
    shifted_sigmas[:-1] = shift * t / (1 + (shift - 1) * t)
    return shifted_sigmas


class AceStepKSampler:
    """
    Audio-optimized KSampler for ACE-Step music generation
    
    ADAPTATION NOTES:
    - Added shift parameter support
    - Removed automatic steps calculation
    - Kept audio-specific CFG guidance and memory optimization
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Scromfy/Ace-Step/sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, shift=1.0):
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        
        # Get sigmas from scheduler
        device = comfy.model_management.get_torch_device()
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).to(device)
        
        # Apply denoise if < 1.0
        if denoise < 1.0:
            sigmas = sigmas[int(len(sigmas) * (1.0 - denoise)):]
            
        # Apply shift
        sigmas = apply_shift(sigmas, shift)
        
        # Internal sampler call using custom sigmas
        samples = comfy.sample.sample_custom(
            model,
            seed,
            cfg,
            sampler_name,
            sigmas,
            positive,
            negative,
            latent_samples,
            disable_noise=False
        )
        
        latent["samples"] = samples
        return (latent,)


class AceStepKSamplerAdvanced:
    """
    Advanced KSampler with additional audio-specific controls
    
    ADAPTATION NOTES:
    - Added shift parameter support (critical for ACE-Step quality)
    - Removed auto-steps calculation
    - Kept advanced step controls
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Scromfy/Ace-Step/sampling"

    def sample(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, start_at_step, end_at_step,
               return_with_leftover_noise, shift=1.0):
        
        force_full_denoise = return_with_leftover_noise == "disable"
        disable_noise = add_noise == "disable"
        
        latent = latent_image.copy()
        latent_samples = latent["samples"]
        
        # Get sigmas
        device = comfy.model_management.get_torch_device()
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).to(device)
        
        # Adjust sigmas for start/end steps
        start_at_step = min(start_at_step, len(sigmas) - 1)
        end_at_step = min(end_at_step, len(sigmas) - 1)
        sigmas = sigmas[start_at_step:end_at_step + 1]
        
        # Apply shift
        sigmas = apply_shift(sigmas, shift)
        
        samples = comfy.sample.sample_custom(
            model,
            noise_seed,
            cfg,
            sampler_name,
            sigmas,
            positive,
            negative,
            latent_samples,
            disable_noise=disable_noise,
            force_full_denoise=force_full_denoise
        )
        
        latent["samples"] = samples
        return (latent,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "AceStepKSampler": AceStepKSampler,
    "AceStepKSamplerAdvanced": AceStepKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepKSampler": "KSampler (Audio-Optimized)",
    "AceStepKSamplerAdvanced": "KSampler Advanced (Audio)",
}
