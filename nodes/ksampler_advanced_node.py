"""AceStepKSamplerAdvanced node for ACE-Step"""
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
from .includes.sampling_utils import apply_shift

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


NODE_CLASS_MAPPINGS = {
    "AceStepKSamplerAdvanced": AceStepKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepKSamplerAdvanced": "KSampler Advanced (Audio)",
}
