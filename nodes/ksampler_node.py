"""AceStepKSampler node for ACE-Step"""
import comfy.samplers
import comfy.sample
import comfy.utils
import comfy.model_management
from .includes.sampling_utils import apply_shift

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


NODE_CLASS_MAPPINGS = {
    "AceStepKSampler": AceStepKSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepKSampler": "KSampler (Audio-Optimized)",
}
