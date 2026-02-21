"""AceStepCustomTimesteps node for ACE-Step"""
import torch

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


NODE_CLASS_MAPPINGS = {
    "AceStepCustomTimesteps": AceStepCustomTimesteps,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepCustomTimesteps": "Custom Timesteps",
}
