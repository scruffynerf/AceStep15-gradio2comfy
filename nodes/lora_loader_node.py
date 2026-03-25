import folder_paths

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the LoRA loader node.

class ScromfyAceStepLoraLoader:
    """Chainable LoRA loader for ScromfySFT.
    Accumulates LoRA specifications into a stack that is applied
    when the Model Loader loads its models.
    
    Inputs:
        lora_name (STRING): Target LoRA (.safetensors).
        strength_model (FLOAT): Diffusion network weight.
        (Optional:) lora_stack (ACESTEP_LORA): Previous stack to append to.
        
    Outputs:
        lora_stack (ACESTEP_LORA): Updated stack array.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"), {
                    "tooltip": "AceStep 1.5 LoRA model file.",
                }),
                "strength_model": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "LoRA strength for the diffusion model (MODEL).",
                }),
            },
            "optional": {
                "lora_stack": ("ACESTEP_LORA",),
            },
        }

    RETURN_TYPES = ("ACESTEP_LORA",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load_lora"
    CATEGORY = "Scromfy/Ace-Step/Lora"

    def load_lora(self, lora_name, strength_model, lora_stack=None):
        stack = list(lora_stack) if lora_stack is not None else []
        stack.append({
            "lora_name": lora_name,
            "strength_model": strength_model,
        })
        return (stack,)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepLoraLoader": ScromfyAceStepLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepLoraLoader": "Scromfy AceStep Lora Stack"}
