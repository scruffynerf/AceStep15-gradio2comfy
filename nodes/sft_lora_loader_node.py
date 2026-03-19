import folder_paths

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the LoRA loader node.

class ScromfySFTLoraLoader:
    """Chainable LoRA loader for ScromfySFT.
    Accumulates LoRA specifications into a stack that is applied
    when the Model Loader loads its models.
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
                "strength_clip": ("FLOAT", {
                    "default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                    "tooltip": "LoRA strength for the text encoders (CLIP).",
                }),
            },
            "optional": {
                "lora_stack": ("ACESTEP_LORA",),
            },
        }

    RETURN_TYPES = ("ACESTEP_LORA",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "load_lora"
    CATEGORY = "Scromfy/Ace-Step/SFT"

    def load_lora(self, lora_name, strength_model, strength_clip, lora_stack=None):
        stack = list(lora_stack) if lora_stack is not None else []
        stack.append({
            "lora_name": lora_name,
            "strength_model": strength_model,
            "strength_clip": strength_clip,
        })
        return (stack,)

NODE_CLASS_MAPPINGS = {"ScromfySFTLoraLoader": ScromfySFTLoraLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfySFTLoraLoader": "ScromfySFT Lora Loader"}
