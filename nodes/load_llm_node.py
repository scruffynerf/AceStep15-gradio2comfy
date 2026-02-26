"""AceStepLLMLoader node for ACE-Step"""
import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import folder_paths
except ImportError:
    folder_paths = None

logger = logging.getLogger(__name__)

class AceStepLLMLoader:
    """Load an ACE-Step 5Hz LM (Qwen) for standalone text generation"""
    
    @classmethod
    def INPUT_TYPES(s):
        checkpoints = []
        if folder_paths:
            checkpoints = folder_paths.get_filename_list("checkpoints")
        
        return {
            "required": {
                "model_name": (checkpoints,),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"}),
            }
        }
    
    RETURN_TYPES = ("ACE_LLM",)
    RETURN_NAMES = ("llm",)
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    def load(self, model_name, device, precision):
        if not folder_paths:
            raise ImportError("ComfyUI folder_paths not found.")
            
        model_path = folder_paths.get_full_path("checkpoints", model_name)
        if not model_path:
            raise FileNotFoundError(f"Model path for {model_name} not found.")

        # Resolve device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve dtype
        dtype = torch.float32
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            dtype = torch.bfloat16
            
        # Transformers AutoModel.from_pretrained expects a DIRECTORY for local paths.
        # If model_path points to a file (like model.safetensors), we use its parent directory.
        model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
            
        logger.info(f"Loading ACE-Step LLM from {model_dir} on {device} (precision: {precision})")
        
        try:
            # We use local_files_only=True to ensure we don't hit the hub
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, 
                trust_remote_code=True, 
                local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_dir, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                local_files_only=True
            ).to(device)
            
            return ({"model": model, "tokenizer": tokenizer, "device": device},)
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise e

NODE_CLASS_MAPPINGS = {
    "AceStepLLMLoader": AceStepLLMLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLLMLoader": "LLM Loader (ACE-Step)",
}
