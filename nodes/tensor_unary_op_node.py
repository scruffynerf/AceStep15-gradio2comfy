"""AceStepTensorUnaryOp node for ACE-Step"""
import torch

class AceStepTensorUnaryOp:
    """Operations that transform a single input A with optional masking"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_A": ("TENSOR",),
                "mode": (["gate", "scale_masked", "noise_masked", "fade_out"], {"default": "gate"}),
                "length_pct": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "process"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def process(self, tensor_A, mode, length_pct, strength, sigma, seed, mask=None):
        A = tensor_A.clone()
        
        # Handle Length Scaling
        if length_pct != 100.0:
            L_idx = 1 if A.dim() == 3 else 0
            curr_len = A.shape[L_idx]
            new_len = max(1, int(curr_len * (length_pct / 100.0)))
            
            if A.dim() == 3:
                # [B, L, D] -> [B, D, L]
                A = A.transpose(1, 2)
                A = torch.nn.functional.interpolate(A, size=new_len, mode='linear', align_corners=False)
                A = A.transpose(1, 2)
            else:
                # [L, D] -> [1, D, L]
                A = A.unsqueeze(0).transpose(1, 2)
                A = torch.nn.functional.interpolate(A, size=new_len, mode='linear', align_corners=False)
                A = A.transpose(1, 2).squeeze(0)

        # Default mask is all ones if not provided
        if mask is None:
            if A.dim() == 3:
                mask = torch.ones((1, A.size(1), 1), device=A.device)
            else:
                mask = torch.ones((A.size(0), 1), device=A.device)
                mask = mask.unsqueeze(0) # [1, L, 1]
        else:
            # Existing mask logic already handles interpolation to A's current length
            mask = mask.to(A.device)
            if mask.dim() == 2: mask = mask.unsqueeze(-1)
            elif mask.dim() == 3: mask = mask.mean(dim=1).unsqueeze(-1)
            
            if mask.shape[1] != A.shape[1] if A.dim() == 3 else A.shape[0]:
                target_len = A.shape[1] if A.dim() == 3 else A.shape[0]
                mask = mask.transpose(1, 2)
                mask = torch.nn.functional.interpolate(mask, size=target_len, mode='linear', align_corners=False)
                mask = mask.transpose(1, 2)
        
        if mode == "gate":
            # Gating: A * mask
            return (A * mask,)
            
        elif mode == "scale_masked":
            # Scale A only where mask > 0
            return (A * (1.0 + mask * (strength - 1.0)),)
            
        elif mode == "noise_masked":
            # Add noise only where mask allows
            generator = torch.Generator(device=A.device)
            generator.manual_seed(seed)
            noise = torch.randn_like(A, generator=generator) * sigma
            return (A + mask * noise,)
            
        elif mode == "fade_out":
            # Special case: linear ramp multiply
            N = A.size(1) if A.dim() == 3 else A.size(0)
            fade = torch.linspace(1, 0, N, device=A.device)[None, :, None]
            return (A * (1.0 - mask) + (A * fade) * mask,)
            
        return (A,)

NODE_CLASS_MAPPINGS = {
    "AceStepTensorUnaryOp": AceStepTensorUnaryOp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepTensorUnaryOp": "Tensor Conditioning Solo Operation",
}
