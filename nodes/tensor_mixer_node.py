"""AceStepTensorMixer node for ACE-Step"""
import torch
import torch.nn.functional as F

class AceStepTensorMixer:
    """Consolidated Binary Toolbox for mixing and combining two tensors with optional masking"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_A": ("TENSOR",),
                "tensor_B": ("TENSOR",),
                "mode": ([
                    "blend", "lerp", "inject", "average", "difference_injection", 
                    "dominant_recessive", "replace", "concatenate", "add", 
                    "multiply", "maximum", "minimum"
                ], {"default": "blend"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "eps": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_mode": (["scale_B_to_A", "scale_A_to_B", "pad_to_match", "none"], {"default": "scale_B_to_A"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "mix"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def mix(self, tensor_A, tensor_B, mode, alpha, ratio, weight, eps, scale_mode, mask=None):
        A = tensor_A.clone()
        B = tensor_B.clone()
        
        # Sequence length dimension is usually index 1 if 3D, 0 if 2D
        L_idx = 1 if A.dim() == 3 else 0
        
        # Scaling / Interpolation (Silent scaling by default)
        if scale_mode == "scale_B_to_A" and A.shape[L_idx] != B.shape[L_idx]:
            B = self.interpolate_tensor(B, A.shape[L_idx], L_idx)
        elif scale_mode == "scale_A_to_B" and A.shape[L_idx] != B.shape[L_idx]:
            A = self.interpolate_tensor(A, B.shape[L_idx], L_idx)
        elif scale_mode == "pad_to_match" and A.shape[L_idx] != B.shape[L_idx]:
            A, B = self.pad_tensors(A, B, L_idx)
        elif scale_mode == "none" and A.shape[L_idx] != B.shape[L_idx] and mode != "concatenate":
            # For non-concatenation modes, if shapes don't match and no scaling requested, 
            # we still need to match for element-wise ops or it will crash.
            # Defaulting to interpolate B to A as the "safest" silent fallback.
            B = self.interpolate_tensor(B, A.shape[L_idx], L_idx)
            
        # Default mask is all ones
        if mask is None:
            mask = torch.ones((1, A.size(1), 1), device=A.device)
            
        # Core Operations
        if mode == "blend":
            # lerp(B, A, mask) -> mask == 1 -> A, mask == 0 -> B
            out = mask * A + (1.0 - mask) * B
        elif mode == "lerp":
            # Blend A and B by alpha, only in masked area
            blended = alpha * A + (1.0 - alpha) * B
            out = mask * blended + (1.0 - mask) * A
        elif mode == "inject":
            # A + mask * B
            out = A + mask * B
        elif mode == "average":
            # 0.5 * (A + B) only in masked area
            res = 0.5 * (A + B)
            out = mask * res + (1.0 - mask) * A
        elif mode == "difference_injection":
            # weight * (B - A) injected via mask
            delta = weight * (B - A)
            out = A + mask * delta
        elif mode == "dominant_recessive":
            # A + eps * B via mask
            out = A + mask * (eps * B)
        elif mode == "replace":
            # Replace A with B where mask == 1
            out = mask * B + (1.0 - mask) * A
        elif mode == "concatenate":
            # Join tensors sequentially (ignoring mask for base op, or should we?)
            # Usually concat is structural, but we applies mask to B before concat
            out = torch.cat([A, B * mask], dim=L_idx) if mask.shape[1] == B.shape[1] else torch.cat([A, B], dim=L_idx)
        elif mode == "add":
            res = A + B
            out = mask * res + (1.0 - mask) * A
        elif mode == "multiply":
            res = A * B
            out = mask * res + (1.0 - mask) * A
        elif mode == "maximum":
            res = torch.max(A, B)
            out = mask * res + (1.0 - mask) * A
        elif mode == "minimum":
            res = torch.min(A, B)
            out = mask * res + (1.0 - mask) * A
        else:
            out = A
            
        return (out,)

    def interpolate_tensor(self, t, target_len, L_idx):
        if t.dim() == 3:
            t = t.transpose(1, 2)
            t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
            t = t.transpose(1, 2)
        elif t.dim() == 2:
            t = t.unsqueeze(0).transpose(1, 2)
            t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
            t = t.squeeze(0).transpose(0, 1)
        return t

    def pad_tensors(self, A, B, L_idx):
        len_A = A.shape[L_idx]
        len_B = B.shape[L_idx]
        if len_A == len_B:
            return A, B
        max_len = max(len_A, len_B)
        
        def pad_one(tensor, current_len, target_len, dim):
            if current_len >= target_len:
                return tensor
            pad_size = target_len - current_len
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            padding = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=dim)

        A = pad_one(A, len_A, max_len, L_idx)
        B = pad_one(B, len_B, max_len, L_idx)
        return A, B

NODE_CLASS_MAPPINGS = {
    "AceStepTensorMixer": AceStepTensorMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepTensorMixer": "Tensor Conditioning Mixer",
}
