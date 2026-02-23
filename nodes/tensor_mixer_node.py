"""AceStepTensorMixer node for ACE-Step"""
import torch
import torch.nn.functional as F

class AceStepTensorMixer:
    """Mix two tensors with various modes and scaling options"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_A": ("TENSOR",),
                "tensor_B": ("TENSOR",),
                "mode": (["linear_blend", "concatenate", "add", "multiply", "maximum", "minimum"], {"default": "linear_blend"}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_mode": (["none", "scale_B_to_A", "scale_A_to_B", "pad_to_match"], {"default": "none"}),
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "mix"
    CATEGORY = "Scromfy/Ace-Step/processing"

    def mix(self, tensor_A, tensor_B, mode, ratio, scale_mode):
        # We assume tensors are [B, L, D] or [L, D]
        # Usually ACE-Step conditioning tensors are [1, L, 1024]
        
        A = tensor_A.clone()
        B = tensor_B.clone()
        
        # Check dimensions
        if A.dim() != B.dim():
             raise ValueError(f"Tensor dimensions mismatch: A={A.shape}, B={B.shape}")
             
        # Sequence length dimension is usually index 1 if 3D, 0 if 2D
        L_idx = 1 if A.dim() == 3 else 0
        
        # Scaling / Interpolation
        if scale_mode == "scale_B_to_A" and A.shape[L_idx] != B.shape[L_idx]:
            B = self.interpolate_tensor(B, A.shape[L_idx], L_idx)
        elif scale_mode == "scale_A_to_B" and A.shape[L_idx] != B.shape[L_idx]:
            A = self.interpolate_tensor(A, B.shape[L_idx], L_idx)
        elif scale_mode == "pad_to_match" and A.shape[L_idx] != B.shape[L_idx]:
            A, B = self.pad_tensors(A, B, L_idx)
            
        if mode == "linear_blend":
            out = A * (1.0 - ratio) + B * ratio
        elif mode == "concatenate":
            out = torch.cat([A, B], dim=L_idx)
        elif mode == "add":
            out = A + B
        elif mode == "multiply":
            out = A * B
        elif mode == "maximum":
            out = torch.max(A, B)
        elif mode == "minimum":
            out = torch.min(A, B)
        else:
            out = A
            
        return (out,)

    def interpolate_tensor(self, t, target_len, L_idx):
        # t is [B, L, D] or [L, D]
        # F.interpolate expects [B, C, L] for 1D linear
        if t.dim() == 3:
            # [B, L, D] -> [B, D, L]
            t = t.transpose(1, 2)
            t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
            # [B, D, L] -> [B, L, D]
            t = t.transpose(1, 2)
        elif t.dim() == 2:
            # [L, D] -> [1, D, L]
            t = t.unsqueeze(0).transpose(1, 2)
            t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
            # [1, D, L] -> [L, D]
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
            # pad format: (left, right, top, bottom, ...)
            # for [B, L, D] and dim 1, we want to pad L. 
            # F.pad handles last dims first. 
            # If we want to pad dim 1 of [B, L, D]:
            # torch.cat might be easier
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
    "AceStepTensorMixer": "Tensor Mixer & Scaler",
}
