"""Audio analysis and FSQ logic for ACE-Step"""
import torch
import logging

logger = logging.getLogger(__name__)

# Try to import librosa for audio analysis
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - audio analysis features will be limited")


class FSQ(torch.nn.Module):
    def __init__(self, levels, device=None, dtype=None):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32, device=device)
        self.register_buffer('_levels', _levels, persistent=False)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int32, device=device), dim=0)
        self.register_buffer('_basis', _basis, persistent=False)
        self.codebook_dim = len(levels)
        self.codebook_size = self._levels.prod().item()
        
        indices = torch.arange(self.codebook_size, device=device)
        self.register_buffer('implicit_codebook', self._indices_to_codes(indices).to(dtype), persistent=False)

    def _indices_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        return codes_non_centered.float() * (2. / (self._levels.float() - 1)) - 1.

    def codes_to_indices(self, zhat):
        zhat_normalized = (zhat + 1.) / (2. / (self._levels.to(zhat.dtype) - 1))
        return (zhat_normalized * self._basis.to(zhat.dtype)).sum(dim=-1).round().to(torch.int32)

    def bound(self, z):
        levels_minus_1 = (self._levels - 1).to(z.dtype)
        scale = 2. / levels_minus_1
        bracket = (levels_minus_1 * (torch.tanh(z) + 1) / 2.) + 0.5
        zhat = bracket.floor()
        return scale * (bracket + (zhat - bracket).detach()) - 1.

    def forward(self, z):
        codes = self.bound(z)
        return codes, self.codes_to_indices(codes)


class ResidualFSQ(torch.nn.Module):
    def __init__(self, levels, num_quantizers, device=None, dtype=None):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            FSQ(levels=levels, device=device, dtype=dtype) for _ in range(num_quantizers)
        ])
        levels_tensor = torch.tensor(levels, device=device)
        scales = [levels_tensor.float() ** -ind for ind in range(num_quantizers)]
        scales_tensor = torch.stack(scales)
        if dtype is not None:
            scales_tensor = scales_tensor.to(dtype)
        self.register_buffer('scales', scales_tensor, persistent=False)
        
        val = 1 + (1 / (levels_tensor.float() - 1))
        self.register_buffer('soft_clamp_input_value', val.to(dtype) if dtype else val, persistent=False)

    def get_output_from_indices(self, indices, dtype=torch.float32):
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)
        all_codes = []
        for i, layer in enumerate(self.layers):
            idx = indices[..., i].long()
            codes = torch.nn.functional.embedding(idx, layer.implicit_codebook.to(device=idx.device, dtype=dtype))
            all_codes.append(codes * self.scales[i].to(device=idx.device, dtype=dtype))
        return torch.stack(all_codes).sum(dim=0)

    def forward(self, x):
        sc_val = self.soft_clamp_input_value.to(x.dtype)
        x = (x / sc_val).tanh() * sc_val
        quantized_out = torch.tensor(0., device=x.device, dtype=x.dtype)
        residual = x
        all_indices = []
        for layer, scale in zip(self.layers, self.scales):
            scale = scale.to(residual.dtype)
            codes, indices = layer(residual / scale)
            quantized = codes * scale
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        return quantized_out, torch.stack(all_indices, dim=-1)
