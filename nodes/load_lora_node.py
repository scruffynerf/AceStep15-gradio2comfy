"""ACE-Step 1.5 LoRA Loader node for scromfyUI-AceStep"""
import os
import glob
import json
import time
import uuid
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    from safetensors.torch import load_file as safetensors_load
except Exception:
    safetensors_load = None


# =========================
# Paths: models/loras/Ace-Step-1.5/<LORA_NAME>/
# =========================

def _get_lora_base_dirs():
    base_dirs = []
    if folder_paths is not None:
        try:
            lora_dirs = folder_paths.get_folder_paths("loras")
            if isinstance(lora_dirs, (list, tuple)):
                for d in lora_dirs:
                    base_dirs.append(os.path.join(d, "Ace-Step-1.5"))
        except Exception:
            pass
    base_dirs.append(os.path.join("models", "loras", "Ace-Step-1.5"))
    seen, out = set(), []
    for d in base_dirs:
        nd = os.path.normpath(d)
        if nd not in seen:
            out.append(nd)
            seen.add(nd)
    return out


def _list_lora_dirs():
    dirs = []
    for bd in _get_lora_base_dirs():
        if os.path.isdir(bd):
            for name in os.listdir(bd):
                full = os.path.join(bd, name)
                if os.path.isdir(full):
                    if os.path.isfile(os.path.join(full, "adapter_config.json")) or os.path.isfile(os.path.join(full, "config.json")):
                        dirs.append(name)
    return sorted(list(dict.fromkeys(dirs)))


def _resolve_lora_dir(lora_name: str):
    for bd in _get_lora_base_dirs():
        candidate = os.path.join(bd, lora_name)
        if os.path.isdir(candidate):
            return candidate
    return None


def _find_lora_weights_file(lora_dir: str):
    common = [
        "adapter_model.safetensors",
        "pytorch_lora_weights.safetensors",
        "adapter.safetensors",
        "adapter_model.bin",
    ]
    for fn in common:
        p = os.path.join(lora_dir, fn)
        if os.path.isfile(p):
            return p
    sts = glob.glob(os.path.join(lora_dir, "*.safetensors"))
    return sts[0] if sts else None


def _read_adapter_config(lora_dir: str) -> dict:
    for fn in ["adapter_config.json", "config.json"]:
        p = os.path.join(lora_dir, fn)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def _get_dict(obj):
    try:
        return getattr(obj, "__dict__", {}) or {}
    except Exception:
        return {}


# =========================
# LoRA wrapper
# =========================

class LoRALinearWrapper(nn.Module):
    def __init__(self, base: nn.Module, A: torch.Tensor, B: torch.Tensor, scale: float, tag: str):
        super().__init__()
        self.base = base
        self.register_buffer("lora_A", A)
        self.register_buffer("lora_B", B)
        self.scale = float(scale)
        self.tag = str(tag)

    def set_scale(self, scale: float):
        self.scale = float(scale)

    def set_lora(self, A: torch.Tensor, B: torch.Tensor, tag: str, scale: float):
        self.lora_A = A
        self.lora_B = B
        self.tag = str(tag)
        self.scale = float(scale)

    def forward(self, x):
        y = self.base(x)
        if self.scale == 0.0:
            return y

        A = self.lora_A.to(device=x.device, dtype=x.dtype)
        B = self.lora_B.to(device=x.device, dtype=x.dtype)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        xr = torch.matmul(x2, A.t())
        delta = torch.matmul(xr, B.t())
        delta = delta.reshape(*orig_shape[:-1], delta.shape[-1])

        return y + delta * self.scale


# =========================
# Key mapping
# =========================

def _extract_module_path_from_lora_key(full_key: str) -> Optional[str]:
    marker = ".layers."
    idx = full_key.find(marker)
    if idx == -1:
        return None
    sub = full_key[idx + 1:]

    for suffix in [".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight", ".lora_B.default.weight"]:
        if sub.endswith(suffix):
            return sub[: -len(suffix)]

    parts = sub.split(".")
    if len(parts) >= 3 and parts[-3].startswith("lora_"):
        return ".".join(parts[:-3])
    return None


def _pair_lora_A_B(sd: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    A_map, B_map = {}, {}
    for k, v in sd.items():
        p = _extract_module_path_from_lora_key(k)
        if p is None:
            continue
        if ".lora_A" in k:
            A_map[p] = v
        elif ".lora_B" in k:
            B_map[p] = v

    out = {}
    for p, A in A_map.items():
        if p in B_map:
            out[p] = (A, B_map[p])
    return out


def _get_decoder(base: nn.Module) -> nn.Module:
    if not hasattr(base, "diffusion_model"):
        raise RuntimeError("ACEStep15 has no diffusion_model attribute.")
    dm = base.diffusion_model
    if not hasattr(dm, "decoder"):
        raise RuntimeError("diffusion_model has no decoder attribute.")
    dec = dm.decoder
    if not isinstance(dec, nn.Module):
        raise RuntimeError("diffusion_model.decoder is not an nn.Module.")
    return dec


def _resolve_module(decoder: nn.Module, module_path: str) -> Optional[nn.Module]:
    parts = module_path.split(".")
    cur: object = decoder
    for part in parts:
        if not isinstance(cur, nn.Module):
            return None
        if part.isdigit():
            try:
                cur = cur[int(part)]
            except Exception:
                return None
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur if isinstance(cur, nn.Module) else None


def _replace_module(decoder: nn.Module, module_path: str, new_module: nn.Module) -> bool:
    parts = module_path.split(".")
    parent_parts = parts[:-1]
    leaf = parts[-1]

    cur: object = decoder
    for part in parent_parts:
        if not isinstance(cur, nn.Module):
            return False
        if part.isdigit():
            try:
                cur = cur[int(part)]
            except Exception:
                return False
        else:
            if part in cur._modules:
                cur = cur._modules[part]
            else:
                if not hasattr(cur, part):
                    return False
                cur = getattr(cur, part)

    if not isinstance(cur, nn.Module):
        return False

    if leaf.isdigit():
        try:
            cur[int(leaf)] = new_module
            return True
        except Exception:
            return False

    if leaf in cur._modules:
        cur._modules[leaf] = new_module
        return True

    return False


def _calc_scale(adapter_cfg: dict, strength: float) -> float:
    r = adapter_cfg.get("r", None)
    alpha = adapter_cfg.get("lora_alpha", None)
    try:
        if r is not None and alpha is not None and float(r) != 0.0:
            return float(strength) * (float(alpha) / float(r))
    except Exception:
        pass
    return float(strength)


def _count_wrappers(decoder: nn.Module) -> int:
    c = 0
    for m in decoder.modules():
        if isinstance(m, LoRALinearWrapper):
            c += 1
    return c


def _restore_originals(decoder: nn.Module, state: dict) -> int:
    backup = state.get("backup_modules", {})
    restored = 0
    for module_path, original_module in list(backup.items()):
        cur = _resolve_module(decoder, module_path)
        if cur is None:
            continue
        if isinstance(cur, LoRALinearWrapper):
            if _replace_module(decoder, module_path, original_module):
                restored += 1
    if restored > 0:
        state["current_lora"] = None
        state["current_scale"] = 0.0
    return restored


# =========================
# Node
# =========================

class AceStepLoRALoader:
    """Specialized LoRA loader for ACE-Step 1.5 decoder"""
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Force re-execution every prompt to avoid cached outputs if needed,
        # but ComfyUI usually handles IS_CHANGED via inputs. 
        # Here we use time to be safe as per reference code.
        return time.time()

    @classmethod        
    def INPUT_TYPES(cls):
        loras = _list_lora_dirs()
        lora_field = (loras,) if loras else ("STRING", {"default": ""})
        return {
            "required": {
                "ace_model": ("MODEL",),
                "lora_name": lora_field,
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mode": (["auto_clean", "disable"], {"default": "auto_clean"}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_lora"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def apply_lora(self, ace_model, lora_name, strength=1.0, mode="auto_clean", debug=False):
        if safetensors_load is None:
            raise RuntimeError("safetensors is not available in this Python environment.")

        mp = _get_dict(ace_model)
        if "model" not in mp:
            raise RuntimeError("ModelPatcher has no 'model' in __dict__.")
        base = mp["model"]
        if not isinstance(base, nn.Module):
            raise RuntimeError(f"Unwrapped base is not nn.Module: {type(base)}")

        decoder = _get_decoder(base)

        if not hasattr(decoder, "_aces_lora_state"):
            decoder._aces_lora_state = {}  # type: ignore
        state: dict = decoder._aces_lora_state  # type: ignore

        if "backup_modules" not in state:
            state["backup_modules"] = {}
        backup: dict = state["backup_modules"]

        run_id = f"{time.time():.3f}-{uuid.uuid4().hex[:6]}"

        if debug:
            print(f"\n[AceStepLoRA] RUN_ID {run_id}")
            print(f"[AceStepLoRA] mode={mode} requested_lora={lora_name} strength={strength}")
            print(f"[AceStepLoRA] Before: wrappers={_count_wrappers(decoder)} current_lora={state.get('current_lora', None)}")

        # disable: restore and exit
        if mode == "disable":
            restored = _restore_originals(decoder, state)
            if debug:
                print(f"[AceStepLoRA] Restore: restored={restored}")
                print(f"[AceStepLoRA] After: wrappers={_count_wrappers(decoder)} current_lora={state.get('current_lora', None)}\n")
            return (ace_model,)

        # auto_clean: ALWAYS restore to base first (strong protection against accumulation)
        restored = _restore_originals(decoder, state)

        if not isinstance(lora_name, str) or not lora_name.strip():
            # If no lora_name, just return model (after potential restoration)
            return (ace_model,)
            
        lora_name = lora_name.strip()

        lora_dir = _resolve_lora_dir(lora_name)
        if not lora_dir:
            raise RuntimeError(f"LoRA folder not found: '{lora_name}'")

        weights_path = _find_lora_weights_file(lora_dir)
        if not weights_path:
            raise RuntimeError(f"LoRA weights not found in: {lora_dir}")

        adapter_cfg = _read_adapter_config(lora_dir)
        scale = _calc_scale(adapter_cfg, strength)

        sd = safetensors_load(weights_path)
        pairs = _pair_lora_A_B(sd)
        if len(pairs) == 0:
            raise RuntimeError("No LoRA A/B pairs found in adapter safetensors.")

        applied = 0
        updated = 0
        skipped = 0

        for module_path, (A, B) in pairs.items():
            cur = _resolve_module(decoder, module_path)
            if cur is None:
                skipped += 1
                continue

            # save originals
            if module_path not in backup and isinstance(cur, nn.Linear):
                backup[module_path] = cur

            orig = backup.get(module_path, None)

            # build wrapper ONLY on original linear
            if isinstance(orig, nn.Linear):
                wrapped = LoRALinearWrapper(orig, A, B, scale, tag=lora_name)
                if _replace_module(decoder, module_path, wrapped):
                    applied += 1
                else:
                    skipped += 1
                continue

            # fallback: if no backup but cur is linear
            if isinstance(cur, nn.Linear):
                backup[module_path] = cur
                wrapped = LoRALinearWrapper(cur, A, B, scale, tag=lora_name)
                if _replace_module(decoder, module_path, wrapped):
                    applied += 1
                else:
                    skipped += 1
                continue

            skipped += 1

        state["current_lora"] = lora_name
        state["current_scale"] = float(scale)

        if debug:
            print(f"[AceStepLoRA] Restore: restored={restored}")
            print(f"[AceStepLoRA] Apply: applied={applied} updated={updated} skipped={skipped} scale={scale}")
            print(f"[AceStepLoRA] After: wrappers={_count_wrappers(decoder)} current_lora={state.get('current_lora', None)}\n")

        if applied == 0:
            raise RuntimeError(
                "Couldn't apply LoRA to diffusion_model.decoder. "
                "Likely module path mismatch between LoRA keys and ComfyUI model structure."
            )

        return (ace_model,)


NODE_CLASS_MAPPINGS = {
    "AceStepLoRALoader": AceStepLoRALoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLoRALoader": "LoRA Loader (ACE-Step)",
}
