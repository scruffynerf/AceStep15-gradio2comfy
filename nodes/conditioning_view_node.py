"""AceStepConditioningExplore node for ACE-Step"""
import json
import torch
import lovely_tensors as lt

class AceStepConditioningExplore:
    """Show conditioning content summarized with lovely-tensors"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_cond": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_text",)
    FUNCTION = "extract"
    CATEGORY = "Scromfy/Ace-Step/metadata"

    def extract(self, text_cond):
        # Conditioning is a list of lists: [[cond, {"pooled_output": ...}]]
        # Convert to JSON string with indentation and lovely-tensors summaries
        seen = set()
        serializable_data = self._to_serializable(text_cond, seen, depth=0)
        json_string = json.dumps(serializable_data, indent=4, default=str)
        return (json_string,)

    MAX_DEPTH = 20

    def _to_serializable(self, obj, seen, depth):
        """Recursively convert any object to JSON-serializable form with max detail."""
        # Depth guard
        if depth > self.MAX_DEPTH:
            return f"<MAX_DEPTH exceeded: {type(obj).__name__}>"

        # Circular reference guard (only for mutable objects with identity)
        obj_id = id(obj)
        if not isinstance(obj, (int, float, str, bool, type(None))):
            if obj_id in seen:
                return f"<circular ref: {type(obj).__name__} id={obj_id}>"
            seen.add(obj_id)

        try:
            return self._convert(obj, seen, depth)
        finally:
            # Remove from seen after processing so the same object
            # can appear in different branches (just not recursively)
            if not isinstance(obj, (int, float, str, bool, type(None))):
                seen.discard(obj_id)

    def _convert(self, obj, seen, depth):
        # --- Primitives (JSON-native) ---
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # --- Tensors (PyTorch) ---
        if isinstance(obj, torch.Tensor):
            return {
                "__type__": "torch.Tensor",
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "lovely": str(lt.lovely(obj)),
            }

        # --- Numpy arrays ---
        try:
            import numpy as np
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "numpy.ndarray",
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "summary": repr(obj) if obj.size < 20 else f"ndarray({obj.shape}, {obj.dtype})",
                }
        except ImportError:
            pass

        # --- Dicts ---
        if isinstance(obj, dict):
            return {
                str(k): self._to_serializable(v, seen, depth + 1)
                for k, v in obj.items()
            }

        # --- Lists / Tuples ---
        if isinstance(obj, (list, tuple)):
            result = [self._to_serializable(item, seen, depth + 1) for item in obj]
            if isinstance(obj, tuple):
                return {"__type__": "tuple", "items": result}
            return result

        # --- Sets / Frozensets ---
        if isinstance(obj, (set, frozenset)):
            return {
                "__type__": type(obj).__name__,
                "items": [self._to_serializable(item, seen, depth + 1) for item in obj],
            }

        # --- Bytes ---
        if isinstance(obj, (bytes, bytearray)):
            return {
                "__type__": type(obj).__name__,
                "length": len(obj),
                "preview": repr(obj[:64]),
            }

        # --- Callables (functions, methods, lambdas) ---
        if callable(obj) and not hasattr(obj, '__dict__'):
            return {
                "__type__": "callable",
                "name": getattr(obj, '__qualname__', getattr(obj, '__name__', repr(obj))),
                "module": getattr(obj, '__module__', '<unknown>'),
            }

        # --- Objects with __dict__ and/or __slots__ ---
        if hasattr(obj, '__dict__') or hasattr(obj, '__slots__'):
            result = {
                "__type__": type(obj).__qualname__,
                "__module__": type(obj).__module__,
                "__mro__": [c.__name__ for c in type(obj).__mro__],
            }

            # Gather attributes from __dict__
            if hasattr(obj, '__dict__'):
                for k, v in obj.__dict__.items():
                    result[k] = self._to_serializable(v, seen, depth + 1)

            # Gather attributes from __slots__
            if hasattr(obj, '__slots__'):
                for slot in obj.__slots__:
                    if hasattr(obj, slot) and slot not in result:
                        result[slot] = self._to_serializable(
                            getattr(obj, slot), seen, depth + 1
                        )

            # If the object looks empty, try repr and dir for extra info
            attr_keys = [k for k in result if not k.startswith('__')]
            if not attr_keys:
                result["__repr__"] = repr(obj)
                # Show non-dunder public attributes via dir()
                public_attrs = [
                    a for a in dir(obj)
                    if not a.startswith('_')
                ]
                if public_attrs:
                    result["__public_attrs__"] = public_attrs

            return result

        # --- Fallback: repr everything else ---
        return {
            "__type__": type(obj).__name__,
            "__repr__": repr(obj),
        }

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningExplore": AceStepConditioningExplore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningExplore": "Conditioning to Json Text",
}
