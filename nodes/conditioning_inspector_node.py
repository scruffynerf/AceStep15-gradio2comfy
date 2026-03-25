import torch

class AceStepConditioningInspector:
    """
    ACE-Step ▸ Conditioning Inspector — passthrough that prints the dict.
    """
    CATEGORY = "Scromfy/Ace-Step/Conditioning"
    FUNCTION = "inspect"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "label": ("STRING", {"default":"COND_DUMP","multiline":False}),
        }}

    def inspect(self, conditioning, label):
        sep = "=" * 70
        print(f"\n{sep}\n[{label}] ({len(conditioning)} entries)\n{sep}")
        for i, (tensor, d) in enumerate(conditioning):
            print(f"\n  ── Entry [{i}] ──")
            if isinstance(tensor, torch.Tensor):
                print(f"    main tensor: {list(tensor.shape)}  {tensor.dtype}")
            print(f"    dict keys  : {sorted(d.keys())}")
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, torch.Tensor):
                    vf = v.float()
                    print(f"      {k:35s}: Tensor {list(v.shape)} {v.dtype} "
                          f"min={vf.min():.4f} max={vf.max():.4f}")
                elif isinstance(v, list):
                    print(f"      {k:35s}: list[{len(v)}]", end="")
                    if v and isinstance(v[0], list):
                        print(f"  → list[{len(v[0])}]  first 6: {v[0][:6]}")
                    elif v and isinstance(v[0], torch.Tensor):
                        print(f"  → Tensor {list(v[0].shape)}")
                    else:
                        print(f"  = {repr(v)[:80]}")
                elif isinstance(v, (int,float,bool,str)):
                    print(f"      {k:35s}: {type(v).__name__} = {repr(v)}")
                else:
                    print(f"      {k:35s}: {type(v).__name__}")
        print(sep + "\n")
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningInspector": AceStepConditioningInspector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningInspector": "ACE-Step ▸ Conditioning Inspector"
}
