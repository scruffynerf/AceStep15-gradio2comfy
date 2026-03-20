import os
import sys
import torch
from safetensors.torch import load_file, save_file
import argparse

def split_checkpoint(checkpoint_path, output_dir, strip=False, list_keys=False):
    """
    Splits a consolidated ACE-Step 1.5 safetensors checkpoint into its components.
    Targeted for Dual CLIP (Qwen), Diffusion, and VAE.
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}...")
    try:
        tensors = load_file(checkpoint_path)
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        return

    if list_keys:
        print("\n--- Key Audit (First 20 keys) ---")
        keys = list(tensors.keys())
        for i, key in enumerate(keys[:20]):
            print(f"{i+1}: {key} - {tensors[key].shape}")
        
        # Analyze prefixes
        prefixes = set()
        for key in keys:
            parts = key.split('.')
            if len(parts) > 1:
                prefixes.add(".".join(parts[:2]))
            else:
                prefixes.add(parts[0])
        
        print("\n--- Detected Prefixes ---")
        for p in sorted(list(prefixes)):
            count = sum(1 for k in keys if k.startswith(p))
            print(f"{p} ({count} keys)")
        return

    # Grouping logic
    groups = {
        "qwen_0.6b": [],
        "qwen_2b": [],
        "diffusion": [],
        "vae": []
    }

    print("Analyzing tensors...")
    for key in tensors.keys():
        if key.startswith("text_encoders.qwen3_06b") or key.startswith("text_encoders.qwen3_6b"):
            groups["qwen_0.6b"].append(key)
        elif key.startswith("text_encoders.qwen3_2b"):
            groups["qwen_2b"].append(key)
        elif key.startswith("model.diffusion_model"):
            groups["diffusion"].append(key)
        elif key.startswith("vae"):
            groups["vae"].append(key)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for group_name, keys in groups.items():
        if not keys:
            continue

        output_path = os.path.join(output_dir, f"{group_name}.safetensors")
        print(f"Saving {len(keys)} tensors to {output_path}...")
        
        out_tensors = {}
        
        # Determine the base prefix for this group
        group_prefix = ""
        if group_name == "qwen_0.6b":
            group_prefix = "text_encoders.qwen3_06b." if any(k.startswith("text_encoders.qwen3_06b.") for k in keys) else "text_encoders.qwen3_6b."
        elif group_name == "qwen_2b":
            group_prefix = "text_encoders.qwen3_2b."
        elif group_name == "diffusion":
            group_prefix = "model.diffusion_model."
        elif group_name == "vae":
            group_prefix = "vae." if any(k.startswith("vae.") for k in keys) else "vae"

        for k in keys:
            new_key = k
            if strip:
                # 1. Strip the group-level "path" prefix
                if group_prefix and k.startswith(group_prefix):
                    new_key = k[len(group_prefix):]
                
                # 2. Strip redundant 'transformer.' wrapper if present after group strip
                if new_key.startswith("transformer."):
                    new_key = new_key[len("transformer."):]
                
                # 3. Clean up leading dots just in case
                if new_key.startswith("."):
                    new_key = new_key[1:]
            
            out_tensors[new_key] = tensors[k]

        try:
            save_file(out_tensors, output_path)
            print(f"  Saved {group_name} component.")
        except Exception as e:
            print(f"  Error saving {group_name}: {e}")

    print("\nSplitting complete!")
    print(f"Outputs are in: {os.path.abspath(output_dir)}")
    if not strip:
        print("\nWARNING: Splitting without --strip keeps consolidated prefixes.")
        print("For ACE-Step standalone loaders, you likely MUST use --strip.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split ACE-Step 1.5 consolidated checkpoints.")
    parser.add_argument("checkpoint", type=str, help="Path to the source .safetensors file.")
    parser.add_argument("--out_dir", type=str, default="split_models", help="Directory to save pieces.")
    parser.add_argument("--strip", action="store_true", help="Strip prefixes from keys (strongly recommended).")
    parser.add_argument("--list", action="store_true", help="Just list prefixes and keys, do not split.")

    args = parser.parse_args()
    split_checkpoint(args.checkpoint, args.out_dir, args.strip, args.list)
