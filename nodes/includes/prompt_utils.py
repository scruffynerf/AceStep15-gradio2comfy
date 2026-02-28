"""Prompt generation utilities and presets for ACE-Step.
Now dynamically loads components from the 'prompt_components' directory.
"""
import os
import json
import sys

# Cache to store loaded components
_COMPONENTS = {}
_HIDDEN_COMPONENTS = set()

def _load_components():
    """Scan the prompt_components directory and load all txt/json files."""
    global _HIDDEN_COMPONENTS
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    components_dir = os.path.join(base_dir, "prompt_components")
    
    if not os.path.exists(components_dir):
        print(f"Warning: prompt_components directory not found at {components_dir}", file=sys.stderr)
        return

    # Load ignore/hide/replace/weight lists first
    total_ignore = set()
    load_but_not_show = set()
    replace_map = {}
    reverse_replace = {}
    weights = {} # Component Name -> float Weight
    
    ignore_path = os.path.join(components_dir, "TOTALIGNORE.list")
    hide_path = os.path.join(components_dir, "LOADBUTNOTSHOW.list")
    replace_path = os.path.join(components_dir, "REPLACE.list")
    weights_path = os.path.join(components_dir, "WEIGHTS.json")
    
    def read_list_file(p):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        return set()

    total_ignore = read_list_file(ignore_path)
    load_but_not_show = read_list_file(hide_path)
    
    if os.path.exists(replace_path):
        try:
            with open(replace_path, "r", encoding="utf-8") as f:
                raw_map = json.load(f)
                for k, v in raw_map.items():
                    if k.startswith("EXAMPLE_") or v.startswith("EXAMPLE_"):
                        continue
                    has_replacement = any(os.path.exists(os.path.join(components_dir, f"{v}{ext}")) for ext in [".txt", ".json"])
                    if has_replacement:
                        replace_map[k] = v
                        reverse_replace[v] = k
        except Exception as e:
            print(f"Error reading REPLACE.list: {e}", file=sys.stderr)

    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                weights = json.load(f)
        except Exception as e:
            print(f"Error reading WEIGHTS.json: {e}", file=sys.stderr)

    _HIDDEN_COMPONENTS = load_but_not_show
    _COMPONENT_WEIGHTS = weights

    for filename in os.listdir(components_dir):
        if filename in ("TOTALIGNORE.list", "LOADBUTNOTSHOW.list", "REPLACE.list", "WEIGHTS.json", "README.md"):
            continue
            
        name, ext = os.path.splitext(filename)
        if filename in total_ignore or name in total_ignore:
            continue
        if name in replace_map:
            continue
            
        assign_name = reverse_replace.get(name, name)
        full_path = os.path.join(components_dir, filename)
        if not os.path.isfile(full_path):
            continue
            
        ext = ext.lower()
        try:
            if ext == ".json":
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _COMPONENTS[assign_name] = data
                    globals()[assign_name] = data
            elif ext == ".txt":
                with open(full_path, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                    _COMPONENTS[assign_name] = lines
                    globals()[assign_name] = lines
        except Exception as e:
            print(f"Error loading prompt component {filename}: {e}", file=sys.stderr)

# Global caches for weighted sorting
_COMPONENT_WEIGHTS = {}

def sort_weighted(names):
    """Sort a list of component names by weight (descending) then alphabetically."""
    # Weight defaults to 0 if not listed. Higher weight comes first.
    return sorted(names, key=lambda x: (-_COMPONENT_WEIGHTS.get(x, 0), x))

# Initialize on import
_load_components()

def get_available_components():
    """Return a list of all dynamically loaded component names (including hidden), sorted by weight."""
    return sort_weighted(_COMPONENTS.keys())

def get_visible_components():
    """Return component names that should be shown in the UI, sorted by weight."""
    all_comps = _COMPONENTS.keys()
    visible = [c for c in all_comps if c not in _HIDDEN_COMPONENTS]
    return sort_weighted(visible)

def get_component(name, default=None):
    """Safely retrieve a component by name."""
    return _COMPONENTS.get(name, default)
