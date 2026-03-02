"""Prompt generation utilities and presets for ACE-Step.
Now dynamically loads components from the 'prompt_components' directory.
"""
import os
import json
import sys

# Cache to store loaded components
_COMPONENTS = {}
_TOP_LEVEL_COMPONENTS = set()
_COMPONENT_WEIGHTS = {}

def get_keyscales():
    """Generate the standard ACE-Step 1.5 keyscale list."""
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    accidentals = ['', '#', 'b']
    modes = ['major', 'minor']
    
    keyscales = ["Auto-detect"]
    for note in notes:
        for acc in accidentals:
            for mode in modes:
                # Basic theory overrides to match ACE-Step constants.py
                if note == 'C' and mode == 'major' and acc != '':
                    continue
                if note == 'A' and mode == 'minor' and acc != '':
                    continue
                keyscales.append(f"{note}{acc} {mode}")
    return keyscales

def _load_components():
    """Scan the prompt_components directory and load all txt/json files."""
    global _HIDDEN_COMPONENTS
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    components_dir = os.path.join(base_dir, "prompt_components")
    
    if not os.path.exists(components_dir):
        print(f"Warning: prompt_components directory not found at {components_dir}", file=sys.stderr)
        return

    # Load ignore/replace/weight lists first
    total_ignore = set()
    replace_map = {}
    reverse_replace = {}
    weights = {} # Component Name -> float Weight
    
    def get_path(filename):
        user_p = os.path.join(components_dir, filename)
        if os.path.exists(user_p):
            return user_p
        
        # Determine default filename mapping
        ext = os.path.splitext(filename)[1]
        base = os.path.splitext(filename)[0]
        # WEIGHTS defaults to .json, REPLACE/IGNORE/HIDE default to .list
        # But we'll try to find any .default files that match the base name
        default_p = os.path.join(components_dir, f"{base}.default{ext}")
        if os.path.exists(default_p):
            return default_p
            
        return user_p

    ignore_path = get_path("TOTALIGNORE.list")
    replace_path = get_path("REPLACE.list")
    weights_path = get_path("WEIGHTS.json")
    
    def read_list_file(p):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}
        return set()

    total_ignore = read_list_file(ignore_path)
    
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
            print(f"Error reading weights from {weights_path}: {e}", file=sys.stderr)

    global _TOP_LEVEL_COMPONENTS, _COMPONENT_WEIGHTS, _COMPONENTS
    
    _TOP_LEVEL_COMPONENTS = set()

    # We also lowercase total_ignore for robust checking
    lower_ignore = {item.lower() for item in total_ignore}
    for item in total_ignore:
        if '.' in item:
            lower_ignore.add(os.path.splitext(item)[0].lower())

    _COMPONENT_WEIGHTS = weights
    
    # Reset components cache
    _COMPONENTS = {}

    for root, dirs, files in os.walk(components_dir):
        # Skip hidden directories (like .git or __pycache__)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for filename in files:
            if filename in ("TOTALIGNORE.list", "LOADBUTNOTSHOW.list", "REPLACE.list", "WEIGHTS.json", "README.md") or ".default." in filename:
                continue
                
            name, ext = os.path.splitext(filename)
            if filename.lower() in lower_ignore or name.lower() in lower_ignore:
                continue
            if name in replace_map:
                continue
                
            assign_name = reverse_replace.get(name, name)
            
            # Key collision check: first one found (alphabetically by path) wins
            if assign_name in _COMPONENTS:
                continue

            full_path = os.path.join(root, filename)
            
            # Visibility logic: Only files in the root of prompt_components are visible in UI
            if root == components_dir:
                _TOP_LEVEL_COMPONENTS.add(assign_name)

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
                print(f"Error loading prompt component {filename} from {root}: {e}", file=sys.stderr)

    # Inject built-in Keyscale component as visible
    _COMPONENTS["KEYSCALE"] = get_keyscales()
    _TOP_LEVEL_COMPONENTS.add("KEYSCALE")

# Initialize global caches
_COMPONENT_WEIGHTS = {}

def sort_weighted(names):
    """Sort a list of component names by weight (descending) then alphabetically."""
    # Weight defaults to 0 if not listed. Higher weight comes first.
    return sorted(names, key=lambda x: (-_COMPONENT_WEIGHTS.get(x, 0), x))

# Initial load
_load_components()

def get_available_components():
    """Return a list of all dynamically loaded component names (including hidden), sorted by weight."""
    return sort_weighted(_COMPONENTS.keys())

def get_visible_components():
    """Return component names that should be shown in the UI, sorted by weight."""
    # Only top-level components are shown in dropdowns
    return sort_weighted(list(_TOP_LEVEL_COMPONENTS))

def get_component(name, default=None):
    """Safely retrieve a component by name, case-insensitively."""
    if not name:
        return default
    if name in _COMPONENTS:
        return _COMPONENTS[name]
    # Check lowercase/uppercase variations for robustness
    if name.upper() in _COMPONENTS:
        return _COMPONENTS[name.upper()]
    if name.lower() in _COMPONENTS:
        return _COMPONENTS[name.lower()]
    return default
