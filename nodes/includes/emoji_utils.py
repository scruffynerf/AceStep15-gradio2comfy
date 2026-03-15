import os
import io
import re
import pyconify
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM, shapes
from PIL import Image
import numpy as np
import torch
import random

# Cache directory for icons/masks
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(BASE_DIR, "masks")

# Global memory cache to prevent redundant disk/API hits in a single session
_ICON_MEMORY_CACHE = {}

def ensure_cache_dir(subdir=None):
    path = CACHE_DIR
    if subdir:
        path = os.path.join(path, subdir)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path

def get_local_icon_names(icon_set=None, count=25, seed=None):
    """
    Fetch a list of already-cached icon names from the masks directory.
    Format: 'collection:name' or just 'name' if at root.
    """
    if not os.path.exists(CACHE_DIR):
        return []
    
    local_icons = []
    
    # Check root level of masks/
    if not icon_set or icon_set == "local":
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".png") and not any(x in f for x in ["_ws", "_wo", "_wb"]):
                name = f[:-4]
                # If there's no collection prefix, we can just use the name
                local_icons.append(name)
    
    # Check subdirs
    possible_subdirs = [icon_set] if icon_set and icon_set != "local" else os.listdir(CACHE_DIR)
    for subdir in possible_subdirs:
        subdir_path = os.path.join(CACHE_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for f in os.listdir(subdir_path):
            if f.endswith(".png") and not any(x in f for x in ["_ws", "_wo", "_wb"]):
                name = f[:-4]
                local_icons.append(f"{subdir}:{name}")
    
    if not local_icons:
        return []
        
    rng = random.Random(seed)
    if len(local_icons) > count:
        return rng.sample(local_icons, count)
    else:
        rng.shuffle(local_icons)
        return local_icons

def get_emoji_icon_names(collection_prefix="fluent-emoji-flat", count=25, seed=None):
    """
    Fetch a random set of icon names from an Iconify collection.
    Falls back to local icons if API fails.
    """
    if collection_prefix == "local":
        return get_local_icon_names(count=count, seed=seed)

    try:
        info = pyconify.collection(collection_prefix)
        icon_names = []
        if 'uncategorized' in info:
            icon_names = info['uncategorized']
        elif 'icons' in info:
            icon_names = list(info['icons'].keys())
        elif 'categories' in info:
            for cat_icons in info['categories'].values():
                icon_names.extend(cat_icons)
        
        if not icon_names:
            raise ValueError(f"No icons found in collection {collection_prefix}")

        rng = random.Random(seed)
        if len(icon_names) > count:
            selection = rng.sample(icon_names, count)
        else:
            rng.shuffle(icon_names)
            selection = icon_names
            
        return [f"{collection_prefix}:{n}" for n in selection]

    except Exception as e:
        print(f"Warning: Error fetching Iconify collection {collection_prefix} ({e}). Falling back to local icons.")
        return get_local_icon_names(count=count, seed=seed)

def _make_drawing_bw(obj, mode="white_outline", stroke_width=0.3):
    """
    Transform a reportlab drawing for different B&W/Mask styles.
    """
    # Safe handling: if color is a string (like 'url(#...)'), reportlab might complain.
    # Explicitly clear any existing complex color objects (gradients)
    
    # Handle direct fill/stroke properties
    if hasattr(obj, 'fillColor'):
        if mode == "white_solid" or mode == "white_solid_black_outline":
             obj.fillColor = shapes.colors.white
        elif mode == "white_outline":
             obj.fillColor = shapes.colors.black
        elif obj.fillColor is not None:
             # Fallback: if not None, make it black for outline mode
             obj.fillColor = shapes.colors.black

    if hasattr(obj, 'strokeColor'):
        if mode == "white_solid":
            obj.strokeColor = shapes.colors.white
        elif mode == "white_outline":
            obj.strokeColor = shapes.colors.white
            if hasattr(obj, 'strokeWidth'): obj.strokeWidth = stroke_width
        elif mode == "white_solid_black_outline":
            obj.strokeColor = shapes.colors.black
            if hasattr(obj, 'strokeWidth'): obj.strokeWidth = stroke_width
        else:
            obj.strokeColor = None

    # Recurse into groups or other container types
    if hasattr(obj, 'contents'):
        for sub in obj.contents:
            _make_drawing_bw(sub, mode=mode, stroke_width=stroke_width)
    elif isinstance(obj, list):
        for sub in obj:
            _make_drawing_bw(sub, mode=mode, stroke_width=stroke_width)

def load_icon_as_image(icon_full_name, size=512, render_mode="color", stroke_width=0.3):
    """
    Load an icon (e.g. 'twemoji:rocket' or 'local_icon') as a PIL Image.
    Uses memory caching and atomic disk writes to prevent race conditions.
    """
    # 0. Check Memory Cache First
    cache_key = f"{icon_full_name}_{size}_{render_mode}_{stroke_width}"
    if cache_key in _ICON_MEMORY_CACHE:
        return _ICON_MEMORY_CACHE[cache_key].copy()

    mode_suffixes = {
        "color": "",
        "white_solid": "_ws",
        "white_outline": "_wo",
        "white_solid_black_outline": "_wb"
    }
    
    suffix = mode_suffixes.get(render_mode, "")
    if render_mode != "color":
        suffix += f"_{stroke_width}"

    # 1. Handle root-level local icons (no colon)
    if ":" not in icon_full_name:
        cache_path = os.path.join(CACHE_DIR, f"{icon_full_name}{suffix}.png")
        if os.path.exists(cache_path):
            try:
                img = Image.open(cache_path).convert("RGBA")
                img.load() # Realize data
                _ICON_MEMORY_CACHE[cache_key] = img
                return img.copy()
            except Exception:
                pass
        
        base_path = os.path.join(CACHE_DIR, f"{icon_full_name}.png")
        if os.path.exists(base_path):
            img = Image.open(base_path).convert("RGBA")
            img.load()
            _ICON_MEMORY_CACHE[cache_key] = img
            return img.copy()
        
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

    # 2. Handle collection-prefixed icons
    collection, name = icon_full_name.split(":", 1)
    safe_collection = collection.replace("/", "_")
    safe_name = name.replace("/", "_")
    
    subdir = ensure_cache_dir(safe_collection)
    cache_path = os.path.join(subdir, f"{safe_name}{suffix}.png")
    
    if os.path.exists(cache_path):
        try:
            img = Image.open(cache_path).convert("RGBA")
            img.load()
            _ICON_MEMORY_CACHE[cache_key] = img
            return img.copy()
        except Exception:
            pass # Fallback to re-fetching if corrupted
            
    try:
        svg_data = pyconify.svg(icon_full_name)
        if isinstance(svg_data, bytes):
            svg_data = svg_data.decode("utf-8")
        
        if "url(#" in svg_data:
            svg_data = re.sub(r'url\(#.*?\)', 'black', svg_data)
        
        svg_file = io.BytesIO(svg_data.encode("utf-8"))
        drawing = svg2rlg(svg_file)
        
        if render_mode != "color":
            _make_drawing_bw(drawing, mode=render_mode, stroke_width=stroke_width)

        scale = min(size / drawing.width, size / drawing.height)
        drawing.scale(scale, scale)
        drawing.width *= scale
        drawing.height *= scale
        
        png_stream = io.BytesIO()
        renderPM.drawToFile(drawing, png_stream, fmt="PNG", bg=None)
        
        img = Image.open(io.BytesIO(png_stream.getvalue())).convert("RGBA")
        img.load()
        
        # 3. Atomic Cache Write
        tmp_path = cache_path + f".tmp_{random.randint(0, 999999)}"
        img.save(tmp_path)
        os.replace(tmp_path, cache_path)
        
        _ICON_MEMORY_CACHE[cache_key] = img
        return img.copy()
    except Exception as e:
        print(f"Error loading icon {icon_full_name} in mode {render_mode}: {e}")
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))

def pil_to_tensor(pil_img, extract_luminance_mask=False):
    """
    Convert PIL Image to ComfyUI Image Tensor (B, H, W, C) with robust mask generation.
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    
    if img_np.shape[2] == 4:
        # RGBA
        image = torch.from_numpy(img_np[:, :, :3]).unsqueeze(0)
        alpha = img_np[:, :, 3]
        
        if extract_luminance_mask:
            luminance = np.mean(img_np[:, :, :3], axis=2)
            mask_data = alpha * luminance
            
            # SILHOUETTE FALLBACK:
            # If the resulting mask is too dark (Max < 0.1) but there is a clear shape (Alpha Max > 0.5),
            # it means we have a black-on-transparent icon in a B&W mode.
            # We fallback to the Alpha channel to ensure the mask isn't empty.
            if np.max(mask_data) < 0.1 and np.max(alpha) > 0.5:
                mask_data = alpha
        else:
            mask_data = alpha
            
        mask = torch.from_numpy(mask_data).unsqueeze(0)
    else:
        # RGB
        image = torch.from_numpy(img_np).unsqueeze(0)
        mask = torch.from_numpy(np.mean(img_np, axis=2)).unsqueeze(0)
        
    return image, mask

def tensor_to_pil(tensor):
    """
    Convert ComfyUI Image Tensor (1, H, W, C) to PIL Image
    """
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    
    img_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)
