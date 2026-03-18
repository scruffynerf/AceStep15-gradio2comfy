import os
import io
import re
import sys
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM, shapes
from PIL import Image

SIZE = 512

def _force_white(obj):
    """
    Recursively force all colors in a reportlab drawing to white.
    """
    if hasattr(obj, 'fillColor'):
        obj.fillColor = shapes.colors.white
    if hasattr(obj, 'strokeColor'):
        obj.strokeColor = shapes.colors.white
        
    if hasattr(obj, 'contents'):
        for sub in obj.contents:
            _force_white(sub)
    elif isinstance(obj, list):
        for sub in obj:
            _force_white(sub)

def convert_svg_file(svg_path):
    """
    Converts a single SVG file to a white-on-black PNG.
    """
    png_path = svg_path.replace(".svg", ".png")
    
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            svg_data = f.read()
        
        # Scrub gradients and force any hardcoded colors to white in the SVG source if necessary
        # but _force_white handles the drawing objects directly which is more reliable.
        if "url(#" in svg_data:
            svg_data = re.sub(r'url\(#.*?\)', 'white', svg_data)
        
        drawing = svg2rlg(io.BytesIO(svg_data.encode("utf-8")))
        _force_white(drawing)

        # Scale to fit
        scale = min(SIZE / drawing.width, SIZE / drawing.height)
        drawing.scale(scale, scale)
        drawing.width *= scale
        drawing.height *= scale
        
        # Render to PNG with black background (bg=shapes.colors.black)
        png_stream = io.BytesIO()
        renderPM.drawToFile(drawing, png_stream, fmt="PNG", bg=shapes.colors.black)
        
        img = Image.open(io.BytesIO(png_stream.getvalue())).convert("RGB")
        
        # Resize exactly to 512x512 if not already (in case aspect ratio was off)
        final_img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        x_off = (SIZE - img.width) // 2
        y_off = (SIZE - img.height) // 2
        final_img.paste(img, (x_off, y_off))
        
        final_img.save(png_path)
        print(f"Converted: {os.path.relpath(svg_path)} -> {os.path.basename(png_path)}")
        return True
        
    except Exception as e:
        print(f"Error converting {svg_path}: {e}")
        return False

def process_directory(root_dir):
    """
    Recursively find and convert all SVGs in a directory.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    found_count = 0
    converted_count = 0

    print(f"Scanning directory: {root_dir}")
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(".svg"):
                found_count += 1
                svg_path = os.path.join(root, file)
                if convert_svg_file(svg_path):
                    converted_count += 1

    print(f"\nProcessing complete!")
    print(f"Found: {found_count} SVGs")
    print(f"Successfully converted: {converted_count}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 masks/convert_svgs.py <directory_path>")
        sys.exit(1)

    target_dir = sys.argv[1]
    process_directory(target_dir)
