import os
from pathlib import Path
from aiohttp import web
from server import PromptServer

# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

routes = PromptServer.instance.routes

# Skins and Presets directories (relative to this file)
_SKINS_DIR = Path(__file__).parent.parent / "webamp_skins"
_VISUALIZERS_DIR = Path(__file__).parent.parent / "webamp_visualizers"


@routes.get("/webamp_skins/list")
async def list_webamp_skins(request: web.Request) -> web.Response:
    """Return available .wsz skins from web/webamp_skins/."""
    _SKINS_DIR.mkdir(exist_ok=True)
    skins = []
    for p in sorted(_SKINS_DIR.glob("*.wsz")):
        skins.append({
            "name": p.stem.replace("_", " "),
            "url": f"/webamp_skins/{p.name}",   # served by route below
            "filename": p.name,
        })
    return web.json_response({"skins": skins})


@routes.get("/webamp_skins/{filename}")
async def serve_webamp_skin(request: web.Request) -> web.Response:
    """Serve a .wsz skin file from web/webamp_skins/."""
    filename = request.match_info["filename"]
    skin_path = _SKINS_DIR / filename
    if not skin_path.is_file() or skin_path.suffix.lower() != ".wsz":
        return web.Response(status=404, text=f"Skin not found: {filename}")
    return web.FileResponse(skin_path, headers={"Content-Type": "application/octet-stream"})


@routes.get("/webamp_visualizers/list")
async def list_webamp_visualizers(request: web.Request) -> web.Response:
    """Return available .json Milkdrop presets from webamp_visualizers/."""
    _VISUALIZERS_DIR.mkdir(exist_ok=True)
    presets = []
    for p in sorted(_VISUALIZERS_DIR.glob("*.json")):
        presets.append({
            "name": p.stem.replace("_", " "),
            "url": f"/webamp_visualizers/{p.name}",
            "filename": p.name,
        })
    return web.json_response({"visualizers": presets})


@routes.get("/webamp_visualizers/{filename}")
async def serve_webamp_visualizer(request: web.Request) -> web.Response:
    """Serve a .json preset file from webamp_visualizers/."""
    filename = request.match_info["filename"]
    preset_path = _VISUALIZERS_DIR / filename
    if not preset_path.is_file() or preset_path.suffix.lower() != ".json":
        return web.Response(status=404, text=f"Visualizer not found: {filename}")
    return web.FileResponse(preset_path, headers={"Content-Type": "application/json"})


@routes.get("/webamp_visualizers/bundle")
async def bundle_webamp_visualizers(request: web.Request) -> web.Response:
    """Return all visualizer presets as one large dictionary with caching."""
    import json
    
    # Simple global cache to avoid re-reading 500+ files on every refresh
    if not hasattr(bundle_webamp_visualizers, "_cache"):
        bundle_webamp_visualizers._cache = None

    if bundle_webamp_visualizers._cache is not None:
        return web.json_response(bundle_webamp_visualizers._cache)

    _VISUALIZERS_DIR.mkdir(exist_ok=True)
    bundle = {}
    success_count = 0
    fail_count = 0
    
    for p in sorted(_VISUALIZERS_DIR.glob("*.json")):
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Butterchurn presets must be dictionaries
                if isinstance(data, dict):
                    bundle[p.stem.replace("_", " ")] = data
                    success_count += 1
                else:
                    print(f"[WebampRadio] Skipping {p.name}: Not a valid preset dictionary.")
                    fail_count += 1
        except Exception as e:
            print(f"[WebampRadio] Error loading visualizer {p.name}: {e}")
            fail_count += 1
            
    print(f"[WebampRadio] Bundle created: {success_count} success, {fail_count} failed.")
    bundle_webamp_visualizers._cache = bundle
    return web.json_response(bundle)


@routes.get("/radio_player/scan")
async def scan_folder(request: web.Request) -> web.Response:
    folder = request.rel_url.query.get("folder", "").strip()
    if not folder:
        return web.json_response({"error": "No folder specified"}, status=400)

    folder_path = Path("./output/" + folder)
    if not folder_path.is_dir():
        return web.json_response({"error": f"Folder not found: {folder}"}, status=404)

    extensions = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".webm"}
    files = []
    for ext in extensions:
        files.extend(folder_path.glob(f"*{ext}"))

    files.sort(key=lambda p: p.stat().st_mtime)

    tracks = []
    for p in files:
        lrc_path = p.with_suffix(".lrc")
        has_lrc = lrc_path.exists()
        
        tracks.append({
            "filename": p.name,
            "path": str(p),
            "mtime": p.stat().st_mtime,
            "url": f"/radio_player/audio?path={p}",
            "lrc_url": f"/radio_player/lrc?path={lrc_path}" if has_lrc else None,
        })

    return web.json_response({"tracks": tracks})


@routes.get("/radio_player/audio")
async def serve_audio(request: web.Request) -> web.Response:
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="No path specified")

    file_path = Path(path)
    if not file_path.is_file():
        return web.Response(status=404, text=f"File not found: {path}")

    mime_map = {
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
        ".ogg":  "audio/ogg",
        ".flac": "audio/flac",
        ".m4a":  "audio/mp4",
        ".aac":  "audio/aac",
        ".opus": "audio/opus",
        ".webm": "audio/webm",
    }
    mime = mime_map.get(file_path.suffix.lower(), "application/octet-stream")
    return web.FileResponse(file_path, headers={"Content-Type": mime})


@routes.get("/radio_player/lrc")
async def serve_lrc(request: web.Request) -> web.Response:
    path = request.rel_url.query.get("path", "").strip()
    if not path:
        return web.Response(status=400, text="No path specified")

    file_path = Path(path)
    if not file_path.is_file():
        return web.Response(status=404, text=f"File not found: {path}")

    return web.FileResponse(file_path, headers={"Content-Type": "text/plain; charset=utf-8"})


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class RadioPlayerNode:
    """A ComfyUI node that spins up an interactive WebAmp radio player.
    Polls a specifiable output directory for generated audio/lyrics and serves them.
    
    Inputs:
        folder (STRING): Directory inside /output to monitor.
        poll_interval_seconds (FLOAT): UI polling refresh rate.
        
    Outputs:
        (None. Visual Web UI element only.)
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "audio",
                    "multiline": False,
                    "placeholder": "audio folder (relative to /output)",
                }),
            },
            "optional": {
                "poll_interval_seconds": ("FLOAT", {
                    "default": 60.0,
                    "min": 15.0,
                    "max": 600.0,
                    "step": 0.5,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Scromfy/Ace-Step/Radio"

    def run(self, folder: str, poll_interval_seconds: float = 60.0):
        return {}


NODE_CLASS_MAPPINGS = {
    "RadioPlayer": RadioPlayerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadioPlayer": "Radio Player 📻",
}
