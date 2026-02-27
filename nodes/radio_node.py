import os
from pathlib import Path
from aiohttp import web
from server import PromptServer

# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

routes = PromptServer.instance.routes


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

    tracks = [
        {
            "filename": p.name,
            "path": str(p),
            "mtime": p.stat().st_mtime,
            "url": f"/radio_player/audio?path={p}",
        }
        for p in files
    ]

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


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class RadioPlayerNode:

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
