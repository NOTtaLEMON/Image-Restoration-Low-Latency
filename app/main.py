"""FastAPI server: POST /restore takes an uploaded image and streams back the u-map, then the
restored image, as newline-delimited JSON. Serves static/ at /.

Each line of the response is one JSON object:
  {"progress": {"stage": "queued"}, "t_ms": int}  another upload is still running; this one waits
  {"progress": {"stage": "denoise"}, "t_ms": int} NAFNet started
  {"progress": {"stage": "umap"}, "t_ms": int}    UMapNet started
  {"umap": <base64 PNG>, "t_umap_ms": int}        as soon as the u-map is ready
  {"progress": {"stage": "diffuse", "done": k, "total": n}, "t_ms": int}
                                                  IR-SDE progress, after each batch of patches
  {"restored": <base64 PNG>, "t_total_ms": int}   when the restored image is ready
  {"error": str}                                  if something fails after streaming has started
Times are measured from when the upload arrives and include PNG encoding.
An unreadable upload gets a plain 400 before any streaming starts.
"""
import base64
import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from app import pipeline
from app.config import DEVICE

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MIN_SIDE = 128  # smallest accepted width/height in pixels

# One restoration at a time: torch already uses every CPU core, so two at once only slow both down.
_busy = threading.Lock()

# Heatmap colour stops (low to high uncertainty), a dark-to-yellow ramp.
_STOPS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_COLORS = np.array([[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]], dtype=np.float64)


@asynccontextmanager
async def lifespan(app):
    pipeline.load()
    pipeline.warm_up()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/info")
def info():
    """Where the models run ("cuda" or "cpu"); the page uses it to warn about slow CPU runs."""
    return {"device": DEVICE}


@app.post("/restore")
async def restore(file: UploadFile = File(...)):
    start = time.perf_counter()
    data = await file.read()
    try:
        image = _decode(data)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="That file is not a readable image.")
    # Smaller images break the gate: a single 64x64 patch makes its ranking divide by zero.
    if min(image.shape[:2]) < MIN_SIDE:
        raise HTTPException(status_code=400, detail=f"Image is too small: both sides must be at least {MIN_SIDE} px.")
    return StreamingResponse(_stream(image, start), media_type="application/x-ndjson")


def _stream(image, start):
    def ms():
        return round((time.perf_counter() - start) * 1000)

    # Tell the page it is waiting rather than stuck, if another restoration holds the lock.
    if not _busy.acquire(blocking=False):
        yield json.dumps({"progress": {"stage": "queued"}, "t_ms": ms()}) + "\n"
        _busy.acquire()
    try:
        for kind, result in pipeline.restore(image):
            if kind == "progress":
                body = {"progress": result, "t_ms": ms()}
            elif kind == "umap":
                body = {"umap": _png_b64(_heatmap(result))}
                body["t_umap_ms"] = ms()
            else:
                body = {"restored": _png_b64(_to_uint8(result))}
                body["t_total_ms"] = ms()
            yield json.dumps(body) + "\n"
    except Exception as exc:
        log.exception("restoration failed")
        yield json.dumps({"error": str(exc)}) + "\n"
    finally:
        _busy.release()


def _decode(data):
    img = Image.open(io.BytesIO(data))
    # Phone photos store rotation in EXIF; apply it so the result matches what the browser shows.
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def _to_uint8(a):
    return (np.clip(a, 0, 1) * 255).round().astype(np.uint8)


def _heatmap(u):
    u = np.clip(u, 0, 1)
    rgb = np.stack([np.interp(u, _STOPS, _COLORS[:, c]) for c in range(3)], axis=-1)
    return rgb.round().astype(np.uint8)


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=1)  # level 1: fast, bigger file
    return base64.b64encode(buf.getvalue()).decode("ascii")


# Mounted last so it does not shadow /restore.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
