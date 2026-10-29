"""FastAPI server: POST /restore takes an uploaded image and streams back the u-map, then the
restored image, as newline-delimited JSON. Serves static/ at /.

Each line of the response is one JSON object:
  {"umap": <base64 PNG>, "t_umap_ms": int}        as soon as the u-map is ready
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

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

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


@app.post("/restore")
async def restore(file: UploadFile = File(...)):
    start = time.perf_counter()
    data = await file.read()
    try:
        image = _decode(data)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="That file is not a readable image.")
    return StreamingResponse(_stream(image, start), media_type="application/x-ndjson")


def _stream(image, start):
    with _busy:
        try:
            for stage, result in pipeline.restore(image):
                if stage == "umap":
                    body = {"umap": _png_b64(_heatmap(result))}
                    key = "t_umap_ms"
                else:
                    body = {"restored": _png_b64(_to_uint8(result))}
                    key = "t_total_ms"
                body[key] = round((time.perf_counter() - start) * 1000)
                yield json.dumps(body) + "\n"
        except Exception as exc:
            log.exception("restoration failed")
            yield json.dumps({"error": str(exc)}) + "\n"


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
