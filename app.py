"""Entry point for hosting on a Hugging Face Space with the Gradio SDK (Docker Spaces are paid).

A Gradio Space simply runs `python app.py` and sends web traffic to port 7860. This file does not
use Gradio's UI: it downloads the weights, then serves the normal FastAPI app from app/main.py,
so the hosted page is exactly the local one. Locally you do not need this file; keep using
`python -m uvicorn app.main:app`.

ZeroGPU hardware: the GPU is attached only while a function decorated with @spaces.GPU runs,
and the Space refuses to start unless it is told about one. So:
  - `spaces` is imported before anything imports torch (it patches torch's CUDA handling);
  - pipeline.restore (the whole per-request pipeline) is wrapped in @spaces.GPU;
  - the models are loaded HERE, before the server starts: ZeroGPU packs the weights that were
    moved to "cuda" at startup and ships them to the GPU worker for each call;
  - spaces.zero.startup() is called by hand. ZeroGPU normally runs it from gr.Blocks.launch(),
    which this app never calls (it serves FastAPI, not a Gradio UI); without it the Space fails
    with "No @spaces.GPU function detected during startup";
  - main.py's startup load and warm-up are then skipped (already loaded; GPU work outside the
    decorated function is not allowed).
Requests carry no Gradio login, so ZeroGPU applies its IP-based (anonymous) quota.
On non-ZeroGPU hardware @spaces.GPU does nothing and spaces.zero.startup does not exist, so this
file also works on a CPU Space.

Name clash note: `from app import ...` below imports the app/ PACKAGE, not this file. Python
prefers a package directory over a module file of the same name, and this file runs as __main__.
The Space's packages come from deploy/space_requirements.txt (uploaded as its requirements.txt).
"""
import spaces  # noqa: I001  must come before torch is imported (by app.* below)

import os
import runpy
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent

# GPU time per request, in seconds. ZeroGPU aborts a call that runs longer, and every call
# counts against the visitor's daily ZeroGPU quota.
GPU_DURATION_S = 120

# Fetch the three checkpoints into hf_weights/ (pinned revision, SHA-256 checked; exits on failure).
runpy.run_path(str(ROOT / "scripts" / "download_weights.py"), run_name="__main__")

from app import pipeline  # noqa: E402  (weights must exist before the models load)

# main.py looks up pipeline.restore / .load / .warm_up when it calls them, so replacing them here
# is enough; nothing under app/ needs to know about ZeroGPU.
pipeline.restore = spaces.GPU(duration=GPU_DURATION_S)(pipeline.restore)
pipeline.load()  # before startup(): ZeroGPU packs the CUDA weights loaded so far
zero_startup = getattr(getattr(spaces, "zero", None), "startup", None)  # only exists on ZeroGPU
if zero_startup is not None:
    zero_startup()
pipeline.load = lambda: None
pipeline.warm_up = lambda: None

from app.main import app as fastapi_app  # noqa: E402

uvicorn.run(fastapi_app, host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
