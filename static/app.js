/*
 * Browser side of the app. Plain JavaScript, no libraries, no build step.
 *
 * What it does: the user picks an image and clicks Restore. This file sends it to the
 * server with one POST request and shows the results as they stream back.
 *
 * Server contract (implemented in app/main.py, POST /restore):
 *   Request:  multipart/form-data with one field named "file" holding the image.
 *   Response: status 200 with a STREAM of newline-delimited JSON ("NDJSON"). Each line is one
 *             complete JSON object, sent as soon as that result is ready:
 *               {"umap": "<base64 PNG>", "t_umap_ms": 1234}       first: the u-map heatmap
 *               {"restored": "<base64 PNG>", "t_total_ms": 56789} later: the restored image
 *               {"error": "message"}                              if it fails after streaming began
 *             Both images have the same width and height as the uploaded image.
 *             Times are milliseconds since the server received the upload.
 *   Errors before streaming (for example the file is not an image): status 400 with the JSON
 *             body {"detail": "message"}.
 *
 * The u-map arrives well before the restored image because the diffusion step (IR-SDE) is slow
 * on a CPU, so the page updates the u-map panel first and the restored panel when it can.
 */

"use strict";

// ---- Page elements (ids are defined in index.html) -------------------------------------
const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const button = document.getElementById("restore-button");
const statusLine = document.getElementById("status");

// One entry per result panel: its <img>, its placeholder text, and its timing label.
const panels = {
  original: panel("original", null),
  umap: panel("umap", "umap-time"),
  restored: panel("restored", "restored-time"),
};

function panel(name, timeId) {
  return {
    img: document.getElementById(`${name}-img`),
    placeholder: document.getElementById(`${name}-placeholder`),
    time: timeId ? document.getElementById(timeId) : null,
  };
}

// ---- Small helpers ---------------------------------------------------------------------

// Show an image in a panel and hide its placeholder. `src` is an object URL or a data URL.
function showImage(p, src) {
  p.img.src = src;
  p.img.hidden = false;
  p.placeholder.hidden = true;
}

// Put a panel back to its empty "Waiting" state.
function resetPanel(p, placeholderText) {
  p.img.hidden = true;
  p.img.removeAttribute("src");
  p.placeholder.textContent = placeholderText;
  p.placeholder.hidden = false;
  if (p.time) p.time.textContent = "";
}

// Milliseconds -> "850 ms" or "12.3 s".
function formatMs(ms) {
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

// Server images are base64 PNGs; a data URL lets <img> display them without another request.
function pngDataUrl(base64) {
  return `data:image/png;base64,${base64}`;
}

// ---- Handling one line of the response stream ------------------------------------------

function handleMessage(msg) {
  if (msg.error) {
    // The server failed part-way through. Keep whatever already arrived, report the problem.
    setStatus(`Restoration failed: ${msg.error}`, true);
  } else if (msg.umap) {
    showImage(panels.umap, pngDataUrl(msg.umap));
    panels.umap.time.textContent = `(${formatMs(msg.t_umap_ms)})`;
    setStatus("U-map ready. Restoring the image, this can take a while on CPU...");
  } else if (msg.restored) {
    showImage(panels.restored, pngDataUrl(msg.restored));
    panels.restored.time.textContent = `(${formatMs(msg.t_total_ms)})`;
    setStatus(`Done in ${formatMs(msg.t_total_ms)}.`);
  }
}

// Read the response body as it arrives. A network chunk can end in the middle of a line, so
// text is collected in `buffer` and only complete lines (ending in "\n") are parsed.
async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let newline;
    while ((newline = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) handleMessage(JSON.parse(line));
    }
  }

  // The last line normally ends with "\n", but handle a final line without one too.
  const rest = (buffer + decoder.decode()).trim();
  if (rest) handleMessage(JSON.parse(rest));
}

// ---- The upload ------------------------------------------------------------------------

let originalUrl = null; // object URL of the chosen file; released when a new one is chosen

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stay on this page instead of letting the browser submit the form
  const file = fileInput.files[0];
  if (!file) return;

  // Show the original and clear the old results.
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  showImage(panels.original, originalUrl);
  resetPanel(panels.umap, "Waiting");
  resetPanel(panels.restored, "Waiting");

  button.disabled = true;
  setStatus("Uploading and computing the u-map...");

  try {
    const body = new FormData();
    body.append("file", file); // the field name must match the "file" parameter in app/main.py

    const response = await fetch("/restore", { method: "POST", body });

    if (!response.ok) {
      // Rejected before streaming started, e.g. 400 "That file is not a readable image."
      const problem = await response.json().catch(() => ({}));
      throw new Error(problem.detail || `Server returned status ${response.status}`);
    }
    await readStream(response);
  } catch (err) {
    setStatus(err.message, true);
    // Keep any result that already arrived; only mark the panels that are still empty.
    for (const p of [panels.umap, panels.restored]) {
      if (p.img.hidden) p.placeholder.textContent = "No result";
    }
  } finally {
    button.disabled = false;
  }
});
