/*
 * Browser side of the app. Plain JavaScript, no libraries, no build step.
 *
 * What it does: the user picks an image and presses Restore. This file sends it to the server
 * with one POST request and updates the page as results and progress events stream back, so the
 * user can always see that the server is working (step lamps, progress bar, elapsed timer).
 *
 * Server contract (implemented in app/main.py, POST /restore):
 *   Request:  multipart/form-data with one field named "file" holding the image.
 *   Response: status 200 with a STREAM of newline-delimited JSON ("NDJSON"). Each line is one
 *             complete JSON object, sent as soon as it is ready, in this order:
 *               {"progress": {"stage": "queued"}, "t_ms": n}    only if another upload is running
 *               {"progress": {"stage": "denoise"}, "t_ms": n}   NAFNet started
 *               {"progress": {"stage": "umap"}, "t_ms": n}      UMapNet started
 *               {"umap": "<base64 PNG>", "t_umap_ms": n}        the u-map heatmap
 *               {"progress": {"stage": "diffuse", "done": k, "total": m}, "t_ms": n}
 *                                                               IR-SDE: k of m flagged patches done
 *                                                               (repeated; absent if none flagged)
 *               {"restored": "<base64 PNG>", "t_total_ms": n}   the final image
 *               {"error": "message"}                            if it fails after streaming began
 *             Both images have the same width and height as the uploaded image.
 *             t_* values are milliseconds since the server received the upload.
 *   Errors before streaming (not an image, too small): status 4xx with JSON {"detail": "message"}.
 *
 * The page is same-origin with the API (main.py serves both), so "/restore" works unchanged
 * behind a tunnel such as ngrok.
 */

"use strict";

// After this long without any message from the server, warn that it may be stuck.
// One IR-SDE patch takes roughly 20 s on a laptop CPU, so this leaves a wide margin.
const STALL_WARNING_S = 120;

// ---- Page elements (ids are defined in index.html) -------------------------------------
const $ = (id) => document.getElementById(id);
const form = $("upload-form");
const fileInput = $("file-input");
const fileName = $("file-name");
const button = $("restore-button");
const statusLine = $("status");
const errorBar = $("error-bar");
const errorText = $("error-text");
const progressTrack = $("progress");
const progressFill = $("progress-fill");
const meterCount = $("meter-count");
const elapsedOut = $("elapsed");
const lastUpdateOut = $("last-update");
const etaOut = $("eta");
const downloadLink = $("download-link");
const steps = [...document.querySelectorAll("#steps li")]; // data-step: denoise, umap, diffuse, done

// One entry per result panel: its <img>, its placeholder text, and its timing label.
const panels = {
  original: panel("original", null),
  umap: panel("umap", "umap-time"),
  restored: panel("restored", "restored-time"),
};

function panel(name, timeId) {
  return {
    img: $(`${name}-img`),
    placeholder: $(`${name}-placeholder`),
    time: timeId ? $(timeId) : null,
  };
}

// ---- Small helpers ---------------------------------------------------------------------

// Milliseconds -> "850 ms", "12.3 s" or "4 min 05 s".
function formatMs(ms) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)} s`;
  const m = Math.floor(s / 60);
  return `${m} min ${String(Math.round(s % 60)).padStart(2, "0")} s`;
}

function showImage(p, src) {
  p.img.src = src;
  p.img.hidden = false;
  p.placeholder.hidden = true;
}

function resetPanel(p, placeholderText) {
  p.img.hidden = true;
  p.img.removeAttribute("src");
  p.placeholder.textContent = placeholderText;
  p.placeholder.hidden = false;
  if (p.time) p.time.textContent = "";
}

function setStatus(text, isWarning = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("is-warning", isWarning);
}

function showError(message) {
  errorText.textContent = message;
  errorBar.hidden = false;
}

// Server images are base64 PNGs; a data URL lets <img> display them without another request.
function pngDataUrl(base64) {
  return `data:image/png;base64,${base64}`;
}

// Step lamps. Every step before `name` is marked done, `name` is active, the rest wait.
// `state` is the text shown under the active step (e.g. "running", "3 / 29").
function setStep(name, state = "running") {
  const index = steps.findIndex((li) => li.dataset.step === name);
  steps.forEach((li, i) => {
    li.classList.toggle("is-done", i < index);
    li.classList.toggle("is-active", i === index);
    const label = li.querySelector(".step-state");
    if (i < index && label.textContent !== "skipped") label.textContent = "done";
    if (i === index) label.textContent = state;
    if (i > index) label.textContent = "waiting";
  });
}

function resetSteps() {
  steps.forEach((li) => {
    li.classList.remove("is-done", "is-active");
    li.querySelector(".step-state").textContent = "waiting";
  });
}

function setProgress(done, total) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;
  progressFill.style.width = `${pct}%`;
  progressTrack.setAttribute("aria-valuenow", String(pct));
  meterCount.textContent = `${done} / ${total}`;
}

// ---- Timers: prove the request is alive --------------------------------------------------

let requestStart = 0; // performance.now() when Restore was pressed
let lastMessageAt = 0; // performance.now() of the latest message from the server
let diffuseStart = 0; // when IR-SDE reported its first progress event
let ticker = null;

function startTicker() {
  requestStart = lastMessageAt = performance.now();
  ticker = setInterval(tick, 1000);
  tick();
}

function stopTicker() {
  clearInterval(ticker);
  ticker = null;
  tick();
}

function tick() {
  const now = performance.now();
  elapsedOut.textContent = formatMs(now - requestStart);
  const silentS = (now - lastMessageAt) / 1000;
  lastUpdateOut.textContent = ticker ? `${Math.round(silentS)} s ago` : "–";
  if (ticker && silentS > STALL_WARNING_S) {
    setStatus(
      `No update from the server for ${Math.round(silentS)} s. It may be stuck: check the server log.`,
      true
    );
  }
}

// ---- Handling one line of the response stream ------------------------------------------

let gotRestored = false;

function handleMessage(msg) {
  lastMessageAt = performance.now();

  if (msg.error) {
    // The server failed part-way through. Keep whatever already arrived, report the problem.
    showError(`Restoration failed: ${msg.error}`);
    setStatus("Stopped because of an error.");
  } else if (msg.progress) {
    const { stage, done, total } = msg.progress;
    if (stage === "queued") {
      setStatus("Another restoration is running on the server. Yours starts when it finishes.");
    } else if (stage === "denoise") {
      setStep("denoise");
      setStatus("Denoising with NAFNet...");
    } else if (stage === "umap") {
      setStep("umap");
      setStatus("Predicting the uncertainty map...");
    } else if (stage === "diffuse") {
      if (done === 0) diffuseStart = performance.now();
      setStep("diffuse", `${done} / ${total}`);
      setProgress(done, total);
      setStatus(`Diffusion (IR-SDE) on the uncertain patches: ${done} of ${total} done.`);
      if (done > 0 && done < total) {
        const perPatch = (performance.now() - diffuseStart) / done;
        etaOut.textContent = formatMs(perPatch * (total - done));
      }
    }
  } else if (msg.umap) {
    showImage(panels.umap, pngDataUrl(msg.umap));
    panels.umap.time.textContent = formatMs(msg.t_umap_ms);
  } else if (msg.restored) {
    gotRestored = true;
    const url = pngDataUrl(msg.restored);
    showImage(panels.restored, url);
    panels.restored.time.textContent = formatMs(msg.t_total_ms);
    downloadLink.href = url;
    downloadLink.hidden = false;
    // No diffuse events means the gate flagged nothing and IR-SDE was skipped.
    const diffuse = steps.find((li) => li.dataset.step === "diffuse");
    if (!diffuse.classList.contains("is-active")) diffuse.querySelector(".step-state").textContent = "skipped";
    setStep("done", "ok");
    steps[steps.length - 1].classList.add("is-done");
    etaOut.textContent = "–";
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

// ---- Device notice ---------------------------------------------------------------------
// GET /info returns {"device": "cuda"} or {"device": "cpu"}. CPU means this copy runs on a
// machine without a GPU (typically someone's own laptop, used when the hosted GPU link is down),
// so the page sets expectations: the u-map is quick, the full restoration takes minutes.

let onCpu = false;

async function showDeviceNote() {
  try {
    const { device } = await (await fetch("/info")).json();
    onCpu = device !== "cuda";
    const note = $("device-note");
    $("device-text").textContent = onCpu
      ? "Running on this computer's CPU: no GPU was found on this machine, so each restoration " +
        "takes about 3+ minutes. The u-map still appears within seconds."
      : "Running on a GPU.";
    note.classList.toggle("is-cpu", onCpu);
    note.hidden = false;
  } catch {
    // Older server without /info, or offline: leave the notice hidden.
  }
}

showDeviceNote();

// ---- The upload ------------------------------------------------------------------------

let originalUrl = null; // object URL of the chosen file; released when a new one is chosen

fileInput.addEventListener("change", () => {
  fileName.textContent = fileInput.files[0] ? fileInput.files[0].name : "No file chosen";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stay on this page instead of letting the browser submit the form
  const file = fileInput.files[0];
  if (!file) return;

  // Show the original and clear the previous run.
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  showImage(panels.original, originalUrl);
  resetPanel(panels.umap, "Waiting");
  resetPanel(panels.restored, "Waiting");
  downloadLink.hidden = true;
  errorBar.hidden = true;
  resetSteps();
  setProgress(0, 0);
  etaOut.textContent = "–";
  gotRestored = false;

  button.disabled = true;
  button.classList.add("is-busy");
  setStatus(onCpu ? "Uploading... CPU mode (no GPU on this machine): expect about 3+ minutes." : "Uploading...");
  startTicker();

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
    if (!gotRestored && errorBar.hidden) {
      throw new Error("The connection closed before the restored image arrived.");
    }
  } catch (err) {
    // Network failure, rejected upload, or a stream that ended early.
    showError(err.message);
    setStatus("Stopped because of an error.");
    // Keep any result that already arrived; only mark the panels that are still empty.
    for (const p of [panels.umap, panels.restored]) {
      if (p.img.hidden) p.placeholder.textContent = "No result";
    }
  } finally {
    stopTicker();
    button.disabled = false;
    button.classList.remove("is-busy");
  }
});
