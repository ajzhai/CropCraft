"""Builds the interactive reconstruction gallery as a standalone HTML page.

Every carved volume is embedded as zlib-compressed uint8 voxel coordinates and
decompressed in the browser, so the whole gallery (both segmenters, every
specimen) is one self-contained file with no external requests.

Rendering is a painter's-algorithm point cloud on a 2D canvas rather than WebGL:
the volumes are a few thousand voxels each, the depth cue is the whole point, and
a 2D canvas has no context-loss or driver-compatibility failure modes on the
lab machines this has to open on.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import WORK_DIR

TEMPLATE_HEAD = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Specimen Reconstructions</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --ground: #e9edf0;
  --surface: #ffffff;
  --surface-2: #f3f6f8;
  --line: #d2dadf;
  --line-strong: #b3bfc7;
  --ink: #10161c;
  --ink-2: #47555f;
  --ink-3: #78868f;
  --accent: #0f7d84;
  --accent-soft: #d7ecec;
  --near: #e8a52c;
  --far: #17606e;
  --ok: #2f7d55;
  --warn: #b0741f;
  --bad: #a8443c;
  --shadow: 0 1px 2px rgba(16,22,28,.06), 0 6px 20px rgba(16,22,28,.05);
  --radius: 3px;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #0d1216;
    --surface: #151d23;
    --surface-2: #1b242b;
    --line: #26323a;
    --line-strong: #3a4952;
    --ink: #e7edf1;
    --ink-2: #a2b1bb;
    --ink-3: #6f8089;
    --accent: #4bc4c9;
    --accent-soft: #143038;
    --near: #f0b543;
    --far: #2b8fa0;
    --ok: #57b581;
    --warn: #d99a3e;
    --bad: #d4726a;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
  }
}
:root[data-theme="dark"] {
  --ground: #0d1216;
  --surface: #151d23;
  --surface-2: #1b242b;
  --line: #26323a;
  --line-strong: #3a4952;
  --ink: #e7edf1;
  --ink-2: #a2b1bb;
  --ink-3: #6f8089;
  --accent: #4bc4c9;
  --accent-soft: #143038;
  --near: #f0b543;
  --far: #2b8fa0;
  --ok: #57b581;
  --warn: #d99a3e;
  --bad: #d4726a;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.3);
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1240px; margin: 0 auto; padding: 28px 20px 64px; }

header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 18px; margin-bottom: 6px; }
h1 {
  font-family: "IBM Plex Sans Condensed", "IBM Plex Sans", sans-serif;
  font-weight: 600; font-size: 26px; letter-spacing: -.01em; margin: 0;
  text-wrap: balance;
}
.sub { color: var(--ink-2); font-size: 14px; margin: 0 0 22px; max-width: 68ch; }

.eyebrow {
  font-family: "IBM Plex Sans Condensed", sans-serif;
  text-transform: uppercase; letter-spacing: .09em; font-size: 11px;
  font-weight: 600; color: var(--ink-3);
}

.stats { display: flex; flex-wrap: wrap; gap: 1px; background: var(--line);
  border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; margin-bottom: 22px; }
.stat { background: var(--surface); padding: 10px 16px; flex: 1 1 130px; }
.stat .v { font-family: "IBM Plex Mono", monospace; font-size: 19px; font-weight: 500;
  font-variant-numeric: tabular-nums; display: block; }
.stat .k { font-size: 11px; color: var(--ink-3); text-transform: uppercase; letter-spacing: .07em;
  font-family: "IBM Plex Sans Condensed", sans-serif; font-weight: 600; }

.toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 16px; }
.seg { display: inline-flex; border: 1px solid var(--line-strong); border-radius: var(--radius);
  overflow: hidden; background: var(--surface); }
.seg button {
  font: inherit; font-size: 13px; padding: 6px 13px; border: 0; background: transparent;
  color: var(--ink-2); cursor: pointer; border-right: 1px solid var(--line);
}
.seg button:last-child { border-right: 0; }
.seg button[aria-pressed="true"] { background: var(--accent); color: #fff; }
.seg button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.spacer { flex: 1 1 auto; }
.hint { font-size: 12px; color: var(--ink-3); }

.layout { display: grid; grid-template-columns: minmax(0,1fr) 300px; gap: 18px; align-items: start; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }

.stage { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); overflow: hidden; }
.stage-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding: 11px 15px; border-bottom: 1px solid var(--line); background: var(--surface-2); }
.stage-head h2 { margin: 0; font-size: 16px; font-weight: 600;
  font-family: "IBM Plex Mono", monospace; letter-spacing: -.01em; }
.chip { font-size: 11px; padding: 2px 7px; border-radius: 100px; background: var(--accent-soft);
  color: var(--accent); font-weight: 600; letter-spacing: .03em; }
.canvases { display: grid; grid-template-columns: 1fr; }
.canvases.pair { grid-template-columns: 1fr 1fr; }
.cwrap { position: relative; border-right: 1px solid var(--line); }
.cwrap:last-child { border-right: 0; }
.cwrap canvas { display: block; width: 100%; height: auto; touch-action: none; cursor: grab; }
.cwrap canvas:active { cursor: grabbing; }
.cwrap .tag { position: absolute; top: 8px; left: 10px; font-size: 11px; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase; color: var(--ink-3);
  font-family: "IBM Plex Sans Condensed", sans-serif; }
.cwrap .miss { position: absolute; inset: 0; display: grid; place-items: center;
  font-size: 13px; color: var(--ink-3); }

.facts { border-top: 1px solid var(--line); display: grid; grid-template-columns: repeat(4, 1fr); }
@media (max-width: 620px) { .facts { grid-template-columns: repeat(2, 1fr); } }
.fact { padding: 10px 15px; border-right: 1px solid var(--line); }
.fact:last-child { border-right: 0; }
.fact .k { font-size: 10px; }
.fact .v { font-family: "IBM Plex Mono", monospace; font-size: 14.5px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }

.side { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); max-height: 78vh; overflow-y: auto; }
.side h3 { margin: 0; padding: 11px 15px; font-size: 12px; position: sticky; top: 0;
  background: var(--surface-2); border-bottom: 1px solid var(--line); z-index: 2;
  font-family: "IBM Plex Sans Condensed", sans-serif; text-transform: uppercase;
  letter-spacing: .09em; color: var(--ink-3); }
.group { font-size: 11px; padding: 7px 15px 5px; color: var(--ink-3);
  font-family: "IBM Plex Sans Condensed", sans-serif; text-transform: uppercase;
  letter-spacing: .08em; font-weight: 600; background: var(--surface-2);
  border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
.item { display: grid; grid-template-columns: 44px 1fr auto; gap: 10px; align-items: center;
  width: 100%; text-align: left; font: inherit; padding: 7px 15px; background: transparent;
  border: 0; border-bottom: 1px solid var(--line); color: var(--ink); cursor: pointer; }
.item:hover { background: var(--surface-2); }
.item[aria-current="true"] { background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }
.item:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.item canvas { display: block; width: 44px; height: 44px; }
.item .id { font-family: "IBM Plex Mono", monospace; font-size: 13px; font-weight: 500; }
.item .meta { font-size: 11px; color: var(--ink-3); }
.item .num { font-family: "IBM Plex Mono", monospace; font-size: 12px; color: var(--ink-2);
  font-variant-numeric: tabular-nums; text-align: right; }

.note { margin-top: 24px; padding: 14px 16px; border: 1px solid var(--line);
  border-left: 3px solid var(--warn); border-radius: var(--radius);
  background: var(--surface); font-size: 13.5px; color: var(--ink-2); }
.note strong { color: var(--ink); }
.note p { margin: 0 0 8px; }
.note p:last-child { margin: 0; }
.note code { font-family: "IBM Plex Mono", monospace; font-size: 12.5px;
  background: var(--surface-2); padding: 1px 4px; border-radius: 2px; }

@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
"""

TEMPLATE_BODY = """<div class="wrap">
  <header>
    <h1>Specimen Reconstructions</h1>
    <span class="eyebrow">GG-SSVT &middot; space-carved occupancy</span>
  </header>
  <p class="sub">
    Every carved volume from the dual-Kinect rig, under both segmenters. Drag to
    rotate, scroll to zoom. Warmer voxels are nearer the camera.
  </p>

  <div class="stats" id="stats"></div>

  <div class="toolbar">
    <div class="seg" id="viewmode" role="group" aria-label="Segmenter">
      <button data-mode="geometric" aria-pressed="true">Geometric</button>
      <button data-mode="sam3d" aria-pressed="false">SAM3D</button>
      <button data-mode="pair" aria-pressed="false">Compare</button>
    </div>
    <div class="seg" id="sortmode" role="group" aria-label="Sort">
      <button data-sort="id" aria-pressed="true">By ID</button>
      <button data-sort="mass" aria-pressed="false">By mass</button>
      <button data-sort="volume" aria-pressed="false">By volume</button>
    </div>
    <span class="spacer"></span>
    <span class="hint" id="hint">Drag to rotate &middot; scroll to zoom</span>
  </div>

  <div class="layout">
    <div class="stage">
      <div class="stage-head">
        <h2 id="title"></h2>
        <span class="chip" id="species"></span>
        <span class="spacer"></span>
        <span class="hint" id="sub"></span>
      </div>
      <div class="canvases" id="canvases"></div>
      <div class="facts" id="facts"></div>
    </div>

    <nav class="side" aria-label="Specimens">
      <h3>Specimens</h3>
      <div id="list"></div>
    </nav>
  </div>

  <div class="note">
    <p><strong>Read these before trusting a volume.</strong> The pot is part of
    every carved hull, because it is opaque and the cameras see it, so space carving
    keeps it. The above-ground figure subtracts a fixed
    <code>POT_HEIGHT_M = 0.28 m</code>, which sits <em>below</em> the actual rim on
    the E001&ndash;E010 pots, so their above-ground volume still contains a slab of
    pot.</p>
    <p>E001&ndash;E010 in particular reconstruct as mostly pot with a small tuft:
    their canopies are thin and the rig sees them poorly. The mango specimens
    recover real canopy structure. That difference, not the method, is what drives
    most of the spread in the biomass results.</p>
  </div>
</div>

<script>
const RAW = __PAYLOAD__;

const byId = new Map();
for (const v of RAW.volumes) {
  if (!byId.has(v.plant_id)) byId.set(v.plant_id, {});
  byId.get(v.plant_id)[v.segmenter] = v;
}
const IDS = [...byId.keys()];

async function inflate(b64) {
  const bin = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  if (typeof DecompressionStream === "undefined") return null;
  const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const cloudCache = new Map();
async function cloud(entry) {
  if (!entry) return null;
  const key = entry.plant_id + "/" + entry.segmenter;
  if (cloudCache.has(key)) return cloudCache.get(key);
  const bytes = await inflate(entry.data);
  let pts = null;
  if (bytes) {
    const n = Math.floor(bytes.length / 3);
    pts = new Float32Array(n * 3);
    // Centre and scale on the specimen's own bounding box. A plant fills a small
    // corner of the 1.5 m working volume, so normalising by the grid instead
    // would leave every reconstruction as a speck in the middle of empty space.
    let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < 3; a++) {
        const v = bytes[i*3+a];
        if (v < lo[a]) lo[a] = v;
        if (v > hi[a]) hi[a] = v;
      }
    }
    const span = Math.max(1, hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]);
    const mid = [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, (lo[2]+hi[2])/2];
    for (let i = 0; i < n; i++) {
      pts[i*3]   = (bytes[i*3]   - mid[0]) / span;
      pts[i*3+1] = (bytes[i*3+1] - mid[1]) / span;
      pts[i*3+2] = (bytes[i*3+2] - mid[2]) / span + 0.5;
    }
  }
  cloudCache.set(key, pts);
  return pts;
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function draw(canvas, pts, yaw, pitch, zoom, dot) {
  const ctx = canvas.getContext("2d");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w*dpr)) { canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!pts) {
    ctx.fillStyle = css("--ink-3"); ctx.font = "13px 'IBM Plex Sans', sans-serif";
    ctx.textAlign = "center"; ctx.fillText("no data", w/2, h/2);
    return;
  }

  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const n = pts.length / 3;
  const proj = new Float32Array(n * 3);

  for (let i = 0; i < n; i++) {
    const x = pts[i*3], y = pts[i*3+1], z = pts[i*3+2] - 0.5;
    const rx =  x*cy + y*sy;
    const rz = -x*sy + y*cy;
    const ry =  z*cp - rz*sp;
    const rd =  z*sp + rz*cp;
    proj[i*3] = rx; proj[i*3+1] = ry; proj[i*3+2] = rd;
  }

  const order = Array.from({length: n}, (_, i) => i).sort((a, b) => proj[a*3+2] - proj[b*3+2]);
  const scale = Math.min(w, h) * zoom;
  const near = css("--near"), far = css("--far");
  const nc = hex(near), fc = hex(far);

  let dmin = Infinity, dmax = -Infinity;
  for (let i = 0; i < n; i++) { const d = proj[i*3+2]; if (d < dmin) dmin = d; if (d > dmax) dmax = d; }
  const span = Math.max(1e-6, dmax - dmin);

  for (const i of order) {
    const sx = w/2 + proj[i*3] * scale;
    const sv = h/2 - proj[i*3+1] * scale;
    const t = (proj[i*3+2] - dmin) / span;
    ctx.fillStyle = `rgb(${mix(fc[0],nc[0],t)},${mix(fc[1],nc[1],t)},${mix(fc[2],nc[2],t)})`;
    ctx.fillRect(sx, sv, dot, dot);
  }
}
function hex(h) {
  h = h.replace("#","");
  if (h.length === 3) h = h.split("").map(c => c+c).join("");
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}
const mix = (a, b, t) => Math.round(a + (b - a) * t);

let current = IDS[0];
let mode = "geometric";
let sort = "id";
let yaw = 0.6, pitch = 0.22, zoom = 0.92;
const panes = [];

function fmt(v, d = 2) { return v === null || v === undefined ? "n/a" : Number(v).toFixed(d); }

function buildStats() {
  const segs = RAW.segmenters;
  const rows = [
    ["Specimens", IDS.length],
    ["Segmenters", segs.length],
    ["Volumes", RAW.volumes.length],
  ];
  for (const s of segs) {
    const vs = RAW.volumes.filter(v => v.segmenter === s);
    const mean = vs.reduce((a, v) => a + v.volume_l, 0) / Math.max(1, vs.length);
    rows.push([s + " mean hull", fmt(mean, 1) + " L"]);
  }
  document.getElementById("stats").innerHTML = rows
    .map(([k, v]) => `<div class="stat"><span class="v">${v}</span><span class="k">${k}</span></div>`)
    .join("");
}

function sorted() {
  const key = v => {
    const e = byId.get(v);
    const g = e.geometric || e.sam3d;
    return sort === "mass" ? -g.target_kg : sort === "volume" ? -g.volume_l : 0;
  };
  const ids = [...IDS];
  if (sort === "id") return ids.sort();
  return ids.sort((a, b) => key(a) - key(b));
}

async function buildList() {
  const host = document.getElementById("list");
  host.innerHTML = "";
  let lastGroup = null;

  for (const id of sorted()) {
    const e = byId.get(id);
    const g = e.geometric || e.sam3d;
    if (sort === "id" && g.species !== lastGroup) {
      lastGroup = g.species;
      const h = document.createElement("div");
      h.className = "group"; h.textContent = g.species;
      host.appendChild(h);
    }
    const b = document.createElement("button");
    b.className = "item"; b.type = "button";
    b.setAttribute("aria-current", id === current ? "true" : "false");
    b.innerHTML = `<canvas width="88" height="88"></canvas>
      <span><span class="id">${id}</span><br><span class="meta">${fmt(g.target_kg)} kg</span></span>
      <span class="num">${fmt(g.volume_l, 1)} L</span>`;
    b.addEventListener("click", () => select(id));
    host.appendChild(b);
    cloud(g).then(pts => draw(b.querySelector("canvas"), pts, 0.6, 0.22, 0.82, 1));
  }
}

function paneList() {
  return mode === "pair" ? ["geometric", "sam3d"] : [mode];
}

async function buildStage() {
  const host = document.getElementById("canvases");
  host.className = "canvases" + (mode === "pair" ? " pair" : "");
  host.innerHTML = "";
  panes.length = 0;

  for (const seg of paneList()) {
    const wrap = document.createElement("div");
    wrap.className = "cwrap";
    const c = document.createElement("canvas");
    c.style.height = (mode === "pair" ? 380 : 460) + "px";
    wrap.innerHTML = `<span class="tag">${seg}</span>`;
    wrap.appendChild(c);
    host.appendChild(wrap);
    panes.push({ seg, canvas: c, wrap });
    attach(c);
  }
  await render();
}

function attach(canvas) {
  let dragging = false, lx = 0, ly = 0;
  canvas.addEventListener("pointerdown", e => {
    dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    yaw += (e.clientX - lx) * 0.01;
    pitch = Math.max(-1.4, Math.min(1.4, pitch + (e.clientY - ly) * 0.01));
    lx = e.clientX; ly = e.clientY;
    render();
  });
  const stop = e => { dragging = false; try { canvas.releasePointerCapture(e.pointerId); } catch (_) {} };
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    zoom = Math.max(0.25, Math.min(2.4, zoom * (e.deltaY > 0 ? 0.92 : 1.08)));
    render();
  }, { passive: false });
}

async function render() {
  const e = byId.get(current);
  for (const pane of panes) {
    const entry = e[pane.seg];
    const pts = await cloud(entry);
    const missing = pane.wrap.querySelector(".miss");
    if (!entry) {
      if (!missing) {
        const d = document.createElement("div");
        d.className = "miss"; d.textContent = "not reconstructed under " + pane.seg;
        pane.wrap.appendChild(d);
      }
    } else if (missing) { missing.remove(); }
    draw(pane.canvas, entry ? pts : null, yaw, pitch, zoom, mode === "pair" ? 2 : 3);
  }
}

function facts() {
  const e = byId.get(current);
  const g = e.geometric, s = e.sam3d;
  const ref = g || s;
  const delta = g && s ? (s.volume_l - g.volume_l) / g.volume_l * 100 : null;
  const rows = [
    ["Fresh mass", fmt(ref.target_kg) + " kg"],
    ["Hull volume", (g ? fmt(g.volume_l, 1) : "n/a") + (s ? " → " + fmt(s.volume_l, 1) : "") + " L"],
    ["Height", fmt(ref.height_m) + " m"],
    ["SAM3D Δ volume", delta === null ? "n/a" : (delta > 0 ? "+" : "") + fmt(delta, 1) + "%"],
  ];
  document.getElementById("facts").innerHTML = rows
    .map(([k, v]) => `<div class="fact"><span class="k eyebrow">${k}</span><br><span class="v">${v}</span></div>`)
    .join("");
}

async function select(id) {
  current = id;
  const e = byId.get(id);
  const ref = e.geometric || e.sam3d;
  document.getElementById("title").textContent = id;
  document.getElementById("species").textContent = ref.species;
  document.getElementById("sub").textContent = `${ref.n_voxels.toLocaleString()} voxels`;
  for (const b of document.querySelectorAll(".item")) {
    b.setAttribute("aria-current", b.querySelector(".id").textContent === id ? "true" : "false");
  }
  facts();
  await render();
}

document.getElementById("viewmode").addEventListener("click", async e => {
  const b = e.target.closest("button"); if (!b) return;
  mode = b.dataset.mode;
  for (const x of e.currentTarget.children) x.setAttribute("aria-pressed", String(x === b));
  await buildStage();
});
document.getElementById("sortmode").addEventListener("click", async e => {
  const b = e.target.closest("button"); if (!b) return;
  sort = b.dataset.sort;
  for (const x of e.currentTarget.children) x.setAttribute("aria-pressed", String(x === b));
  await buildList();
});
window.addEventListener("resize", () => render());

buildStats();
buildList().then(buildStage).then(() => select(current));
</script>
"""


def build_html(manifest: dict, out_path: Path) -> Path:
    """Write the standalone gallery page."""
    payload = json.dumps(manifest, separators=(",", ":"))
    html = TEMPLATE_HEAD + TEMPLATE_BODY.replace("__PAYLOAD__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def build_gallery_page(
    plant_ids: list[str],
    *,
    out_path: Path = WORK_DIR / "reports" / "gallery" / "reconstructions.html",
    **kwargs,
) -> Path:
    """Render every reconstruction and write the interactive page."""
    from .render import build_gallery

    manifest = build_gallery(plant_ids, write_ply=False, write_sheets=False, **kwargs)
    return build_html(manifest, out_path)


__all__ = ["build_gallery_page", "build_html"]
