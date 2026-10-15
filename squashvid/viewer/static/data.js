// Data loading, time <-> frame mapping and camera maths. Everything hangs off the global V.
'use strict';
const V = {
  D: null,            // loaded clip
  names: { A: 'Player A', B: 'Player B' },
  frame: 0,
  listeners: [],
  COLORS: { A: '#ff9f43', B: '#3ec7e6' },
  opts: {},
};

// Index of the frame showing at media time t: the last frame whose PTS <= t (same rule as export.frame_of_time).
function frameOfTime(ts, t) {
  let lo = 0, hi = ts.length - 1;
  const x = t + 1e-6;
  if (x < ts[0]) return 0;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ts[mid] <= x) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// Time that lands safely inside frame f (halfway to the next frame), for seeking.
function seekTimeOfFrame(ts, f) {
  f = Math.max(0, Math.min(ts.length - 1, f));
  return f + 1 < ts.length ? (ts[f] + ts[f + 1]) / 2 : ts[f] + 0.004;
}

const DTYPES = { int16: Int16Array, uint8: Uint8Array, float32: Float32Array, float64: Float64Array, int32: Int32Array, uint16: Uint16Array };

async function loadArray(stem, name, info) {
  const r = await fetch(`/api/${stem}/array/${name}`);
  if (!r.ok) return null;
  const raw = new (DTYPES[info.dtype])(await r.arrayBuffer());
  if (info.dtype === 'float32' || info.dtype === 'float64') return raw;
  if (info.dtype === 'uint8' && info.scale === 1) return raw;
  // Quantised: decode to Float32 with NaN for the sentinel; uint8 score/conf arrays decode to 0..1.
  const out = new Float32Array(raw.length);
  const miss = info.missing, k = 1 / info.scale;
  if (info.dtype === 'uint8') { for (let i = 0; i < raw.length; i++) out[i] = raw[i] * k; return out; }
  for (let i = 0; i < raw.length; i++) out[i] = raw[i] === miss ? NaN : raw[i] * k;
  return out;
}

async function loadClip(stem) {
  const meta = await (await fetch(`/api/${stem}/meta`)).json();
  const D = { stem, meta, n: meta.n_frames, A: {}, B: {} };
  const jobs = Object.entries(meta.arrays).map(async ([name, info]) => {
    const a = await loadArray(stem, name, info);
    if (a) D[name] = a;
  });
  await Promise.all(jobs);
  for (const p of ['A', 'B']) {
    const P = D[p];
    P.pos = D[`${p}_pos`]; P.box = D[`${p}_box`]; P.kp = D[`${p}_kp`]; P.ks = D[`${p}_ks`]; P.status = D[`${p}_status`];
    P.body = D[`${p}_body`]; P.bconf = D[`${p}_bconf`];  // 3D joints (H36M-17, court metres) and confidence; absent without bodies.parquet
  }
  D.t = D.t || new Float64Array(0);
  D.rallies = (meta.rallies || []).map((r, i) => ({ ...r, id: i, label: r.rally ?? i + 1 }));
  D.events = meta.events || [];
  D.hits = (meta.hits || []).slice().sort((a, b) => a.t - b.t);
  D.bounces = D.events.filter(e => e.source === 'ball');
  D.court = meta.court;
  D.dist = meta.camera_dist || [0, 0, 0, 0, 0];
  D.camCache = new Map();
  return D;
}

// ---- camera: OpenCV pinhole + radial/tangential distortion, pose = Rodrigues rvec, tvec ----
function rodrigues(r) {
  const th = Math.hypot(r[0], r[1], r[2]);
  if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const x = r[0] / th, y = r[1] / th, z = r[2] / th, c = Math.cos(th), s = Math.sin(th), C = 1 - c;
  return [c + x * x * C, x * y * C - z * s, x * z * C + y * s,
          y * x * C + z * s, c + y * y * C, y * z * C - x * s,
          z * x * C - y * s, z * y * C + x * s, c + z * z * C];
}

function cameraAt(D, f) {
  if (!D.cam) return null;
  const o = Math.max(0, Math.min(D.n - 1, f)) * 9, a = D.cam;
  if (!Number.isFinite(a[o])) return null;
  const R = rodrigues([a[o + 3], a[o + 4], a[o + 5]]), t = [a[o + 6], a[o + 7], a[o + 8]];
  // centre = -R^T t
  const c = [-(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]), -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]), -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2])];
  return { f: a[o], cx: a[o + 1], cy: a[o + 2], R, t, c, dist: D.dist };
}

// World point -> [u, v, depth] in original video pixels.
function project(cam, X) {
  const R = cam.R, t = cam.t;
  const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + t[0];
  const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + t[1];
  const zc = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
  const x = xc / zc, y = yc / zc, [k1, k2, p1, p2, k3] = cam.dist;
  const r2 = x * x + y * y, rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x);
  const yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y;
  return [cam.f * xd + cam.cx, cam.f * yd + cam.cy, zc];
}

// Pixel -> unit world ray direction (undistorted by fixed-point iteration).
function rayOf(cam, u, v) {
  const xd = (u - cam.cx) / cam.f, yd = (v - cam.cy) / cam.f, [k1, k2, p1, p2, k3] = cam.dist;
  let x = xd, y = yd;
  for (let i = 0; i < 6; i++) {
    const r2 = x * x + y * y, rad = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
    x = (xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad;
    y = (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad;
  }
  const d = [x, y, 1], R = cam.R;
  const w = [R[0] * d[0] + R[3] * d[1] + R[6] * d[2], R[1] * d[0] + R[4] * d[1] + R[7] * d[2], R[2] * d[0] + R[5] * d[1] + R[8] * d[2]];
  const n = Math.hypot(w[0], w[1], w[2]);
  return [w[0] / n, w[1] / n, w[2] / n];
}

// COCO keypoint u,v -> world, assuming the body lies in the vertical plane through the player's floor point that
// faces the camera. Returns null if the ray misses (behind camera).
function liftKeypoint(cam, floorXY, u, v) {
  const nx = cam.c[0] - floorXY[0], ny = cam.c[1] - floorXY[1], nn = Math.hypot(nx, ny) || 1;
  const n = [nx / nn, ny / nn, 0], d = rayOf(cam, u, v);
  const den = n[0] * d[0] + n[1] * d[1];
  if (Math.abs(den) < 1e-6) return null;
  const tt = (n[0] * (floorXY[0] - cam.c[0]) + n[1] * (floorXY[1] - cam.c[1])) / den;
  if (!(tt > 0)) return null;
  const p = [cam.c[0] + d[0] * tt, cam.c[1] + d[1] * tt, cam.c[2] + d[2] * tt];
  p[2] = Math.max(0, p[2]);
  return p;
}

function fmtTime(t) {
  const cs = Math.round(Math.max(t, 0) * 100);  // round once, so 119.999 s reads 2:00.00, not 1:60.00
  const m = Math.floor(cs / 6000), s = (cs - m * 6000) / 100;
  return `${m}:${s.toFixed(2).padStart(5, '0')}`;
}

function rallyAt(D, t) {
  for (const r of D.rallies) if (t >= r.start_t && t <= r.end_t) return r;
  return null;
}

function onFrame(fn) { V.listeners.push(fn); }
