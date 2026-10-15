// Canvas overlay on the video: court lines through the per-frame camera, players, ball + trail.
'use strict';
const COCO_EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const SURFACE_COLOR = { floor: '#ffe27a', front: '#8fd0ff', left: '#b6a0ff', right: '#b6a0ff', back: '#b6a0ff' };

const Overlay = (() => {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  const wrap = document.getElementById('video-wrap');
  const ctx = canvas.getContext('2d');
  let rect = { x: 0, y: 0, w: 1, h: 1 };

  function layout() {
    const vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
    const W = wrap.clientWidth, H = wrap.clientHeight, s = Math.min(W / vw, H / vh);
    rect = { x: (W - vw * s) / 2, y: (H - vh * s) / 2, w: vw * s, h: vh * s };
    const dpr = window.devicePixelRatio || 1;
    canvas.style.left = rect.x + 'px'; canvas.style.top = rect.y + 'px';
    canvas.style.width = rect.w + 'px'; canvas.style.height = rect.h + 'px';
    canvas.width = Math.round(rect.w * dpr); canvas.height = Math.round(rect.h * dpr);
    V.videoScale = canvas.width / vw;
  }

  function drawCourt(cam, k) {
    ctx.lineWidth = Math.max(2, 2.5 * k);
    for (const ln of V.D.court.lines) {
      ctx.strokeStyle = ln.kind === 'edge' ? 'rgba(255,255,255,0.35)' : SURFACE_COLOR[ln.surface] || '#fff';
      ctx.beginPath();
      let pen = false;
      for (const p of ln.pts) {
        const q = project(cam, p);
        if (q[2] < 0.3) { pen = false; continue; }
        if (pen) ctx.lineTo(q[0] * k, q[1] * k); else { ctx.moveTo(q[0] * k, q[1] * k); pen = true; }
      }
      ctx.stroke();
    }
  }

  function drawPlayer(p, f, k) {
    const D = V.D, P = D[p];
    if (!P.status || !P.status[f]) return;
    const col = V.COLORS[p], interp = P.status[f] === 1;
    const b = f * 4;
    ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = Math.max(1.5, 2 * k);
    if (V.opts.boxes && P.box && Number.isFinite(P.box[b])) {
      ctx.setLineDash(interp ? [6 * k, 4 * k] : []);
      ctx.strokeRect(P.box[b] * k, P.box[b + 1] * k, (P.box[b + 2] - P.box[b]) * k, (P.box[b + 3] - P.box[b + 1]) * k);
      ctx.setLineDash([]);
      ctx.font = `${Math.round(16 * Math.max(1, k * 1.2))}px sans-serif`;
      ctx.fillText(V.names[p] + (interp ? ' (interp)' : ''), P.box[b] * k + 2, P.box[b + 1] * k - 4);
    }
    if (V.opts.skel && P.kp && P.ks) {
      const o = f * 34, so = f * 17;
      const ok = i => Number.isFinite(P.kp[o + 2 * i]) && P.ks[so + i] > 0.3;
      ctx.lineWidth = Math.max(1.5, 2 * k);
      for (const [i, j] of COCO_EDGES) {
        if (!ok(i) || !ok(j)) continue;
        ctx.beginPath(); ctx.moveTo(P.kp[o + 2 * i] * k, P.kp[o + 2 * i + 1] * k); ctx.lineTo(P.kp[o + 2 * j] * k, P.kp[o + 2 * j + 1] * k); ctx.stroke();
      }
      for (let i = 0; i < 17; i++) if (ok(i)) { ctx.beginPath(); ctx.arc(P.kp[o + 2 * i] * k, P.kp[o + 2 * i + 1] * k, 2.5 * k, 0, 6.3); ctx.fill(); }
    }
  }

  function drawBall(f, k) {
    const D = V.D;
    if (!D.ball_uv) return;
    const TRAIL = 12;
    // trail: connect consecutive known points, fading with age
    let prev = null;
    for (let g = Math.max(0, f - TRAIL); g <= f; g++) {
      const u = D.ball_uv[2 * g], v = D.ball_uv[2 * g + 1];
      if (!Number.isFinite(u)) { prev = null; continue; }
      if (prev) {
        const age = (g - (f - TRAIL)) / TRAIL;
        ctx.strokeStyle = `rgba(255,213,74,${(0.15 + 0.7 * age).toFixed(2)})`;
        ctx.lineWidth = Math.max(1, (1 + 2 * age) * k);
        ctx.beginPath(); ctx.moveTo(prev[0] * k, prev[1] * k); ctx.lineTo(u * k, v * k); ctx.stroke();
      }
      prev = [u, v];
    }
    const u = D.ball_uv[2 * f], v = D.ball_uv[2 * f + 1], st = D.ball_state[f];
    if (!Number.isFinite(u)) return;
    const r = 9 * k;
    ctx.lineWidth = Math.max(2, 2.5 * k);
    if (st === 1) { ctx.strokeStyle = '#ffd54a'; ctx.beginPath(); ctx.arc(u * k, v * k, r, 0, 6.3); ctx.stroke(); }
    else if (st === 2) { ctx.strokeStyle = '#ff9f43'; ctx.beginPath(); ctx.arc(u * k, v * k, r, 0, 6.3); ctx.stroke(); }
    else { ctx.strokeStyle = '#ffffff'; ctx.setLineDash([4 * k, 3 * k]); ctx.beginPath(); ctx.arc(u * k, v * k, r, 0, 6.3); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(u * k, v * k, 1.8 * k, 0, 6.3); ctx.fill();
  }

  function draw(f) {
    if (!V.D) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const k = V.videoScale || 1;
    const cam = cameraAt(V.D, f);
    if (V.opts.court && cam) drawCourt(cam, k);
    for (const p of ['A', 'B']) drawPlayer(p, f, k);
    if (V.opts.ball) drawBall(f, k);
    const hud = document.getElementById('video-hud');
    const st = V.D.ball_state ? ['no ball', 'ball: detected', 'ball: linked', 'ball: interpolated'][V.D.ball_state[f]] : 'no ball data';
    hud.textContent = `frame ${f}  ${st}`;
  }

  new ResizeObserver(() => { layout(); if (V.D) draw(V.frame); }).observe(wrap);
  video.addEventListener('loadedmetadata', () => { layout(); });
  return { draw, layout };
})();
