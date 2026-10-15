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

  // A ring on the ball and a comet tail along its path over the last TRAIL_S, widening and brightening toward the
  // ball; stretches filled between tracked points are fainter and the ring dashed. Drawn on the shown frame only, so
  // it stays on the picture.
  const TRAIL_S = 0.5;
  function ring(u, v, k, color, dashed, z = 1) {
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, 2.5 * k * z);
    if (dashed) ctx.setLineDash([4 * k * z, 3 * k * z]);
    ctx.beginPath(); ctx.arc(u * k, v * k, 9 * k * z, 0, 6.3); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(u * k, v * k, 1.8 * k * z, 0, 6.3); ctx.fill();
  }

  function drawBall(f, k) {
    const D = V.D;
    if (D.ball_path_uv) {
      const z = Math.max(k, 0.6 * (window.devicePixelRatio || 1)) / k;  // sizes hold up when the video is drawn small
      const tr = ballTrail(D, f, TRAIL_S, false);
      ctx.lineCap = 'round'; ctx.lineJoin = 'round';
      for (let i = 1; i < tr.length; i++) {
        const a = i / (tr.length - 1), filled = tr[i].k === 2 || tr[i - 1].k === 2;
        ctx.strokeStyle = `rgba(255,213,74,${((0.08 + 0.72 * a * a) * (filled ? 0.55 : 1)).toFixed(3)})`;
        ctx.lineWidth = (1.2 + 5 * a) * k * z;
        ctx.beginPath(); ctx.moveTo(tr[i - 1].p[0] * k, tr[i - 1].p[1] * k); ctx.lineTo(tr[i].p[0] * k, tr[i].p[1] * k); ctx.stroke();
      }
      ctx.lineCap = 'butt';
      if (!tr.length) return;
      const [u, v] = tr[tr.length - 1].p, st = D.ball_state ? D.ball_state[f] : 0;
      if (tr[tr.length - 1].k === 2) ring(u, v, k, 'rgba(255,240,190,0.85)', true, z);
      else ring(u, v, k, st === 2 ? '#ff9f43' : st === 3 ? '#ffffff' : '#ffd54a', st === 3, z);
      return;
    }
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
    ring(u, v, k, st === 1 ? '#ffd54a' : st === 2 ? '#ff9f43' : '#ffffff', st !== 1 && st !== 2);
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
    let st = V.D.ball_state ? ['no ball', 'ball: detected', 'ball: linked', 'ball: interpolated'][V.D.ball_state[f]] : 'no ball data';
    if (V.D.ball_path_kind && V.D.ball_path_kind[f] === 2) st = 'ball: filled between tracked points';
    hud.textContent = `frame ${f}  ${st}`;
  }

  new ResizeObserver(() => { layout(); if (V.D) draw(V.frame); }).observe(wrap);
  video.addEventListener('loadedmetadata', () => { layout(); });
  return { draw, layout };
})();
