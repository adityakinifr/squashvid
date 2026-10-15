// Scrubber: rally bands, event ticks, ball-coverage strip, playhead. Click or drag to seek.
'use strict';
const Timeline = (() => {
  const cv = document.getElementById('timeline');
  const ctx = cv.getContext('2d');
  const KIND_COLOR = { front_wall: '#6fc3ff', side_wall_left: '#b6a0ff', side_wall_right: '#b6a0ff', floor: '#ffe27a', back_wall: '#d0a0a0', racket: '#ff7f7f',
    audio_racket: '#ff7f7f55', audio_front_wall: '#6fc3ff55', audio_floor: '#ffe27a55', audio_side_wall: '#b6a0ff55' };
  const STATE_COLOR = ['', '#ffd54a', '#ff9f43', '#7a7a7a'];
  let bg = null, key = '', W = 0, H = 0, dpr = 1, dragging = false;

  function size() {
    dpr = window.devicePixelRatio || 1;
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); key = '';
  }
  const dur = () => V.D.t[V.D.n - 1];
  const xOf = t => (t / dur()) * W;
  const tOf = x => Math.max(0, Math.min(dur(), (x / W) * dur()));

  function paintBackground() {
    const D = V.D;
    bg = document.createElement('canvas'); bg.width = cv.width; bg.height = cv.height;
    const c = bg.getContext('2d'); c.scale(dpr, dpr);
    const style = getComputedStyle(document.body);
    c.fillStyle = style.getPropertyValue('--bg'); c.fillRect(0, 0, W, H);
    const bandH = H * 0.5, stripY = H * 0.62, stripH = H * 0.16;
    for (const r of D.rallies) {
      c.fillStyle = 'rgba(80,160,255,0.35)'; c.fillRect(xOf(r.start_t), 0, Math.max(1, xOf(r.end_t) - xOf(r.start_t)), bandH);
    }
    c.fillStyle = style.getPropertyValue('--muted'); c.font = '10px sans-serif';
    D.rallies.forEach(r => { if (xOf(r.end_t) - xOf(r.start_t) > 14) c.fillText(String(r.label), xOf(r.start_t) + 2, 10); });
    for (const e of D.events) {
      c.fillStyle = e.kind === 'hit' ? (V.COLORS[e.player] || '#ccc') : (KIND_COLOR[e.kind] || '#ccc');
      const x = xOf(e.t);
      if (e.source === 'audio') c.fillRect(x, bandH - 8, 1, 8); else if (e.source === 'hits') c.fillRect(x - 0.5, bandH - 14, 1.5, 14); else c.fillRect(x - 0.5, 12, 1.5, bandH - 12);
    }
    // ball coverage: one column per pixel, the best state seen in that span
    if (D.ball_state) {
      const cols = Math.ceil(W);
      for (let x = 0; x < cols; x++) {
        const f0 = frameOfTime(D.t, tOf(x)), f1 = Math.max(f0, frameOfTime(D.t, tOf(x + 1)));
        let best = 0, any = false;
        for (let f = f0; f <= f1; f++) { const s = D.ball_state[f]; if (s) { any = true; if (best === 0 || s < best) best = s; } }
        if (any) { c.fillStyle = STATE_COLOR[best]; c.fillRect(x, stripY, 1, stripH); }
      }
    }
    c.fillStyle = style.getPropertyValue('--muted'); c.fillText('ball', 2, H - 6);
    if (D.A.status) {
      for (let x = 0; x < W; x++) {
        const f0 = frameOfTime(D.t, tOf(x));
        c.fillStyle = D.A.status[f0] === 2 ? V.COLORS.A : D.A.status[f0] === 1 ? '#6a5a4a' : 'transparent'; c.fillRect(x, stripY + stripH + 2, 1, 3);
        c.fillStyle = D.B.status[f0] === 2 ? V.COLORS.B : D.B.status[f0] === 1 ? '#3a5a60' : 'transparent'; c.fillRect(x, stripY + stripH + 6, 1, 3);
      }
    }
    key = `${D.stem}|${W}|${H}|${D.events.length}`;
  }

  function draw(f) {
    if (!V.D) return;
    if (key !== `${V.D.stem}|${W}|${H}|${V.D.events.length}`) paintBackground();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(bg, 0, 0);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const x = xOf(V.D.t[f]);
    ctx.fillStyle = '#ff3b3b'; ctx.fillRect(x - 1, 0, 2, H);
    document.getElementById('tl-time').textContent = fmtTime(V.D.t[f]);
    document.getElementById('tl-frame').textContent = `frame ${f} / ${V.D.n - 1}` + rallyText(rallyAt(V.D, V.D.t[f]));
  }

  const seekX = ev => V.seekTo(tOf(ev.clientX - cv.getBoundingClientRect().left));
  cv.addEventListener('mousedown', ev => { dragging = true; seekX(ev); });
  window.addEventListener('mousemove', ev => { if (dragging) seekX(ev); });
  window.addEventListener('mouseup', () => { dragging = false; });
  new ResizeObserver(() => { size(); if (V.D) draw(V.frame); }).observe(cv);
  return { draw, invalidate() { key = ''; } };
})();

// The current rally, who served it and from which side, and who won it (inferred from who served next).
function rallyText(r) {
  if (!r) return '';
  const side = { L: 'left', R: 'right' }[r.server_side];
  let s = ` · rally ${r.label}`;
  if (r.server) s += `: ${V.names[r.server]} served${side ? ' from the ' + side : ''}`;
  if (r.winner) s += `, won by ${V.names[r.winner]} (inferred from the next serve)`;
  return s;
}
