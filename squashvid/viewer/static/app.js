// Wiring: clip loading, the video as master clock, controls, keyboard.
'use strict';
(() => {
  const video = document.getElementById('video');
  const $ = id => document.getElementById(id);
  let lastDrawn = -1;

  function readOpts() {
    V.opts = { court: $('o-court').checked, boxes: $('o-boxes').checked, skel: $('o-skel').checked, ball: $('o-ball').checked,
      skel3d: $('o-skel3d').checked, bounces: $('o-bounces').checked };
  }

  function renderFrame(f, force) {
    if (!V.D) return;
    if (f === lastDrawn && !force) return;
    lastDrawn = f; V.frame = f;
    Overlay.draw(f); View3D.update(f); TopView.update(f); FrontView.update(f); Timeline.draw(f); Panel.update(f);
    drawBalls(ballFrame(f));
  }

  // The ball in the 3D, top and front views moves between video frames: while playing, at display rate, by how far
  // the clock is between one frame and the next (smooth at slow speeds too). Paused, it sits on the frame. The video
  // overlay always stays on the shown frame.
  function ballFrame(f) {
    if (!V.D || video.paused || video.seeking) return f;
    const ts = V.D.t, t = video.currentTime, g = frameOfTime(ts, t);
    return g + 1 < ts.length ? g + Math.min(1, Math.max(0, (t - ts[g]) / (ts[g + 1] - ts[g]))) : g;
  }
  function drawBalls(ft) { View3D.ball(ft); TopView.ball(ft); FrontView.ball(ft); }
  (function tick() { if (V.D && !video.paused && !video.seeking) drawBalls(ballFrame(V.frame)); requestAnimationFrame(tick); })();
  const redraw = () => { readOpts(); renderFrame(V.frame, true); };

  // Media time -> frame, via the exported per-frame t array (binary search; VFR safe).
  function syncFromVideo(mediaTime) {
    if (!V.D) return;
    renderFrame(frameOfTime(V.D.t, mediaTime === undefined ? video.currentTime : mediaTime));
  }
  if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
    const cb = (now, md) => { syncFromVideo(md.mediaTime); video.requestVideoFrameCallback(cb); };
    video.requestVideoFrameCallback(cb);
    for (const ev of ['seeked', 'pause', 'loadeddata']) video.addEventListener(ev, () => syncFromVideo());
  } else {
    (function loop() { syncFromVideo(); requestAnimationFrame(loop); })();
  }

  V.seekTo = t => {
    if (!V.D) return;
    const f = frameOfTime(V.D.t, Math.max(0, Math.min(V.D.t[V.D.n - 1], t)));
    video.currentTime = seekTimeOfFrame(V.D.t, f);
    renderFrame(f);
  };
  const stepFrame = d => {
    if (!V.D) return;
    video.pause();
    const f = Math.max(0, Math.min(V.D.n - 1, frameOfTime(V.D.t, video.currentTime) + d));
    video.currentTime = seekTimeOfFrame(V.D.t, f); renderFrame(f);
  };
  const jumpRally = dir => {
    const D = V.D; if (!D || !D.rallies.length) return;
    const t = video.currentTime;
    if (dir > 0) { const r = D.rallies.find(r => r.start_t > t + 0.05); if (r) V.seekTo(r.start_t - 0.02); }
    else {
      const prev = D.rallies.filter(r => r.start_t < t - 1.0);
      V.seekTo(prev.length ? prev[prev.length - 1].start_t - 0.02 : 0);
    }
  };

  document.addEventListener('keydown', e => {
    if ((e.target.matches && e.target.matches('input, select, textarea')) || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === ' ') { e.preventDefault(); video.paused ? video.play() : video.pause(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); e.shiftKey ? V.seekTo(video.currentTime - 5) : stepFrame(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); e.shiftKey ? V.seekTo(video.currentTime + 5) : stepFrame(1); }
    else if (e.key === '[') jumpRally(-1);
    else if (e.key === ']') jumpRally(1);
  });
  document.addEventListener('click', e => { if (e.target.matches('button')) e.target.blur(); });
  video.addEventListener('click', () => { video.paused ? video.play() : video.pause(); });

  for (const id of ['o-court', 'o-boxes', 'o-skel', 'o-ball', 'o-skel3d', 'o-bounces', 'heat', 'wall-scope', 'wall-color']) $(id).addEventListener('change', redraw);
  $('rate').addEventListener('change', () => { video.playbackRate = parseFloat($('rate').value); });

  // ---- names ----
  async function loadSettings() {
    try {
      const s = await (await fetch(URLS.settings)).json();
      V.names = { ...V.names, ...s.names };
      if (s.looks) { V.looks = s.looks; $('look-A').textContent = `(${s.looks.A})`; $('look-B').textContent = `(${s.looks.B})`; $('names-btn').title = `Name the players: A wears the ${s.looks.A}, B the ${s.looks.B}`; }
    } catch (e) { console.warn('settings', e); }
    // A blank box keeps the default label; the server stores 'Player A' / 'Player B' for blanks.
    $('name-A').value = V.names.A === 'Player A' ? '' : V.names.A; $('name-B').value = V.names.B === 'Player B' ? '' : V.names.B;
  }
  $('names-btn').hidden = !!window.SV_STATIC;  // a static build has no server to save names to
  $('names-btn').addEventListener('click', () => { $('names-form').hidden = !$('names-form').hidden; });
  $('names-save').addEventListener('click', async () => {
    const r = await fetch('/api/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ names: { A: $('name-A').value, B: $('name-B').value } }) });
    if (r.ok) { V.names = { ...V.names, ...(await r.json()).names }; $('names-form').hidden = true; TopView.rebuild(); Panel.render(); renderFrame(V.frame, true); }
  });

  // ---- clip loading ----
  async function openClip(stem) {
    $('status').textContent = `loading ${stem}…`;
    video.pause();
    const D = await loadClip(stem);
    V.D = D; annotateEvents(D);
    video.src = URLS.video(stem);
    View3D.buildCourt(D); TopView.rebuild(); FrontView.rebuild(); Timeline.invalidate();
    Panel.render();
    const s = D.meta.sources;
    const miss = ['camera', 'players', 'ball', 'ball_3d', 'events', 'stats', 'insights'].filter(k => !s[k]);
    $('status').textContent = `${stem}: ${D.n} frames` + (D.meta.proxy ? '' : ' (no proxy video: run the proxy command)') + (miss.length ? ` · missing: ${miss.join(', ')}` : '') + (s.ball && String(s.ball).startsWith('raw') ? ' · ball = raw detector' : '');
    location.hash = stem;
    lastDrawn = -1; readOpts(); renderFrame(0, true);
    video.addEventListener('loadedmetadata', () => { Overlay.layout(); renderFrame(V.frame, true); seekableOrBlob(stem); }, { once: true });
  }

  // A static host that ignores byte ranges leaves the video unseekable: then play it from memory (a static build
  // keeps each video under 15 MB).
  async function seekableOrBlob(stem) {
    const s = video.seekable;
    if (!window.SV_STATIC || (s.length && s.end(s.length - 1) >= video.duration - 1)) return;
    console.warn('video not seekable from the host; loading it whole');
    const url = URL.createObjectURL(await (await fetch(URLS.video(stem))).blob());
    if (V.D?.stem !== stem) return URL.revokeObjectURL(url);
    const t = video.currentTime;
    video.src = url;
    video.addEventListener('loadedmetadata', () => { video.currentTime = t; }, { once: true });
  }

  (async () => {
    readOpts();
    await loadSettings();
    const { clips } = await (await fetch(URLS.clips)).json();
    if (!clips.length) { $('status').textContent = 'no exported clips: run `python -I -m squashvid.viewer export --all`'; return; }
    $('clip').innerHTML = clips.map(c => `<option>${c.stem}</option>`).join('');
    V.stems = clips.map(c => c.stem);
    const want = location.hash.slice(1);
    if (clips.some(c => c.stem === want)) $('clip').value = want;
    $('clip').addEventListener('change', () => openClip($('clip').value));
    await openClip($('clip').value);
    const q = new URLSearchParams(location.search);
    if (q.has('t')) V.seekTo(parseFloat(q.get('t')));
    if (q.has('tab')) [...document.querySelectorAll('#tabs button')].find(x => x.dataset.tab === q.get('tab'))?.click();
  })();
  V.openClip = stem => { $('clip').value = stem; return openClip(stem); };
  window.V = V; window.__viewer = { renderFrame, stepFrame };
})();
