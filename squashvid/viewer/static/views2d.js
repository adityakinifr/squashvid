// Top (floor) view and front-wall view as SVG. Events are clickable and seek the video.
'use strict';
const RALLY_PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075'];

function eventColor(e, mode) {
  if (mode === 'rally') { const r = e._rally; return r == null ? '#999' : RALLY_PALETTE[((r % RALLY_PALETTE.length) + RALLY_PALETTE.length) % RALLY_PALETTE.length]; }
  return V.COLORS[e.hitter || e.player] || '#bbbbbb';  // impacts take the hitter of the preceding shot
}

function annotateEvents(D) {
  for (const e of D.events) {
    const r = D.rallies.find(r => e.t >= r.start_t - 0.3 && e.t <= r.end_t + 0.3);
    e._rally = r ? r.id : null;
  }
}

// Events to show for a scope: the rally containing t (or a +-4 s window outside rallies), or the whole clip.
function scopedEvents(D, kinds, t, scope) {
  const sel = D.bounces.filter(e => kinds.includes(e.kind) && Number.isFinite(e.x));
  if (scope === 'clip') return sel;
  const r = rallyAt(D, t);
  if (r) return sel.filter(e => e.t >= r.start_t - 0.3 && e.t <= r.end_t + 0.3);
  return sel.filter(e => Math.abs(e.t - t) < 4);
}

// The ball and its path over the last BALL_TRAIL_S in a 2D view: pt maps a court point [x, y, z] to view coordinates.
// Drawn at a fractional frame, so it moves between video frames.
const BALL_TRAIL_S = 0.8;
function makeBall(parent, r) {
  const trail = parent.append('polyline').attr('fill', 'none').attr('stroke', '#ffd54a').attr('stroke-width', r * 0.6).attr('stroke-opacity', 0.55)
    .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round');
  const dot = parent.append('circle').attr('r', r).attr('fill', '#ffd54a').attr('stroke', '#3a2f00').attr('stroke-width', r * 0.3);
  return (D, ft, pt) => {
    const tr = D.ball_path_xyz ? ballTrail(D, ft, BALL_TRAIL_S, true) : [];
    const b = tr[tr.length - 1];
    trail.attr('points', tr.map(q => pt(q.p).join(',')).join(' '));
    dot.style('display', b ? null : 'none');
    if (b) { const [x, y] = pt(b.p); dot.attr('cx', x).attr('cy', y).attr('fill-opacity', b.k === 2 ? 0.6 : 1); }
  };
}

const TopView = (() => {
  const svg = d3.select('#top');
  let g = null, heatG, evG, plG, trails, dots, drawBall = null, key = '', W = 6.4, L = 9.75;
  const X = x => x, Y = y => L - y;

  function build(D) {
    const s = D.court.spec; W = s.width; L = s.length;
    svg.selectAll('*').remove();
    svg.attr('viewBox', `${-0.9} ${-0.9} ${W + 1.8} ${L + 1.8}`);
    g = svg.append('g');
    g.append('rect').attr('x', 0).attr('y', 0).attr('width', W).attr('height', L).attr('fill', '#8a6a3a').attr('fill-opacity', 0.25).attr('stroke', '#fff').attr('stroke-width', 0.04);
    heatG = g.append('g');
    for (const ln of D.court.lines) {
      if (ln.surface !== 'floor' || ln.kind === 'edge') continue;
      const a = ln.pts[0], b = ln.pts[ln.pts.length - 1];
      g.append('line').attr('x1', X(a[0])).attr('y1', Y(a[1])).attr('x2', X(b[0])).attr('y2', Y(b[1])).attr('stroke', '#ffe27a').attr('stroke-width', 0.05).attr('stroke-opacity', 0.9);
    }
    const T = [W / 2, s.short_line_near_edge + 0.025];
    g.append('circle').attr('cx', T[0]).attr('cy', Y(T[1])).attr('r', 0.1).attr('fill', '#ffe27a');
    g.append('text').attr('x', T[0] + 0.15).attr('y', Y(T[1]) - 0.1).attr('font-size', 0.3).text('T');
    g.append('text').attr('x', W / 2).attr('y', -0.35).attr('text-anchor', 'middle').attr('font-size', 0.32).text('front wall');
    g.append('text').attr('x', W / 2).attr('y', L + 0.6).attr('text-anchor', 'middle').attr('font-size', 0.32).text('back wall (camera side)');
    evG = g.append('g'); plG = g.append('g');
    trails = {}; dots = {};
    for (const p of ['A', 'B']) {
      trails[p] = plG.append('polyline').attr('fill', 'none').attr('stroke', V.COLORS[p]).attr('stroke-width', 0.07).attr('stroke-opacity', 0.6).attr('stroke-linejoin', 'round');
      dots[p] = plG.append('g');
      dots[p].append('circle').attr('r', 0.28).attr('fill', V.COLORS[p]).attr('fill-opacity', 0.85).attr('stroke', '#fff').attr('stroke-width', 0.04);
      dots[p].append('text').attr('font-size', 0.3).attr('text-anchor', 'middle').attr('y', 0.42).style('fill', V.COLORS[p]);
    }
    drawBall = makeBall(g.append('g'), 0.1);
    key = '';
  }

  function heat(D, spec, t) {
    heatG.selectAll('*').remove();
    if (spec === 'off') return;
    const [scope, p] = spec.split(':');
    const P = D[p]; if (!P.pos) return;
    let f0 = 0, f1 = D.n - 1;
    if (scope === 'rally') { const r = rallyAt(D, t); if (!r) return; f0 = frameOfTime(D.t, r.start_t); f1 = frameOfTime(D.t, r.end_t); }
    const cell = 0.25, nx = Math.ceil(W / cell), ny = Math.ceil(L / cell), cnt = new Float32Array(nx * ny);
    for (let f = f0; f <= f1; f++) {
      const x = P.pos[2 * f], y = P.pos[2 * f + 1];
      if (!Number.isFinite(x) || x < 0 || y < 0 || x >= W || y >= L) continue;
      cnt[Math.floor(y / cell) * nx + Math.floor(x / cell)]++;
    }
    const mx = Math.max(1, ...cnt);
    const col = d3.interpolateYlOrRd;
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const c = cnt[j * nx + i]; if (!c) continue;
      heatG.append('rect').attr('x', i * cell).attr('y', Y((j + 1) * cell)).attr('width', cell).attr('height', cell).attr('fill', col(Math.sqrt(c / mx))).attr('fill-opacity', 0.75);
    }
  }

  function update(f) {
    const D = V.D; if (!D) return;
    if (!g || g.datum() !== D) { build(D); g.datum(D); }
    const t = D.t[f];
    const h = document.getElementById('heat').value;
    const r = rallyAt(D, t);
    const k = [h.startsWith('rally') ? (r ? r.id : 'none') : h, V.opts.bounces, r ? r.id : 'x' + Math.floor(t / 4)].join('|') + D.stem + V.names.A + V.names.B;
    if (k !== key) {
      key = k;
      heat(D, h, t);
      evG.selectAll('*').remove();
      if (V.opts.bounces) {
        for (const e of scopedEvents(D, ['floor'], t, 'rally')) {
          const rx = Math.max(0.06, e.sigma_a || 0.08), ry = Math.max(0.06, e.sigma_b || 0.08);
          evG.append('ellipse').attr('class', 'ev').datum(e).attr('cx', e.x).attr('cy', Y(e.y)).attr('rx', rx).attr('ry', ry)
            .attr('fill', eventColor(e, 'player')).attr('fill-opacity', 0.35).attr('stroke', eventColor(e, 'player')).attr('stroke-width', 0.05)
            .on('click', (_, d) => V.seekTo(d.t - 0.02)).append('title').text(`${e.kind} @ ${fmtTime(e.t)}`);
        }
      }
    }
    evG.selectAll('ellipse').attr('stroke-width', d => Math.abs(d.t - t) < 0.5 ? 0.14 : 0.05).attr('fill-opacity', d => d.t > t ? 0.12 : 0.4);
    for (const p of ['A', 'B']) {
      const P = D[p], x = P.pos ? P.pos[2 * f] : NaN, y = P.pos ? P.pos[2 * f + 1] : NaN;
      const ok = Number.isFinite(x);
      dots[p].style('display', ok ? null : 'none');
      if (ok) { dots[p].attr('transform', `translate(${X(x)},${Y(y)})`); dots[p].select('text').text(V.names[p]); }
      const pts = [];
      for (let q = Math.max(0, f - 45); q <= f; q++) { const px = P.pos ? P.pos[2 * q] : NaN; if (Number.isFinite(px)) pts.push(`${X(px)},${Y(P.pos[2 * q + 1])}`); }
      trails[p].attr('points', pts.join(' '));
    }
  }
  // Seen from above, the ball is drawn on the floor under it.
  const ball = ft => { if (g && V.D && g.datum() === V.D) drawBall(V.D, ft, q => [X(q[0]), Y(q[1])]); };
  return { update, ball, rebuild() { g = null; key = ''; } };
})();

const FrontView = (() => {
  const svg = d3.select('#front');
  let g = null, evG, drawBall = null, W = 6.4, H = 5, key = '';
  const Y = z => H - z;

  function build(D) {
    const s = D.court.spec, d = D.court.derived; W = s.width; H = d.front_out_z + 0.45;
    svg.selectAll('*').remove();
    svg.attr('viewBox', `${-0.5} ${-0.3} ${W + 1.6} ${H + 0.7}`);
    g = svg.append('g');
    g.append('rect').attr('x', 0).attr('y', Y(d.front_out_z)).attr('width', W).attr('height', d.front_out_z).attr('fill', '#6fa8dc').attr('fill-opacity', 0.14).attr('stroke', '#fff').attr('stroke-width', 0.03);
    g.append('rect').attr('x', 0).attr('y', Y(s.tin_top)).attr('width', W).attr('height', s.tin_top).attr('fill', '#d9534f').attr('fill-opacity', 0.35);
    const lab = (z, txt) => {
      g.append('line').attr('x1', 0).attr('x2', W).attr('y1', Y(z)).attr('y2', Y(z)).attr('stroke', '#ffe27a').attr('stroke-width', 0.04);
      g.append('text').attr('x', W + 0.06).attr('y', Y(z) + 0.1).attr('font-size', 0.24).text(txt);
    };
    lab(d.front_out_z, 'out'); lab(d.service_z, 'service'); lab(d.tin_z, 'tin');
    g.append('line').attr('x1', W / 2).attr('x2', W / 2).attr('y1', Y(0)).attr('y2', Y(s.service_line)).attr('stroke', '#ffe27a').attr('stroke-opacity', 0.25).attr('stroke-width', 0.02);
    g.append('text').attr('x', 0).attr('y', Y(0) + 0.35).attr('font-size', 0.26).text('left wall');
    g.append('text').attr('x', W).attr('y', Y(0) + 0.35).attr('font-size', 0.26).attr('text-anchor', 'end').text('right wall');
    evG = g.append('g');
    drawBall = makeBall(g.append('g'), 0.08);
    key = '';
  }

  function update(f) {
    const D = V.D; if (!D) return;
    if (!g || g.datum() !== D) { build(D); g.datum(D); }
    const t = D.t[f], scope = document.getElementById('wall-scope').value, mode = document.getElementById('wall-color').value;
    const r = rallyAt(D, t);
    const k = [scope, mode, scope === 'clip' ? '' : (r ? r.id : 'x' + Math.floor(t / 4)), D.stem].join('|');
    if (k !== key) {
      key = k; evG.selectAll('*').remove();
      for (const e of scopedEvents(D, ['front_wall'], t, scope)) {
        if (!Number.isFinite(e.z)) continue;
        const rx = Math.max(0.05, e.sigma_a || 0.07), ry = Math.max(0.05, e.sigma_b || 0.07);
        evG.append('ellipse').attr('class', 'ev').datum(e).attr('cx', e.x).attr('cy', Y(e.z)).attr('rx', rx).attr('ry', ry)
          .attr('fill', eventColor(e, mode)).attr('stroke', eventColor(e, mode)).attr('stroke-width', 0.05)
          .on('click', (_, d) => V.seekTo(d.t - 0.02)).append('title').text(`front wall @ ${fmtTime(e.t)}`);
      }
      if (!evG.selectAll('ellipse').size()) evG.append('text').attr('x', W / 2).attr('y', Y(H / 2)).attr('text-anchor', 'middle').attr('font-size', 0.3).text(D.bounces.length ? 'no front-wall impacts here' : 'no ball events yet');
    }
    evG.selectAll('ellipse').attr('fill-opacity', d => d.t > t ? 0.12 : 0.4).attr('stroke-width', d => Math.abs(d.t - t) < 0.5 ? 0.14 : 0.05);
  }
  // Seen from behind, across and up: the ball's height over the court, wherever it is along the court.
  const ball = ft => { if (g && V.D && g.datum() === V.D) drawBall(V.D, ft, q => [q[0], Y(q[2])]); };
  return { update, ball, rebuild() { g = null; key = ''; } };
})();
