// Side panel: stats, insights, events, report. Renders whatever the analysis stage wrote, generically.
'use strict';
const Panel = (() => {
  const body = document.getElementById('tab-body');
  let tab = 'stats', lastNow = -1;

  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const num = v => (typeof v === 'number' ? (Number.isInteger(v) ? String(v) : (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2))) : esc(v));
  const isObj = v => v && typeof v === 'object' && !Array.isArray(v);
  const nice = k => esc(String(k).replace(/_/g, ' '));
  function playerKey(k) {
    const m = /^(?:player[_ ]?)?([ab])$/i.exec(String(k));
    return m ? m[1].toUpperCase() : null;
  }
  const label = k => { const p = playerKey(k); return p ? `<span class="p${p.toLowerCase()}">${esc(V.names[p])}</span>` : nice(k); };
  const chip = (t, txt) => `<span class="chip" data-t="${t}">${txt || fmtTime(t)}</span>`;

  function renderObj(o, depth = 0) {
    if (Array.isArray(o)) {
      if (o.every(x => !x || typeof x !== 'object')) return `<div>${o.map(num).join(', ')}</div>`;
      if (o.every(isObj)) {
        const cols = [...new Set(o.flatMap(r => Object.keys(r)))].filter(c => o.every(r => !isObj(r[c]) && !Array.isArray(r[c])));
        return `<table class="kv"><tr>${cols.map(c => `<th>${nice(c)}</th>`).join('')}</tr>${o.map(r => `<tr>${cols.map(c => `<td>${r[c] == null ? '' : num(r[c])}</td>`).join('')}</tr>`).join('')}</table>`;
      }
      return o.map(x => renderObj(x, depth + 1)).join('');
    }
    if (!isObj(o)) return `<div>${num(o)}</div>`;
    const keys = Object.keys(o);
    const kids = keys.filter(k => isObj(o[k]));
    if (kids.length >= 2 && kids.length === keys.length && kids.some(k => playerKey(k))) {
      const rows = [...new Set(kids.flatMap(k => Object.keys(o[k])))];
      return `<table class="kv"><tr><th></th>${kids.map(k => `<th>${label(k)}</th>`).join('')}</tr>` +
        rows.map(r => `<tr><td>${nice(r)}</td>${kids.map(k => `<td>${isObj(o[k][r]) || Array.isArray(o[k][r]) ? '…' : (o[k][r] == null ? '' : num(o[k][r]))}</td>`).join('')}</tr>`).join('') + '</table>';
    }
    let scal = '', rest = '';
    for (const k of keys) {
      const v = o[k];
      if (isObj(v) || Array.isArray(v)) rest += `<h4>${label(k)}</h4>${renderObj(v, depth + 1)}`;
      else scal += `<tr><td>${label(k)}</td><td>${v == null ? '' : num(v)}</td></tr>`;
    }
    return (scal ? `<table class="kv">${scal}</table>` : '') + rest;
  }

  // Stats computed from tracking alone, so the panel is useful before the analysis stage exists.
  function tracked(D) {
    const out = {};
    const inRally = new Uint8Array(D.n);
    for (const r of D.rallies) { const a = frameOfTime(D.t, r.start_t), b = frameOfTime(D.t, r.end_t); for (let f = a; f <= b; f++) inRally[f] = 1; }
    const T = [D.court.spec.width / 2, D.court.derived.short_y];
    const fps = D.meta.fps || 30;
    for (const p of ['A', 'B']) {
      const P = D[p]; if (!P.pos) continue;
      let dist = 0, nIn = 0, nT = 0, nFront = 0, prev = null, det = 0, ysum = 0;
      for (let f = 0; f < D.n; f++) {
        const x = P.pos[2 * f], y = P.pos[2 * f + 1];
        if (P.status[f] === 2) det++;
        if (!Number.isFinite(x)) { prev = null; continue; }
        if (inRally[f]) {
          nIn++; ysum += y;
          if (Math.hypot(x - T[0], y - T[1]) < 1.0) nT++;
          if (y > T[1]) nFront++;
          if (prev) { const d = Math.hypot(x - prev[0], y - prev[1]); if (d < 0.5) dist += d; }
        }
        prev = inRally[f] ? [x, y] : null;
      }
      const sec = nIn / fps;
      out[p] = {
        'detected (% of frames)': 100 * det / D.n,
        'distance in rallies (m)': dist,
        'mean speed in rallies (m/s)': sec ? dist / sec : null,
        'within 1 m of T (% of rally time)': nIn ? 100 * nT / nIn : null,
        'beyond the short line (% of rally time)': nIn ? 100 * nFront / nIn : null,
        'mean depth from back wall (m)': nIn ? ysum / nIn : null,
      };
    }
    return out;
  }

  function statsHtml(D) {
    let h = '';
    if (D.meta.stats) h += `<h4>Analysis</h4>${renderObj(D.meta.stats)}`;
    else h += '<div class="note">No analysis/stats.json yet.</div>';
    h += `<h4>Tracking summary</h4><div class="note">Computed here from the tracked positions during audio-detected rallies.</div>${renderObj(tracked(D))}`;
    const lens = D.rallies.map(r => r.end_t - r.start_t);
    if (lens.length) h += `<h4>Rallies</h4><table class="kv"><tr><td>count</td><td>${lens.length}</td></tr><tr><td>mean length (s)</td><td>${num(lens.reduce((a, b) => a + b, 0) / lens.length)}</td></tr><tr><td>longest (s)</td><td>${num(Math.max(...lens))}</td></tr></table>`;
    return h;
  }

  function evidenceTimes(item) {
    const raw = item.evidence ?? item.timestamps ?? item.times ?? item.evidence_t ?? item.clips ?? item.examples ?? [];
    const arr = Array.isArray(raw) ? raw : [raw];
    const out = [];
    for (const e of arr) {
      if (typeof e === 'number') out.push([e, null]);
      else if (isObj(e)) {
        if (e.clip && V.D && e.clip !== V.D.stem) continue;
        const t = e.t ?? e.time ?? e.t0 ?? e.start_t ?? e.t_start ?? e.timestamp ?? e.start;
        if (typeof t === 'number') out.push([t, e.label ?? e.note ?? e.desc ?? null]);
      }
    }
    return out;
  }
  function insightCards(items, cls) {
    return items.map(it => {
      if (typeof it === 'string') return `<div class="card ${cls}">${esc(it)}</div>`;
      const title = it.title ?? it.headline ?? it.behaviour ?? it.name ?? it.summary ?? it.insight ?? it.text ?? '';
      const text = it.text ?? it.description ?? it.detail ?? it.explanation ?? (it.summary !== title ? it.summary : '') ?? '';
      const extra = [it.cue && `Cue: ${it.cue}`, it.drill && `Drill: ${it.drill}`].filter(Boolean).map(x => `<div class="note">${esc(x)}</div>`).join('');
      const pk = it.player && playerKey(it.player);
      const pl = pk ? `<span class="p${pk.toLowerCase()}">${esc(V.names[pk])}</span> ` : '';
      return `<div class="card ${cls}"><div class="h">${pl}${esc(title)}</div>${text && text !== title ? `<div>${esc(text)}</div>` : ''}${extra}${evidenceTimes(it).map(([t, l]) => chip(t, l ? `${fmtTime(t)} ${esc(l)}` : null)).join('')}</div>`;
    }).join('');
  }
  const SKIP_INSIGHT_KEYS = new Set(['candidates', 'rejected_sentences']);
  function insightWalk(v, key, cls) {
    if (Array.isArray(v)) {
      const c = /strength|good|positive/i.test(key) ? 'good' : /improv|weak|bad|fix/i.test(key) ? 'bad' : cls;
      return v.length ? `<div class="note">${nice(key)}</div>${insightCards(v, c)}` : '';
    }
    if (!isObj(v)) return '';
    return Object.entries(v).filter(([k]) => !SKIP_INSIGHT_KEYS.has(k)).map(([k, x]) => {
      const inner = insightWalk(x, k, cls);
      return inner && isObj(x) ? `<h4>${label(k)}</h4>${inner}` : inner;
    }).join('');
  }
  function insightsHtml(D) {
    const ins = D.meta.insights;
    if (!ins) return '<div class="empty">No analysis/insights.json yet. Strengths and improvements, each with clickable evidence timestamps, appear here when the analysis stage writes it.</div>';
    const note = isObj(ins) && typeof ins.note === 'string' ? `<div class="note">${esc(ins.note)}</div>` : '';
    const h = insightWalk(Array.isArray(ins) ? { insights: ins } : ins, '', '');
    return note + (h || `<pre>${esc(JSON.stringify(ins, null, 1))}</pre>`);
  }

  function eventsHtml(D) {
    const evs = D.events.filter(e => Panel.showAudio || e.source !== 'audio');
    let h = `<label class="small"><input type="checkbox" id="ev-audio" ${Panel.showAudio ? 'checked' : ''}> include audio onsets</label>`;
    if (!evs.length) return h + '<div class="empty">No ball events yet (ball_events.parquet).</div>';
    return h + evs.slice(0, 1500).map(e => {
      const pk = e.player && playerKey(e.player);
      const who = pk ? ` <span class="p${pk.toLowerCase()}">${esc(V.names[pk])}</span>` : '';
      return `<div class="ev-row" data-t="${e.t}"><span class="k">${nice(e.kind)}${who}</span><span>${fmtTime(e.t)}</span></div>`;
    }).join('');
  }

  function mdToHtml(md) {
    let out = '', list = false;
    for (const ln of esc(md).split('\n')) {
      const li = /^\s*[-*]\s+(.*)/.exec(ln), hd = /^(#{1,4})\s+(.*)/.exec(ln);
      if (li) { if (!list) { out += '<ul>'; list = true; } out += `<li>${li[1]}</li>`; continue; }
      if (list) { out += '</ul>'; list = false; }
      if (hd) out += `<h4>${hd[2]}</h4>`; else if (ln.trim()) out += `<p>${ln}</p>`;
    }
    if (list) out += '</ul>';
    return out.replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/\b(\d{1,2}):(\d\d)(?:\.\d+)?\b/g, (m, a, b) => `<span class="chip" data-t="${+a * 60 + +b}">${m}</span>`);
  }
  const reportHtml = D => D.meta.report ? mdToHtml(D.meta.report) : '<div class="empty">No analysis/report.md yet.</div>';

  function render() {
    const D = V.D; lastNow = -1;
    if (!D) { body.innerHTML = ''; return; }
    body.innerHTML = { stats: statsHtml, insights: insightsHtml, events: eventsHtml, report: reportHtml }[tab](D);
    const au = document.getElementById('ev-audio');
    if (au) au.onchange = () => { Panel.showAudio = au.checked; render(); };
  }

  body.addEventListener('click', ev => {
    const el = ev.target.closest('[data-t]');
    if (el) V.seekTo(Math.max(0, parseFloat(el.dataset.t) - (el.classList.contains('chip') ? 1.0 : 0.02)));
  });
  document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {
    tab = b.dataset.tab; document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b)); render();
  }));

  function update(f) {
    if (tab !== 'events' || !V.D) return;
    const t = V.D.t[f], rows = body.querySelectorAll('.ev-row');
    let best = -1, bd = 1e9;
    rows.forEach((r, i) => { const d = Math.abs(parseFloat(r.dataset.t) - t); if (d < bd) { bd = d; best = i; } });
    if (best === lastNow) return;
    lastNow = best; rows.forEach((r, i) => r.classList.toggle('now', i === best && bd < 0.5));
  }
  return { render, update, showAudio: false };
})();
