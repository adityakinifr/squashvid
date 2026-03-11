const form = document.getElementById("analyze-form");
const resultsEl = document.getElementById("results");
const statusText = document.getElementById("status-text");
const analyzeBtn = document.getElementById("analyze-btn");
const sourceInput = document.getElementById("source-input");
const fileInput = document.getElementById("video-file");
const pathGroup = document.getElementById("path-group");
const uploadGroup = document.getElementById("upload-group");

const sourceLabel = document.getElementById("source-label");
const kpiGrid = document.getElementById("kpi-grid");
const tacticalBars = document.getElementById("tactical-bars");
const movementCards = document.getElementById("movement-cards");
const rallyChart = document.getElementById("rally-chart");
const rallyList = document.getElementById("rally-list");
const rawJson = document.getElementById("raw-json");

const insightPanel = document.getElementById("insight-panel");
const insightModel = document.getElementById("insight-model");
const insightBody = document.getElementById("insight-body");
const patternList = document.getElementById("pattern-list");
const drillList = document.getElementById("drill-list");

const kpiTemplate = document.getElementById("kpi-template");
const barTemplate = document.getElementById("bar-template");
const movementTemplate = document.getElementById("movement-template");
const rallyTemplate = document.getElementById("rally-template");

const modeRadios = Array.from(document.querySelectorAll("input[name='source_mode']"));
let progressTicker = null;
let progressStartTs = 0;

function activeMode() {
  const checked = modeRadios.find((radio) => radio.checked);
  return checked ? checked.value : "youtube";
}

function setMode(mode) {
  const isUpload = mode === "upload";
  pathGroup.classList.toggle("hidden", isUpload);
  uploadGroup.classList.toggle("hidden", !isUpload);

  if (mode === "youtube") {
    sourceInput.placeholder = "https://www.youtube.com/watch?v=...";
  }
  if (mode === "path") {
    sourceInput.placeholder = "/absolute/path/to/match.mp4";
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}

function fmtPct(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "n/a";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const out = [];
  let listOpen = false;

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (listOpen) {
        out.push("</ul>");
        listOpen = false;
      }
      continue;
    }

    if (line.startsWith("### ")) {
      if (listOpen) {
        out.push("</ul>");
        listOpen = false;
      }
      out.push(`<h3>${escapeHtml(line.slice(4))}</h3>`);
      continue;
    }
    if (line.startsWith("## ")) {
      if (listOpen) {
        out.push("</ul>");
        listOpen = false;
      }
      out.push(`<h2>${escapeHtml(line.slice(3))}</h2>`);
      continue;
    }
    if (line.startsWith("# ")) {
      if (listOpen) {
        out.push("</ul>");
        listOpen = false;
      }
      out.push(`<h2>${escapeHtml(line.slice(2))}</h2>`);
      continue;
    }

    if (line.startsWith("- ")) {
      if (!listOpen) {
        out.push("<ul>");
        listOpen = true;
      }
      out.push(`<li>${escapeHtml(line.slice(2))}</li>`);
      continue;
    }

    if (listOpen) {
      out.push("</ul>");
      listOpen = false;
    }
    out.push(`<p>${escapeHtml(line)}</p>`);
  }

  if (listOpen) {
    out.push("</ul>");
  }

  return out.join("\n");
}

function normalizedValue(metricKey, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return 0;
  }

  const raw = Number(value);
  if (metricKey.includes("rate") || metricKey.includes("frequency") || metricKey.includes("usage")) {
    return Math.max(0, Math.min(1, raw));
  }
  if (metricKey === "winner_rate") {
    return Math.max(0, Math.min(1, raw));
  }
  if (metricKey === "avg_shots_per_rally") {
    return Math.max(0, Math.min(1, raw / 18));
  }
  return Math.max(0, Math.min(1, raw));
}

function setLoading(isLoading, message = "") {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.textContent = isLoading ? "Analyzing..." : "Analyze Match";
  statusText.textContent = message;
}

function startProgressTicker(mode, includeLlm) {
  if (progressTicker) {
    clearInterval(progressTicker);
    progressTicker = null;
  }

  progressStartTs = Date.now();
  const messages = [
    mode === "youtube"
      ? "Downloading YouTube video (or loading from cache)..."
      : "Loading video source...",
    "Segmenting rallies from match motion...",
    "Tracking players and ball inside rally windows...",
    includeLlm ? "Generating coaching insight with LLM..." : "Skipping LLM and building local report...",
    "Finalizing dashboard output...",
  ];

  let idx = 0;
  statusText.textContent = messages[idx];
  progressTicker = setInterval(() => {
    idx += 1;
    if (idx < messages.length) {
      statusText.textContent = messages[idx];
      return;
    }
    const elapsed = Math.floor((Date.now() - progressStartTs) / 1000);
    statusText.textContent = `Still processing (${elapsed}s). Tip: reduce max rallies or increase frame-step values for faster runs.`;
  }, 8000);
}

function stopProgressTicker() {
  if (progressTicker) {
    clearInterval(progressTicker);
    progressTicker = null;
  }
}

async function submitAnalyze(event) {
  event.preventDefault();

  const mode = activeMode();
  const includeLlm = document.getElementById("include-llm").checked;
  const llmModel = document.getElementById("llm-model").value.trim() || "gpt-4.1-mini";
  const openaiApiKey = document.getElementById("openai-api-key").value.trim();
  const maxRallies = document.getElementById("max-rallies").value.trim();
  const youtubeCacheDir = document.getElementById("youtube-cache").value.trim();

  const motionThreshold = Number(document.getElementById("motion-threshold").value || "0.018");
  const minRallySec = Number(document.getElementById("min-rally-sec").value || "4.0");
  const idleGapSec = Number(document.getElementById("idle-gap-sec").value || "1.2");
  const segmentFrameStep = Number(document.getElementById("segment-frame-step").value || "2");
  const trackingFrameStep = Number(document.getElementById("tracking-frame-step").value || "4");

  if (mode === "upload" && (!fileInput.files || fileInput.files.length === 0)) {
    setLoading(false, "Select a video file to upload.");
    return;
  }

  if (mode !== "upload" && !sourceInput.value.trim()) {
    setLoading(false, "Enter a YouTube URL or local file path.");
    return;
  }

  setLoading(true, "");
  startProgressTicker(mode, includeLlm);

  try {
    let response;

    if (mode === "upload") {
      const body = new FormData();
      body.append("file", fileInput.files[0]);
      body.append("include_llm", includeLlm ? "true" : "false");
      body.append("llm_model", llmModel);
      if (openaiApiKey) {
        body.append("openai_api_key", openaiApiKey);
      }
      body.append("motion_threshold", String(motionThreshold));
      body.append("min_rally_sec", String(minRallySec));
      body.append("idle_gap_sec", String(idleGapSec));
      body.append("segment_frame_step", String(segmentFrameStep));
      body.append("tracking_frame_step", String(trackingFrameStep));
      if (maxRallies) {
        body.append("max_rallies", maxRallies);
      }
      if (youtubeCacheDir) {
        body.append("youtube_cache_dir", youtubeCacheDir);
      }
      response = await fetch("/analyze/upload", { method: "POST", body });
    } else {
      const body = {
        video_path: sourceInput.value.trim(),
        include_llm: includeLlm,
        llm_model: llmModel,
        motion_threshold: motionThreshold,
        min_rally_sec: minRallySec,
        idle_gap_sec: idleGapSec,
        segment_frame_step: segmentFrameStep,
        tracking_frame_step: trackingFrameStep,
      };
      if (openaiApiKey) {
        body.openai_api_key = openaiApiKey;
      }
      if (maxRallies) {
        body.max_rallies = Number(maxRallies);
      }
      if (youtubeCacheDir) {
        body.youtube_cache_dir = youtubeCacheDir;
      }
      response = await fetch("/analyze/path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    }

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed with ${response.status}`);
    }

    const data = await response.json();
    renderAll(data);
    stopProgressTicker();
    setLoading(false, `Analysis complete. ${data.timeline.rallies.length} rallies rendered.`);
  } catch (error) {
    stopProgressTicker();
    setLoading(false, String(error.message || error));
  }
}

function renderKpis(timeline) {
  clearChildren(kpiGrid);
  const tactical = timeline.tactical_patterns || {};

  const items = [
    ["Rallies", timeline.rallies.length],
    ["FPS", fmt(timeline.fps, 1)],
    ["Avg Shots/Rally", fmt(tactical.avg_shots_per_rally, 1)],
    ["Winner Rate", fmtPct(tactical.winner_rate)],
    ["Backhand Pressure", fmtPct(tactical.backhand_pressure_rate)],
    ["Crosscourt Freq", fmtPct(tactical.crosscourt_frequency)],
  ];

  for (const [label, value] of items) {
    const node = kpiTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".kpi-label").textContent = label;
    node.querySelector(".kpi-value").textContent = String(value);
    kpiGrid.appendChild(node);
  }
}

function renderTacticalBars(tactical) {
  clearChildren(tacticalBars);

  const labels = {
    backhand_pressure_rate: "Backhand Pressure",
    boast_usage: "Boast Usage",
    crosscourt_frequency: "Crosscourt Frequency",
    short_ball_punish_rate: "Short-Ball Punish",
    winner_rate: "Winner Rate",
    avg_shots_per_rally: "Avg Shots / Rally",
  };

  for (const [key, label] of Object.entries(labels)) {
    const value = tactical[key] ?? 0;
    const row = barTemplate.content.firstElementChild.cloneNode(true);
    row.querySelector(".metric-label").textContent = label;

    const isPct = key !== "avg_shots_per_rally";
    row.querySelector(".metric-value").textContent = isPct ? fmtPct(value) : fmt(value, 1);

    const fill = row.querySelector(".bar-fill");
    const percent = normalizedValue(key, value) * 100;
    requestAnimationFrame(() => {
      fill.style.width = `${percent}%`;
    });

    tacticalBars.appendChild(row);
  }
}

function renderMovementCards(movement) {
  clearChildren(movementCards);

  const items = [
    ["A Avg T Recovery", movement.A_avg_T_recovery_sec, "s"],
    ["B Avg T Recovery", movement.B_avg_T_recovery_sec, "s"],
    ["A T Occupancy", movement.A_T_occupancy, "%"],
    ["B T Occupancy", movement.B_T_occupancy, "%"],
  ];

  for (const [label, value, unit] of items) {
    const node = movementTemplate.content.firstElementChild.cloneNode(true);
    node.querySelector(".movement-label").textContent = label;

    let text = "n/a";
    if (value !== null && value !== undefined) {
      if (unit === "%") {
        text = fmtPct(value);
      } else {
        text = `${fmt(value, 2)}${unit}`;
      }
    }
    node.querySelector(".movement-value").textContent = text;
    movementCards.appendChild(node);
  }
}

function buildRallyChart(rallies) {
  clearChildren(rallyChart);

  if (!rallies.length) {
    rallyChart.textContent = "No rally data detected for charting.";
    return;
  }

  const width = Math.max(700, rallies.length * 42);
  const height = 220;
  const margin = { top: 20, right: 18, bottom: 34, left: 40 };
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const durations = rallies.map((r) => Number(r.duration_sec || 0));
  const shotCounts = rallies.map((r) => Number((r.shots || []).length));

  const maxDuration = Math.max(1, ...durations);
  const maxShots = Math.max(1, ...shotCounts);

  const x = (idx) => margin.left + (idx * innerW) / Math.max(1, rallies.length - 1);
  const yDuration = (value) => margin.top + innerH - (value / maxDuration) * innerH;
  const yShots = (value) => margin.top + innerH - (value / maxShots) * innerH;

  const bars = rallies
    .map((_, idx) => {
      const bw = Math.max(10, innerW / Math.max(8, rallies.length) - 6);
      const xPos = x(idx) - bw / 2;
      const yPos = yShots(shotCounts[idx]);
      const h = margin.top + innerH - yPos;
      return `<rect x="${xPos.toFixed(2)}" y="${yPos.toFixed(2)}" width="${bw.toFixed(
        2
      )}" height="${h.toFixed(2)}" rx="4" fill="rgba(15,157,88,0.23)"/>`;
    })
    .join("\n");

  const linePath = durations
    .map((value, idx) => `${idx === 0 ? "M" : "L"}${x(idx).toFixed(2)} ${yDuration(value).toFixed(2)}`)
    .join(" ");

  const dots = durations
    .map(
      (value, idx) =>
        `<circle cx="${x(idx).toFixed(2)}" cy="${yDuration(value).toFixed(2)}" r="3.4" fill="#0f6f9d"/>`
    )
    .join("\n");

  const xTicks = rallies
    .map((rally, idx) => {
      if (rallies.length > 28 && idx % 3 !== 0) {
        return "";
      }
      return `<text x="${x(idx).toFixed(2)}" y="${(height - 10).toFixed(
        2
      )}" text-anchor="middle" font-size="10" fill="#4a616e">R${escapeHtml(String(
        rally.rally_id
      ))}</text>`;
    })
    .join("\n");

  const gridY = [0.25, 0.5, 0.75].map((p) => {
    const y = margin.top + innerH * p;
    return `<line x1="${margin.left}" y1="${y.toFixed(2)}" x2="${
      width - margin.right
    }" y2="${y.toFixed(2)}" stroke="rgba(12,24,30,0.12)" stroke-dasharray="3 3"/>`;
  });

  const svg = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Rally duration and shot count chart">
      <defs>
        <linearGradient id="durationGradient" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%" stop-color="#0f6f9d" />
          <stop offset="100%" stop-color="#f96e2a" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width="${width}" height="${height}" fill="transparent" />
      ${gridY.join("\n")}
      <line x1="${margin.left}" y1="${margin.top + innerH}" x2="${width - margin.right}" y2="${
        margin.top + innerH
      }" stroke="rgba(12,24,30,0.25)"/>
      ${bars}
      <path d="${linePath}" fill="none" stroke="url(#durationGradient)" stroke-width="2.5"/>
      ${dots}
      ${xTicks}
      <text x="${margin.left}" y="14" font-size="11" fill="#4a616e">Duration (line)</text>
      <text x="${margin.left + 110}" y="14" font-size="11" fill="#4a616e">Shot count (bars)</text>
    </svg>
  `;

  rallyChart.innerHTML = svg;
}

function renderRallies(rallies) {
  clearChildren(rallyList);

  if (!rallies.length) {
    const empty = document.createElement("p");
    empty.textContent = "No rally segments were detected from this video.";
    empty.className = "rally-empty";
    rallyList.appendChild(empty);
    return;
  }

  for (const rally of rallies) {
    const node = rallyTemplate.content.firstElementChild.cloneNode(true);

    const start = fmt(rally.start_time, 1);
    const end = fmt(rally.end_time, 1);

    node.querySelector(".rally-title").textContent = `Rally ${rally.rally_id}`;
    node.querySelector(".rally-meta").textContent = `${start}s -> ${end}s | ${fmt(
      rally.duration_sec,
      1
    )}s`;
    node.querySelector(".rally-outcome").textContent = rally.outcome || "Outcome unknown";

    const strip = node.querySelector(".shot-strip");
    const shots = Array.isArray(rally.shots) ? rally.shots : [];
    if (!shots.length) {
      const noShots = document.createElement("span");
      noShots.className = "shot-pill";
      noShots.textContent = "No shots inferred";
      strip.appendChild(noShots);
    } else {
      for (const shot of shots) {
        const pill = document.createElement("span");
        const label = shot.player || "Unknown";
        const side = shot.side ? ` ${shot.side}` : "";
        pill.className = `shot-pill ${String(label).toLowerCase()}`;
        pill.textContent = `${label}: ${shot.type || "shot"}${side}`;
        strip.appendChild(pill);
      }
    }

    const metrics = node.querySelector(".rally-metrics");
    const tags = [
      `A T recover: ${rally.positions?.A_avg_T_recovery_sec ? `${fmt(rally.positions.A_avg_T_recovery_sec)}s` : "n/a"}`,
      `B T recover: ${rally.positions?.B_avg_T_recovery_sec ? `${fmt(rally.positions.B_avg_T_recovery_sec)}s` : "n/a"}`,
      `A T occ: ${rally.positions?.A_T_occupancy !== null && rally.positions?.A_T_occupancy !== undefined ? fmtPct(rally.positions.A_T_occupancy) : "n/a"}`,
      `B T occ: ${rally.positions?.B_T_occupancy !== null && rally.positions?.B_T_occupancy !== undefined ? fmtPct(rally.positions.B_T_occupancy) : "n/a"}`,
    ];
    for (const tagText of tags) {
      const tag = document.createElement("span");
      tag.className = "rally-tag";
      tag.textContent = tagText;
      metrics.appendChild(tag);
    }

    rallyList.appendChild(node);
  }
}

function renderInsight(insight) {
  if (!insight) {
    insightPanel.classList.add("hidden");
    return;
  }

  insightPanel.classList.remove("hidden");
  insightModel.textContent = `${insight.model || "model-unknown"} | confidence: ${
    insight.confidence || "n/a"
  }`;

  insightBody.innerHTML = markdownToHtml(insight.report_markdown || "");

  clearChildren(patternList);
  const patterns = Array.isArray(insight.key_patterns) ? insight.key_patterns : [];
  if (!patterns.length) {
    const li = document.createElement("li");
    li.textContent = "No recurring patterns returned.";
    patternList.appendChild(li);
  } else {
    for (const item of patterns) {
      const li = document.createElement("li");
      li.textContent = item;
      patternList.appendChild(li);
    }
  }

  clearChildren(drillList);
  const drills = Array.isArray(insight.drills) ? insight.drills : [];
  if (!drills.length) {
    const li = document.createElement("li");
    li.textContent = "No drills returned.";
    drillList.appendChild(li);
  } else {
    for (const item of drills) {
      const li = document.createElement("li");
      li.textContent = item;
      drillList.appendChild(li);
    }
  }
}

function renderAll(data) {
  const timeline = data.timeline || { rallies: [], tactical_patterns: {}, movement_summary: {} };
  const tactical = timeline.tactical_patterns || {};
  const movement = timeline.movement_summary || {};

  sourceLabel.textContent = timeline.video_path || "Source unavailable";
  renderKpis(timeline);
  renderTacticalBars(tactical);
  renderMovementCards(movement);
  buildRallyChart(Array.isArray(timeline.rallies) ? timeline.rallies : []);
  renderRallies(Array.isArray(timeline.rallies) ? timeline.rallies : []);
  renderInsight(data.insight || null);

  rawJson.textContent = JSON.stringify(data, null, 2);
  resultsEl.classList.remove("hidden");
}

for (const radio of modeRadios) {
  radio.addEventListener("change", () => {
    setMode(activeMode());
  });
}

setMode(activeMode());
form.addEventListener("submit", submitAnalyze);
