const form = document.getElementById("analyze-form");
const resultsEl = document.getElementById("results");
const statusText = document.getElementById("status-text");
const analyzeBtn = document.getElementById("analyze-btn");
const sourceInput = document.getElementById("source-input");
const fileInput = document.getElementById("video-file");
const playerANameInput = document.getElementById("player-a-name");
const playerBNameInput = document.getElementById("player-b-name");
const pathGroup = document.getElementById("path-group");
const uploadGroup = document.getElementById("upload-group");

const sourceLabel = document.getElementById("source-label");
const kpiGrid = document.getElementById("kpi-grid");
const tacticalBars = document.getElementById("tactical-bars");
const movementCards = document.getElementById("movement-cards");
const rallyChart = document.getElementById("rally-chart");
const diagnosticsPanel = document.getElementById("diagnostics-panel");
const diagnosticsCards = document.getElementById("diagnostics-cards");
const segmentPreview = document.getElementById("segment-preview");
const rallyList = document.getElementById("rally-list");
const rallySort = document.getElementById("rally-sort");
const winnerFilter = document.getElementById("winner-filter");
const durationMinInput = document.getElementById("duration-min");
const durationMaxInput = document.getElementById("duration-max");
const shotsMinInput = document.getElementById("shots-min");
const shotsMaxInput = document.getElementById("shots-max");
const rallyPicker = document.getElementById("rally-picker");
const rallyPickerBtn = document.getElementById("rally-picker-btn");
const rallyToolbarStatus = document.getElementById("rally-toolbar-status");
const rawJson = document.getElementById("raw-json");
const reviewPanel = document.getElementById("review-panel");
const reviewMeta = document.getElementById("review-meta");
const reviewCanvas = document.getElementById("review-canvas");
const reviewVideo = document.getElementById("review-video");
const rallyScrubber = document.getElementById("rally-scrubber");
const shotMarkers = document.getElementById("shot-markers");
const shotCaption = document.getElementById("shot-caption");
const selectedRallyKpis = document.getElementById("selected-rally-kpis");
const selectedRallyShots = document.getElementById("selected-rally-shots");
const boundaryStartInput = document.getElementById("boundary-start");
const boundaryEndInput = document.getElementById("boundary-end");
const boundaryApplyBtn = document.getElementById("boundary-apply");
const boundaryResetBtn = document.getElementById("boundary-reset");
const boundaryNote = document.getElementById("boundary-note");

const insightPanel = document.getElementById("insight-panel");
const insightModel = document.getElementById("insight-model");
const insightSummaryCards = document.getElementById("insight-summary-cards");
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
const reviewCtx = reviewCanvas ? reviewCanvas.getContext("2d") : null;
let currentResult = null;
let activeRally = null;
let reviewRaf = null;
let reviewTimeSync = false;
let allRallies = [];
let displayedRallies = [];
let playerNames = { A: "Player A", B: "Player B" };
let selectedOriginalBounds = null;

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

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

function updatePlayerNamesFromInputs() {
  const a = (playerANameInput?.value || "").trim();
  const b = (playerBNameInput?.value || "").trim();
  playerNames = {
    A: a || "Player A",
    B: b || "Player B",
  };
}

function playerLabel(value) {
  if (value === "A") {
    return playerNames.A;
  }
  if (value === "B") {
    return playerNames.B;
  }
  return "Unknown";
}

function withNamedPlayers(text) {
  return String(text || "")
    .replaceAll("Player A", playerNames.A)
    .replaceAll("Player B", playerNames.B)
    .replaceAll("A avg", `${playerNames.A} avg`)
    .replaceAll("B avg", `${playerNames.B} avg`)
    .replaceAll("A T", `${playerNames.A} T`)
    .replaceAll("B T", `${playerNames.B} T`);
}

function formatOutcomeText(outcome) {
  const raw = String(outcome || "Outcome unknown");
  if (raw === "unknown") {
    return "Outcome unknown";
  }
  return withNamedPlayers(raw)
    .replaceAll("A winner", `${playerNames.A} winner`)
    .replaceAll("B winner", `${playerNames.B} winner`)
    .replaceAll("A pressure", `${playerNames.A} pressure`)
    .replaceAll("B pressure", `${playerNames.B} pressure`)
    .replaceAll("A forced error", `${playerNames.A} forced error`)
    .replaceAll("B forced error", `${playerNames.B} forced error`);
}

function refreshWinnerFilterLabels() {
  if (!winnerFilter) {
    return;
  }
  const aOpt = winnerFilter.querySelector('option[value="A"]');
  const bOpt = winnerFilter.querySelector('option[value="B"]');
  if (aOpt) {
    aOpt.textContent = `${playerNames.A} Wins`;
  }
  if (bOpt) {
    bOpt.textContent = `${playerNames.B} Wins`;
  }
}

function parseBound(value) {
  if (value === undefined || value === null) {
    return null;
  }
  const trimmed = String(value).trim();
  if (!trimmed) {
    return null;
  }
  const num = Number(trimmed);
  if (!Number.isFinite(num)) {
    return null;
  }
  return num;
}

function parsePositiveNumber(value, fallback = null) {
  const parsed = parseBound(value);
  if (parsed === null || parsed < 0) {
    return fallback;
  }
  return parsed;
}

function cloneRallyForReview(rally) {
  if (!rally || typeof rally !== "object") {
    return null;
  }
  return {
    ...rally,
    shots: Array.isArray(rally.shots) ? [...rally.shots] : [],
    positions: rally.positions ? { ...rally.positions } : {},
    metadata: rally.metadata ? { ...rally.metadata } : {},
  };
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

function normalizeInsightItem(item) {
  const raw = String(item ?? "").trim();
  if (!raw) {
    return "";
  }
  if (raw === "[object Object]") {
    return "Structured recommendation";
  }

  if (
    (raw.startsWith("{") && raw.endsWith("}")) ||
    (raw.startsWith("[") && raw.endsWith("]"))
  ) {
    let jsonCandidate = raw;
    if (raw.includes("'") && !raw.includes('"')) {
      jsonCandidate = raw
        .replace(/([{,]\s*)'([^']+?)'\s*:/g, '$1"$2":')
        .replace(/:\s*'([^']*?)'(?=\s*[,}])/g, ': "$1"');
    }
    try {
      const parsed = JSON.parse(jsonCandidate);
      if (typeof parsed === "string") {
        return parsed.trim();
      }
      if (Array.isArray(parsed)) {
        return parsed.map((x) => String(x)).join(" | ").trim();
      }
      if (parsed && typeof parsed === "object") {
        for (const key of ["drill", "title", "description", "focus", "text", "summary"]) {
          if (parsed[key]) {
            return String(parsed[key]).trim();
          }
        }
        return Object.entries(parsed)
          .map(([k, v]) => `${k}: ${String(v)}`)
          .join(" | ")
          .trim();
      }
    } catch {
      if (raw.startsWith("{") && raw.endsWith("}")) {
        return raw
          .slice(1, -1)
          .replaceAll('"', "")
          .replaceAll("'", "")
          .replaceAll(":", " ")
          .replaceAll(",", " | ")
          .replace(/\s+/g, " ")
          .trim();
      }
    }
  }
  return raw;
}

function classifyDrillType(text) {
  const value = String(text || "").toLowerCase();
  if (
    value.includes("ghost") ||
    value.includes("footwork") ||
    value.includes("recover") ||
    value.includes("lunge") ||
    value.includes("split")
  ) {
    return "footwork";
  }
  if (
    value.includes("length") ||
    value.includes("drive") ||
    value.includes("target") ||
    value.includes("backhand")
  ) {
    return "length";
  }
  return "decision";
}

function drillIconSvg(type) {
  if (type === "footwork") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 13l4-3 2 2 4 1 1 5H7z"/><path d="M11 10l-1-4 3-1 2 3"/></svg>';
  }
  if (type === "length") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1.5"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 9a4 4 0 118 0c0 2-1 3-2 4-1 1-1 2-1 3"/><path d="M11 20h2"/><path d="M9 6l-2-2"/><path d="M15 6l2-2"/></svg>';
}

function reportTone(title) {
  const value = String(title || "").toLowerCase();
  if (value.includes("movement") || value.includes("coverage") || value.includes("recovery")) {
    return "movement";
  }
  if (value.includes("difference") || value.includes("early") || value.includes("late")) {
    return "compare";
  }
  if (value.includes("cause") || value.includes("lost") || value.includes("error")) {
    return "risk";
  }
  if (value.includes("adjustment") || value.includes("recommend") || value.includes("drill")) {
    return "action";
  }
  return "tactical";
}

function reportToneLabel(tone) {
  if (tone === "movement") {
    return "Movement";
  }
  if (tone === "compare") {
    return "Comparison";
  }
  if (tone === "risk") {
    return "Risk";
  }
  if (tone === "action") {
    return "Action Plan";
  }
  return "Tactical";
}

function reportToneIconSvg(tone) {
  if (tone === "movement") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 17l4-5 3 2 5-7"/><circle cx="7" cy="7" r="2"/></svg>';
  }
  if (tone === "compare") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 8h6"/><path d="M5 12h10"/><path d="M5 16h6"/><path d="M16 6l3 3-3 3"/><path d="M19 15l-3 3-3-3"/></svg>';
  }
  if (tone === "risk") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l8 14H4z"/><path d="M12 10v4"/><path d="M12 17h.01"/></svg>';
  }
  if (tone === "action") {
    return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12l5 5L20 6"/><path d="M4 6h6"/><path d="M4 18h10"/></svg>';
  }
  return '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12h16"/><path d="M12 4v16"/><circle cx="12" cy="12" r="8"/></svg>';
}

function cleanSectionTitle(rawTitle) {
  return String(rawTitle || "Section")
    .replace(/^\d+\s*[\)\].:-]\s*/, "")
    .replace(/\s+/g, " ")
    .trim();
}

function parseReportItem(item) {
  const text = normalizeInsightItem(item)
    .replace(/\s+/g, " ")
    .trim();
  if (!text) {
    return { number: "", title: "", body: "" };
  }

  let value = text;
  let number = "";
  const numbered = value.match(/^(\d+)\s*[\)\].:-]\s*(.+)$/);
  if (numbered) {
    number = numbered[1];
    value = numbered[2].trim();
  }

  let title = "";
  let body = "";
  const strongSplit = value.match(/^\*\*([^*]+)\*\*\s*[:\-]?\s*(.*)$/);
  if (strongSplit) {
    title = strongSplit[1].trim();
    body = strongSplit[2].trim();
  } else {
    const colonSplit = value.match(/^([^:]{3,56}):\s+(.+)$/);
    if (colonSplit) {
      title = colonSplit[1].trim();
      body = colonSplit[2].trim();
    } else {
      body = value;
    }
  }

  return { number, title, body };
}

function parseReportSections(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const sections = [];
  let current = { title: "Summary", items: [] };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    if (line.startsWith("#")) {
      if (current.items.length > 0) {
        sections.push(current);
      }
      current = { title: line.replace(/^#+\s*/, "").trim() || "Section", items: [] };
      continue;
    }
    if (line.startsWith("- ")) {
      current.items.push(line.slice(2).trim());
      continue;
    }
    current.items.push(line);
  }
  if (current.items.length > 0) {
    sections.push(current);
  }
  return sections;
}

function renderReportCards(markdown) {
  clearChildren(insightBody);
  const sections = parseReportSections(markdown);
  if (!sections.length) {
    const empty = document.createElement("p");
    empty.textContent = "No report text available.";
    insightBody.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "report-cards";
  for (const section of sections) {
    const tone = reportTone(section.title);
    const title = cleanSectionTitle(withNamedPlayers(section.title));
    const card = document.createElement("article");
    card.className = `report-section-card tone-${tone}`;
    const head = document.createElement("div");
    head.className = "report-section-head";
    head.innerHTML = `
      <span class="report-section-badge">
        <span class="report-section-icon">${reportToneIconSvg(tone)}</span>
        <span>${escapeHtml(reportToneLabel(tone))}</span>
      </span>
      <p class="report-section-title">${escapeHtml(title)}</p>
      <p class="report-section-sub">${escapeHtml(String(section.items.length))} points</p>
    `;
    card.appendChild(head);

    const grid = document.createElement("div");
    grid.className = "report-item-grid";
    for (const item of section.items) {
      const parsed = parseReportItem(withNamedPlayers(item));
      const itemCard = document.createElement("article");
      itemCard.className = "report-item-card";
      const numberText = parsed.number ? escapeHtml(parsed.number) : "•";
      const titleText = parsed.title ? escapeHtml(withNamedPlayers(parsed.title)) : "";
      const bodyText = escapeHtml(withNamedPlayers(parsed.body || ""));
      itemCard.innerHTML = `
        <span class="report-item-number">${numberText}</span>
        <div class="report-item-copy">
          ${titleText ? `<p class="report-item-title">${titleText}</p>` : ""}
          <p class="report-item-body">${bodyText}</p>
        </div>
      `;
      grid.appendChild(itemCard);
    }
    card.appendChild(grid);
    wrap.appendChild(card);
  }
  insightBody.appendChild(wrap);
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

function rallyShotsCount(rally) {
  return Array.isArray(rally?.shots) ? rally.shots.length : 0;
}

function inferRallyWinnerCode(rally) {
  const outcome = String(rally?.outcome || "");
  if (!outcome || outcome === "unknown") {
    return "unknown";
  }
  if (outcome.includes("A winner") || outcome.includes("B forced error")) {
    return "A";
  }
  if (outcome.includes("B winner") || outcome.includes("A forced error")) {
    return "B";
  }
  return "unknown";
}

function rallyPassesFilters(rally) {
  const winnerCode = inferRallyWinnerCode(rally);
  const winnerValue = winnerFilter?.value || "all";
  if (winnerValue !== "all" && winnerCode !== winnerValue) {
    return false;
  }

  const duration = Number(rally.duration_sec || 0);
  const minDuration = parseBound(durationMinInput?.value);
  const maxDuration = parseBound(durationMaxInput?.value);
  if (minDuration !== null && duration < minDuration) {
    return false;
  }
  if (maxDuration !== null && duration > maxDuration) {
    return false;
  }

  const shotCount = rallyShotsCount(rally);
  const minShots = parseBound(shotsMinInput?.value);
  const maxShots = parseBound(shotsMaxInput?.value);
  if (minShots !== null && shotCount < minShots) {
    return false;
  }
  if (maxShots !== null && shotCount > maxShots) {
    return false;
  }

  return true;
}

function filterRallies(rallies) {
  return rallies.filter((rally) => rallyPassesFilters(rally));
}

function sortRallies(rallies, mode) {
  const rows = [...rallies];
  if (mode === "longest") {
    rows.sort((a, b) => Number(b.duration_sec || 0) - Number(a.duration_sec || 0));
  } else if (mode === "shortest") {
    rows.sort((a, b) => Number(a.duration_sec || 0) - Number(b.duration_sec || 0));
  } else if (mode === "most_shots") {
    rows.sort((a, b) => rallyShotsCount(b) - rallyShotsCount(a));
  } else {
    rows.sort((a, b) => Number(a.rally_id || 0) - Number(b.rally_id || 0));
  }
  return rows;
}

function updateRallyToolbarStatus() {
  if (!Array.isArray(displayedRallies) || displayedRallies.length === 0) {
    rallyToolbarStatus.textContent = "No rallies match current filters.";
    return;
  }

  const top = displayedRallies[0];
  const mode = rallySort?.value || "match";
  const countText = `${displayedRallies.length} shown / ${allRallies.length} total`;
  if (mode === "longest") {
    rallyToolbarStatus.textContent = `${countText}. Longest first: Rally ${top.rally_id} (${fmt(top.duration_sec, 1)}s).`;
  } else if (mode === "shortest") {
    rallyToolbarStatus.textContent = `${countText}. Shortest first: Rally ${top.rally_id} (${fmt(top.duration_sec, 1)}s).`;
  } else if (mode === "most_shots") {
    rallyToolbarStatus.textContent = `${countText}. Most shots first: Rally ${top.rally_id} (${rallyShotsCount(top)} shots).`;
  } else {
    rallyToolbarStatus.textContent = `${countText}. Sorted by match order.`;
  }
}

function applyRallySortRender() {
  const filtered = filterRallies(allRallies);
  displayedRallies = sortRallies(filtered, rallySort?.value || "match");
  if (displayedRallies.length === 0 && allRallies.length > 0) {
    renderRallies(displayedRallies, { emptyMessage: "No rallies match current filters." });
  } else {
    renderRallies(displayedRallies);
  }
  if (activeRally) {
    const refreshed = displayedRallies.find(
      (item) => Number(item.rally_id) === Number(activeRally.rally_id)
    );
    if (refreshed) {
      activeRally = refreshed;
    }
  }

  if (!activeRally && displayedRallies.length > 0) {
    activeRally = displayedRallies[0];
  } else if (displayedRallies.length === 0) {
    activeRally = null;
    clearChildren(selectedRallyKpis);
    clearChildren(selectedRallyShots);
    clearChildren(shotMarkers);
    shotCaption.textContent = "";
    if (reviewVideo && !reviewVideo.paused) {
      reviewVideo.pause();
    }
    if (reviewCtx) {
      reviewCtx.clearRect(0, 0, reviewCanvas.width, reviewCanvas.height);
    }
    updateReviewMeta();
  }
  highlightActiveRallyCard();
  updateRallyToolbarStatus();
}

function jumpToRallyById(rallyId) {
  const target = allRallies.find((item) => Number(item.rally_id) === Number(rallyId));
  if (!target) {
    rallyToolbarStatus.textContent = `Rally ${rallyId} was not found in this analysis.`;
    return;
  }

  if (rallySort) {
    rallySort.value = "match";
  }
  if (winnerFilter) {
    winnerFilter.value = "all";
  }
  if (durationMinInput) {
    durationMinInput.value = "";
  }
  if (durationMaxInput) {
    durationMaxInput.value = "";
  }
  if (shotsMinInput) {
    shotsMinInput.value = "";
  }
  if (shotsMaxInput) {
    shotsMaxInput.value = "";
  }
  applyRallySortRender();
  selectRally(target);

  const node = rallyList.querySelector(`[data-rally-id="${target.rally_id}"]`);
  if (node) {
    node.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  rallyToolbarStatus.textContent = `Jumped to Rally ${target.rally_id}.`;
}

function stopReviewLoop() {
  if (reviewRaf) {
    cancelAnimationFrame(reviewRaf);
    reviewRaf = null;
  }
}

function rallyDuration(rally) {
  return Math.max(0.001, Number(rally.end_time) - Number(rally.start_time));
}

function getCropNorm(rally) {
  const fallback = { x: 0, y: 0, w: 1, h: 1 };
  const crop = rally?.metadata?.focus_crop_norm;
  if (!crop || typeof crop !== "object") {
    return fallback;
  }
  return {
    x: clamp(Number(crop.x ?? 0), 0, 1),
    y: clamp(Number(crop.y ?? 0), 0, 1),
    w: clamp(Number(crop.w ?? 1), 0.05, 1),
    h: clamp(Number(crop.h ?? 1), 0.05, 1),
  };
}

function drawReviewFrame() {
  if (!reviewCtx || !activeRally || !reviewVideo || reviewVideo.readyState < 2) {
    return;
  }

  const vw = Number(reviewVideo.videoWidth || 0);
  const vh = Number(reviewVideo.videoHeight || 0);
  if (!vw || !vh) {
    return;
  }

  const crop = getCropNorm(activeRally);
  const sx = clamp(crop.x * vw, 0, vw - 1);
  const sy = clamp(crop.y * vh, 0, vh - 1);
  const sw = clamp(crop.w * vw, 1, vw - sx);
  const sh = clamp(crop.h * vh, 1, vh - sy);

  reviewCtx.clearRect(0, 0, reviewCanvas.width, reviewCanvas.height);
  reviewCtx.drawImage(reviewVideo, sx, sy, sw, sh, 0, 0, reviewCanvas.width, reviewCanvas.height);

  if (!reviewVideo.paused && !reviewVideo.ended) {
    reviewRaf = requestAnimationFrame(drawReviewFrame);
  }
}

function updateReviewMeta() {
  if (!activeRally) {
    reviewMeta.textContent = "Select a rally card to inspect the cropped view.";
    return;
  }
  reviewMeta.textContent =
    `Rally ${activeRally.rally_id}: ` +
    `${fmt(activeRally.start_time, 1)}s -> ${fmt(activeRally.end_time, 1)}s ` +
    `(${fmt(activeRally.duration_sec, 1)}s) | ${formatOutcomeText(activeRally.outcome)}`;
}

function describeShot(shot) {
  const label = playerLabel(shot?.player || "Unknown");
  const type = shot?.type || "shot";
  const side = shot?.side ? ` ${shot.side}` : "";
  return `${label} ${type}${side}`.trim();
}

function summarizeShotTypes(shots) {
  if (!Array.isArray(shots) || shots.length === 0) {
    return [];
  }
  const counts = {};
  for (const shot of shots) {
    const key = String(shot.type || "shot");
    counts[key] = (counts[key] || 0) + 1;
  }
  return Object.entries(counts)
    .sort((a, b) => Number(b[1]) - Number(a[1]))
    .slice(0, 3)
    .map(([name, count]) => `${name} x${count}`);
}

function renderSelectedRallyDetails(rally) {
  clearChildren(selectedRallyKpis);
  clearChildren(selectedRallyShots);
  if (!rally) {
    return;
  }

  const tags = [
    `Duration ${fmt(rally.duration_sec, 1)}s`,
    `Shots ${rallyShotsCount(rally)}`,
    `${playerNames.A} T ${rally.positions?.A_T_occupancy !== null && rally.positions?.A_T_occupancy !== undefined ? fmtPct(rally.positions.A_T_occupancy) : "n/a"}`,
    `${playerNames.B} T ${rally.positions?.B_T_occupancy !== null && rally.positions?.B_T_occupancy !== undefined ? fmtPct(rally.positions.B_T_occupancy) : "n/a"}`,
    `${playerNames.A} recover ${rally.positions?.A_avg_T_recovery_sec ? `${fmt(rally.positions.A_avg_T_recovery_sec)}s` : "n/a"}`,
    `${playerNames.B} recover ${rally.positions?.B_avg_T_recovery_sec ? `${fmt(rally.positions.B_avg_T_recovery_sec)}s` : "n/a"}`,
  ];
  for (const text of tags) {
    const el = document.createElement("span");
    el.className = "selected-rally-tag";
    el.textContent = text;
    selectedRallyKpis.appendChild(el);
  }

  const shots = Array.isArray(rally.shots) ? rally.shots : [];
  if (!shots.length) {
    const empty = document.createElement("p");
    empty.className = "selected-rally-empty";
    empty.textContent = "No shot sequence available for this rally.";
    selectedRallyShots.appendChild(empty);
    return;
  }

  for (const shot of shots) {
    const row = document.createElement("div");
    row.className = "selected-shot-row";
    row.innerHTML = `
      <span class="selected-shot-time">${escapeHtml(fmt(shot.timestamp, 2))}s</span>
      <span class="selected-shot-player">${escapeHtml(playerLabel(shot.player))}</span>
      <span class="selected-shot-type">${escapeHtml(shot.type || "shot")}</span>
      <span class="selected-shot-side">${escapeHtml(shot.side || "")}</span>
    `;
    selectedRallyShots.appendChild(row);
  }
}

function updateShotCaption() {
  if (!activeRally) {
    shotCaption.textContent = "";
    return;
  }
  const shots = Array.isArray(activeRally.shots) ? activeRally.shots : [];
  const current = Number(reviewVideo.currentTime || 0);
  if (!shots.length) {
    shotCaption.textContent = `t=${fmt(current, 2)}s | no shot markers available for this rally`;
    return;
  }

  let nearest = null;
  let nearestDelta = Number.POSITIVE_INFINITY;
  for (const shot of shots) {
    const delta = Math.abs(Number(shot.timestamp || 0) - current);
    if (delta < nearestDelta) {
      nearestDelta = delta;
      nearest = shot;
    }
  }

  if (nearest && nearestDelta <= 0.35) {
    shotCaption.textContent = `${describeShot(nearest)} @ ${fmt(nearest.timestamp, 2)}s`;
  } else {
    shotCaption.textContent = `t=${fmt(current, 2)}s`;
  }
}

function renderShotMarkers(rally) {
  clearChildren(shotMarkers);
  const shots = Array.isArray(rally.shots) ? rally.shots : [];
  const start = Number(rally.start_time || 0);
  const duration = rallyDuration(rally);

  for (const shot of shots) {
    const ts = Number(shot.timestamp || 0);
    if (ts < start || ts > start + duration) {
      continue;
    }
    const ratio = clamp((ts - start) / duration, 0, 1);
    const marker = document.createElement("button");
    marker.type = "button";
    marker.className = `shot-marker ${String(shot.player || "unknown").toLowerCase()}`;
    marker.style.left = `${ratio * 100}%`;
    marker.title = `${describeShot(shot)} @ ${fmt(ts, 2)}s`;
    marker.addEventListener("click", (event) => {
      event.stopPropagation();
      reviewVideo.currentTime = ts;
      drawReviewFrame();
      updateShotCaption();
    });
    shotMarkers.appendChild(marker);
  }
}

function highlightActiveRallyCard() {
  const selectedId = activeRally ? String(activeRally.rally_id) : "";
  for (const node of rallyList.querySelectorAll(".rally-card")) {
    node.classList.toggle("active", node.dataset.rallyId === selectedId);
  }
}

function syncBoundaryInputs(rally) {
  if (!boundaryStartInput || !boundaryEndInput || !rally) {
    return;
  }
  boundaryStartInput.value = fmt(rally.start_time, 1);
  boundaryEndInput.value = fmt(rally.end_time, 1);
  if (boundaryNote) {
    boundaryNote.textContent =
      "Preview only. Shot detection is not recomputed until saved boundary corrections are added.";
  }
}

function selectRally(rally) {
  activeRally = cloneRallyForReview(rally);
  selectedOriginalBounds = activeRally
    ? {
        start_time: Number(activeRally.start_time || 0),
        end_time: Number(activeRally.end_time || 0),
        duration_sec: Number(activeRally.duration_sec || 0),
      }
    : null;
  highlightActiveRallyCard();
  updateReviewMeta();
  renderShotMarkers(activeRally);
  renderSelectedRallyDetails(activeRally);
  syncBoundaryInputs(activeRally);

  const start = Number(activeRally?.start_time || 0);
  rallyScrubber.value = "0";
  if (reviewVideo.readyState >= 1) {
    reviewVideo.currentTime = start;
    drawReviewFrame();
    updateShotCaption();
  }
}

function previewBoundaryBounds() {
  if (!activeRally || !boundaryStartInput || !boundaryEndInput) {
    return;
  }
  const start = parsePositiveNumber(boundaryStartInput.value, null);
  const end = parsePositiveNumber(boundaryEndInput.value, null);
  if (start === null || end === null || end <= start) {
    if (boundaryNote) {
      boundaryNote.textContent = "Enter a valid end time greater than the start time.";
    }
    return;
  }

  activeRally = {
    ...activeRally,
    start_time: start,
    end_time: end,
    duration_sec: end - start,
  };
  updateReviewMeta();
  renderShotMarkers(activeRally);
  renderSelectedRallyDetails(activeRally);
  rallyScrubber.value = "0";
  if (reviewVideo.readyState >= 1) {
    reviewVideo.currentTime = start;
    drawReviewFrame();
  }
  updateShotCaption();
  if (boundaryNote) {
    boundaryNote.textContent =
      "Previewing adjusted bounds locally. Rally cards and coaching metrics are unchanged.";
  }
}

function resetBoundaryBounds() {
  if (!activeRally || !selectedOriginalBounds) {
    return;
  }
  activeRally = {
    ...activeRally,
    start_time: selectedOriginalBounds.start_time,
    end_time: selectedOriginalBounds.end_time,
    duration_sec: selectedOriginalBounds.duration_sec,
  };
  syncBoundaryInputs(activeRally);
  updateReviewMeta();
  renderShotMarkers(activeRally);
  renderSelectedRallyDetails(activeRally);
  rallyScrubber.value = "0";
  if (reviewVideo.readyState >= 1) {
    reviewVideo.currentTime = Number(activeRally.start_time || 0);
    drawReviewFrame();
  }
  updateShotCaption();
}

function setupReviewPanel(data, rallies) {
  const sourceVideoUrl = data?.source_video_url;
  if (!sourceVideoUrl || !Array.isArray(rallies) || rallies.length === 0) {
    reviewPanel.classList.add("hidden");
    activeRally = null;
    selectedOriginalBounds = null;
    clearChildren(selectedRallyKpis);
    clearChildren(selectedRallyShots);
    stopReviewLoop();
    return;
  }

  reviewPanel.classList.remove("hidden");
  if (reviewVideo.getAttribute("data-src") !== sourceVideoUrl) {
    reviewVideo.setAttribute("data-src", sourceVideoUrl);
    reviewVideo.src = sourceVideoUrl;
    reviewVideo.load();
  }

  selectRally(rallies[0]);
}

async function submitAnalyze(event) {
  event.preventDefault();

  const mode = activeMode();
  const includeLlm = document.getElementById("include-llm").checked;
  const llmModel = document.getElementById("llm-model").value.trim() || "gpt-4.1-mini";
  const openaiApiKey = document.getElementById("openai-api-key").value.trim();
  const playerAName = (playerANameInput?.value || "").trim() || "Player A";
  const playerBName = (playerBNameInput?.value || "").trim() || "Player B";
  const maxRallies = document.getElementById("max-rallies").value.trim();
  const youtubeCacheDir = document.getElementById("youtube-cache").value.trim();
  const cvWorkersRaw = document.getElementById("cv-workers").value.trim();
  const analysisStartMinuteRaw = document.getElementById("analysis-start-minute").value.trim();
  const maxVideoMinutesRaw = document.getElementById("max-video-minutes").value.trim();

  const motionThreshold = Number(document.getElementById("motion-threshold").value || "0.018");
  const minRallySec = Number(document.getElementById("min-rally-sec").value || "4.0");
  const idleGapSec = Number(document.getElementById("idle-gap-sec").value || "1.2");
  const segmentFrameStep = Number(document.getElementById("segment-frame-step").value || "2");
  const trackingFrameStep = Number(document.getElementById("tracking-frame-step").value || "4");
  const cvWorkers = cvWorkersRaw ? Number(cvWorkersRaw) : null;
  const analysisStartMinute = analysisStartMinuteRaw ? Number(analysisStartMinuteRaw) : 0;
  const maxVideoMinutes = maxVideoMinutesRaw ? Number(maxVideoMinutesRaw) : null;

  if (mode === "upload" && (!fileInput.files || fileInput.files.length === 0)) {
    setLoading(false, "Select a video file to upload.");
    return;
  }

  if (mode !== "upload" && !sourceInput.value.trim()) {
    setLoading(false, "Enter a YouTube URL or local file path.");
    return;
  }

  if (!Number.isFinite(analysisStartMinute) || analysisStartMinute < 0) {
    setLoading(false, "Start minute must be zero or greater.");
    return;
  }

  if (maxVideoMinutes !== null && (!Number.isFinite(maxVideoMinutes) || maxVideoMinutes <= 0)) {
    setLoading(false, "Analyze duration must be blank or greater than zero.");
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
      body.append("player_a_name", playerAName);
      body.append("player_b_name", playerBName);
      if (openaiApiKey) {
        body.append("openai_api_key", openaiApiKey);
      }
      body.append("motion_threshold", String(motionThreshold));
      body.append("min_rally_sec", String(minRallySec));
      body.append("idle_gap_sec", String(idleGapSec));
      body.append("segment_frame_step", String(segmentFrameStep));
      body.append("tracking_frame_step", String(trackingFrameStep));
      body.append("analysis_start_minute", String(analysisStartMinute));
      if (cvWorkers !== null && Number.isFinite(cvWorkers) && cvWorkers > 0) {
        body.append("cv_workers", String(Math.round(cvWorkers)));
      }
      if (maxVideoMinutes !== null && Number.isFinite(maxVideoMinutes) && maxVideoMinutes > 0) {
        body.append("max_video_minutes", String(maxVideoMinutes));
      }
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
        player_a_name: playerAName,
        player_b_name: playerBName,
        motion_threshold: motionThreshold,
        min_rally_sec: minRallySec,
        idle_gap_sec: idleGapSec,
        segment_frame_step: segmentFrameStep,
        tracking_frame_step: trackingFrameStep,
        analysis_start_minute: analysisStartMinute,
      };
      if (cvWorkers !== null && Number.isFinite(cvWorkers) && cvWorkers > 0) {
        body.cv_workers = Math.round(cvWorkers);
      }
      if (maxVideoMinutes !== null && Number.isFinite(maxVideoMinutes) && maxVideoMinutes > 0) {
        body.max_video_minutes = maxVideoMinutes;
      }
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
    [`${playerNames.A} Avg T Recovery`, movement.A_avg_T_recovery_sec, "s"],
    [`${playerNames.B} Avg T Recovery`, movement.B_avg_T_recovery_sec, "s"],
    [`${playerNames.A} T Occupancy`, movement.A_T_occupancy, "%"],
    [`${playerNames.B} T Occupancy`, movement.B_T_occupancy, "%"],
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

function diagnosticValue(value, digits = 2) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : fmt(value, digits);
  }
  return String(value);
}

function renderDiagnostics(diagnostics) {
  if (!diagnosticsPanel || !diagnosticsCards || !segmentPreview) {
    return;
  }

  const segmentation = diagnostics?.segmentation || null;
  if (!segmentation) {
    diagnosticsPanel.classList.add("hidden");
    clearChildren(diagnosticsCards);
    clearChildren(segmentPreview);
    return;
  }

  diagnosticsPanel.classList.remove("hidden");
  clearChildren(diagnosticsCards);
  clearChildren(segmentPreview);

  const motion = segmentation.motion || {};
  const cards = [
    {
      label: "Analyzed Window",
      value: `${diagnosticValue(segmentation.window_start_sec, 1)}s -> ${diagnosticValue(
        segmentation.window_end_sec,
        1
      )}s`,
      detail: `${diagnosticValue(segmentation.window_duration_sec, 1)}s total`,
    },
    {
      label: "Final Rallies",
      value: diagnosticValue(segmentation.final_segment_count, 0),
      detail: `${diagnosticValue(segmentation.base_segment_count, 0)} base candidates`,
    },
    {
      label: "Threshold",
      value: diagnosticValue(segmentation.selected_threshold, 4),
      detail: segmentation.adaptive_used ? "adaptive threshold used" : "base threshold used",
    },
    {
      label: "Motion Samples",
      value: diagnosticValue(motion.sample_count, 0),
      detail: `${fmtPct(motion.active_sample_rate || 0)} active`,
    },
    {
      label: "Frame Step",
      value: diagnosticValue(segmentation.frame_step, 0),
      detail: "higher is faster, lower is denser",
    },
    {
      label: "Fallback",
      value: segmentation.fallback_full_window_used ? "Used" : "No",
      detail: segmentation.fallback_full_window_used
        ? "no confident breaks found"
        : "boundaries came from motion",
    },
  ];

  for (const card of cards) {
    const el = document.createElement("article");
    el.className = "diagnostic-card";
    el.innerHTML = `
      <p class="diagnostic-label">${escapeHtml(card.label)}</p>
      <p class="diagnostic-value">${escapeHtml(card.value)}</p>
      <p class="diagnostic-detail">${escapeHtml(card.detail)}</p>
    `;
    diagnosticsCards.appendChild(el);
  }

  const finalSegments = Array.isArray(segmentation.final_segments)
    ? segmentation.final_segments.slice(0, 40)
    : [];
  if (!finalSegments.length) {
    segmentPreview.textContent = "No final segment candidates were emitted.";
    return;
  }

  for (const [idx, segment] of finalSegments.entries()) {
    const chip = document.createElement("span");
    chip.className = "segment-chip";
    chip.textContent = `R${idx + 1}: ${fmt(segment.start_sec, 1)}s -> ${fmt(
      segment.end_sec,
      1
    )}s`;
    segmentPreview.appendChild(chip);
  }
}

function renderRallies(rallies, options = {}) {
  clearChildren(rallyList);

  if (!rallies.length) {
    const empty = document.createElement("p");
    const message =
      typeof options.emptyMessage === "string" && options.emptyMessage.trim()
        ? options.emptyMessage.trim()
        : "No rally segments were detected from this video.";
    empty.textContent = message;
    empty.className = "rally-empty";
    rallyList.appendChild(empty);
    return;
  }

  for (const [index, rally] of rallies.entries()) {
    const node = rallyTemplate.content.firstElementChild.cloneNode(true);
    node.classList.add(`burst-${index % 3}`);
    node.dataset.rallyId = String(rally.rally_id);

    const start = fmt(rally.start_time, 1);
    const end = fmt(rally.end_time, 1);

    const rankingTag = `<span class="rally-rank">#${index + 1}</span> `;
    node.querySelector(".rally-title").innerHTML = `${rankingTag}Rally ${escapeHtml(String(rally.rally_id))}`;
    node.querySelector(".rally-meta").textContent = `${start}s -> ${end}s`;
    node.querySelector(".rally-outcome").textContent = formatOutcomeText(rally.outcome);

    const strip = node.querySelector(".shot-strip");
    const shots = Array.isArray(rally.shots) ? rally.shots : [];
    const shotSummary = summarizeShotTypes(shots);
    if (!shotSummary.length) {
      const noShots = document.createElement("span");
      noShots.className = "shot-pill";
      noShots.textContent = "No shot profile";
      strip.appendChild(noShots);
    } else {
      for (const item of shotSummary) {
        const pill = document.createElement("span");
        pill.className = "shot-pill summary";
        pill.textContent = item;
        strip.appendChild(pill);
      }
    }

    const metrics = node.querySelector(".rally-metrics");
    const tags = [
      `${fmt(rally.duration_sec, 1)}s`,
      `${rallyShotsCount(rally)} shots`,
      `${playerNames.A} T ${rally.positions?.A_T_occupancy !== null && rally.positions?.A_T_occupancy !== undefined ? fmtPct(rally.positions.A_T_occupancy) : "n/a"}`,
      `${playerNames.B} T ${rally.positions?.B_T_occupancy !== null && rally.positions?.B_T_occupancy !== undefined ? fmtPct(rally.positions.B_T_occupancy) : "n/a"}`,
    ];
    for (const tagText of tags) {
      const tag = document.createElement("span");
      tag.className = "rally-tag";
      tag.textContent = tagText;
      metrics.appendChild(tag);
    }

    node.addEventListener("click", () => {
      selectRally(rally);
    });

    rallyList.appendChild(node);
  }

  highlightActiveRallyCard();
}

function renderInsight(insight) {
  if (!insight) {
    insightPanel.classList.add("hidden");
    return;
  }

  insightPanel.classList.remove("hidden");
  const confidence = insight.confidence || "n/a";
  insightModel.textContent = `${insight.model || "model-unknown"} | confidence: ${confidence}`;

  clearChildren(insightSummaryCards);
  const summaryCards = [
    { label: "Confidence", value: confidence },
    { label: "Pattern Count", value: String((insight.key_patterns || []).length) },
    { label: "Drill Count", value: String((insight.drills || []).length) },
  ];
  for (const card of summaryCards) {
    const el = document.createElement("div");
    el.className = "insight-summary-card";
    el.innerHTML = `
      <p class="insight-summary-label">${escapeHtml(card.label)}</p>
      <p class="insight-summary-value">${escapeHtml(card.value)}</p>
    `;
    insightSummaryCards.appendChild(el);
  }

  renderReportCards(withNamedPlayers(insight.report_markdown || ""));

  clearChildren(patternList);
  const patterns = Array.isArray(insight.key_patterns) ? insight.key_patterns : [];
  if (!patterns.length) {
    const card = document.createElement("article");
    card.className = "insight-item-card empty";
    card.textContent = "No recurring patterns returned.";
    patternList.appendChild(card);
  } else {
    for (const [idx, item] of patterns.entries()) {
      const normalized = withNamedPlayers(normalizeInsightItem(item));
      const card = document.createElement("article");
      card.className = "insight-item-card pattern";
      card.innerHTML = `
        <div class="insight-item-copy">
          <p class="insight-item-index">Pattern ${idx + 1}</p>
          <p class="insight-item-text">${escapeHtml(normalized)}</p>
        </div>
      `;
      patternList.appendChild(card);
    }
  }

  clearChildren(drillList);
  const drills = Array.isArray(insight.drills) ? insight.drills : [];
  if (!drills.length) {
    const card = document.createElement("article");
    card.className = "insight-item-card empty";
    card.textContent = "No drills returned.";
    drillList.appendChild(card);
  } else {
    for (const [idx, item] of drills.entries()) {
      const normalized = withNamedPlayers(normalizeInsightItem(item));
      const drillType = classifyDrillType(normalized);
      const card = document.createElement("article");
      card.className = "insight-item-card";
      card.innerHTML = `
        <span class="drill-icon ${escapeHtml(drillType)}">${drillIconSvg(drillType)}</span>
        <div class="insight-item-copy">
          <p class="insight-item-index">Drill ${idx + 1}</p>
          <p class="insight-item-text">${escapeHtml(normalized)}</p>
        </div>
      `;
      drillList.appendChild(card);
    }
  }
}

function renderAll(data) {
  currentResult = data;
  updatePlayerNamesFromInputs();
  refreshWinnerFilterLabels();
  const timeline = data.timeline || { rallies: [], tactical_patterns: {}, movement_summary: {} };
  const rallies = Array.isArray(timeline.rallies) ? timeline.rallies : [];
  const tactical = timeline.tactical_patterns || {};
  const movement = timeline.movement_summary || {};

  sourceLabel.textContent = timeline.video_path || "Source unavailable";
  renderKpis(timeline);
  renderTacticalBars(tactical);
  renderMovementCards(movement);
  buildRallyChart(rallies);
  renderDiagnostics(timeline.diagnostics || {});
  allRallies = [...rallies];
  if (winnerFilter) {
    winnerFilter.value = "all";
  }
  if (durationMinInput) {
    durationMinInput.value = "";
  }
  if (durationMaxInput) {
    durationMaxInput.value = "";
  }
  if (shotsMinInput) {
    shotsMinInput.value = "";
  }
  if (shotsMaxInput) {
    shotsMaxInput.value = "";
  }
  if (rallySort) {
    rallySort.value = "match";
  }
  applyRallySortRender();
  if (rallyPicker) {
    rallyPicker.value = "";
    rallyPicker.min = "1";
    rallyPicker.max = String(rallies.length || 1);
  }
  setupReviewPanel(data, displayedRallies);
  renderInsight(data.insight || null);

  rawJson.textContent = JSON.stringify(data, null, 2);
  resultsEl.classList.remove("hidden");
}

rallyScrubber.addEventListener("input", () => {
  if (!activeRally) {
    return;
  }
  const start = Number(activeRally.start_time || 0);
  const duration = rallyDuration(activeRally);
  const ratio = Number(rallyScrubber.value || 0) / 1000;
  const target = start + ratio * duration;
  reviewTimeSync = true;
  reviewVideo.currentTime = target;
  drawReviewFrame();
  updateShotCaption();
  reviewTimeSync = false;
});

reviewVideo.addEventListener("loadedmetadata", () => {
  if (activeRally) {
    reviewVideo.currentTime = Number(activeRally.start_time || 0);
  }
  drawReviewFrame();
  updateShotCaption();
});

reviewVideo.addEventListener("play", () => {
  stopReviewLoop();
  drawReviewFrame();
});

reviewVideo.addEventListener("pause", () => {
  stopReviewLoop();
  drawReviewFrame();
});

reviewVideo.addEventListener("seeked", () => {
  drawReviewFrame();
  updateShotCaption();
});

reviewVideo.addEventListener("timeupdate", () => {
  if (!activeRally) {
    return;
  }
  const start = Number(activeRally.start_time || 0);
  const end = Number(activeRally.end_time || 0);
  const current = Number(reviewVideo.currentTime || 0);

  if (current > end) {
    reviewVideo.pause();
    reviewVideo.currentTime = end;
  }

  if (!reviewTimeSync) {
    const ratio = clamp((current - start) / rallyDuration(activeRally), 0, 1);
    rallyScrubber.value = String(Math.round(ratio * 1000));
  }
  updateShotCaption();
});

for (const radio of modeRadios) {
  radio.addEventListener("change", () => {
    setMode(activeMode());
  });
}

if (rallySort) {
  rallySort.addEventListener("change", () => {
    applyRallySortRender();
    if (displayedRallies.length > 0) {
      selectRally(displayedRallies[0]);
    }
  });
}

function onRallyFilterChange() {
  applyRallySortRender();
  if (displayedRallies.length > 0) {
    selectRally(displayedRallies[0]);
  }
}

for (const input of [winnerFilter, durationMinInput, durationMaxInput, shotsMinInput, shotsMaxInput]) {
  if (!input) {
    continue;
  }
  input.addEventListener("change", onRallyFilterChange);
}

if (boundaryApplyBtn) {
  boundaryApplyBtn.addEventListener("click", previewBoundaryBounds);
}

if (boundaryResetBtn) {
  boundaryResetBtn.addEventListener("click", resetBoundaryBounds);
}

for (const input of [durationMinInput, durationMaxInput, shotsMinInput, shotsMaxInput]) {
  if (!input) {
    continue;
  }
  input.addEventListener("input", onRallyFilterChange);
}

function submitRallyPick() {
  const value = Number(rallyPicker?.value || 0);
  if (!value) {
    rallyToolbarStatus.textContent = "Enter a rally number to jump.";
    return;
  }
  jumpToRallyById(value);
}

if (rallyPickerBtn) {
  rallyPickerBtn.addEventListener("click", submitRallyPick);
}

if (rallyPicker) {
  rallyPicker.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      submitRallyPick();
    }
  });
}

function rerenderWithCurrentResult() {
  if (!currentResult) {
    return;
  }
  renderAll(currentResult);
}

if (playerANameInput) {
  playerANameInput.addEventListener("change", rerenderWithCurrentResult);
}
if (playerBNameInput) {
  playerBNameInput.addEventListener("change", rerenderWithCurrentResult);
}

setMode(activeMode());
form.addEventListener("submit", submitAnalyze);
