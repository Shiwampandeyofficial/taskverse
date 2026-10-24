/* StatVerse front-end */
const $ = (id) => document.getElementById(id);

/* ───────── Markdown + LaTeX rendering ─────────
   1. pull every math expression out (so markdown can't mangle \( \) _ * etc.)
   2. render markdown  3. sanitize  4. put KaTeX-rendered math back */
function renderRich(md) {
  const store = [];
  const protectedText = md.replace(
    /\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|\$([^$\n]+?)\$/g,
    (m, a, b, c, d) => {
      const display = a !== undefined || b !== undefined;
      store.push({ tex: (a ?? b ?? c ?? d).trim(), display });
      return `@@M${store.length - 1}@@`;
    }
  );
  const html = DOMPurify.sanitize(marked.parse(protectedText, { breaks: false, gfm: true }));
  return html.replace(/@@M(\d+)@@/g, (_, i) => {
    const { tex, display } = store[+i];
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
    } catch (e) {
      return tex.replace(/</g, "&lt;");
    }
  });
}

/* ───────── Tabs ───────── */
document.querySelectorAll(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
    if (t.dataset.tab === "library") loadLibrary();
    window.scrollTo({ top: 0, behavior: "smooth" });
  })
);

/* ───────── Segmented controls (remember choice) ───────── */
function seg(id, key, def) {
  const el = $(id);
  let val = localStorage.getItem(key) || def;
  const paint = () => el.querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.v === val));
  el.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => { val = b.dataset.v; localStorage.setItem(key, val); paint(); })
  );
  paint();
  return () => val;
}
const getLang = seg("langSeg", "sv_lang", "hinglish");
const getLevel = seg("levelSeg", "sv_level", "basic");

/* ───────── Suggestion chips ───────── */
$("chips").addEventListener("click", (e) => {
  if (e.target.tagName === "BUTTON") { $("topicInput").value = e.target.textContent; $("topicInput").focus(); }
});

/* ───────── Ask a topic ───────── */
const SECTION_ICONS = [
  ["concept", "💡"], ["example", "✏️"], ["case", "🏭"], ["graph", "📈"], ["chart", "📈"],
  ["advantage", "✅"], ["disadvantage", "⚠️"], ["limitation", "⚠️"], ["exam", "🎯"],
];
function iconFor(title) {
  const t = title.toLowerCase();
  // "disadvantage" contains "advantage" — check the longer word first
  const order = [["disadvantage", "⚠️"], ["limitation", "⚠️"], ...SECTION_ICONS];
  for (const [k, i] of order) if (t.includes(k)) return i;
  return "📘";
}

function splitSections(text) {
  const parts = text.split(/^##\s+/m).map((s) => s.trim()).filter(Boolean);
  if (!/^##\s+/m.test(text)) return [{ title: "Explanation", body: text }];
  return parts.map((p) => {
    const nl = p.indexOf("\n");
    const title = (nl === -1 ? p : p.slice(0, nl)).replace(/^\d+[.)]\s*/, "").replace(/\*\*/g, "").trim();
    return { title, body: nl === -1 ? "" : p.slice(nl + 1).trim() };
  });
}


/* ───────── Charts (Chart.js draws what the server computed) ───────── */
function buildChartConfig(spec, th) {
  const o = spec.options || {};
  const base = { responsive: true, maintainAspectRatio: false, animation: { duration: 600 }, interaction: { mode: "nearest", intersect: false } };
  const title = { display: true, text: spec.title, color: th.ink, font: { size: 14, weight: "700" }, padding: { bottom: 10 } };
  const legendOn = o.legend !== false && (spec.datasets.length > 1 || o.legend === true);
  const legend = {
    display: legendOn, position: "bottom",
    labels: { color: th.muted, usePointStyle: true, boxWidth: 8, font: { size: 11 }, filter: (it) => !it.text.startsWith("_") },
  };

  if (spec.type === "pie" || spec.type === "doughnut") {
    const { interaction, ...pieBase } = base;   // "nearest" interaction is for axis charts only
    return { type: spec.type, data: { labels: spec.labels, datasets: spec.datasets },
             options: { ...pieBase, plugins: { title, legend: { ...legend, display: true } } } };
  }

  const axis = (label) => ({
    title: { display: !!label, text: label, color: th.muted, font: { size: 12, weight: "600" } },
    ticks: { color: th.muted, maxTicksLimit: 9 }, grid: { color: th.grid },
  });
  const scales = { x: axis(spec.xlabel), y: axis(spec.ylabel) };
  if (spec.type === "scatter") scales.x.type = "linear";
  if (o.xmin !== undefined) scales.x.min = o.xmin;
  if (o.xmax !== undefined) scales.x.max = o.xmax;
  if (o.ymin !== undefined) scales.y.min = o.ymin;
  if (o.ymax !== undefined) scales.y.max = o.ymax;
  if (o.xticklabels) {
    scales.x.ticks = { ...scales.x.ticks, callback: (v) => o.xticklabels[Math.round(v) - 1] || "" };
    scales.x.afterBuildTicks = (ax) => { ax.ticks = o.xticklabels.map((_, i) => ({ value: i + 1 })); };
  }
  if (spec.type === "bar" || spec.type === "line") {
    if (!o.indexAxis) scales.x.ticks = { ...scales.x.ticks, maxTicksLimit: spec.labels && spec.labels.length > 30 ? 12 : 14 };
  }

  const vlinePlugin = {
    id: "vline",
    afterDatasetsDraw(chart) {
      if (o.vline === undefined) return;
      const { ctx, chartArea, scales: sc } = chart;
      const x = sc.x.getPixelForValue(o.vline);
      ctx.save(); ctx.setLineDash([6, 4]); ctx.strokeStyle = "#e5484d"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke(); ctx.restore();
    },
  };

  if (o.indexAxis === "y") scales.y.ticks = { ...scales.y.ticks, maxTicksLimit: 30, autoSkip: false };
  const options = { ...base, scales, plugins: { title, legend } };
  if (o.indexAxis) options.indexAxis = o.indexAxis;
  if (o.grouped === false) options.datasets = { bar: { grouped: false } };
  return { type: spec.type, data: { labels: spec.labels || undefined, datasets: spec.datasets }, options, plugins: [vlinePlugin] };
}

let liveCharts = [];
function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return { ink: g("--ink"), muted: g("--muted"), grid: g("--line") };
}
function mountCharts(container, charts) {
  const th = themeColors();
  Chart.defaults.font.family = "'Inter','Noto Sans Devanagari',system-ui,sans-serif";
  charts.forEach((spec) => {
    const fig = document.createElement("figure");
    fig.className = "chart";
    fig.innerHTML = `<div class="chart-wrap"><canvas></canvas></div><figcaption></figcaption>`;
    fig.querySelector("figcaption").textContent = spec.caption || "";
    container.appendChild(fig);
    try {
      liveCharts.push(new Chart(fig.querySelector("canvas"), buildChartConfig(spec, th)));
    } catch (err) {
      fig.querySelector(".chart-wrap").innerHTML = `<p class="hint">Couldn’t draw this chart (${err.message}).</p>`;
    }
  });
}

let lastPlainText = "";

$("queryForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const topic = $("topicInput").value.trim();
  if (!topic) return;

  const btn = $("askBtn");
  btn.disabled = true;
  btn.querySelector(".spinner").hidden = false;
  btn.querySelector(".btn-text").textContent = "Preparing…";
  $("errorBox").hidden = true;
  $("result").hidden = true;
  $("skeleton").hidden = false;

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ topic, language: getLang(), level: getLevel() }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Something went wrong.");
    showResult(topic, data);
  } catch (err) {
    $("errorBox").textContent = err.message;
    $("errorBox").hidden = false;
  } finally {
    $("skeleton").hidden = true;
    btn.disabled = false;
    btn.querySelector(".spinner").hidden = true;
    btn.querySelector(".btn-text").textContent = "Explain this topic";
  }
});

function showResult(topic, data) {
  $("resultTitle").textContent = topic;
  const fromBook = data.source.includes("reference") || data.source.includes("mixed");
  const label = data.source.includes("mixed") ? "Books + general knowledge"
              : fromBook ? "From your books" : "General knowledge";
  $("resultMeta").innerHTML =
    `<span class="pill ${fromBook ? "book" : "gen"}">${label}</span>` +
    `<span class="pill">${{ english: "English", hindi: "हिन्दी", hinglish: "Hinglish" }[getLang()]}</span>` +
    `<span class="pill">${getLevel() === "advanced" ? "MSc level" : "BSc level"}</span>`;

  const box = $("sections");
  box.innerHTML = "";
  splitSections(data.explanation).forEach((s, i) => {
    const el = document.createElement("article");
    el.className = `sec c${i % 7}`;
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML =
      `<div class="sec-head"><div class="sec-ico">${iconFor(s.title)}</div><h3></h3></div>` +
      `<div class="prose">${renderRich(s.body)}</div>`;
    el.querySelector("h3").textContent = s.title;
    box.appendChild(el);
  });

  liveCharts.forEach((c) => c.destroy());
  liveCharts = [];
  if (data.charts && data.charts.length && typeof Chart !== "undefined") {
    const secs = [...box.querySelectorAll(".sec")];
    const target = secs.find((el) => /graph|chart/i.test(el.querySelector("h3").textContent)) || secs[secs.length - 1];
    mountCharts(target || box, data.charts);
  }

  const refs = $("refs");
  if (data.references && data.references.length) {
    $("refsBody").innerHTML = data.references
      .map((r) => `<div class="ref"><b></b><div></div></div>`).join("");
    [...$("refsBody").children].forEach((node, i) => {
      const r = data.references[i];
      node.querySelector("b").textContent = r.book + (r.page ? ` · page ${r.page}` : "");
      node.querySelector("div").textContent = "“" + r.snippet + "…”";
    });
    refs.hidden = false;
  } else refs.hidden = true;

  lastPlainText = topic + "\n\n" + data.explanation;
  $("result").hidden = false;
  try { $("result").scrollIntoView({ behavior: "smooth", block: "start" }); } catch {}
}

$("copyBtn").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(lastPlainText); $("copyBtn").textContent = "Copied ✓"; }
  catch { $("copyBtn").textContent = "Copy failed"; }
  setTimeout(() => ($("copyBtn").textContent = "Copy"), 1600);
});

/* ───────── Library / upload ───────── */
const fileInput = $("bookFile");
function fmtSize(b) { return b > 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB"; }

fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  $("drop").classList.toggle("has", !!f);
  $("uploadBtn").disabled = !f;
  $("dropTitle").textContent = f ? f.name : "Tap to choose a file";
  $("dropSub").textContent = f ? fmtSize(f.size) : "PDF or TXT · up to 300 MB";
});

function setProgress(pct, msg) {
  $("progressBox").hidden = false;
  $("barFill").style.width = pct + "%";
  $("progPct").textContent = Math.round(pct) + "%";
  if (msg) $("progMsg").textContent = msg;
}
function uploadAlert(kind, msg) {
  const a = $("uploadAlert");
  a.className = "alert " + kind; a.textContent = msg; a.hidden = !msg;
}

$("uploadForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const f = fileInput.files[0];
  if (!f) return;
  uploadAlert("", "");
  $("uploadBtn").disabled = true;
  setProgress(0, "Uploading file…");

  const fd = new FormData();
  fd.append("book_file", f);
  fd.append("topic_tag", $("topicTag").value);

  // XHR so we get real upload progress on slow mobile networks
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = (ev) => {
    if (ev.lengthComputable) setProgress((ev.loaded / ev.total) * 25, `Uploading file… ${fmtSize(ev.loaded)} / ${fmtSize(ev.total)}`);
  };
  xhr.onerror = () => { uploadAlert("error", "Upload failed — check your connection and try again."); $("uploadBtn").disabled = false; };
  xhr.onload = () => {
    let data = {};
    try { data = JSON.parse(xhr.responseText); } catch {}
    if (!data.ok) { uploadAlert("error", data.error || `Upload failed (HTTP ${xhr.status}).`); $("uploadBtn").disabled = false; return; }
    pollJob(data.job_id);
  };
  xhr.send(fd);
});

async function pollJob(id) {
  try {
    const r = await fetch("/api/job/" + id);
    const j = await r.json();
    if (!j.ok) throw new Error(j.error);
    // map server progress (0-100) into the remaining 25-100 part of the bar
    setProgress(25 + j.progress * 0.75, j.message);
    if (j.status === "running") return setTimeout(() => pollJob(id), 900);
    if (j.status === "done") {
      setProgress(100, j.message);
      uploadAlert("ok", `✓ “${j.book}” is ready — ${j.chunks} sections indexed. Go to Learn and ask a topic!`);
      fileInput.value = ""; $("drop").classList.remove("has");
      $("dropTitle").textContent = "Tap to choose a file"; $("dropSub").textContent = "PDF or TXT · up to 300 MB";
      $("topicTag").value = "";
      loadLibrary();
    } else {
      uploadAlert("error", j.error || "Indexing failed.");
      $("progressBox").hidden = true;
    }
  } catch (err) {
    uploadAlert("error", "Lost connection while indexing: " + err.message + ". Reopen the Library tab to check progress.");
  }
  $("uploadBtn").disabled = !fileInput.files[0];
}

async function loadLibrary() {
  try {
    const r = await fetch("/api/library");
    const d = await r.json();
    if (!d.ok) throw new Error(d.error);
    const badge = $("libBadge");
    badge.hidden = d.books.length === 0; badge.textContent = d.books.length;
    $("libSummary").textContent = d.books.length
      ? `${d.books.length} book${d.books.length > 1 ? "s" : ""} · ${d.total_chunks} searchable sections`
      : "No books yet. Add your first one above — explanations will then be grounded in it.";
    const list = $("bookList");
    list.innerHTML = "";
    d.books.forEach((b) => {
      const row = document.createElement("div");
      row.className = "book";
      row.innerHTML = `<div class="book-ico">📗</div><div class="book-info"><b></b><span></span></div><button class="del">Remove</button>`;
      row.querySelector("b").textContent = b.title;
      row.querySelector("span").textContent = `${b.chunks} sections`;
      row.querySelector(".del").addEventListener("click", async () => {
        if (!confirm(`Remove “${b.title}” from the library?`)) return;
        await fetch("/api/library/" + encodeURIComponent(b.title), { method: "DELETE" });
        loadLibrary();
      });
      list.appendChild(row);
    });
  } catch (err) {
    $("libSummary").textContent = "Couldn’t load library: " + err.message;
  }
}
loadLibrary();

/* ═══════════════════════ Analyze tab (Industry Analysis Engine) ═══════════════════════ */
const getAnalyzeLang = seg("analyzeLangSeg", "sv_alang", "english");
let currentDatasetId = null;
let analysesCatalog = null;

$("dropDs").addEventListener("click", (e) => e.preventDefault());
$("datasetFile").addEventListener("change", () => {
  const f = $("datasetFile").files[0];
  if (!f) return;
  $("dropDsTitle").textContent = f.name;
  $("dropDsSub").textContent = fmtSize(f.size);
  uploadDataset(f);
});

function dsAlert(kind, msg) {
  const a = $("dsAlert");
  a.className = "alert " + kind; a.textContent = msg; a.hidden = !msg;
}

async function uploadDataset(file) {
  dsAlert("", "");
  $("dsProfile").hidden = true;
  $("objectiveCard").hidden = true;
  $("analyzeResult").hidden = true;
  const fd = new FormData();
  fd.append("dataset_file", file);
  try {
    const res = await fetch("/api/dataset/upload", { method: "POST", body: fd });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Upload failed.");
    currentDatasetId = data.dataset_id;
    renderProfile(data.name, data.profile);
    dsAlert("ok", `✓ "${data.name}" loaded — describe your objective below.`);
    $("objectiveCard").hidden = false;
    $("objectiveInput").focus();
  } catch (err) {
    dsAlert("error", err.message);
  }
}

function renderProfile(name, profile) {
  $("dsSummary").textContent = `${name} — ${profile.n_rows} rows × ${profile.n_cols} columns`;
  const box = $("dsCols");
  box.innerHTML = "";
  profile.columns.forEach((c) => {
    const el = document.createElement("div");
    el.className = "ds-col";
    if (c.type === "numeric") {
      el.innerHTML = `<b></b><span></span>`;
      el.querySelector("b").append(c.name, Object.assign(document.createElement("span"), { className: "tag", textContent: "numeric" }));
      el.querySelector("span:last-child").textContent = `mean ${c.mean ?? "–"} · sd ${c.sd ?? "–"} · n ${c.n}` + (c.missing ? ` · ${c.missing} missing` : "");
    } else {
      el.innerHTML = `<b></b><span></span>`;
      el.querySelector("b").append(c.name, Object.assign(document.createElement("span"), { className: "tag", textContent: "category" }));
      const top = (c.top_categories || []).slice(0, 3).map((t) => `${t.value} (${t.count})`).join(", ");
      el.querySelector("span:last-child").textContent = `${c.distinct} distinct · ${top}` + (c.missing ? ` · ${c.missing} missing` : "");
    }
    box.appendChild(el);
  });
  $("dsProfile").hidden = false;
}

$("pickManualBtn").addEventListener("click", async () => {
  const box = $("manualPicker");
  if (!box.hidden) { box.hidden = true; return; }
  if (!analysesCatalog) {
    try {
      const r = await fetch("/api/analyses");
      const d = await r.json();
      analysesCatalog = d.help.split("\n").map((line) => {
        const m = line.match(/^(\w+)\{[^}]*\}\s*(?:—|-)\s*(.*)$/);
        return m ? { name: m[1], desc: m[2] } : null;
      }).filter(Boolean);
    } catch { analysesCatalog = []; }
  }
  box.innerHTML = "";
  analysesCatalog.forEach((a) => {
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<b></b><br><span style="color:var(--muted);font-size:.8rem"></span>`;
    b.querySelector("b").textContent = a.name;
    b.querySelector("span").textContent = a.desc;
    b.addEventListener("click", () => runAnalysis({ analysis: a.name }));
    box.appendChild(b);
  });
  box.hidden = false;
});

$("analyzeBtn").addEventListener("click", () => {
  const objective = $("objectiveInput").value.trim();
  if (!objective) { $("objectiveInput").focus(); return; }
  runAnalysis({ objective });
});

function stepLine(label, detail) {
  return `<div class="step-row"><div class="step-num"></div><div class="step-body"><b></b><span></span></div></div>`;
}

async function runAnalysis(payload) {
  if (!currentDatasetId) return;
  const btn = $("analyzeBtn");
  btn.disabled = true;
  btn.querySelector(".spinner").hidden = false;
  $("manualPicker").hidden = true;
  $("analyzeError").hidden = true;
  $("analyzeResult").hidden = true;
  $("analyzeSkeleton").hidden = false;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: currentDatasetId, language: getAnalyzeLang(), objective: $("objectiveInput").value.trim(), ...payload }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Analysis failed.");
    showAnalysisResult(data);
  } catch (err) {
    $("analyzeError").textContent = err.message;
    $("analyzeError").hidden = false;
  } finally {
    $("analyzeSkeleton").hidden = true;
    btn.disabled = false;
    btn.querySelector(".spinner").hidden = true;
  }
}

let analyzeCharts = [];

function showAnalysisResult(data) {
  $("planTitle").textContent = data.title;
  $("planReasoning").textContent = data.plan && data.plan.reasoning ? data.plan.reasoning : "";
  const rel = data.reliability;
  const pill = $("reliabilityPill");
  pill.className = "pill rel-" + rel.level.toLowerCase();
  pill.textContent = `Reliability: ${rel.level} (${rel.score}/100)`;

  const fb = $("fallbackNote");
  if (data.fallback_note) { fb.textContent = data.fallback_note; fb.hidden = false; } else fb.hidden = true;

  const box = $("analyzeSections");
  box.innerHTML = "";
  splitSections(data.narrative).forEach((s, i) => {
    const el = document.createElement("article");
    el.className = `sec c${i % 7}`;
    el.style.animationDelay = `${i * 60}ms`;
    el.innerHTML = `<div class="sec-head"><div class="sec-ico">${iconFor(s.title)}</div><h3></h3></div><div class="prose">${renderRich(s.body)}</div>`;
    el.querySelector("h3").textContent = s.title;
    box.appendChild(el);
  });

  analyzeCharts.forEach((c) => c.destroy());
  analyzeCharts = [];
  if (data.chart && typeof Chart !== "undefined") {
    const host = document.createElement("div");
    host.className = "sec c1";
    host.innerHTML = `<div class="sec-head"><div class="sec-ico">📈</div><h3>Chart</h3></div>`;
    box.appendChild(host);
    const before = liveCharts.length;
    mountCharts(host, [data.chart]);
    analyzeCharts = liveCharts.slice(before);
  }

  if (data.table && Array.isArray(data.table)) {
    const host = document.createElement("div");
    host.className = "sec c3";
    host.innerHTML = `<div class="sec-head"><div class="sec-ico">📋</div><h3>Column summary</h3></div><div class="prose" style="overflow-x:auto"></div>`;
    host.querySelector(".prose").appendChild(buildTable(data.table));
    box.appendChild(host);
  } else if (data.table && data.table.rows) {
    const host = document.createElement("div");
    host.className = "sec c3";
    host.innerHTML = `<div class="sec-head"><div class="sec-ico">📋</div><h3>Contingency table</h3></div><div class="prose" style="overflow-x:auto"></div>`;
    host.querySelector(".prose").appendChild(buildCrossTable(data.table));
    box.appendChild(host);
  }

  const stepsBody = $("stepsBody");
  stepsBody.innerHTML = "";
  (data.steps || []).forEach((s, i) => {
    const row = document.createElement("div");
    row.className = "step-row";
    row.innerHTML = `<div class="step-num"></div><div class="step-body"><b></b><span></span></div>`;
    row.querySelector(".step-num").textContent = i + 1;
    row.querySelector("b").textContent = s.label;
    row.querySelector("span").textContent = s.detail;
    stepsBody.appendChild(row);
  });
  $("stepsDetails").open = false;

  const relBody = $("reliabilityBody");
  relBody.innerHTML = "";
  rel.factors.forEach((f) => {
    const row = document.createElement("div");
    row.className = "rel-factor";
    row.innerHTML = `<div class="rel-dot"></div><div><b></b><span></span></div>`;
    row.querySelector(".rel-dot").classList.add(f.status);
    row.querySelector("b").textContent = f.name;
    row.querySelector("span").textContent = f.note;
    relBody.appendChild(row);
  });
  $("reliabilityDetails").open = false;

  $("analyzeResult").hidden = false;
  try { $("analyzeResult").scrollIntoView({ behavior: "smooth", block: "start" }); } catch {}
}

function buildTable(cols) {
  const t = document.createElement("table");
  t.className = "result-table";
  t.innerHTML = "<thead><tr><th>Column</th><th>Type</th><th>n</th><th>Mean / Top value</th><th>Missing</th></tr></thead>";
  const tb = document.createElement("tbody");
  cols.forEach((c) => {
    const tr = document.createElement("tr");
    const mainVal = c.type === "numeric" ? `mean ${c.mean ?? "–"}, sd ${c.sd ?? "–"}` : (c.top_categories?.[0] ? `${c.top_categories[0].value} (${c.top_categories[0].count})` : "–");
    [c.name, c.type, c.n, mainVal, `${c.missing_pct}%`].forEach((v) => {
      const td = document.createElement("td"); td.textContent = v; tr.appendChild(td);
    });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  return t;
}

function buildCrossTable(ct) {
  const t = document.createElement("table");
  t.className = "result-table";
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  hr.appendChild(document.createElement("th"));
  ct.cols.forEach((c) => { const th = document.createElement("th"); th.textContent = c; hr.appendChild(th); });
  thead.appendChild(hr); t.appendChild(thead);
  const tb = document.createElement("tbody");
  ct.rows.forEach((r, i) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th"); th.textContent = r; tr.appendChild(th);
    ct.values[i].forEach((v) => { const td = document.createElement("td"); td.textContent = v; tr.appendChild(td); });
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  return t;
}
