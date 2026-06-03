const state = { records: [], checkpoints: [], currentGame: null };
const $ = id => document.getElementById(id);

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function fmt(value, digits = 3) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "n/a";
}

function avg(rows, key) {
  const values = rows.map(row => row[key]).filter(value => Number.isFinite(Number(value))).map(Number);
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}

function metric(label, value, note) {
  return `<div class="card"><span class="metric-label">${label}</span><span class="metric-value">${value}</span><span class="metric-note">${note}</span></div>`;
}

async function boot() {
  const [health, logs, checkpoints, prompts] = await Promise.all([
    api("/api/health"), api("/api/logs"), api("/api/checkpoints"), api("/api/prompts")
  ]);
  $("codexHealth").textContent = health.codex.available ? `Codex: ${health.codex.stdout}` : `Codex unavailable: ${health.codex.error || health.codex.stderr || "not found"}`;
  $("runAnalysis").disabled = !health.codex.available;
  $("logSelect").innerHTML = logs.logs.map(log => `<option value="${log.id}">${log.id}</option>`).join("");
  state.checkpoints = checkpoints.checkpoints;
  const checkpointOptions = state.checkpoints.map(cp => `<option value="${cp.id}">${cp.name}</option>`).join("");
  $("modelSelect").innerHTML = checkpointOptions;
  $("opponentModelSelect").innerHTML = checkpointOptions;
  $("promptSelect").innerHTML = prompts.prompts.map(id => `<option value="${id}">${id.replaceAll("_", " ")}</option>`).join("");
  if (logs.logs.length) await loadSummary();
}

async function loadSummary() {
  const id = $("logSelect").value;
  if (!id) return;
  const data = await api(`/api/log-summary?id=${encodeURIComponent(id)}`);
  state.records = data.records;
  renderSummary(data);
  renderCharts();
}

function renderSummary(data) {
  const last = data.summary.latest || {};
  const latest100 = state.records.slice(-100);
  $("summary").innerHTML = [
    metric("Batches", data.summary.rows || 0, `${data.summary.first_batch || "n/a"} to ${data.summary.latest_batch || "n/a"}`),
    metric("Episode Length", last.episode_len ?? "n/a", `latest 100 avg ${fmt(avg(latest100, "episode_len"), 1)}`),
    metric("KL", fmt(last.kl, 5), `latest 100 avg ${fmt(avg(latest100, "kl"), 5)}`),
    metric("LR Multiplier", fmt(last.lr_multiplier, 3), `effective ${fmt(last.effective_lr, 6)}`),
    metric("Loss", fmt(last.loss, 3), `policy ${fmt(last.policy_loss, 3)}, value ${fmt(last.value_loss, 3)}`),
    metric("Entropy", fmt(last.entropy, 3), `latest 100 avg ${fmt(avg(latest100, "entropy"), 3)}`),
    metric("Explained Var", fmt(last.explained_var_new, 3), `old ${fmt(last.explained_var_old, 3)}`),
    metric("Replay Buffer", last.replay_buffer ?? "n/a", "latest logged size")
  ].join("");
  $("currentRead").textContent = data.current_read || "";
  $("alerts").innerHTML = (data.alerts || []).map(alert => `<p>${alert}</p>`).join("") || "<p>No dashboard alerts.</p>";
  $("windowTable").innerHTML = table(data.windows || []);
}

function table(rows) {
  if (!rows.length) return "<p class='muted'>No window data.</p>";
  const cols = ["label", "rows", "episode_len", "kl", "lr_multiplier", "loss", "policy_loss", "value_loss", "entropy", "explained_var_new"];
  return `<table><thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${cols.map(c => `<td>${typeof row[c] === "number" ? fmt(row[c], c === "kl" ? 5 : 3) : row[c] ?? "n/a"}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function rolling(key, window = 25) {
  const out = [];
  let sum = 0, count = 0;
  const queue = [];
  for (const row of state.records) {
    const value = Number(row[key]);
    if (!Number.isFinite(value)) continue;
    queue.push(value); sum += value; count += 1;
    if (queue.length > window) { sum -= queue.shift(); count -= 1; }
    out.push({ x: Number(row.batch), y: sum / count });
  }
  return out;
}

function drawChart(canvas, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.scale(dpr, dpr);
  const pad = { left: 45, right: 12, top: 12, bottom: 28 };
  const points = series.flatMap(s => s.points);
  if (!points.length) return;
  const xMin = Math.min(...points.map(p => p.x)), xMax = Math.max(...points.map(p => p.x));
  let yMin = options.yMin ?? Math.min(...points.map(p => p.y));
  let yMax = options.yMax ?? Math.max(...points.map(p => p.y));
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * .08; yMin = options.yMin ?? yMin - yPad; yMax = options.yMax ?? yMax + yPad;
  const sx = x => pad.left + ((x - xMin) / Math.max(xMax - xMin, 1)) * (rect.width - pad.left - pad.right);
  const sy = y => pad.top + (1 - (y - yMin) / Math.max(yMax - yMin, 1e-9)) * (rect.height - pad.top - pad.bottom);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, rect.width, rect.height);
  ctx.strokeStyle = "#e5e7eb"; ctx.fillStyle = "#64707d"; ctx.font = "12px Segoe UI, Arial";
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + ((rect.height - pad.top - pad.bottom) / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(rect.width - pad.right, y); ctx.stroke();
    ctx.fillText(fmt(yMax - ((yMax - yMin) / 4) * i, options.digits ?? 2), 4, y + 4);
  }
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    s.points.forEach((p, i) => i ? ctx.lineTo(sx(p.x), sy(p.y)) : ctx.moveTo(sx(p.x), sy(p.y)));
    ctx.stroke();
  });
}

function renderCharts() {
  drawChart($("episodeChart"), [{ color: "#0f766e", points: rolling("episode_len") }], { yMin: 0, digits: 0 });
  drawChart($("lossChart"), [
    { color: "#2563eb", points: rolling("loss") },
    { color: "#b45309", points: rolling("policy_loss") },
    { color: "#7c3aed", points: rolling("value_loss") }
  ]);
  drawChart($("klChart"), [{ color: "#b91c1c", points: rolling("kl") }, { color: "#15803d", points: rolling("lr_multiplier") }], { yMin: 0 });
  drawChart($("evChart"), [{ color: "#7c3aed", points: rolling("explained_var_new") }, { color: "#64707d", points: rolling("explained_var_old") }], { yMin: -0.4, yMax: 1.05 });
}

async function startJob(path, payload, outputEl) {
  const job = await api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  outputEl.textContent = `Job ${job.job_id} queued...`;
  const timer = setInterval(async () => {
    const current = await api(`/api/jobs/${job.job_id}`);
    outputEl.textContent = current.status === "succeeded" ? JSON.stringify(current.result, null, 2) : `${current.status}\n${current.error || ""}`;
    if (["succeeded", "failed"].includes(current.status)) clearInterval(timer);
  }, 1200);
}

function renderBoard(snapshot) {
  const board = $("board");
  board.style.gridTemplateColumns = `repeat(${snapshot.width}, 1fr)`;
  board.innerHTML = "";
  for (let row = snapshot.height - 1; row >= 0; row--) {
    for (let col = 0; col < snapshot.width; col++) {
      const value = snapshot.cells[row][col];
      const button = document.createElement("button");
      button.className = "cell";
      button.onclick = () => playMove(row, col);
      button.disabled = value || snapshot.ended;
      button.innerHTML = value ? `<span class="stone p${value}"></span>` : "";
      board.appendChild(button);
    }
  }
  $("gameStatus").textContent = snapshot.ended ? (snapshot.winner === -1 ? "Draw." : `Winner: player ${snapshot.winner}`) : `Current player: ${snapshot.current_player}`;
}

async function newGame() {
  const data = await api("/api/games/human-vs-model", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: $("modelSelect").value, playouts: Number($("playoutsInput").value), human_first: $("humanFirst").checked })
  });
  state.currentGame = data.game_id;
  renderBoard(data.state);
}

async function playMove(row, col) {
  if (!state.currentGame) return;
  const data = await api(`/api/games/${state.currentGame}/move`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ row, col })
  });
  renderBoard(data.state);
}

$("reload").onclick = loadSummary;
$("logSelect").onchange = loadSummary;
$("runAnalysis").onclick = () => startJob("/api/analysis/codex", { log_id: $("logSelect").value, prompt_id: $("promptSelect").value }, $("analysisOutput"));
$("runEval").onclick = () => startJob("/api/evaluate", {
  model: $("modelSelect").value,
  opponent: $("opponentSelect").value,
  opponent_model: $("opponentModelSelect").value,
  games: Number($("gamesInput").value),
  playouts: Number($("playoutsInput").value)
}, $("evalOutput"));
$("newGame").onclick = newGame;
window.addEventListener("resize", renderCharts);
boot().catch(error => { document.body.innerHTML = `<pre>${error.stack || error}</pre>`; });
