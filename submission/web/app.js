const AXES = ["Protection Impact", "Chip-to-Route Value", "Route Threat", "Coverage Stress", "Open-Field Creation", "Run-Game Impact"];
const VALIDATED_AXES = AXES.filter((axis) => axis !== "Run-Game Impact");
const ROLE_LABELS = {
  wshare_shrunk_inline_route: "Inline route",
  wshare_shrunk_detached_route: "Detached route",
  wshare_shrunk_chip_release: "Chip and release",
  wshare_shrunk_stay_in_block: "Stay-in block",
  wshare_shrunk_standard_pass_block: "Standard pass block",
};
let players = [];
let selected = null;

const $ = (selector) => document.querySelector(selector);
const number = (value) => Number(value || 0);
const title = (key) => key.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function filters() {
  const term = $("#search").value.trim().toLowerCase();
  const performance = $("#performance-filter").value;
  const deployment = $("#deployment-filter").value;
  const minSnaps = number($("#min-snaps").value);
  return players
    .filter((player) => player.profile_eligible)
    .filter((player) => !term || `${player.displayName} ${player.team}`.toLowerCase().includes(term))
    .filter((player) => !performance || player.performance_archetype === performance)
    .filter((player) => !deployment || player.deployment_archetype === deployment)
    .filter((player) => number(player.snaps) >= minSnaps);
}

function sortColumn() {
  return $("#sort-by").value;
}

function renderList() {
  const sorted = filters().sort((a, b) => number(b[sortColumn()]) - number(a[sortColumn()]));
  $("#player-list").innerHTML = sorted.map((player) => `
    <button class="player ${selected?.nflId === player.nflId ? "selected" : ""}" data-id="${player.nflId}">
      <span class="mini-score">${number(player[sortColumn()]).toFixed(0)}</span>
      <strong>${player.displayName}</strong>
      <small>${player.team} · ${number(player.snaps)} snaps · ${player.performance_archetype}</small>
    </button>`).join("") || "<p class='empty-state'>No players match these filters.</p>";
  document.querySelectorAll(".player").forEach((node) => node.addEventListener("click", () => selectPlayer(node.dataset.id)));
}

function drawRadar(player) {
  const axes = AXES.filter((axis) => player[axis] !== null && player[axis] !== undefined);
  const canvas = $("#radar");
  const context = canvas.getContext("2d");
  const width = canvas.width, height = canvas.height, centerX = width / 2, centerY = height / 2 + 16, radius = 145;
  const point = (value, index) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * index) / axes.length;
    return [centerX + (radius * value / 100) * Math.cos(angle), centerY + (radius * value / 100) * Math.sin(angle)];
  };
  const polygon = (values, stroke, fill, dashed = false) => {
    context.beginPath();
    values.map(point).forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath(); context.setLineDash(dashed ? [6, 6] : []); context.strokeStyle = stroke; context.lineWidth = 2;
    context.stroke(); context.setLineDash([]); if (fill) { context.fillStyle = fill; context.fill(); }
  };
  context.clearRect(0, 0, width, height);
  context.font = "13px system-ui"; context.fillStyle = "#99adbc"; context.textAlign = "center";
  [25, 50, 75, 100].forEach((level) => polygon(Array(axes.length).fill(level), "#31536a", null));
  axes.forEach((axis, index) => {
    const [x, y] = point(112, index);
    context.fillText(axis, x, y + 5);
  });
  polygon(Array(axes.length).fill(50), "#99adbc", null, true);
  polygon(axes.map((axis) => number(player[axis])), "#6ca8ff", "rgba(108,168,255,.27)");
}

function renderBars(player) {
  $("#axis-bars").innerHTML = AXES.map((axis) => {
    const available = player[axis] !== null && player[axis] !== undefined;
    return `<div class="axis-row"><span>${axis}${axis === "Run-Game Impact" ? " †" : ""}</span>
      <div class="bar"><i style="width:${available ? number(player[axis]) : 0}%"></i></div>
      <strong class="value">${available ? number(player[axis]).toFixed(0) : "N/A"}</strong></div>`;
  }).join("");
  $("#role-bars").innerHTML = Object.entries(ROLE_LABELS).map(([key, label]) => `
    <div class="role-row"><span>${label}</span><div class="bar"><i style="width:${number(player[key]) * 100}%"></i></div>
    <strong class="value">${(number(player[key]) * 100).toFixed(0)}%</strong></div>`).join("");
}

function similarities(player, mode) {
  return players.filter((candidate) => candidate.profile_eligible && candidate.nflId !== player.nflId).map((candidate) => {
    const performance = 100 - Math.sqrt(VALIDATED_AXES.reduce((sum, axis) => sum + (number(candidate[axis]) - number(player[axis])) ** 2, 0) / VALIDATED_AXES.length);
    const roles = Object.keys(ROLE_LABELS);
    const role = 100 - 100 * Math.sqrt(roles.reduce((sum, key) => sum + (number(candidate[key]) - number(player[key])) ** 2, 0) / roles.length);
    const score = mode === "performance" ? performance : mode === "role" ? role : performance * .7 + role * .3;
    return {...candidate, score};
  }).sort((a, b) => b.score - a.score).slice(0, 5);
}

function renderSimilar() {
  const mode = $("#similarity-mode").value;
  $("#similar-list").innerHTML = similarities(selected, mode).map((player) => `
    <div class="similar" data-id="${player.nflId}">
      <span><b>${player.displayName}</b><br><small>${player.team} · ${player.deployment_archetype}</small></span>
      <strong>${player.score.toFixed(0)}%</strong>
    </div>`).join("");
  document.querySelectorAll(".similar").forEach((node) => node.addEventListener("click", () => selectPlayer(node.dataset.id)));
}

function selectPlayer(id) {
  selected = players.find((player) => String(player.nflId) === String(id));
  if (!selected) return;
  $("#empty-state").hidden = true; $("#profile-content").hidden = false;
  $("#team").textContent = `${selected.team} · ${number(selected.snaps)} snaps`;
  $("#player-name").textContent = selected.displayName;
  $("#profile-subtitle").textContent = `${selected.performance_archetype} · ${selected.deployment_archetype} deployment`;
  $("#profile-note").textContent = selected["Run-Game Impact"] === null || selected["Run-Game Impact"] === undefined
    ? "Run-Game Impact is unavailable for this player: the provisional source table has no matching row. The radar therefore shows the five validated pass-game axes."
    : "† Run-Game Impact is provisional and does not affect the overall score or performance-similarity results.";
  $("#overall-score").textContent = number(selected.versatility_profile_score).toFixed(0);
  $("#deployment-summary").textContent = `Deployment Breadth: ${(number(selected.deployment_breadth) * 100).toFixed(0)}th percentile-style index · ${selected.deployment_archetype}. This is role context, not part of the performance score.`;
  drawRadar(selected); renderBars(selected); renderSimilar(); renderList();
}

function populateSelect(selector, values) {
  $(selector).insertAdjacentHTML("beforeend", [...new Set(values)].filter(Boolean).sort().map((value) => `<option>${value}</option>`).join(""));
}

async function start() {
  const response = await fetch("../output/coach_report/search_index.json");
  if (!response.ok) throw new Error("Could not load the coach profile index.");
  players = await response.json();
  $("#eligible-count").textContent = players.filter((player) => player.profile_eligible).length;
  $("#axis-count").textContent = AXES.length;
  $("#snap-count").textContent = `${(players.reduce((sum, player) => sum + number(player.snaps), 0) / 1000).toFixed(1)}k`;
  populateSelect("#performance-filter", players.map((player) => player.performance_archetype));
  populateSelect("#deployment-filter", players.map((player) => player.deployment_archetype));
  const options = { versatility_profile_score: "Overall profile", "Protection Impact": "Protection impact", "Chip-to-Route Value": "Chip-to-route value", "Route Threat": "Route threat", "Coverage Stress": "Coverage stress", "Open-Field Creation": "Open-field creation", "Run-Game Impact": "Run-game impact †", deployment_breadth: "Deployment breadth" };
  $("#sort-by").innerHTML = Object.entries(options).map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
  ["#search", "#performance-filter", "#deployment-filter", "#min-snaps", "#sort-by"].forEach((selector) => $(selector).addEventListener("input", renderList));
  $("#similarity-mode").addEventListener("input", renderSimilar);
  renderList();
  selectPlayer(players.find((player) => player.profile_eligible)?.nflId);
}

start().catch((error) => { $("#empty-state").textContent = error.message; });
