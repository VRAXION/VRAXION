import {
  ACTIONS,
  WORLD,
  createWorld,
  makeScenario,
  makeAdversarialScenario,
  stepWorld,
  distance,
} from "./sim.js";
import { EvolutionTrainer } from "./trainer.js";
import { chooseD51Action } from "./d51_controller.js";

const canvas = document.getElementById("arena");
const ctx = canvas.getContext("2d");
const chart = document.getElementById("chart");
const chartCtx = chart.getContext("2d");
const params = new URLSearchParams(window.location.search);
const testMode = params.get("test") === "1";

const ui = {
  gen: document.getElementById("gen"),
  best: document.getElementById("best"),
  success: document.getElementById("success"),
  caught: document.getElementById("caught"),
  goals: document.getElementById("goals"),
  walls: document.getElementById("walls"),
  brain: document.getElementById("brain"),
  infoCost: document.getElementById("info-cost"),
  pause: document.getElementById("pause-btn"),
  train: document.getElementById("train-btn"),
  burst: document.getElementById("burst-btn"),
  reset: document.getElementById("reset-btn"),
  speed: document.getElementById("speed"),
  stateText: document.getElementById("state-text"),
};

const trainer = new EvolutionTrainer({
  populationSize: testMode ? 28 : 56,
  eliteCount: testMode ? 4 : 7,
  seed: 73001,
  includePrior: true,
});

const app = {
  paused: false,
  autoTrain: !testMode,
  simAccumulator: 0,
  trainAccumulator: 0,
  scenarioCursor: 0,
  scenario: makeScenario(44001, 0),
  world: null,
  latestDecision: null,
  lastMetric: null,
};
app.world = createWorld(app.scenario);

function resetWorld(nextScenario = false) {
  if (nextScenario) {
    app.scenarioCursor += 1;
    app.scenario = app.scenarioCursor % 3 === 2
      ? makeAdversarialScenario(44001 + app.scenarioCursor * 19, app.scenarioCursor)
      : makeScenario(44001 + app.scenarioCursor * 19, app.scenarioCursor);
  }
  app.world = createWorld(app.scenario);
  app.latestDecision = null;
}

function trainOne() {
  app.lastMetric = trainer.stepGeneration();
  updateUi();
}

function trainBurst(count) {
  for (let i = 0; i < count; i += 1) {
    trainOne();
  }
}

function update(dtMs) {
  if (!app.paused) {
    const speed = Number(ui.speed.value);
    app.simAccumulator += dtMs * speed;
    while (app.simAccumulator >= WORLD.dt * 1000) {
      app.latestDecision = chooseD51Action(trainer.bestPolicy(), app.world);
      stepWorld(app.world, app.latestDecision.actionIndex, WORLD.dt, app.latestDecision);
      if (app.world.done) {
        resetWorld(true);
      }
      app.simAccumulator -= WORLD.dt * 1000;
    }
  }
  if (app.autoTrain) {
    app.trainAccumulator += dtMs;
    const threshold = 210 / Number(ui.speed.value);
    while (app.trainAccumulator >= threshold && trainer.generation < 160) {
      trainOne();
      app.trainAccumulator -= threshold;
    }
  }
}

function drawGrid() {
  ctx.fillStyle = "#edf3eb";
  ctx.fillRect(0, 0, WORLD.width, WORLD.height);
  ctx.strokeStyle = "rgba(65, 93, 83, 0.09)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= WORLD.width; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, WORLD.height);
    ctx.stroke();
  }
  for (let y = 0; y <= WORLD.height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(WORLD.width, y);
    ctx.stroke();
  }
}

function drawObstacles() {
  for (const rect of app.world.obstacles) {
    ctx.fillStyle = "#384853";
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = "rgba(255,255,255,0.08)";
    ctx.fillRect(rect.x + 4, rect.y + 4, rect.w - 8, 5);
  }
}

function drawTrail() {
  const trail = app.world.trail;
  if (trail.length < 2) {
    return;
  }
  ctx.lineWidth = 3;
  for (let i = 1; i < trail.length; i += 1) {
    const a = trail[i - 1];
    const b = trail[i];
    ctx.strokeStyle = `rgba(35, 108, 204, ${0.08 + (i / trail.length) * 0.28})`;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function drawGoalLine() {
  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = "rgba(28, 124, 109, 0.28)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(app.world.agent.x, app.world.agent.y);
  ctx.lineTo(app.world.goal.x, app.world.goal.y);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawEntity(pos, radius, color, ring = null) {
  if (ring) {
    ctx.fillStyle = ring;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, radius + 13, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "rgba(0,0,0,0.18)";
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawActionSpokes() {
  const decision = app.latestDecision ?? chooseD51Action(trainer.bestPolicy(), app.world);
  const scores = decision.scores;
  const maxScore = Math.max(...scores);
  const minScore = Math.min(...scores);
  const span = Math.max(1e-6, maxScore - minScore);
  for (let i = 0; i < ACTIONS.length; i += 1) {
    const action = ACTIONS[i];
    const t = (scores[i] - minScore) / span;
    ctx.strokeStyle = i === decision.actionIndex
      ? "rgba(20, 32, 51, 0.82)"
      : `rgba(28, 124, 109, ${0.16 + t * 0.34})`;
    ctx.lineWidth = i === decision.actionIndex ? 4 : 2;
    ctx.beginPath();
    ctx.moveTo(app.world.agent.x, app.world.agent.y);
    ctx.lineTo(app.world.agent.x + action.x * (28 + t * 34), app.world.agent.y + action.y * (28 + t * 34));
    ctx.stroke();
  }
}

function drawEvidenceHalo() {
  const decision = app.latestDecision;
  if (!decision) {
    return;
  }
  const colors = {
    DECIDE: "rgba(40, 107, 214, 0.18)",
    REQUEST_SUPPORT: "rgba(208, 154, 45, 0.24)",
    REQUEST_COUNTER_TOP1_TOP2: "rgba(208, 154, 45, 0.30)",
    REQUEST_JOINT_COUNTER: "rgba(28, 124, 109, 0.30)",
    REQUEST_EXTERNAL_TEST: "rgba(120, 72, 170, 0.32)",
    ABSTAIN: "rgba(197, 75, 75, 0.26)",
  };
  ctx.strokeStyle = colors[decision.evidenceAction] ?? colors.DECIDE;
  ctx.lineWidth = decision.evidenceAction === "DECIDE" ? 2 : 5;
  ctx.setLineDash(decision.evidenceAction === "DECIDE" ? [] : [10, 8]);
  ctx.beginPath();
  ctx.arc(app.world.agent.x, app.world.agent.y, 38, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}

function render() {
  drawGrid();
  drawGoalLine();
  drawTrail();
  drawObstacles();
  const threatDist = distance(app.world.agent, app.world.threat);
  const threatAlpha = Math.max(0.08, 0.26 - threatDist / 1000);
  drawEntity(app.world.goal, WORLD.goalRadius, "#28a56f", "rgba(40, 165, 111, 0.15)");
  drawEntity(app.world.threat, WORLD.threatRadius, "#c54b4b", `rgba(197, 75, 75, ${threatAlpha})`);
  drawActionSpokes();
  drawEvidenceHalo();
  drawEntity(app.world.agent, WORLD.agentRadius, "#286bd6", "rgba(40, 107, 214, 0.16)");
  drawHud();
  drawChart();
  updateLiveUi();
  ui.stateText.textContent = renderGameToText();
}

function drawHud() {
  ctx.fillStyle = "rgba(255,255,255,0.82)";
  ctx.fillRect(18, 18, 258, 70);
  ctx.strokeStyle = "rgba(20,32,51,0.12)";
  ctx.strokeRect(18.5, 18.5, 257, 69);
  ctx.fillStyle = "#142033";
  ctx.font = "600 14px ui-sans-serif, system-ui";
  ctx.fillText(`gen ${trainer.generation}`, 34, 43);
  ctx.fillText(`goals ${app.world.collected}`, 34, 67);
  ctx.fillStyle = "#637083";
  ctx.font = "12px ui-sans-serif, system-ui";
  const action = app.latestDecision ? ACTIONS[app.latestDecision.actionIndex].id : "-";
  const brain = app.latestDecision?.evidenceAction ?? "DECIDE";
  ctx.fillText(`intent ${action}  ${app.world.scenarioKind}`, 118, 43);
  ctx.fillText(`${brain.slice(0, 22)}  cost ${app.world.infoCost.toFixed(1)}`, 118, 67);
}

function drawChart() {
  const w = chart.width;
  const h = chart.height;
  chartCtx.clearRect(0, 0, w, h);
  chartCtx.fillStyle = "#ffffff";
  chartCtx.fillRect(0, 0, w, h);
  chartCtx.strokeStyle = "rgba(20,32,51,0.12)";
  chartCtx.strokeRect(0.5, 0.5, w - 1, h - 1);
  const rows = trainer.history.slice(-80);
  if (rows.length < 2) {
    return;
  }
  const vals = rows.map((row) => row.best_score);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(1e-6, max - min);
  chartCtx.strokeStyle = "#1c7c6d";
  chartCtx.lineWidth = 2;
  chartCtx.beginPath();
  rows.forEach((row, idx) => {
    const x = 10 + (idx / (rows.length - 1)) * (w - 20);
    const y = h - 12 - ((row.best_score - min) / span) * (h - 26);
    if (idx === 0) {
      chartCtx.moveTo(x, y);
    } else {
      chartCtx.lineTo(x, y);
    }
  });
  chartCtx.stroke();
}

function updateUi() {
  const latest = app.lastMetric ?? trainer.history[trainer.history.length - 1];
  if (!latest) {
    return;
  }
  ui.gen.textContent = String(trainer.generation);
  ui.best.textContent = latest.best_score.toFixed(2);
  ui.success.textContent = `${Math.round(latest.test_success_rate * 100)}%`;
  ui.caught.textContent = `${Math.round(latest.test_caught_rate * 100)}%`;
  ui.goals.textContent = latest.test_mean_goals.toFixed(2);
  ui.walls.textContent = latest.train_wall_hits.toFixed(1);
  updateLiveUi();
}

function updateLiveUi() {
  ui.brain.textContent = app.latestDecision?.evidenceAction ?? "DECIDE";
  ui.infoCost.textContent = app.world.infoCost.toFixed(2);
}

function renderGameToText() {
  const latest = app.lastMetric ?? trainer.history[trainer.history.length - 1] ?? null;
  const action = app.latestDecision ? ACTIONS[app.latestDecision.actionIndex].id : "-";
  return JSON.stringify({
    coordinate_system: "origin top-left, x right, y down",
    mode: app.paused ? "paused" : "running",
    generation: trainer.generation,
    best_score: latest ? Number(latest.best_score.toFixed(3)) : null,
    test_success_rate: latest ? Number(latest.test_success_rate.toFixed(3)) : null,
    test_caught_rate: latest ? Number(latest.test_caught_rate.toFixed(3)) : null,
    adversarial_success_rate: latest ? Number(latest.adversarial_success_rate.toFixed(3)) : null,
    controller: "D51_MUTABLE_RULE_TABLE_CONTROLLER",
    world: {
      tick: app.world.tick,
      scenario_kind: app.world.scenarioKind,
      agent: roundPos(app.world.agent),
      threat: roundPos(app.world.threat),
      goal: roundPos(app.world.goal),
      chosen_action: action,
      evidence_action: app.latestDecision?.evidenceAction ?? "DECIDE",
      confidence: app.latestDecision ? Number(app.latestDecision.confidence.toFixed(3)) : null,
      collected: app.world.collected,
      caught: app.world.caught,
      wall_hits: app.world.wallHits,
      info_cost: Number(app.world.infoCost.toFixed(3)),
      evidence_actions: app.world.evidenceActions,
      distance_to_goal: Math.round(distance(app.world.agent, app.world.goal)),
      distance_to_threat: Math.round(distance(app.world.agent, app.world.threat)),
      d51_features: app.latestDecision?.d51Features ?? null,
    },
  }, null, 2);
}

function roundPos(pos) {
  return { x: Math.round(pos.x), y: Math.round(pos.y) };
}

function setPaused(paused) {
  app.paused = paused;
  ui.pause.textContent = paused ? "Resume" : "Pause";
}

ui.pause.addEventListener("click", () => setPaused(!app.paused));
ui.train.addEventListener("click", () => trainOne());
ui.burst.addEventListener("click", () => trainBurst(25));
ui.reset.addEventListener("click", () => {
  trainer.reset();
  resetWorld(false);
  app.lastMetric = null;
  updateUi();
});

document.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    event.preventDefault();
    setPaused(!app.paused);
  } else if (event.key === "f" || event.key === "F") {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen?.();
    } else {
      document.exitFullscreen?.();
    }
  } else if (event.code === "ArrowRight") {
    trainOne();
  }
});

let lastTime = performance.now();
function frame(now) {
  const dt = Math.min(80, now - lastTime);
  lastTime = now;
  update(dt);
  render();
  requestAnimationFrame(frame);
}

window.advanceTime = (ms) => {
  const steps = Math.max(1, Math.round(ms / (1000 / 30)));
  const slice = ms / steps;
  for (let i = 0; i < steps; i += 1) {
    update(slice);
  }
  render();
};
window.render_game_to_text = renderGameToText;
window.__colonyArena = { trainer, app };

trainBurst(testMode ? 4 : 3);
render();
requestAnimationFrame(frame);
