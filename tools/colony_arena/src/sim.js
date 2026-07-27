export const WORLD = Object.freeze({
  width: 960,
  height: 600,
  dt: 1 / 30,
  horizonTicks: 300,
  agentRadius: 10,
  threatRadius: 13,
  goalRadius: 26,
  agentSpeed: 108,
  threatSpeed: 56,
  catchDistance: 21,
  maxDistance: Math.hypot(960, 600),
});

export const ACTIONS = Object.freeze([
  { id: "N", x: 0, y: -1 },
  { id: "NE", x: Math.SQRT1_2, y: -Math.SQRT1_2 },
  { id: "E", x: 1, y: 0 },
  { id: "SE", x: Math.SQRT1_2, y: Math.SQRT1_2 },
  { id: "S", x: 0, y: 1 },
  { id: "SW", x: -Math.SQRT1_2, y: Math.SQRT1_2 },
  { id: "W", x: -1, y: 0 },
  { id: "NW", x: -Math.SQRT1_2, y: -Math.SQRT1_2 },
]);

export const FEATURE_NAMES = Object.freeze([
  "bias",
  "goal_alignment",
  "threat_escape_alignment",
  "close_threat_escape",
  "one_step_goal_progress",
  "collision_ahead",
  "clearance",
  "momentum",
  "near_goal_pressure",
  "future_threat_distance",
  "diagonal_bias",
]);

export const GENE_COUNT = FEATURE_NAMES.length;

export function makeRng(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function randRange(rng, min, max) {
  return min + (max - min) * rng();
}

export function randomNormal(rng) {
  const u1 = Math.max(1e-9, rng());
  const u2 = Math.max(1e-9, rng());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function normVector(x, y) {
  const len = Math.hypot(x, y);
  if (len < 1e-9) {
    return { x: 0, y: 0 };
  }
  return { x: x / len, y: y / len };
}

function dot(a, b) {
  return a.x * b.x + a.y * b.y;
}

export function baseObstacles() {
  return [
    { id: "wall-a", x: 286, y: 80, w: 42, h: 252 },
    { id: "wall-b", x: 500, y: 268, w: 46, h: 258 },
    { id: "wall-c", x: 690, y: 80, w: 38, h: 220 },
    { id: "wall-d", x: 160, y: 410, w: 244, h: 36 },
    { id: "wall-e", x: 590, y: 390, w: 226, h: 34 },
  ];
}

function pointInRect(p, rect, pad = 0) {
  return (
    p.x >= rect.x - pad &&
    p.x <= rect.x + rect.w + pad &&
    p.y >= rect.y - pad &&
    p.y <= rect.y + rect.h + pad
  );
}

export function collidesCircleRect(circle, rect, radius) {
  const nearestX = clamp(circle.x, rect.x, rect.x + rect.w);
  const nearestY = clamp(circle.y, rect.y, rect.y + rect.h);
  return Math.hypot(circle.x - nearestX, circle.y - nearestY) <= radius;
}

export function inBounds(pos, radius = WORLD.agentRadius) {
  return (
    pos.x >= radius &&
    pos.x <= WORLD.width - radius &&
    pos.y >= radius &&
    pos.y <= WORLD.height - radius
  );
}

export function collidesWithWorld(pos, radius, obstacles) {
  if (!inBounds(pos, radius)) {
    return true;
  }
  return obstacles.some((rect) => collidesCircleRect(pos, rect, radius));
}

function candidatePoint(rng, area) {
  return {
    x: randRange(rng, area.x0, area.x1),
    y: randRange(rng, area.y0, area.y1),
  };
}

function pickOpenPoint(rng, obstacles, area, radius, avoid = []) {
  for (let i = 0; i < 120; i += 1) {
    const p = candidatePoint(rng, area);
    if (collidesWithWorld(p, radius + 8, obstacles)) {
      continue;
    }
    if (avoid.every((other) => distance(p, other) > 110)) {
      return p;
    }
  }
  return candidatePoint(rng, area);
}

export function makeScenario(seed, index = 0) {
  const rng = makeRng((seed + index * 104729) >>> 0);
  const obstacles = baseObstacles().map((rect, idx) => ({
    ...rect,
    y: rect.y + Math.round(randRange(rng, -14, 14)),
    id: `${rect.id}-${idx}`,
  }));
  const agent = pickOpenPoint(
    rng,
    obstacles,
    { x0: 54, x1: 185, y0: 70, y1: WORLD.height - 70 },
    WORLD.agentRadius
  );
  const goal = pickOpenPoint(
    rng,
    obstacles,
    { x0: WORLD.width - 180, x1: WORLD.width - 54, y0: 70, y1: WORLD.height - 70 },
    WORLD.goalRadius,
    [agent]
  );
  const threat = pickOpenPoint(
    rng,
    obstacles,
    { x0: 360, x1: 650, y0: 100, y1: WORLD.height - 100 },
    WORLD.threatRadius,
    [agent, goal]
  );
  return {
    seed,
    index,
    kind: "standard",
    agent,
    goal,
    threat,
    obstacles,
  };
}

export function makeAdversarialScenario(seed, index = 0) {
  const rng = makeRng((seed + index * 130363) >>> 0);
  const yBand = 100 + (index % 4) * 110 + Math.round(randRange(rng, -12, 12));
  const mirror = index % 2 === 1;
  const obstacles = [
    ...baseObstacles().map((rect, idx) => ({
      ...rect,
      y: rect.y + Math.round(randRange(rng, -10, 10)),
      id: `adv-${rect.id}-${idx}`,
    })),
    {
      id: "adv-bait-wall",
      x: 365,
      y: mirror ? 370 + Math.round(randRange(rng, -16, 16)) : 150 + Math.round(randRange(rng, -16, 16)),
      w: 150,
      h: 28,
    },
  ];
  const agent = pickOpenPoint(
    rng,
    obstacles,
    { x0: 62, x1: 150, y0: Math.max(64, yBand - 42), y1: Math.min(WORLD.height - 64, yBand + 42) },
    WORLD.agentRadius
  );
  const goal = pickOpenPoint(
    rng,
    obstacles,
    {
      x0: WORLD.width - 152,
      x1: WORLD.width - 58,
      y0: Math.max(64, yBand - 54),
      y1: Math.min(WORLD.height - 64, yBand + 92),
    },
    WORLD.goalRadius,
    [agent]
  );
  const threat = pickOpenPoint(
    rng,
    obstacles,
    {
      x0: 365,
      x1: 620,
      y0: Math.max(78, (agent.y + goal.y) * 0.5 - 58),
      y1: Math.min(WORLD.height - 78, (agent.y + goal.y) * 0.5 + 58),
    },
    WORLD.threatRadius,
    [agent, goal]
  );
  return {
    seed,
    index,
    kind: "adversarial",
    agent,
    goal,
    threat,
    obstacles,
  };
}

export function cloneScenario(scenario) {
  return JSON.parse(JSON.stringify(scenario));
}

export function createWorld(scenario) {
  const s = cloneScenario(scenario);
  return {
    tick: 0,
    done: false,
    caught: false,
    collected: 0,
    wallHits: 0,
    infoCost: 0,
    decisionCount: 0,
    evidenceActions: {},
    pathLength: 0,
    score: 0,
    agent: { ...s.agent },
    threat: { ...s.threat },
    goal: { ...s.goal },
    obstacles: s.obstacles,
    lastAction: 2,
    lastDir: { x: 1, y: 0 },
    trail: [{ ...s.agent }],
    events: [],
    scenarioSeed: s.seed,
    scenarioIndex: s.index,
    scenarioKind: s.kind ?? "standard",
  };
}

function nearestObstacleDistance(pos, obstacles) {
  let best = Infinity;
  for (const rect of obstacles) {
    const nx = clamp(pos.x, rect.x, rect.x + rect.w);
    const ny = clamp(pos.y, rect.y, rect.y + rect.h);
    best = Math.min(best, Math.hypot(pos.x - nx, pos.y - ny));
  }
  const border = Math.min(pos.x, WORLD.width - pos.x, pos.y, WORLD.height - pos.y);
  return Math.min(best, border);
}

export function actionFeatures(state, actionIndex) {
  const action = ACTIONS[actionIndex];
  const goalVec = normVector(state.goal.x - state.agent.x, state.goal.y - state.agent.y);
  const awayThreat = normVector(state.agent.x - state.threat.x, state.agent.y - state.threat.y);
  const goalDist = distance(state.agent, state.goal);
  const threatDist = distance(state.agent, state.threat);
  const lookahead = WORLD.agentSpeed * WORLD.dt * 4.2;
  const next = {
    x: state.agent.x + action.x * lookahead,
    y: state.agent.y + action.y * lookahead,
  };
  const nextGoalDist = distance(next, state.goal);
  const nextThreatDist = distance(next, state.threat);
  const collision = collidesWithWorld(next, WORLD.agentRadius + 2, state.obstacles) ? 1 : 0;
  const clearance = clamp(nearestObstacleDistance(next, state.obstacles) / 95, 0, 1);
  const threatPressure = clamp(1 - threatDist / 180, 0, 1);
  return [
    1,
    dot(action, goalVec),
    dot(action, awayThreat),
    threatPressure * dot(action, awayThreat),
    clamp((goalDist - nextGoalDist) / 42, -1, 1),
    collision,
    clearance,
    dot(action, state.lastDir),
    clamp(1 - goalDist / WORLD.maxDistance, 0, 1),
    clamp(nextThreatDist / 260, 0, 1),
    actionIndex % 2 === 1 ? 1 : 0,
  ];
}

export function scoreAction(policy, state, actionIndex) {
  const features = actionFeatures(state, actionIndex);
  let score = 0;
  for (let i = 0; i < GENE_COUNT; i += 1) {
    score += policy.weights[i] * features[i];
  }
  return score;
}

export function chooseAction(policy, state) {
  let bestIndex = 0;
  let bestScore = -Infinity;
  const scores = [];
  for (let i = 0; i < ACTIONS.length; i += 1) {
    const score = scoreAction(policy, state, i);
    scores.push(score);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return { actionIndex: bestIndex, scores };
}

function moveCircle(pos, dir, speed, dt, radius, obstacles) {
  const target = {
    x: pos.x + dir.x * speed * dt,
    y: pos.y + dir.y * speed * dt,
  };
  if (!collidesWithWorld(target, radius, obstacles)) {
    return { pos: target, blocked: false };
  }
  const xOnly = { x: target.x, y: pos.y };
  if (!collidesWithWorld(xOnly, radius, obstacles)) {
    return { pos: xOnly, blocked: true };
  }
  const yOnly = { x: pos.x, y: target.y };
  if (!collidesWithWorld(yOnly, radius, obstacles)) {
    return { pos: yOnly, blocked: true };
  }
  return { pos: { ...pos }, blocked: true };
}

function respawnGoal(state) {
  const rng = makeRng((state.scenarioSeed + state.collected * 7919 + 17) >>> 0);
  const side = state.collected % 2 === 0 ? 0 : 1;
  const area = side === 0
    ? { x0: WORLD.width - 180, x1: WORLD.width - 50, y0: 60, y1: WORLD.height - 60 }
    : { x0: 54, x1: 230, y0: 60, y1: WORLD.height - 60 };
  state.goal = pickOpenPoint(rng, state.obstacles, area, WORLD.goalRadius, [state.agent, state.threat]);
}

function applyDecisionAccounting(state, decision) {
  if (!decision) {
    return;
  }
  const evidenceAction = decision.evidenceAction ?? "DECIDE";
  const evidenceCost = Math.max(0, decision.evidenceCost ?? 0);
  state.decisionCount += 1;
  state.evidenceActions[evidenceAction] = (state.evidenceActions[evidenceAction] ?? 0) + 1;
  if (evidenceCost > 0) {
    state.infoCost += evidenceCost;
    state.score -= evidenceCost;
  }
}

export function stepWorld(state, actionIndex, dt = WORLD.dt, decision = null) {
  if (state.done) {
    return state;
  }
  applyDecisionAccounting(state, decision);
  const previousAgent = { ...state.agent };
  const previousGoalDist = distance(state.agent, state.goal);
  const previousThreatDist = distance(state.agent, state.threat);
  const action = ACTIONS[actionIndex];
  const moved = moveCircle(state.agent, action, WORLD.agentSpeed, dt, WORLD.agentRadius, state.obstacles);
  state.agent = moved.pos;
  state.pathLength += distance(previousAgent, state.agent);
  if (moved.blocked) {
    state.wallHits += 1;
    state.score -= 1.35;
    state.events.push({ tick: state.tick, type: "wall" });
  }

  const chase = normVector(state.agent.x - state.threat.x, state.agent.y - state.threat.y);
  const threatMoved = moveCircle(state.threat, chase, WORLD.threatSpeed, dt, WORLD.threatRadius, state.obstacles);
  state.threat = threatMoved.pos;

  const goalDist = distance(state.agent, state.goal);
  const threatDist = distance(state.agent, state.threat);
  const goalProgress = previousGoalDist - goalDist;
  const threatDelta = threatDist - previousThreatDist;
  state.score += 0.22 * goalProgress;
  state.score += 0.035 * threatDelta;
  state.score += 0.012;
  if (threatDist < 110) {
    state.score -= (110 - threatDist) * 0.012;
  }
  if (distance(previousAgent, state.agent) < 0.1) {
    state.score -= 0.08;
  }
  if (dot(action, state.lastDir) < -0.5) {
    state.score -= 0.035;
  }

  state.lastAction = actionIndex;
  state.lastDir = action;
  state.tick += 1;

  if (goalDist <= WORLD.agentRadius + WORLD.goalRadius + 2) {
    state.collected += 1;
    state.score += 45;
    state.events.push({ tick: state.tick, type: "goal", count: state.collected });
    respawnGoal(state);
  }

  if (threatDist <= WORLD.catchDistance) {
    state.caught = true;
    state.done = true;
    state.score -= 70;
    state.events.push({ tick: state.tick, type: "caught" });
  } else if (state.tick >= WORLD.horizonTicks) {
    state.done = true;
    state.events.push({ tick: state.tick, type: "timeout" });
  }

  if (state.tick % 3 === 0) {
    state.trail.push({ ...state.agent });
    if (state.trail.length > 150) {
      state.trail.shift();
    }
  }
  return state;
}

export function simulateEpisode(policy, scenario, options = {}) {
  const state = createWorld(scenario);
  const record = Boolean(options.record);
  const maxTicks = options.maxTicks ?? WORLD.horizonTicks;
  const chooseActionFn = options.chooseActionFn ?? chooseAction;
  const frames = [];
  for (let i = 0; i < maxTicks && !state.done; i += 1) {
    const decision = chooseActionFn(policy, state);
    if (record && i % 3 === 0) {
      frames.push(snapshotState(state, decision));
    }
    stepWorld(state, decision.actionIndex, WORLD.dt, decision);
  }
  if (record) {
    frames.push(snapshotState(state, chooseActionFn(policy, state)));
  }
  const direct = distance(scenario.agent, scenario.goal);
  const efficiency = state.pathLength > 1 ? clamp(direct / state.pathLength, 0, 1.5) : 0;
  const finalGoalDistance = distance(state.agent, state.goal);
  const noGoalPenalty = state.collected === 0 ? Math.min(70, finalGoalDistance * 0.1) : 0;
  const summary = {
    score: state.score + state.collected * 45 + efficiency * 8 - state.wallHits * 0.25 - noGoalPenalty,
    rawScore: state.score,
    collected: state.collected,
    caught: state.caught,
    wallHits: state.wallHits,
    infoCost: state.infoCost,
    decisionCount: state.decisionCount,
    evidenceActions: { ...state.evidenceActions },
    nonDecideRate: state.decisionCount
      ? 1 - ((state.evidenceActions.DECIDE ?? 0) / state.decisionCount)
      : 0,
    pathLength: state.pathLength,
    efficiency,
    finalGoalDistance,
    ticks: state.tick,
    scenarioKind: state.scenarioKind,
  };
  return { summary, frames, finalState: state };
}

export function snapshotState(state, decision = null) {
  return {
    tick: state.tick,
    agent: { ...state.agent },
    threat: { ...state.threat },
    goal: { ...state.goal },
    collected: state.collected,
    caught: state.caught,
    wallHits: state.wallHits,
    infoCost: state.infoCost,
    evidenceActions: { ...state.evidenceActions },
    score: state.score,
    lastAction: ACTIONS[state.lastAction]?.id ?? "E",
    actionScores: decision?.scores ?? [],
    chosenAction: decision ? ACTIONS[decision.actionIndex].id : ACTIONS[state.lastAction]?.id ?? "E",
    evidenceAction: decision?.evidenceAction ?? "DECIDE",
    confidence: decision?.confidence ?? null,
    d51Features: decision?.d51Features ?? null,
  };
}

export function makeScenarioSet(seed, count, kind = "standard") {
  return Array.from({ length: count }, (_, index) => {
    if (kind === "adversarial") {
      return makeAdversarialScenario(seed + index * 37, index);
    }
    if (kind === "mixed") {
      return index % 3 === 2
        ? makeAdversarialScenario(seed + index * 37, index)
        : makeScenario(seed + index * 37, index);
    }
    return makeScenario(seed + index * 37, index);
  });
}
