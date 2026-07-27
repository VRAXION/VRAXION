import {
  ACTIONS,
  WORLD,
  actionFeatures,
  clamp,
  collidesWithWorld,
  distance,
  scoreAction,
} from "./sim.js";

export const D51_FEATURE_NAMES = Object.freeze([
  "bias",
  "scalar_confidence",
  "inverse_margin",
  "entropy_norm",
  "collision_norm",
  "dominant_cluster_fraction",
  "support_cluster_count_norm",
  "top1_factorised_disagreement",
  "cell_confidence",
  "operator_confidence",
  "joint_confidence",
  "internal_unresolvable_indicator",
  "external_channel_available",
]);

export const D51_EVIDENCE_ACTIONS = Object.freeze([
  "DECIDE",
  "REQUEST_SUPPORT",
  "REQUEST_COUNTER_TOP1_TOP2",
  "REQUEST_JOINT_COUNTER",
  "REQUEST_EXTERNAL_TEST",
  "ABSTAIN",
]);

export const D51_RULE_TABLE_POLICY = Object.freeze({
  kind: "rule_table",
  source: "D51_MUTABLE_RULE_TABLE_CONTROLLER",
  margin_threshold: 0.5150859971581,
  dominant_threshold: 0.6375598465004098,
  entropy_threshold: 0.8799876217343521,
  confidence_threshold: 0.08,
  action_external: "REQUEST_EXTERNAL_TEST",
  action_unresolvable: "ABSTAIN",
  action_dominant: "REQUEST_JOINT_COUNTER",
  action_low_margin: "REQUEST_COUNTER_TOP1_TOP2",
  action_uncertain: "REQUEST_COUNTER_TOP1_TOP2",
  action_default: "DECIDE",
});

export const D51_EVIDENCE_COST = Object.freeze({
  DECIDE: 0,
  REQUEST_SUPPORT: 0.025,
  REQUEST_COUNTER_TOP1_TOP2: 0.035,
  REQUEST_JOINT_COUNTER: 0.06,
  REQUEST_EXTERNAL_TEST: 0.09,
  ABSTAIN: 0.025,
});

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

function softmaxStats(scores) {
  const maxScore = Math.max(...scores);
  const mean = scores.reduce((acc, value) => acc + value, 0) / scores.length;
  const variance = scores.reduce((acc, value) => acc + (value - mean) ** 2, 0) / scores.length;
  const temperature = Math.max(0.35, Math.sqrt(variance));
  const exps = scores.map((score) => Math.exp(clamp((score - maxScore) / temperature, -30, 0)));
  const total = exps.reduce((acc, value) => acc + value, 0) || 1;
  const probs = exps.map((value) => value / total);
  const entropy = -probs.reduce((acc, value) => acc + (value > 0 ? value * Math.log(value) : 0), 0);
  return {
    probs,
    entropyNorm: clamp(entropy / Math.log(scores.length), 0, 1),
  };
}

function sortedIndexes(values) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .map((row) => row.index);
}

function nearestActionIndex(vector) {
  let bestIndex = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < ACTIONS.length; i += 1) {
    const score = dot(ACTIONS[i], vector);
    if (score > bestScore) {
      bestScore = score;
      bestIndex = i;
    }
  }
  return bestIndex;
}

function lookaheadPoint(state, actionIndex, ticks) {
  const action = ACTIONS[actionIndex];
  return {
    x: state.agent.x + action.x * WORLD.agentSpeed * WORLD.dt * ticks,
    y: state.agent.y + action.y * WORLD.agentSpeed * WORLD.dt * ticks,
  };
}

function pointInExpandedRect(point, rect, pad) {
  return (
    point.x >= rect.x - pad &&
    point.x <= rect.x + rect.w + pad &&
    point.y >= rect.y - pad &&
    point.y <= rect.y + rect.h + pad
  );
}

function firstRouteBlocker(state) {
  const a = state.agent;
  const b = state.goal;
  for (const rect of state.obstacles) {
    for (let i = 1; i <= 24; i += 1) {
      const t = i / 24;
      const p = {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
      };
      if (pointInExpandedRect(p, rect, WORLD.agentRadius + 10)) {
        return rect;
      }
    }
  }
  return null;
}

function externalTarget(state) {
  const blocker = firstRouteBlocker(state);
  if (!blocker) {
    return state.goal;
  }
  const pad = WORLD.agentRadius + 30;
  const corners = [
    { x: blocker.x - pad, y: blocker.y - pad },
    { x: blocker.x + blocker.w + pad, y: blocker.y - pad },
    { x: blocker.x - pad, y: blocker.y + blocker.h + pad },
    { x: blocker.x + blocker.w + pad, y: blocker.y + blocker.h + pad },
  ].filter((point) => !collidesWithWorld(point, WORLD.agentRadius + 3, state.obstacles));
  if (!corners.length) {
    return state.goal;
  }
  let best = corners[0];
  let bestScore = Infinity;
  for (const point of corners) {
    const threatPenalty = clamp(155 - distance(point, state.threat), 0, 155) * 3.4;
    const score = distance(state.agent, point) + distance(point, state.goal) + threatPenalty;
    if (score < bestScore) {
      bestScore = score;
      best = point;
    }
  }
  return best;
}

function tacticalInfo(policy, state, actionIndex) {
  const features = actionFeatures(state, actionIndex);
  const action = ACTIONS[actionIndex];
  const baseScore = scoreAction(policy, state, actionIndex);
  const threatDist = distance(state.agent, state.threat);
  const goalDist = distance(state.agent, state.goal);
  const near = lookaheadPoint(state, actionIndex, 4.2);
  const far = lookaheadPoint(state, actionIndex, 12);
  const nearCollision = collidesWithWorld(near, WORLD.agentRadius + 2, state.obstacles) ? 1 : 0;
  const farCollision = collidesWithWorld(far, WORLD.agentRadius + 2, state.obstacles) ? 1 : 0;
  const nearGoalDist = distance(near, state.goal);
  const farGoalDist = distance(far, state.goal);
  const nearThreatDist = distance(near, state.threat);
  const farThreatDist = distance(far, state.threat);
  const threatPressure = clamp(1 - threatDist / 185, 0, 1);
  const goalProgress = clamp((goalDist - nearGoalDist) / 42, -1, 1);
  const farGoalProgress = clamp((goalDist - farGoalDist) / 115, -1, 1);
  const escapeProgress = clamp((nearThreatDist - threatDist) / 58, -1, 1);
  const farEscape = clamp((farThreatDist - threatDist) / 130, -1, 1);
  const clearance = features[6];
  const tacticalScore =
    baseScore +
    goalProgress * 1.35 +
    farGoalProgress * 1.1 +
    escapeProgress * (0.75 + threatPressure * 1.75) +
    farEscape * threatPressure * 1.25 +
    clearance * 0.85 -
    nearCollision * 3.4 -
    farCollision * 1.55 -
    clamp(1 - nearThreatDist / 135, 0, 1) * 4.2 -
    clamp(1 - farThreatDist / 115, 0, 1) * 2.4 +
    dot(action, { x: features[1], y: features[2] }) * 0.03;
  return {
    actionIndex,
    baseScore,
    tacticalScore,
    features,
    nearCollision,
    farCollision,
    clearance,
    goalProgress,
    farGoalProgress,
    escapeProgress,
    nearThreatDist,
    farThreatDist,
    threatPressure,
  };
}

function externalProbeScore(info, state) {
  const action = ACTIONS[info.actionIndex];
  const target = externalTarget(state);
  let pos = { ...state.agent };
  let blocked = 0;
  let minThreatDist = distance(pos, state.threat);
  let lastOpen = pos;
  for (let i = 0; i < 16; i += 1) {
    const next = {
      x: pos.x + action.x * WORLD.agentSpeed * WORLD.dt,
      y: pos.y + action.y * WORLD.agentSpeed * WORLD.dt,
    };
    if (collidesWithWorld(next, WORLD.agentRadius + 2, state.obstacles)) {
      blocked += 1;
      break;
    }
    pos = next;
    lastOpen = next;
    minThreatDist = Math.min(minThreatDist, distance(pos, state.threat));
  }
  const goalGain = clamp((distance(state.agent, state.goal) - distance(lastOpen, state.goal)) / 150, -1, 1);
  const targetGain = clamp((distance(state.agent, target) - distance(lastOpen, target)) / 90, -1, 1);
  const escape = clamp((distance(lastOpen, state.threat) - distance(state.agent, state.threat)) / 150, -1, 1);
  const threatRisk = clamp(1 - minThreatDist / 92, 0, 1);
  const detourVector = normVector(target.x - state.agent.x, target.y - state.agent.y);
  return (
    info.tacticalScore +
    goalGain * 1.8 +
    targetGain * 2.6 +
    escape * 2.2 +
    dot(action, detourVector) * 1.55 -
    threatRisk * 7.2 -
    blocked * 2.2
  );
}

function featureMapFromValues(values) {
  return Object.fromEntries(D51_FEATURE_NAMES.map((name, index) => [name, values[index]]));
}

export function chooseD51EvidenceAction(policy, features) {
  const feature = featureMapFromValues(features);
  if (feature.external_channel_available >= 0.5) {
    return policy.action_external;
  }
  if (feature.internal_unresolvable_indicator >= 0.5) {
    return policy.action_unresolvable;
  }
  if (feature.dominant_cluster_fraction >= policy.dominant_threshold) {
    return policy.action_dominant;
  }
  if (feature.inverse_margin >= policy.margin_threshold) {
    return policy.action_low_margin;
  }
  if (
    feature.entropy_norm >= policy.entropy_threshold ||
    feature.scalar_confidence <= policy.confidence_threshold
  ) {
    return policy.action_uncertain;
  }
  return policy.action_default;
}

export function d51FeaturesForState(policy, state) {
  const infos = ACTIONS.map((_, index) => tacticalInfo(policy, state, index));
  const baseScores = infos.map((info) => info.baseScore);
  const tacticalScores = infos.map((info) => info.tacticalScore);
  const baseOrder = sortedIndexes(baseScores);
  const tacticalOrder = sortedIndexes(tacticalScores);
  const top = baseOrder[0];
  const top2 = baseOrder[1];
  const stats = softmaxStats(baseScores);
  const rawMargin = baseScores[top] - baseScores[top2];
  const rawSpan = Math.max(0.25, Math.max(...baseScores) - Math.min(...baseScores));
  const inverseMargin = clamp(1 - rawMargin / (rawSpan * 0.25), 0, 1);
  const collisionNorm = infos.reduce((acc, info) => acc + (info.nearCollision || info.farCollision ? 1 : 0), 0) / ACTIONS.length;
  const goalVec = normVector(state.goal.x - state.agent.x, state.goal.y - state.agent.y);
  const awayThreat = normVector(state.agent.x - state.threat.x, state.agent.y - state.threat.y);
  const votes = [top, nearestActionIndex(goalVec), tacticalOrder[0]];
  if (distance(state.agent, state.threat) < 205) {
    votes.push(nearestActionIndex(awayThreat));
  }
  votes.push(infos.reduce((best, info) => (info.clearance > infos[best].clearance ? info.actionIndex : best), 0));
  votes.push(state.lastAction ?? top);
  const counts = new Map();
  for (const vote of votes) {
    counts.set(vote, (counts.get(vote) ?? 0) + 1);
  }
  const rawDominantClusterFraction = Math.max(...counts.values()) / votes.length;
  const supportClusterCountNorm = counts.size / ACTIONS.length;
  const topInfo = infos[top];
  const safeCount = infos.filter((info) => !info.nearCollision && info.nearThreatDist > WORLD.catchDistance + 34).length;
  const threatDist = distance(state.agent, state.threat);
  const topDisagreement = top !== tacticalOrder[0] ? 1 : 0;
  const routeBlocked = Boolean(firstRouteBlocker(state));
  const dominantHazard = clamp(
    collisionNorm * 0.85 +
    topInfo.threatPressure * 0.55 +
    topDisagreement * 0.35 +
    (routeBlocked ? 0.12 : 0),
    0,
    1
  );
  const dominantClusterFraction = clamp(rawDominantClusterFraction * dominantHazard, 0, 1);
  const corridorRisk =
    ((topInfo.nearCollision || topInfo.farCollision) && threatDist < 175) ||
    (topInfo.clearance < 0.14 && threatDist < 150) ||
    (collisionNorm >= 0.62 && threatDist < 145) ||
    (topDisagreement && topInfo.threatPressure > 0.58 && topInfo.goalProgress > 0.12) ||
    (routeBlocked && (topInfo.nearCollision || topInfo.farCollision || topInfo.clearance < 0.22 || state.wallHits > 1));
  const values = [
    1,
    clamp(stats.probs[top], 0, 1),
    inverseMargin,
    stats.entropyNorm,
    clamp(collisionNorm, 0, 1),
    clamp(dominantClusterFraction, 0, 1),
    clamp(supportClusterCountNorm, 0, 1),
    topDisagreement,
    clamp(Math.max(0, ...infos.map((info) => info.goalProgress)) * 0.5 + stats.probs[top] * 0.5, 0, 1),
    clamp(Math.max(0, ...infos.map((info) => info.clearance)) * 0.7 + (1 - collisionNorm) * 0.3, 0, 1),
    clamp(stats.probs[top] * (1 - topInfo.threatPressure * 0.5) * (1 - collisionNorm * 0.35), 0, 1),
    safeCount <= 1 || (threatDist < 64 && safeCount <= 2) ? 1 : 0,
    corridorRisk ? 1 : 0,
  ];
  return {
    values,
    map: featureMapFromValues(values),
    infos,
    baseScores,
    tacticalScores,
    baseOrder,
    tacticalOrder,
  };
}

function chooseBest(scores) {
  let bestIndex = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < scores.length; i += 1) {
    if (scores[i] > bestScore) {
      bestScore = scores[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function refinedScoresForEvidence(evidenceAction, analysis, state) {
  const scores = analysis.baseScores.slice();
  if (evidenceAction === "REQUEST_SUPPORT") {
    for (const info of analysis.infos) {
      scores[info.actionIndex] += info.clearance * 0.65 + info.goalProgress * 0.4 - info.nearCollision * 1.1;
    }
  } else if (evidenceAction === "REQUEST_COUNTER_TOP1_TOP2") {
    for (const index of analysis.baseOrder.slice(0, 2)) {
      scores[index] = analysis.baseScores[index] * 0.65 + analysis.tacticalScores[index] * 0.35;
    }
  } else if (evidenceAction === "REQUEST_JOINT_COUNTER") {
    return analysis.tacticalScores.map((score, index) => analysis.baseScores[index] * 0.45 + score * 0.55);
  } else if (evidenceAction === "REQUEST_EXTERNAL_TEST") {
    return analysis.infos.map((info, index) => analysis.baseScores[index] * 0.55 + externalProbeScore(info, state) * 0.45);
  } else if (evidenceAction === "ABSTAIN") {
    return analysis.infos.map((info) =>
      info.escapeProgress * 2.5 +
      info.clearance * 1.2 -
      info.nearCollision * 3.2 -
      clamp(1 - info.nearThreatDist / 115, 0, 1) * 3.5
    );
  }
  return scores;
}

export function chooseD51Action(policy, state, controller = D51_RULE_TABLE_POLICY) {
  const analysis = d51FeaturesForState(policy, state);
  const evidenceAction = chooseD51EvidenceAction(controller, analysis.values);
  const scores = refinedScoresForEvidence(evidenceAction, analysis, state);
  const actionIndex = chooseBest(scores);
  const confidence = analysis.map.scalar_confidence;
  return {
    actionIndex,
    scores,
    baseScores: analysis.baseScores,
    tacticalScores: analysis.tacticalScores,
    evidenceAction,
    evidenceCost: D51_EVIDENCE_COST[evidenceAction] ?? 0,
    d51Features: analysis.map,
    d51FeatureValues: analysis.values,
    confidence,
    controller: controller.source,
  };
}
