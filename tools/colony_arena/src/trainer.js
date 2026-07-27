import {
  GENE_COUNT,
  makeRng,
  randomNormal,
  makeScenarioSet,
  simulateEpisode,
} from "./sim.js";
import { chooseD51Action } from "./d51_controller.js";

export function makePolicy(weights, label = "policy") {
  return { label, weights: weights.slice(0, GENE_COUNT) };
}

export function priorPolicy() {
  return makePolicy([
    0.0,
    1.15,
    0.55,
    1.65,
    1.35,
    -2.25,
    0.38,
    0.16,
    0.12,
    0.72,
    -0.04,
  ], "seeded_goal_flee_prior");
}

export function randomPolicy(rng, label = "random") {
  return makePolicy(
    Array.from({ length: GENE_COUNT }, () => randomNormal(rng) * 0.9),
    label
  );
}

export function mutatePolicy(parent, rng, sigma, label = "mutant") {
  return makePolicy(
    parent.weights.map((w) => {
      const jump = rng() < 0.08 ? randomNormal(rng) * sigma * 2.8 : 0;
      return w + randomNormal(rng) * sigma + jump;
    }),
    label
  );
}

export function crossoverPolicy(a, b, rng, label = "crossover") {
  return makePolicy(
    a.weights.map((w, idx) => {
      const mix = rng();
      const base = mix < 0.45 ? w : mix < 0.9 ? b.weights[idx] : (w + b.weights[idx]) * 0.5;
      return base;
    }),
    label
  );
}

export function evaluatePolicy(policy, scenarios, options = {}) {
  const chooseActionFn = options.chooseActionFn ?? chooseD51Action;
  const episodes = scenarios.map((scenario) => simulateEpisode(policy, scenario, { chooseActionFn }).summary);
  const n = episodes.length || 1;
  const mean = (key) => episodes.reduce((acc, row) => acc + row[key], 0) / n;
  const evidenceCounts = {};
  for (const episode of episodes) {
    for (const [action, count] of Object.entries(episode.evidenceActions ?? {})) {
      evidenceCounts[action] = (evidenceCounts[action] ?? 0) + count;
    }
  }
  const decisionCount = episodes.reduce((acc, row) => acc + (row.decisionCount ?? 0), 0) || 1;
  const successRate = episodes.filter((row) => row.collected > 0 && !row.caught).length / n;
  const caughtRate = episodes.filter((row) => row.caught).length / n;
  const score =
    mean("score") +
    successRate * 120 +
    mean("collected") * 35 -
    caughtRate * 70 -
    mean("wallHits") * 1.25 -
    mean("infoCost") * 0.35;
  return {
    policy,
    score,
    successRate,
    caughtRate,
    meanGoals: mean("collected"),
    meanWallHits: mean("wallHits"),
    meanInfoCost: mean("infoCost"),
    nonDecideRate: 1 - ((evidenceCounts.DECIDE ?? 0) / decisionCount),
    evidenceCounts,
    meanEfficiency: mean("efficiency"),
    meanFinalGoalDistance: mean("finalGoalDistance"),
    episodes,
  };
}

export class EvolutionTrainer {
  constructor(options = {}) {
    this.populationSize = options.populationSize ?? 56;
    this.eliteCount = options.eliteCount ?? 7;
    this.mutationSigma = options.mutationSigma ?? 0.42;
    this.seed = options.seed ?? 73001;
    this.includePrior = options.includePrior ?? true;
    this.trainingScenarios = options.trainingScenarios ?? makeScenarioSet(9101, 15, "mixed");
    this.testScenarios = options.testScenarios ?? makeScenarioSet(12001, 8, "mixed");
    this.adversarialScenarios = options.adversarialScenarios ?? makeScenarioSet(53001, 10, "adversarial");
    this.rng = makeRng(this.seed);
    this.generation = 0;
    this.population = [];
    this.history = [];
    this.best = null;
    this.reset();
  }

  reset() {
    this.rng = makeRng(this.seed);
    this.generation = 0;
    this.history = [];
    this.population = [];
    if (this.includePrior) {
      this.population.push(priorPolicy());
      this.population.push(mutatePolicy(priorPolicy(), this.rng, 0.7, "prior_mutant"));
    }
    while (this.population.length < this.populationSize) {
      this.population.push(randomPolicy(this.rng, `random_${this.population.length}`));
    }
    this.best = null;
  }

  stepGeneration() {
    const scored = this.population
      .map((policy) => evaluatePolicy(policy, this.trainingScenarios))
      .sort((a, b) => b.score - a.score);
    const bestTrain = scored[0];
    const test = evaluatePolicy(bestTrain.policy, this.testScenarios);
    const adversarial = evaluatePolicy(bestTrain.policy, this.adversarialScenarios);
    const validationScore = test.score + adversarial.score * 0.35;
    if (!this.best || validationScore > this.best.validationScore) {
      this.best = {
        train: bestTrain,
        test,
        adversarial,
        validationScore,
        policy: makePolicy(bestTrain.policy.weights, `best_g${this.generation}`),
      };
    }
    const avgScore = scored.reduce((acc, row) => acc + row.score, 0) / scored.length;
    const metric = {
      generation: this.generation,
      best_score: bestTrain.score,
      avg_score: avgScore,
      train_success_rate: bestTrain.successRate,
      train_caught_rate: bestTrain.caughtRate,
      train_mean_goals: bestTrain.meanGoals,
      train_wall_hits: bestTrain.meanWallHits,
      test_score: test.score,
      validation_score: validationScore,
      test_success_rate: test.successRate,
      test_caught_rate: test.caughtRate,
      test_mean_goals: test.meanGoals,
      test_info_cost: test.meanInfoCost,
      test_non_decide_rate: test.nonDecideRate,
      adversarial_success_rate: adversarial.successRate,
      adversarial_caught_rate: adversarial.caughtRate,
      adversarial_mean_goals: adversarial.meanGoals,
      adversarial_info_cost: adversarial.meanInfoCost,
      policy_weights: bestTrain.policy.weights.slice(),
      evidence_counts: test.evidenceCounts,
    };
    this.history.push(metric);
    this.population = this.nextPopulation(scored);
    this.generation += 1;
    return metric;
  }

  nextPopulation(scored) {
    const next = [];
    const elites = scored.slice(0, this.eliteCount).map((row) => row.policy);
    for (const elite of elites) {
      next.push(makePolicy(elite.weights, "elite"));
    }
    if (this.includePrior && next.length < this.populationSize) {
      next.push(mutatePolicy(priorPolicy(), this.rng, this.mutationSigma * 0.45, "prior_refresh"));
    }
    while (next.length < this.populationSize) {
      const roll = this.rng();
      if (roll < 0.62) {
        const parent = elites[Math.floor(this.rng() * elites.length)];
        next.push(mutatePolicy(parent, this.rng, this.mutationSigma, "elite_mutant"));
      } else if (roll < 0.88) {
        const a = elites[Math.floor(this.rng() * elites.length)];
        const b = elites[Math.floor(this.rng() * elites.length)];
        next.push(mutatePolicy(crossoverPolicy(a, b, this.rng), this.rng, this.mutationSigma * 0.6, "cross_mutant"));
      } else {
        next.push(randomPolicy(this.rng, "immigrant"));
      }
    }
    return next;
  }

  trainGenerations(count, onMetric = null) {
    const metrics = [];
    for (let i = 0; i < count; i += 1) {
      const metric = this.stepGeneration();
      metrics.push(metric);
      if (onMetric) {
        onMetric(metric);
      }
    }
    return metrics;
  }

  bestPolicy() {
    if (this.best?.policy) {
      return makePolicy(this.best.policy.weights, this.best.policy.label);
    }
    return priorPolicy();
  }

  sampleReplay(index = 0) {
    const scenario = this.testScenarios[index % this.testScenarios.length];
    return simulateEpisode(this.bestPolicy(), scenario, { record: true, chooseActionFn: chooseD51Action });
  }

  adversarialReplay(index = 0) {
    const scenario = this.adversarialScenarios[index % this.adversarialScenarios.length];
    return simulateEpisode(this.bestPolicy(), scenario, { record: true, chooseActionFn: chooseD51Action });
  }
}
