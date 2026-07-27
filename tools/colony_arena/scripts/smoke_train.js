import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { EvolutionTrainer, evaluatePolicy } from "../src/trainer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../../..");

function parseArgs(argv) {
  const out = {
    generations: 55,
    population: 56,
    out: null,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--generations") {
      i += 1;
      out.generations = Number(argv[i]);
    } else if (arg === "--population") {
      i += 1;
      out.population = Number(argv[i]);
    } else if (arg === "--out") {
      i += 1;
      out.out = argv[i];
    }
  }
  return out;
}

function nowId() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function writeJsonAtomic(file, value) {
  ensureDir(path.dirname(file));
  const tmp = `${file}.tmp`;
  fs.writeFileSync(tmp, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  fs.renameSync(tmp, file);
}

function appendJsonl(file, value) {
  ensureDir(path.dirname(file));
  fs.appendFileSync(file, `${JSON.stringify(value)}\n`, "utf8");
}

function writeReport(file, summary) {
  const lines = [
    "# Colony Arena Smoke",
    "",
    "Status:",
    "",
    "```text",
    summary.pass ? "PASS" : "FAIL",
    "```",
    "",
    "Metrics:",
    "",
    "```text",
    `initial_best_score = ${summary.initial_best_score.toFixed(4)}`,
    `final_best_score = ${summary.final_best_score.toFixed(4)}`,
    `improvement = ${summary.improvement.toFixed(4)}`,
    `final_test_success_rate = ${summary.final_test_success_rate.toFixed(4)}`,
    `final_test_caught_rate = ${summary.final_test_caught_rate.toFixed(4)}`,
    `final_test_mean_goals = ${summary.final_test_mean_goals.toFixed(4)}`,
    `final_adversarial_success_rate = ${summary.final_adversarial_success_rate.toFixed(4)}`,
    `final_adversarial_caught_rate = ${summary.final_adversarial_caught_rate.toFixed(4)}`,
    `final_test_non_decide_rate = ${summary.final_test_non_decide_rate.toFixed(4)}`,
    `final_test_info_cost = ${summary.final_test_info_cost.toFixed(4)}`,
    "```",
    "",
    "Boundary: this proves only a D51-gated movement/avoidance policy in the local arena, not raw visual game intelligence.",
    "",
  ];
  fs.writeFileSync(file, lines.join("\n"), "utf8");
}

const args = parseArgs(process.argv);
const outRoot = path.resolve(
  args.out ?? path.join(repoRoot, "target", "colony_arena_smoke", nowId())
);
ensureDir(outRoot);
const started = Date.now();
const progressPath = path.join(outRoot, "progress.jsonl");

writeJsonAtomic(path.join(outRoot, "queue.json"), {
  task: "colony_arena_evolutionary_movement_smoke",
  status: "running",
  no_black_box: true,
  generations: args.generations,
  population: args.population,
  heartbeat: "progress.jsonl receives one row per generation",
  started_unix_ms: started,
});
appendJsonl(progressPath, {
  event: "queue_written",
  elapsed_sec: 0,
  out: outRoot,
});

const trainer = new EvolutionTrainer({
  populationSize: args.population,
  eliteCount: Math.max(4, Math.round(args.population * 0.13)),
  seed: 73001,
  includePrior: true,
});

let initial = null;
let latest = null;
for (let gen = 0; gen < args.generations; gen += 1) {
  latest = trainer.stepGeneration();
  if (!initial) {
    initial = latest;
  }
  appendJsonl(progressPath, {
    event: "generation",
    generation: latest.generation,
    elapsed_sec: (Date.now() - started) / 1000,
    best_score: latest.best_score,
    avg_score: latest.avg_score,
    train_success_rate: latest.train_success_rate,
    test_success_rate: latest.test_success_rate,
    test_caught_rate: latest.test_caught_rate,
    adversarial_success_rate: latest.adversarial_success_rate,
    adversarial_caught_rate: latest.adversarial_caught_rate,
    test_non_decide_rate: latest.test_non_decide_rate,
  });
  writeJsonAtomic(path.join(outRoot, "partial_summary.json"), {
    generation: latest.generation,
    elapsed_sec: (Date.now() - started) / 1000,
    best_score: latest.best_score,
    test_success_rate: latest.test_success_rate,
    test_caught_rate: latest.test_caught_rate,
  });
}

const replay = trainer.sampleReplay(0);
const adversarialReplay = trainer.adversarialReplay(0);
const finalPolicy = trainer.bestPolicy();
const finalTest = evaluatePolicy(finalPolicy, trainer.testScenarios);
const finalAdversarial = evaluatePolicy(finalPolicy, trainer.adversarialScenarios);
const finalBestScore = trainer.best?.validationScore ?? latest.validation_score ?? latest.best_score;
const summary = {
  pass: false,
  generations: args.generations,
  population: args.population,
  initial_best_score: initial.best_score,
  final_best_score: finalBestScore,
  improvement: finalBestScore - (initial.validation_score ?? initial.best_score),
  initial_test_success_rate: initial.test_success_rate,
  final_test_success_rate: finalTest.successRate,
  final_test_caught_rate: finalTest.caughtRate,
  final_test_mean_goals: finalTest.meanGoals,
  final_test_info_cost: finalTest.meanInfoCost,
  final_test_non_decide_rate: finalTest.nonDecideRate,
  final_adversarial_success_rate: finalAdversarial.successRate,
  final_adversarial_caught_rate: finalAdversarial.caughtRate,
  final_adversarial_mean_goals: finalAdversarial.meanGoals,
  final_adversarial_info_cost: finalAdversarial.meanInfoCost,
  final_train_success_rate: latest.train_success_rate,
  final_train_caught_rate: latest.train_caught_rate,
  evidence_counts: finalTest.evidenceCounts,
  final_policy_label: finalPolicy.label,
  controller: "D51_MUTABLE_RULE_TABLE_CONTROLLER",
  smoke_gates: {
    improvement_min: 4.0,
    final_test_success_rate_min: 0.5,
    final_test_caught_rate_max: 0.55,
    final_adversarial_success_rate_min: 0.35,
    final_adversarial_caught_rate_max: 0.60,
    final_test_non_decide_rate_min: 0.05,
  },
};
summary.pass =
  summary.improvement >= summary.smoke_gates.improvement_min &&
  summary.final_test_success_rate >= summary.smoke_gates.final_test_success_rate_min &&
  summary.final_test_caught_rate <= summary.smoke_gates.final_test_caught_rate_max &&
  summary.final_adversarial_success_rate >= summary.smoke_gates.final_adversarial_success_rate_min &&
  summary.final_adversarial_caught_rate <= summary.smoke_gates.final_adversarial_caught_rate_max &&
  summary.final_test_non_decide_rate >= summary.smoke_gates.final_test_non_decide_rate_min;

writeJsonAtomic(path.join(outRoot, "best_policy.json"), finalPolicy);
writeJsonAtomic(path.join(outRoot, "sample_replay.json"), replay);
writeJsonAtomic(path.join(outRoot, "adversarial_replay.json"), adversarialReplay);
writeJsonAtomic(path.join(outRoot, "summary.json"), summary);
writeReport(path.join(outRoot, "report.md"), summary);
writeJsonAtomic(path.join(outRoot, "queue.json"), {
  task: "colony_arena_evolutionary_movement_smoke",
  status: summary.pass ? "complete" : "failed",
  no_black_box: true,
  generations: args.generations,
  population: args.population,
  elapsed_sec: (Date.now() - started) / 1000,
  summary: "summary.json",
});
appendJsonl(progressPath, {
  event: "final",
  elapsed_sec: (Date.now() - started) / 1000,
  pass: summary.pass,
  final_best_score: summary.final_best_score,
  final_test_success_rate: summary.final_test_success_rate,
  final_test_caught_rate: summary.final_test_caught_rate,
});

console.log(JSON.stringify({
  status: summary.pass ? "pass" : "fail",
  out: outRoot,
  improvement: summary.improvement,
  final_test_success_rate: summary.final_test_success_rate,
  final_test_caught_rate: summary.final_test_caught_rate,
  final_adversarial_success_rate: summary.final_adversarial_success_rate,
  final_adversarial_caught_rate: summary.final_adversarial_caught_rate,
  final_test_non_decide_rate: summary.final_test_non_decide_rate,
}, null, 2));

if (!summary.pass) {
  process.exitCode = 1;
}
