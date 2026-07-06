import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const root = process.cwd();
const workerPath = path.join(root, "workers", "instnct-notify", "src", "index.mjs");
const migrationPath = path.join(root, "workers", "instnct-notify", "migrations", "0001_init.sql");
const wranglerExamplePath = path.join(root, "workers", "instnct-notify", "wrangler.example.jsonc");
const readmePath = path.join(root, "workers", "instnct-notify", "README.md");
const deployWorkflowPath = path.join(root, ".github", "workflows", "deploy-instnct-notify.yml");
const source = fs.readFileSync(workerPath, "utf8");
const migration = fs.readFileSync(migrationPath, "utf8");
const wranglerExample = fs.readFileSync(wranglerExamplePath, "utf8");
const readme = fs.readFileSync(readmePath, "utf8");
const deployWorkflow = fs.readFileSync(deployWorkflowPath, "utf8");

class FakeD1 {
  constructor() {
    this.subscribers = new Map();
    this.rateLimits = new Map();
  }

  prepare(sql) {
    return new FakeStatement(this, sql);
  }
}

class FakeStatement {
  constructor(db, sql) {
    this.db = db;
    this.sql = sql.replace(/\s+/g, " ").trim().toLowerCase();
    this.args = [];
  }

  bind(...args) {
    this.args = args;
    return this;
  }

  async first() {
    if (this.sql.includes("count(*) as count from notify_subscribers")) {
      return { count: this.db.subscribers.size };
    }
    if (this.sql.includes("from notify_rate_limits where key")) {
      const row = this.db.rateLimits.get(this.args[0]);
      return row ? { ...row } : null;
    }
    if (this.sql.includes("from notify_subscribers where email_hash")) {
      const row = this.db.subscribers.get(this.args[0]);
      return row ? { ...row } : null;
    }
    throw new Error(`unexpected first SQL: ${this.sql}`);
  }

  async all() {
    if (this.sql.startsWith("select email, source, created_at from notify_subscribers")) {
      const limit = Number(this.args[0] || 10000);
      const results = [...this.db.subscribers.values()]
        .sort((a, b) => String(a.created_at).localeCompare(String(b.created_at)))
        .slice(0, limit)
        .map(({ email, source, created_at }) => ({ email, source, created_at }));
      return { results };
    }
    throw new Error(`unexpected all SQL: ${this.sql}`);
  }

  async run() {
    if (this.sql.startsWith("insert into notify_rate_limits")) {
      const key = this.args[0];
      const windowStart = this.args[1];
      const current = this.db.rateLimits.get(key);
      this.db.rateLimits.set(key, {
        key,
        window_start: windowStart,
        count: current ? current.count + 1 : 1,
      });
      return { success: true };
    }
    if (this.sql.startsWith("insert or ignore into notify_subscribers")) {
      const [id, email, emailHash, source, ipHash, userAgentHash] = this.args;
      if (this.db.subscribers.has(emailHash)) {
        return { success: true, meta: { changes: 0 } };
      }
      this.db.subscribers.set(emailHash, {
        id,
        email,
        email_hash: emailHash,
        source,
        ip_hash: ipHash,
        user_agent_hash: userAgentHash,
        created_at: new Date().toISOString(),
      });
      return { success: true, meta: { changes: 1 } };
    }
    if (this.sql.startsWith("delete from notify_subscribers where email_hash")) {
      const deleted = this.db.subscribers.delete(this.args[0]) ? 1 : 0;
      return { success: true, meta: { changes: deleted } };
    }
    if (this.sql.startsWith("delete from notify_rate_limits where window_start")) {
      const cutoff = this.args[0];
      let deleted = 0;
      for (const [key, row] of this.db.rateLimits) {
        if (row.window_start < cutoff) {
          this.db.rateLimits.delete(key);
          deleted += 1;
        }
      }
      return { success: true, meta: { changes: deleted } };
    }
    throw new Error(`unexpected run SQL: ${this.sql}`);
  }
}

function request(method, pathName, body, origin = "https://vraxion.github.io", extraHeaders = {}) {
  const headers = new Headers();
  if (origin) headers.set("origin", origin);
  headers.set("user-agent", "instnct-audit");
  headers.set("cf-connecting-ip", "203.0.113.10");
  for (const [key, value] of Object.entries(extraHeaders)) headers.set(key, value);
  const init = { method, headers };
  if (body !== undefined) {
    headers.set("content-type", "application/json");
    init.body = JSON.stringify(body);
  }
  return new Request(`https://notify.example.test${pathName}`, init);
}

async function readJson(response) {
  return response.text().then((text) => (text ? JSON.parse(text) : null));
}

function assertSourceGuards() {
  for (const token of ["Math.random", "console.log", "localStorage", "sessionStorage", "eval("]) {
    assert.equal(source.includes(token), false, `worker source must not contain ${token}`);
  }
  for (const token of ["EMAIL_HASH_PEPPER", "ADMIN_TOKEN", "notify_subscribers", "notify_rate_limits", "crypto.randomUUID", "INSERT OR IGNORE", "scheduled("]) {
    assert.equal(source.includes(token), true, `worker source missing ${token}`);
  }
  for (const token of ["CREATE TABLE IF NOT EXISTS notify_subscribers", "email_hash TEXT NOT NULL UNIQUE", "CREATE TABLE IF NOT EXISTS notify_rate_limits"]) {
    assert.equal(migration.includes(token), true, `migration missing ${token}`);
  }
  for (const token of ["d1_databases", "vraxion-instnct-notify", "migrations_dir", "RATE_LIMIT_RETENTION_HOURS", "crons", "2026-07-04"]) {
    assert.equal(wranglerExample.includes(token), true, `wrangler example missing ${token}`);
  }
  for (const token of [
    "Preflight required secrets",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
    "INSTNCT_NOTIFY_D1_DATABASE_ID",
    "INSTNCT_NOTIFY_EMAIL_HASH_PEPPER",
    "INSTNCT_NOTIFY_ADMIN_TOKEN",
    "INSTNCT_NOTIFY_API_BASE",
    "Missing GitHub secret",
    "secret put ADMIN_TOKEN",
    "ADMIN_TOKEN",
  ]) {
    assert.equal(deployWorkflow.includes(token), true, `deploy workflow missing ${token}`);
  }
  for (const token of [
    "npx wrangler secret put ADMIN_TOKEN",
    "GET /admin/notify/export",
    "cleanup-rate-limits",
    "scheduled cleanup trigger",
    "Local config hygiene",
    "Do not\ncommit local config or operator output",
    "`wrangler.jsonc`",
    ".dev.vars",
    "real D1 database ids",
    "operator export/delete output",
    "node scripts\\audit_public_secrets.mjs",
    "node scripts\\audit_instnct_notify_worker.mjs",
  ]) {
    assert.equal(readme.includes(token), true, `worker README missing ${token}`);
  }
}

const worker = (await import(pathToFileURL(workerPath))).default;
const env = {
  DB: new FakeD1(),
  EMAIL_HASH_PEPPER: "test-pepper-value-with-length",
  ADMIN_TOKEN: "test-admin-token-with-enough-length",
  ALLOWED_ORIGIN: "https://vraxion.github.io",
  SUBSCRIBER_GOAL: "1000",
  RATE_LIMIT_PER_HOUR: "20",
};

assertSourceGuards();

let response = await worker.fetch(request("GET", "/health"), env);
assert.equal(response.status, 200);
assert.deepEqual(await readJson(response), { ok: true });

response = await worker.fetch(request("OPTIONS", "/api/notify"), env);
assert.equal(response.status, 204);
assert.equal(response.headers.get("access-control-allow-origin"), "https://vraxion.github.io");

response = await worker.fetch(request("GET", "/api/notify"), env);
assert.equal(response.status, 200);
assert.deepEqual(await readJson(response), { count: 0, goal: 1000 });

response = await worker.fetch(request("POST", "/api/notify", { email: "not-an-email" }), env);
assert.equal(response.status, 400);

response = await worker.fetch(request("POST", "/api/notify", { email: "bot@example.com", website: "filled" }), env);
assert.equal(response.status, 202);
assert.equal(env.DB.subscribers.size, 0);

response = await worker.fetch(request("POST", "/api/notify", { email: "User@Example.COM", source: "instnct-site" }), env);
assert.equal(response.status, 201);
assert.equal(env.DB.subscribers.size, 1);
const stored = [...env.DB.subscribers.values()][0];
assert.equal(stored.email, "user@example.com");
assert.equal(typeof stored.email_hash, "string");
assert.notEqual(stored.email_hash, stored.email);

response = await worker.fetch(request("POST", "/api/notify", { email: "user@example.com" }), env);
assert.equal(response.status, 200);
assert.equal(env.DB.subscribers.size, 1);

response = await worker.fetch(request("GET", "/api/notify"), env);
assert.equal(response.status, 200);
assert.deepEqual(await readJson(response), { count: 1, goal: 1000 });

response = await worker.fetch(request("POST", "/api/notify", { email: "other@example.com" }, "https://bad.example"), env);
assert.equal(response.status, 403);

response = await worker.fetch(request("GET", "/admin/notify/export", undefined, null), env);
assert.equal(response.status, 401);

response = await worker.fetch(
  request("GET", "/admin/notify/export", undefined, null, { authorization: "Bearer wrong-token" }),
  env
);
assert.equal(response.status, 403);

response = await worker.fetch(
  request("GET", "/admin/notify/export", undefined, null, { authorization: `Bearer ${env.ADMIN_TOKEN}` }),
  env
);
assert.equal(response.status, 200);
const exported = await readJson(response);
assert.equal(exported.count, 1);
assert.deepEqual(exported.subscribers.map((row) => row.email), ["user@example.com"]);
assert.equal(exported.subscribers[0].email_hash, undefined);

const oldWindow = "2026-01-01T00:00:00.000Z";
env.DB.rateLimits.set("old", { key: "old", count: 1, window_start: oldWindow });
response = await worker.fetch(
  request("POST", "/admin/notify/cleanup-rate-limits", { olderThanHours: 1 }, null, { authorization: `Bearer ${env.ADMIN_TOKEN}` }),
  env
);
assert.equal(response.status, 200);
assert.equal((await readJson(response)).deleted >= 1, true);
assert.equal(env.DB.rateLimits.has("old"), false);

env.DB.rateLimits.set("scheduled-old", { key: "scheduled-old", count: 1, window_start: oldWindow });
const scheduledTasks = [];
assert.equal(typeof worker.scheduled, "function");
await worker.scheduled({ cron: "17 */6 * * *" }, env, {
  waitUntil(task) {
    scheduledTasks.push(task);
  },
});
await Promise.all(scheduledTasks);
assert.equal(env.DB.rateLimits.has("scheduled-old"), false);

response = await worker.fetch(
  request("POST", "/admin/notify/delete", { email: "user@example.com" }, null, { authorization: `Bearer ${env.ADMIN_TOKEN}` }),
  env
);
assert.equal(response.status, 200);
assert.deepEqual(await readJson(response), { deleted: 1 });
assert.equal(env.DB.subscribers.size, 0);

response = await worker.fetch(request("POST", "/api/notify", { email: "third@example.com" }), { ...env, EMAIL_HASH_PEPPER: "" });
assert.equal(response.status, 500);
assert.deepEqual(await readJson(response), { error: "notify backend is not configured" });

const rateEnv = { ...env, DB: new FakeD1(), RATE_LIMIT_PER_HOUR: "1" };
response = await worker.fetch(request("POST", "/api/notify", { email: "one@example.com" }), rateEnv);
assert.equal(response.status, 201);
response = await worker.fetch(request("POST", "/api/notify", { email: "two@example.com" }), rateEnv);
assert.equal(response.status, 429);

console.log("instnct_notify_worker_audit=pass");
