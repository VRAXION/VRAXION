import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const root = process.cwd();
const workerPath = path.join(root, "workers", "instnct-notify", "src", "index.mjs");
const migrationPath = path.join(root, "workers", "instnct-notify", "migrations", "0001_init.sql");
const wranglerExamplePath = path.join(root, "workers", "instnct-notify", "wrangler.example.jsonc");
const source = fs.readFileSync(workerPath, "utf8");
const migration = fs.readFileSync(migrationPath, "utf8");
const wranglerExample = fs.readFileSync(wranglerExamplePath, "utf8");

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
    if (this.sql.startsWith("insert into notify_subscribers")) {
      const [id, email, emailHash, source, ipHash, userAgentHash] = this.args;
      if (!this.db.subscribers.has(emailHash)) {
        this.db.subscribers.set(emailHash, {
          id,
          email,
          email_hash: emailHash,
          source,
          ip_hash: ipHash,
          user_agent_hash: userAgentHash,
        });
      }
      return { success: true };
    }
    throw new Error(`unexpected run SQL: ${this.sql}`);
  }
}

function request(method, pathName, body, origin = "https://vraxion.github.io") {
  const headers = new Headers();
  if (origin) headers.set("origin", origin);
  headers.set("user-agent", "instnct-audit");
  headers.set("cf-connecting-ip", "203.0.113.10");
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
  for (const token of ["EMAIL_HASH_PEPPER", "notify_subscribers", "notify_rate_limits", "crypto.randomUUID"]) {
    assert.equal(source.includes(token), true, `worker source missing ${token}`);
  }
  for (const token of ["CREATE TABLE IF NOT EXISTS notify_subscribers", "email_hash TEXT NOT NULL UNIQUE", "CREATE TABLE IF NOT EXISTS notify_rate_limits"]) {
    assert.equal(migration.includes(token), true, `migration missing ${token}`);
  }
  for (const token of ["d1_databases", "vraxion-instnct-notify", "migrations_dir", "2026-07-04"]) {
    assert.equal(wranglerExample.includes(token), true, `wrangler example missing ${token}`);
  }
}

const worker = (await import(pathToFileURL(workerPath))).default;
const env = {
  DB: new FakeD1(),
  EMAIL_HASH_PEPPER: "test-pepper-value-with-length",
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

response = await worker.fetch(request("POST", "/api/notify", { email: "third@example.com" }), { ...env, EMAIL_HASH_PEPPER: "" });
assert.equal(response.status, 500);
assert.deepEqual(await readJson(response), { error: "notify backend is not configured" });

const rateEnv = { ...env, DB: new FakeD1(), RATE_LIMIT_PER_HOUR: "1" };
response = await worker.fetch(request("POST", "/api/notify", { email: "one@example.com" }), rateEnv);
assert.equal(response.status, 201);
response = await worker.fetch(request("POST", "/api/notify", { email: "two@example.com" }), rateEnv);
assert.equal(response.status, 429);

console.log("instnct_notify_worker_audit=pass");
