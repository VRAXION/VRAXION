const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_EMAIL_LENGTH = 320;
const MAX_BODY_BYTES = 4096;
const DEFAULT_GOAL = 1000;
const DEFAULT_RATE_LIMIT = 20;
const DEFAULT_ALLOWED_ORIGIN = "https://vraxion.github.io";

function json(data, init = {}) {
  const headers = new Headers(init.headers || {});
  headers.set("content-type", "application/json; charset=utf-8");
  headers.set("cache-control", "no-store");
  headers.set("x-content-type-options", "nosniff");
  return new Response(JSON.stringify(data), { ...init, headers });
}

function parsePositiveInt(value, fallback) {
  const parsed = Number.parseInt(String(value || ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function allowedOrigins(env) {
  const raw = String(env.ALLOWED_ORIGIN || DEFAULT_ALLOWED_ORIGIN);
  return raw
    .split(",")
    .map((origin) => origin.trim())
    .filter(Boolean);
}

function isAllowedOrigin(request, env) {
  const origin = request.headers.get("origin");
  if (!origin) return true;
  return allowedOrigins(env).includes(origin);
}

function corsHeaders(request, env) {
  const headers = new Headers({
    vary: "Origin",
    "access-control-allow-methods": "GET, POST, OPTIONS",
    "access-control-allow-headers": "content-type",
    "access-control-max-age": "86400",
  });
  const origin = request.headers.get("origin");
  if (origin && allowedOrigins(env).includes(origin)) {
    headers.set("access-control-allow-origin", origin);
  }
  return headers;
}

function withCors(response, request, env) {
  const headers = new Headers(response.headers);
  for (const [key, value] of corsHeaders(request, env)) headers.set(key, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function methodNotAllowed(request, env) {
  return withCors(json({ error: "method not allowed" }, { status: 405, headers: { allow: "GET, POST, OPTIONS" } }), request, env);
}

function normalizeEmail(value) {
  return String(value || "").trim().toLowerCase();
}

function normalizeSource(value) {
  const source = String(value || "instnct-site").trim().toLowerCase();
  return /^[a-z0-9._:-]{1,64}$/.test(source) ? source : "instnct-site";
}

async function sha256Hex(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function hourWindow(now = new Date()) {
  const d = new Date(now);
  d.setUTCMinutes(0, 0, 0);
  return d.toISOString();
}

async function subscriberCount(env) {
  const row = await env.DB.prepare("SELECT COUNT(*) AS count FROM notify_subscribers").first();
  return Number(row?.count || 0);
}

function assertConfig(env) {
  if (!env.DB) throw new Error("missing DB binding");
  if (!env.EMAIL_HASH_PEPPER || String(env.EMAIL_HASH_PEPPER).length < 16) {
    throw new Error("missing EMAIL_HASH_PEPPER secret");
  }
}

async function readJsonBody(request) {
  const text = await request.text();
  if (text.length > MAX_BODY_BYTES) {
    return { error: json({ error: "request too large" }, { status: 413 }) };
  }
  try {
    return { data: text ? JSON.parse(text) : null };
  } catch {
    return { error: json({ error: "invalid JSON" }, { status: 400 }) };
  }
}

async function applyRateLimit(request, env) {
  const limit = parsePositiveInt(env.RATE_LIMIT_PER_HOUR, DEFAULT_RATE_LIMIT);
  const ip = request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "unknown";
  const key = await sha256Hex(`notify-rate:${hourWindow()}:${ip}:${env.EMAIL_HASH_PEPPER}`);
  const windowStart = hourWindow();

  await env.DB.prepare(
    `INSERT INTO notify_rate_limits (key, count, window_start, updated_at)
     VALUES (?1, 1, ?2, CURRENT_TIMESTAMP)
     ON CONFLICT(key) DO UPDATE SET count = count + 1, updated_at = CURRENT_TIMESTAMP`
  )
    .bind(key, windowStart)
    .run();

  const row = await env.DB.prepare("SELECT count FROM notify_rate_limits WHERE key = ?1").bind(key).first();
  const count = Number(row?.count || 0);
  if (count > limit) {
    return json({ error: "rate limited" }, { status: 429 });
  }
  return null;
}

async function handleGet(request, env) {
  if (!env.DB) return withCors(json({ count: 0, goal: parsePositiveInt(env.SUBSCRIBER_GOAL, DEFAULT_GOAL) }), request, env);
  const count = await subscriberCount(env);
  return withCors(json({ count, goal: parsePositiveInt(env.SUBSCRIBER_GOAL, DEFAULT_GOAL) }), request, env);
}

async function handlePost(request, env) {
  if (!isAllowedOrigin(request, env)) {
    return withCors(json({ error: "origin not allowed" }, { status: 403 }), request, env);
  }

  assertConfig(env);

  const type = request.headers.get("content-type") || "";
  if (!type.toLowerCase().includes("application/json")) {
    return withCors(json({ error: "content-type must be application/json" }, { status: 415 }), request, env);
  }

  const { data, error } = await readJsonBody(request);
  if (error) return withCors(error, request, env);

  if (typeof data?.website === "string" && data.website.trim()) {
    return withCors(json({ accepted: true }, { status: 202 }), request, env);
  }

  const email = normalizeEmail(data?.email);
  if (!email || email.length > MAX_EMAIL_LENGTH || !EMAIL_RE.test(email)) {
    return withCors(json({ error: "Please provide a valid email." }, { status: 400 }), request, env);
  }

  const rateLimited = await applyRateLimit(request, env);
  if (rateLimited) return withCors(rateLimited, request, env);

  const emailHash = await sha256Hex(`notify-email:${email}:${env.EMAIL_HASH_PEPPER}`);
  const existing = await env.DB.prepare("SELECT id FROM notify_subscribers WHERE email_hash = ?1").bind(emailHash).first();
  if (existing) {
    return withCors(json({ message: "You're already on the list. We'll signal when T1 is ready." }), request, env);
  }

  const ip = request.headers.get("cf-connecting-ip") || request.headers.get("x-forwarded-for") || "";
  const ua = request.headers.get("user-agent") || "";
  const ipHash = ip ? await sha256Hex(`notify-ip:${ip}:${env.EMAIL_HASH_PEPPER}`) : null;
  const userAgentHash = ua ? await sha256Hex(`notify-ua:${ua}:${env.EMAIL_HASH_PEPPER}`) : null;
  const id = crypto.randomUUID();
  const source = normalizeSource(data?.source);

  const insert = await env.DB.prepare(
    `INSERT OR IGNORE INTO notify_subscribers (id, email, email_hash, source, ip_hash, user_agent_hash, created_at)
     VALUES (?1, ?2, ?3, ?4, ?5, ?6, CURRENT_TIMESTAMP)`
  )
    .bind(id, email, emailHash, source, ipHash, userAgentHash)
    .run();

  const changes = Number(insert?.meta?.changes ?? insert?.changes ?? 1);
  if (changes === 0) {
    return withCors(json({ message: "You're already on the list. We'll signal when T1 is ready." }), request, env);
  }

  return withCors(json({ message: "You're on the list. We'll signal when T1 is ready." }, { status: 201 }), request, env);
}

async function route(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/health") return json({ ok: true });

  if (url.pathname !== "/api/notify") {
    return json({ error: "not found" }, { status: 404 });
  }

  if (request.method === "OPTIONS") {
    if (!isAllowedOrigin(request, env)) return withCors(json({ error: "origin not allowed" }, { status: 403 }), request, env);
    return new Response(null, { status: 204, headers: corsHeaders(request, env) });
  }
  if (request.method === "GET") return handleGet(request, env);
  if (request.method === "POST") return handlePost(request, env);
  return methodNotAllowed(request, env);
}

export default {
  async fetch(request, env) {
    try {
      return await route(request, env);
    } catch (err) {
      const message = err instanceof Error && err.message.includes("EMAIL_HASH_PEPPER")
        ? "notify backend is not configured"
        : "server error";
      return withCors(json({ error: message }, { status: 500 }), request, env || {});
    }
  },
};
