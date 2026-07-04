const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_EMAIL_LENGTH = 320;
const MAX_BODY_BYTES = 4096;
const DEFAULT_GOAL = 1000;
const DEFAULT_RATE_LIMIT = 20;
const DEFAULT_ALLOWED_ORIGIN = "https://vraxion.github.io";
const DEFAULT_RATE_LIMIT_RETENTION_HOURS = 48;
const MAX_EXPORT_LIMIT = 10000;
const MIN_ADMIN_TOKEN_LENGTH = 24;

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

function clampPositiveInt(value, fallback, max) {
  return Math.min(parsePositiveInt(value, fallback), max);
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

function adminMethodNotAllowed(allow) {
  return json({ error: "method not allowed" }, { status: 405, headers: { allow } });
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

async function timingSafeEqual(left, right) {
  const leftBytes = new TextEncoder().encode(String(left || ""));
  const rightBytes = new TextEncoder().encode(String(right || ""));
  const maxLength = Math.max(leftBytes.length, rightBytes.length);
  let diff = leftBytes.length ^ rightBytes.length;
  for (let index = 0; index < maxLength; index += 1) {
    diff |= (leftBytes[index] || 0) ^ (rightBytes[index] || 0);
  }
  return diff === 0;
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

function bearerToken(request) {
  const value = request.headers.get("authorization") || "";
  const match = value.match(/^Bearer\s+(.+)$/i);
  return match ? match[1].trim() : "";
}

async function requireAdmin(request, env) {
  if (!env.DB) return { response: json({ error: "missing DB binding" }, { status: 503 }) };
  if (!env.ADMIN_TOKEN || String(env.ADMIN_TOKEN).length < MIN_ADMIN_TOKEN_LENGTH) {
    return { response: json({ error: "notify admin is not configured" }, { status: 503 }) };
  }

  const token = bearerToken(request);
  if (!token) {
    return { response: json({ error: "admin authorization required" }, { status: 401, headers: { "www-authenticate": "Bearer" } }) };
  }
  if (!(await timingSafeEqual(token, env.ADMIN_TOKEN))) {
    return { response: json({ error: "admin authorization failed" }, { status: 403 }) };
  }
  return { ok: true };
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

async function handleAdminExport(request, env, url) {
  if (request.method !== "GET") return adminMethodNotAllowed("GET");
  const limit = clampPositiveInt(url.searchParams.get("limit"), MAX_EXPORT_LIMIT, MAX_EXPORT_LIMIT);
  const result = await env.DB.prepare(
    `SELECT email, source, created_at
     FROM notify_subscribers
     ORDER BY created_at ASC
     LIMIT ?1`
  )
    .bind(limit)
    .all();
  const subscribers = Array.isArray(result?.results) ? result.results : [];
  return json({ subscribers, count: subscribers.length, limited: subscribers.length === limit });
}

async function handleAdminDelete(request, env) {
  if (request.method !== "POST") return adminMethodNotAllowed("POST");
  if (!env.EMAIL_HASH_PEPPER || String(env.EMAIL_HASH_PEPPER).length < 16) {
    return json({ error: "notify duplicate key is not configured" }, { status: 503 });
  }

  const type = request.headers.get("content-type") || "";
  if (!type.toLowerCase().includes("application/json")) {
    return json({ error: "content-type must be application/json" }, { status: 415 });
  }
  const { data, error } = await readJsonBody(request);
  if (error) return error;

  const email = normalizeEmail(data?.email);
  if (!email || email.length > MAX_EMAIL_LENGTH || !EMAIL_RE.test(email)) {
    return json({ error: "Please provide a valid email." }, { status: 400 });
  }

  const emailHash = await sha256Hex(`notify-email:${email}:${env.EMAIL_HASH_PEPPER}`);
  const result = await env.DB.prepare("DELETE FROM notify_subscribers WHERE email_hash = ?1").bind(emailHash).run();
  const deleted = Number(result?.meta?.changes ?? result?.changes ?? 0);
  return json({ deleted });
}

async function cleanupRateLimits(env, olderThanHours) {
  const hours = clampPositiveInt(olderThanHours || env.RATE_LIMIT_RETENTION_HOURS, DEFAULT_RATE_LIMIT_RETENTION_HOURS, 24 * 30);
  const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000).toISOString();
  if (!env.DB) return { deleted: 0, cutoff, olderThanHours: hours };

  const result = await env.DB.prepare("DELETE FROM notify_rate_limits WHERE window_start < ?1").bind(cutoff).run();
  const deleted = Number(result?.meta?.changes ?? result?.changes ?? 0);
  return { deleted, cutoff, olderThanHours: hours };
}

async function handleAdminCleanupRateLimits(request, env) {
  if (request.method !== "POST") return adminMethodNotAllowed("POST");
  let data = null;
  if ((request.headers.get("content-type") || "").toLowerCase().includes("application/json")) {
    const body = await readJsonBody(request);
    if (body.error) return body.error;
    data = body.data;
  }

  return json(await cleanupRateLimits(env, data?.olderThanHours));
}

async function handleAdmin(request, env, url) {
  const auth = await requireAdmin(request, env);
  if (!auth.ok) return auth.response;

  if (url.pathname === "/admin/notify/export") return handleAdminExport(request, env, url);
  if (url.pathname === "/admin/notify/delete") return handleAdminDelete(request, env);
  if (url.pathname === "/admin/notify/cleanup-rate-limits") return handleAdminCleanupRateLimits(request, env);
  return json({ error: "not found" }, { status: 404 });
}

async function route(request, env) {
  const url = new URL(request.url);
  if (url.pathname === "/health") return json({ ok: true });
  if (url.pathname.startsWith("/admin/notify/")) return handleAdmin(request, env, url);

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
  async scheduled(_event, env, ctx) {
    const task = cleanupRateLimits(env || {}, undefined);
    if (ctx?.waitUntil) {
      ctx.waitUntil(task);
      return;
    }
    await task;
  },
};
