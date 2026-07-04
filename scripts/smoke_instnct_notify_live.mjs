import assert from "node:assert/strict";

const base = String(process.env.INSTNCT_NOTIFY_API_BASE || "").replace(/\/+$/, "");
const allowedOrigin = process.env.INSTNCT_NOTIFY_ALLOWED_ORIGIN || "https://vraxion.github.io";
const writeMode = process.env.INSTNCT_NOTIFY_SMOKE_WRITE === "1";
const rateLimitMode = process.env.INSTNCT_NOTIFY_SMOKE_RATE_LIMIT === "1";
const failures = [];

function fail(message) {
  failures.push(message);
}

async function fetchJson(url, init = {}) {
  const response = await fetch(url, init);
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      fail(`${url} did not return JSON: ${text.slice(0, 80)}`);
    }
  }
  return { response, data };
}

function expectCors(response, label) {
  const allowOrigin = response.headers.get("access-control-allow-origin");
  if (allowOrigin !== allowedOrigin) {
    fail(`${label} returned wrong access-control-allow-origin: ${allowOrigin}`);
  }
}

if (!base) {
  console.error("INSTNCT_NOTIFY_API_BASE is required");
  process.exit(2);
}

let result = await fetchJson(`${base}/health`);
if (result.response.status !== 200 || result.data?.ok !== true) fail("health check failed");

result = await fetchJson(`${base}/api/notify`, {
  headers: { origin: allowedOrigin },
});
if (result.response.status !== 200) fail(`GET /api/notify status ${result.response.status}`);
expectCors(result.response, "GET /api/notify");
if (typeof result.data?.count !== "number") fail("GET /api/notify missing numeric count");
if (typeof result.data?.goal !== "number") fail("GET /api/notify missing numeric goal");

let response = await fetch(`${base}/api/notify`, {
  method: "OPTIONS",
  headers: {
    origin: allowedOrigin,
    "access-control-request-method": "POST",
    "access-control-request-headers": "content-type",
  },
});
if (response.status !== 204) fail(`OPTIONS /api/notify status ${response.status}`);
expectCors(response, "OPTIONS /api/notify");

result = await fetchJson(`${base}/api/notify`, {
  method: "POST",
  headers: {
    origin: allowedOrigin,
    "content-type": "application/json",
  },
  body: JSON.stringify({ email: "not-an-email" }),
});
if (result.response.status !== 400) fail(`invalid email POST status ${result.response.status}`);
expectCors(result.response, "invalid email POST");

result = await fetchJson(`${base}/api/notify`, {
  method: "POST",
  headers: {
    origin: allowedOrigin,
    "content-type": "application/json",
  },
  body: JSON.stringify({ email: "bot@example.com", website: "filled" }),
});
if (result.response.status !== 202) fail(`honeypot POST status ${result.response.status}`);
expectCors(result.response, "honeypot POST");

result = await fetchJson(`${base}/api/notify`, {
  method: "POST",
  headers: {
    origin: "https://example.invalid",
    "content-type": "application/json",
  },
  body: JSON.stringify({ email: "blocked@example.com" }),
});
if (result.response.status !== 403) fail(`blocked origin POST status ${result.response.status}`);

if (writeMode) {
  const smokeEmail =
    process.env.INSTNCT_NOTIFY_SMOKE_EMAIL ||
    `instnct-smoke+${Date.now().toString(36)}@example.com`;
  result = await fetchJson(`${base}/api/notify`, {
    method: "POST",
    headers: {
      origin: allowedOrigin,
      "content-type": "application/json",
    },
    body: JSON.stringify({ email: smokeEmail, source: "instnct-smoke" }),
  });
  if (![200, 201].includes(result.response.status)) {
    fail(`write-mode valid POST status ${result.response.status}`);
  }
  expectCors(result.response, "write-mode valid POST");

  result = await fetchJson(`${base}/api/notify`, {
    method: "POST",
    headers: {
      origin: allowedOrigin,
      "content-type": "application/json",
    },
    body: JSON.stringify({ email: smokeEmail, source: "instnct-smoke" }),
  });
  if (result.response.status !== 200) {
    fail(`write-mode duplicate POST status ${result.response.status}`);
  }
  expectCors(result.response, "write-mode duplicate POST");

  if (rateLimitMode) {
    const attempts = Number.parseInt(process.env.INSTNCT_NOTIFY_SMOKE_RATE_ATTEMPTS || "24", 10);
    let limited = false;
    for (let index = 0; index < attempts; index += 1) {
      result = await fetchJson(`${base}/api/notify`, {
        method: "POST",
        headers: {
          origin: allowedOrigin,
          "content-type": "application/json",
        },
        body: JSON.stringify({ email: smokeEmail, source: "instnct-smoke" }),
      });
      expectCors(result.response, `write-mode rate-limit POST ${index + 1}`);
      if (result.response.status === 429) {
        limited = true;
        break;
      }
    }
    if (!limited) fail(`write-mode rate-limit did not return 429 within ${attempts} duplicate attempts`);
  }
}

assert.equal(failures.length, 0, failures.join("\n"));
console.log("instnct_notify_live_smoke=pass");
