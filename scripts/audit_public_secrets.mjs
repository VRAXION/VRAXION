#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const MAX_TEXT_FILE_BYTES = 5 * 1024 * 1024;
const BINARY_EXTENSIONS = new Set([
  ".jpg",
  ".jpeg",
  ".png",
  ".webp",
  ".woff",
  ".woff2",
  ".ico",
  ".pdf",
  ".zip",
  ".gz",
  ".tar",
]);

const make = (...parts) => parts.join("");
const SECRET_PATTERNS = [
  {
    label: "pem_private_key_block",
    regex: new RegExp(make("-----BEGIN ", "(?:RSA |DSA |EC |OPENSSH )?", "PRIVATE", " KEY-----")),
  },
  {
    label: "git_host_credential_value",
    regex: /gh[pousr]_[A-Za-z0-9_]{30,}/,
  },
  {
    label: "openai_key_value",
    regex: /sk-(?:proj-)?[A-Za-z0-9_-]{32,}/,
  },
  {
    label: "anthropic_key_value",
    regex: /sk-ant-[A-Za-z0-9_-]{24,}/,
  },
  {
    label: "slack_token_value",
    regex: /xox[baprs]-[A-Za-z0-9-]{24,}/,
  },
  {
    label: "stripe_secret_value",
    regex: /(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}/,
  },
  {
    label: "aws_access_key_id",
    regex: /AKIA[0-9A-Z]{16}/,
  },
  {
    label: "google_api_key_value",
    regex: /AIza[0-9A-Za-z_-]{35}/,
  },
  {
    label: "jwt_value",
    regex: /eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}/,
  },
  {
    label: "credential_assignment",
    regex: new RegExp(
      make(
        "\\b(?:api[_-]?key|secret|token|password|private[_-]?key|client[_-]?secret)",
        "\\b\\s*[:=]\\s*['\"]?(?!example\\b|placeholder\\b|redacted\\b|none\\b|null\\b)",
        "[A-Za-z0-9_./+=:-]{16,}",
      ),
      "i",
    ),
  },
  {
    label: "cloud_vendor_key_assignment",
    regex: new RegExp(
      make(
        "\\b(?:openai|anthropic|github|cloudflare|stripe|slack|google|aws)",
        "[A-Z0-9_-]*(?:key|token|secret|password)",
        "\\b\\s*[:=]\\s*['\"]?(?!example\\b|placeholder\\b|redacted\\b|none\\b|null\\b)",
        "[A-Za-z0-9_./+=:-]{16,}",
      ),
      "i",
    ),
  },
  {
    label: "absolute_drive_path",
    regex: /(?:^|[^A-Za-z0-9_])[A-Za-z]:[\\/][^\s"'<>|]+/,
  },
  {
    label: "unc_local_path",
    regex: /\\\\[A-Za-z0-9_.-]+\\[A-Za-z0-9_.-]+/,
  },
];

const ALLOWED_MATCHES = new Map([
  [
    "scripts/check_public_export.ps1",
    new Set([
      "absolute_drive_path",
    ]),
  ],
]);

function trackedFiles() {
  return execFileSync("git", ["ls-files"], { cwd: ROOT, encoding: "utf8" })
    .split(/\r?\n/)
    .map((line) => line.trim().replaceAll("\\", "/"))
    .filter(Boolean);
}

function isBinaryPath(relativePath) {
  return BINARY_EXTENSIONS.has(path.extname(relativePath).toLowerCase());
}

function readTextIfSafe(relativePath) {
  const absolutePath = path.join(ROOT, relativePath);
  const stat = fs.statSync(absolutePath);
  if (!stat.isFile() || stat.size > MAX_TEXT_FILE_BYTES || isBinaryPath(relativePath)) {
    return null;
  }
  const buffer = fs.readFileSync(absolutePath);
  if (buffer.includes(0)) {
    return null;
  }
  return buffer.toString("utf8");
}

function lineForIndex(text, index) {
  return text.slice(0, index).split(/\r?\n/).length;
}

function snippet(value) {
  return value.replace(/\s+/g, " ").slice(0, 96);
}

const failures = [];
const files = trackedFiles();
let scannedFiles = 0;

for (const relativePath of files) {
  const text = readTextIfSafe(relativePath);
  if (text === null) {
    continue;
  }
  scannedFiles += 1;
  const allowedLabels = ALLOWED_MATCHES.get(relativePath) ?? new Set();
  for (const pattern of SECRET_PATTERNS) {
    const match = pattern.regex.exec(text);
    if (!match) {
      continue;
    }
    if (allowedLabels.has(pattern.label)) {
      continue;
    }
    failures.push(
      `${relativePath}:${lineForIndex(text, match.index)} matched ${pattern.label}: ${snippet(match[0])}`,
    );
  }
}

console.log("PUBLIC_SECRET_SCAN");
console.log(`tracked_files=${files.length}`);
console.log(`scanned_text_files=${scannedFiles}`);
console.log(`failure_count=${failures.length}`);
for (const failure of failures) {
  console.log(`failure: ${failure}`);
}
if (failures.length > 0) {
  process.exit(1);
}
console.log("public_secret_scan=pass");
