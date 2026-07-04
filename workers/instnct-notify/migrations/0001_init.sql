CREATE TABLE IF NOT EXISTS notify_subscribers (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  email_hash TEXT NOT NULL UNIQUE,
  source TEXT NOT NULL DEFAULT 'instnct-site',
  ip_hash TEXT,
  user_agent_hash TEXT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notify_subscribers_created_at
  ON notify_subscribers(created_at);

CREATE TABLE IF NOT EXISTS notify_rate_limits (
  key TEXT PRIMARY KEY,
  count INTEGER NOT NULL DEFAULT 0,
  window_start TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_notify_rate_limits_window_start
  ON notify_rate_limits(window_start);
