// Val.town HTTP endpoint
// GET  /       → install counter dashboard
// GET  /debug  → raw row dump
// POST /       → record a unique install (INSERT OR IGNORE)

import sqlite from "https://esm.town/v/std/sqlite/main.ts";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

async function ensureTable() {
  await sqlite.execute(`
    CREATE TABLE IF NOT EXISTS installs (
      instance_id TEXT NOT NULL PRIMARY KEY,
      created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);
  await sqlite.execute(`DELETE FROM installs WHERE instance_id IS NULL`);
}

function getCount(res: { rows: unknown[] }): number {
  const row = res.rows[0] as Record<string, unknown> | unknown[];
  return Number(Array.isArray(row) ? row[0] : row["n"]) || 0;
}

function extractRows(res: { rows: unknown[] }): { instance_id: string; created_at: string }[] {
  return res.rows.map((r: unknown) => {
    const row = r as Record<string, string>;
    const id = row["instance_id"] ?? (Array.isArray(r) ? (r as unknown[])[0] : null);
    const ts = row["created_at"]  ?? (Array.isArray(r) ? (r as unknown[])[1] : null);
    return {
      instance_id: id ? String(id).slice(0, 8) + "••••••••" : "unknown",
      created_at:  ts ? String(ts) : "—",
    };
  });
}

export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS });

  await ensureTable();

  const url = new URL(req.url);

  // ── GET /debug ────────────────────────────────────────────────────────────
  if (req.method === "GET" && url.pathname === "/debug") {
    const raw = await sqlite.execute(`SELECT instance_id, created_at FROM installs LIMIT 3`);
    return new Response(JSON.stringify({ columns: raw.columns, rows: raw.rows }, null, 2), {
      headers: { ...CORS, "Content-Type": "application/json" },
    });
  }

  // ── GET / — dashboard ────────────────────────────────────────────────────
  if (req.method === "GET") {
    const totalCount = getCount(await sqlite.execute(`SELECT COUNT(*) as n FROM installs`));
    const last7Count  = getCount(await sqlite.execute(`SELECT COUNT(*) as n FROM installs WHERE created_at >= datetime('now', '-7 days')`));
    const last30Count = getCount(await sqlite.execute(`SELECT COUNT(*) as n FROM installs WHERE created_at >= datetime('now', '-30 days')`));
    const recentRows  = extractRows(await sqlite.execute(`SELECT instance_id, created_at FROM installs ORDER BY created_at DESC LIMIT 10`));

    const rows = recentRows.length === 0
      ? `<tr><td colspan="2" class="empty">No installs yet.</td></tr>`
      : recentRows.map(r => `<tr><td>${r.instance_id}</td><td class="dim">${r.created_at}</td></tr>`).join("");

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>YourMemory — Install Counter</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;padding:48px 24px}
    .wrap{max-width:680px;margin:0 auto}
    header{display:flex;align-items:center;gap:12px;margin-bottom:40px}
    h1{font-size:20px;font-weight:700;letter-spacing:-.3px}
    .pill{font-size:11px;font-family:monospace;border:1px solid #30363d;border-radius:999px;padding:2px 10px;color:#8b949e}
    .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:40px}
    .stat{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px}
    .val{font-size:36px;font-weight:700;font-family:monospace;line-height:1;color:#58a6ff}
    .val.g{color:#3fb950}.val.c{color:#79c0ff}
    .lbl{font-size:12px;color:#8b949e;margin-top:8px}
    .ttl{font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}
    table{width:100%;border-collapse:collapse}
    thead th{font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.6px;text-align:left;padding:8px 12px;border-bottom:1px solid #21262d}
    tbody tr{border-bottom:1px solid #161b22}
    tbody tr:hover{background:#161b22}
    tbody td{font-family:monospace;font-size:13px;padding:10px 12px;color:#c9d1d9}
    .dim{color:#8b949e}.empty{text-align:center;color:#8b949e;font-size:13px;padding:32px}
    footer{margin-top:40px;font-size:12px;color:#484f58;text-align:center}
    footer a{color:#58a6ff;text-decoration:none}
  </style>
</head>
<body><div class="wrap">
  <header>
    <svg width="24" height="24" viewBox="0 0 100 100" fill="none">
      <rect x="10" y="80" width="80" height="10" rx="2" fill="#e6edf3"/>
      <rect x="25" y="60" width="50" height="10" rx="2" fill="#e6edf3"/>
      <rect x="40" y="40" width="15" height="10" rx="2" fill="#e6edf3"/>
      <rect x="60" y="40" width="5"  height="10" rx="1" fill="#58a6ff"/>
      <rect x="47.5" y="20" width="5" height="10" rx="1" fill="#58a6ff" fill-opacity=".6"/>
    </svg>
    <h1>YourMemory</h1><span class="pill">Install Counter</span>
  </header>
  <div class="stats">
    <div class="stat"><div class="val">${totalCount}</div><div class="lbl">Total installs</div></div>
    <div class="stat"><div class="val g">${last7Count}</div><div class="lbl">Last 7 days</div></div>
    <div class="stat"><div class="val c">${last30Count}</div><div class="lbl">Last 30 days</div></div>
  </div>
  <p class="ttl">Recent installs</p>
  <table>
    <thead><tr><th>Instance ID</th><th>Installed at (UTC)</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <footer><a href="https://github.com/sachitrafa/YourMemory">YourMemory</a> · anonymous install counter · instance IDs truncated</footer>
</div></body></html>`;

    return new Response(html, { headers: { ...CORS, "Content-Type": "text/html; charset=utf-8" } });
  }

  // ── POST / — record install ──────────────────────────────────────────────
  if (req.method !== "POST")
    return new Response("Method not allowed", { status: 405, headers: CORS });

  let instance_id: string;
  try {
    ({ instance_id } = await req.json());
    if (!instance_id || typeof instance_id !== "string" || instance_id.length > 64)
      throw new Error("invalid");
  } catch {
    return new Response("Bad request", { status: 400, headers: CORS });
  }

  // Single-argument form — no "Expected 2 arguments" error
  await sqlite.execute({
    sql: `INSERT OR IGNORE INTO installs (instance_id) VALUES (?)`,
    args: [instance_id],
  });

  return new Response(JSON.stringify({ ok: true }), {
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}
