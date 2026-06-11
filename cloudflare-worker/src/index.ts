// Cloudflare Worker — YourMemory activation backend
// GET  /             → install counter dashboard
// GET  /debug        → raw row dump
// GET  /emails       → registered emails (private)
// POST /             → record a unique install
// POST /send-otp     → send 6-digit OTP to email
// POST /verify-otp   → verify OTP, return activation token
// GET  /verify-token → verify CLI token

export interface Env {
  DB: D1Database;
  RESEND_API_KEY: string;
  EMAILS_SECRET: string;
}

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};


let tableReady = false;

async function ensureTable(db: D1Database): Promise<void> {
  if (tableReady) return;
  await db.batch([
    db.prepare(`CREATE TABLE IF NOT EXISTS installs (
      instance_id TEXT NOT NULL PRIMARY KEY,
      created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )`),
    db.prepare(`CREATE TABLE IF NOT EXISTS emails (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      email       TEXT NOT NULL,
      instance_id TEXT,
      token       TEXT,
      created_at  TEXT NOT NULL DEFAULT (datetime('now')),
      UNIQUE(email)
    )`),
    db.prepare(`CREATE TABLE IF NOT EXISTS register_attempts (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      instance_id TEXT NOT NULL,
      created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )`),
    db.prepare(`CREATE TABLE IF NOT EXISTS otp (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      email      TEXT NOT NULL,
      code       TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now')),
      used       INTEGER NOT NULL DEFAULT 0
    )`),
  ]);
  try { await db.exec("ALTER TABLE emails ADD COLUMN token TEXT"); } catch {}
  tableReady = true;
}

async function sendEmail(
  env: Env,
  to: string,
  subject: string,
  html: string,
): Promise<void> {
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: `YourMemory <noreply@yourmemoryai.xyz>`,
      to: [to],
      subject,
      html,
    }),
  });
  if (!res.ok) {
    const body = await res.text();
    console.error(`Resend error ${res.status}: ${body}`);
    throw new Error(`Resend ${res.status}: ${body}`);
  }
}

function tokenEmailHtml(token: string): string {
  return `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;background:#fff">
      <h2 style="color:#0A192F;margin:0 0 8px">Your YourMemory access token</h2>
      <p style="color:#555;margin:0 0 24px">Run this command in your terminal to activate your install:</p>
      <div style="background:#0A192F;border-radius:10px;padding:16px 20px;font-family:monospace;font-size:14px;color:#e6edf3;margin-bottom:20px">
        yourmemory-register ${token}
      </div>
      <p style="color:#888;font-size:13px;margin:0 0 8px">If you haven't installed yet, run all three commands in order:</p>
      <div style="background:#f4f4f4;border-radius:10px;padding:12px 16px;font-family:monospace;font-size:13px;color:#333;margin-bottom:24px">
        <div>pip install yourmemory</div>
        <div>yourmemory-register ${token}</div>
        <div>yourmemory-setup</div>
      </div>
      <p style="color:#aaa;font-size:11px;margin:0">Keep this token safe. If you lose it, sign in again at <a href="https://yourmemoryai.xyz" style="color:#00D4FF">yourmemoryai.xyz</a> to get a new one.</p>
    </div>`;
}

function otpEmailHtml(code: string): string {
  return `
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:480px;margin:0 auto;padding:40px 24px;background:#fff">
      <h2 style="color:#0A192F;margin:0 0 8px">Your verification code</h2>
      <p style="color:#555;margin:0 0 24px">Enter this code on the YourMemory website to get your activation token:</p>
      <div style="background:#0A192F;border-radius:10px;padding:20px;text-align:center;font-family:monospace;font-size:36px;font-weight:700;letter-spacing:10px;color:#00D4FF;margin-bottom:20px">
        ${code}
      </div>
      <p style="color:#888;font-size:13px;margin:0">This code expires in 10 minutes. If you didn't request this, ignore this email.</p>
    </div>`;
}

const DISPOSABLE_DOMAINS = new Set([
  "mailinator.com", "guerrillamail.com", "tempmail.com", "throwaway.email",
  "yopmail.com", "sharklasers.com", "guerrillamail.info", "guerrillamail.biz",
  "guerrillamail.de", "guerrillamail.net", "guerrillamail.org", "spam4.me",
  "trashmail.com", "trashmail.me", "trashmail.net", "dispostable.com",
  "mailnull.com", "spamgourmet.com", "maildrop.cc", "fakeinbox.com",
  "discard.email", "spamfree24.org", "temporarymail.com", "throwam.com",
  "emailondeck.com", "10minutemail.com", "10minutemail.net", "tempinbox.com",
  "getairmail.com", "guerrillamailblock.com", "grr.la",
]);

function isValidEmail(email: string): { valid: boolean; reason?: string } {
  if (!email || typeof email !== "string")
    return { valid: false, reason: "Email is required." };
  const trimmed = email.trim();
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmed))
    return { valid: false, reason: "That doesn't look like a valid email address." };
  const domain = trimmed.split("@")[1].toLowerCase();
  if (DISPOSABLE_DOMAINS.has(domain))
    return { valid: false, reason: "Please use a real email — disposable addresses won't receive updates." };
  if (domain.length < 4 || !domain.includes("."))
    return { valid: false, reason: "That doesn't look like a valid email address." };
  return { valid: true };
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}

export default {
  async fetch(req: Request, env: Env): Promise<Response> {
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS });

    await ensureTable(env.DB);
    const url = new URL(req.url);

    // ── GET /debug
    if (req.method === "GET" && url.pathname === "/debug") {
      if (url.searchParams.get("token") !== env.EMAILS_SECRET)
        return new Response("Forbidden", { status: 403, headers: CORS });
      const raw = await env.DB.prepare(
        "SELECT instance_id, created_at FROM installs LIMIT 3",
      ).all();
      return json(raw.results);
    }


    // ── GET /emails
    if (req.method === "GET" && url.pathname === "/emails") {
      if (url.searchParams.get("token") !== env.EMAILS_SECRET)
        return new Response("Forbidden", { status: 403, headers: CORS });

      const rows = await env.DB.prepare(
        "SELECT email, instance_id, created_at FROM emails ORDER BY created_at DESC",
      ).all<{ email: string; instance_id: string; created_at: string }>();

      const emailList = rows.results;
      const emailRows = emailList.length === 0
        ? `<tr><td colspan="3" class="empty">No registrations yet.</td></tr>`
        : emailList.map((r) =>
            `<tr><td>${r.email}</td><td class="dim">${r.instance_id ? r.instance_id.slice(0, 8) + "••••••••" : "—"}</td><td class="dim">${r.created_at}</td></tr>`,
          ).join("");

      return new Response(
        `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>YourMemory — Registered Emails</title>
        <style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;padding:48px 24px}.wrap{max-width:800px;margin:0 auto}header{display:flex;align-items:center;gap:12px;margin-bottom:40px}h1{font-size:20px;font-weight:700}.pill{font-size:11px;font-family:monospace;border:1px solid #30363d;border-radius:999px;padding:2px 10px;color:#8b949e}.stat{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px;margin-bottom:32px;display:inline-block;min-width:160px}.val{font-size:36px;font-weight:700;font-family:monospace;color:#3fb950}.lbl{font-size:12px;color:#8b949e;margin-top:8px}.ttl{font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}table{width:100%;border-collapse:collapse}thead th{font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.6px;text-align:left;padding:8px 12px;border-bottom:1px solid #21262d}tbody tr{border-bottom:1px solid #161b22}tbody tr:hover{background:#161b22}tbody td{font-size:13px;padding:10px 12px;color:#c9d1d9}.dim{color:#8b949e;font-family:monospace}.empty{text-align:center;color:#8b949e;font-size:13px;padding:32px}</style>
        </head><body><div class="wrap">
        <header><h1>YourMemory</h1><span class="pill">Registered Emails</span></header>
        <div class="stat"><div class="val">${emailList.length}</div><div class="lbl">Total registrations</div></div>
        <p class="ttl">Email list</p>
        <table><thead><tr><th>Email</th><th>Instance ID</th><th>Registered at (UTC)</th></tr></thead>
        <tbody>${emailRows}</tbody></table>
        </div></body></html>`,
        { headers: { ...CORS, "Content-Type": "text/html; charset=utf-8" } },
      );
    }

    // ── GET /verify-token
    if (req.method === "GET" && url.pathname === "/verify-token") {
      const token = url.searchParams.get("token");
      if (!token) return json({ valid: false });
      const row = await env.DB.prepare(
        "SELECT email FROM emails WHERE token = ?",
      ).bind(token).first<{ email: string }>();
      return json({ valid: !!row });
    }

    // ── POST /send-otp
    if (req.method === "POST" && url.pathname === "/send-otp") {
      let email: string;
      try {
        const body = await req.json() as { email?: string };
        email = body.email?.trim() ?? "";
      } catch {
        return json({ ok: false, reason: "Invalid request body." }, 400);
      }

      const validation = isValidEmail(email);
      if (!validation.valid) return json({ ok: false, reason: validation.reason }, 422);

      const attempts = await env.DB.prepare(
        "SELECT COUNT(*) as n FROM otp WHERE email = ? AND created_at >= datetime('now', '-1 hour')",
      ).bind(email).first<{ n: number }>();
      if ((attempts?.n ?? 0) >= 3)
        return json({ ok: false, reason: "Too many requests. Please wait an hour and try again." }, 429);

      const code = Math.floor(100000 + Math.random() * 900000).toString();
      await env.DB.prepare("INSERT INTO otp (email, code) VALUES (?, ?)").bind(email, code).run();

      try {
        await sendEmail(env, email, `${code} is your YourMemory verification code`, otpEmailHtml(code));
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e);
        return json({ ok: false, reason: `Email failed: ${msg}` }, 500);
      }

      return json({ ok: true });
    }

    // ── POST /verify-otp
    if (req.method === "POST" && url.pathname === "/verify-otp") {
      let email: string, code: string;
      try {
        const body = await req.json() as { email?: string; code?: string };
        email = body.email?.trim() ?? "";
        code = body.code?.trim() ?? "";
      } catch {
        return json({ ok: false, reason: "Invalid request body." }, 400);
      }

      if (!email || !code)
        return json({ ok: false, reason: "Email and code are required." }, 422);

      const row = await env.DB.prepare(
        "SELECT id FROM otp WHERE email = ? AND code = ? AND used = 0 AND created_at >= datetime('now', '-10 minutes') ORDER BY created_at DESC LIMIT 1",
      ).bind(email, code).first<{ id: number }>();

      if (!row)
        return json({ ok: false, reason: "Invalid or expired code. Request a new one." }, 422);

      await env.DB.prepare("UPDATE otp SET used = 1 WHERE id = ?").bind(row.id).run();

      const token = crypto.randomUUID();
      await env.DB.prepare(
        "INSERT INTO emails (email, instance_id, token) VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET token = excluded.token",
      ).bind(email, "web-otp", token).run();

      sendEmail(env, email, "Your YourMemory access token", tokenEmailHtml(token)).catch(() => {});

      return json({ ok: true, token });
    }

    // ── GET / — dashboard
    if (req.method === "GET") {
      const [total, last7, last30, emailCount, recent] = await Promise.all([
        env.DB.prepare("SELECT COUNT(*) as n FROM installs").first<{ n: number }>(),
        env.DB.prepare("SELECT COUNT(*) as n FROM installs WHERE created_at >= datetime('now', '-7 days')").first<{ n: number }>(),
        env.DB.prepare("SELECT COUNT(*) as n FROM installs WHERE created_at >= datetime('now', '-30 days')").first<{ n: number }>(),
        env.DB.prepare("SELECT COUNT(*) as n FROM emails").first<{ n: number }>(),
        env.DB.prepare("SELECT instance_id, created_at FROM installs ORDER BY created_at DESC LIMIT 10").all<{ instance_id: string; created_at: string }>(),
      ]);

      const rows = recent.results.length === 0
        ? `<tr><td colspan="2" class="empty">No installs yet.</td></tr>`
        : recent.results.map((r) =>
            `<tr><td>${r.instance_id.slice(0, 8)}••••••••</td><td class="dim">${r.created_at}</td></tr>`,
          ).join("");

      return new Response(
        `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>YourMemory — Install Counter</title>
        <style>*{box-sizing:border-box;margin:0;padding:0}body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#0d1117;color:#e6edf3;min-height:100vh;padding:48px 24px}.wrap{max-width:680px;margin:0 auto}header{display:flex;align-items:center;gap:12px;margin-bottom:40px}h1{font-size:20px;font-weight:700;letter-spacing:-.3px}.pill{font-size:11px;font-family:monospace;border:1px solid #30363d;border-radius:999px;padding:2px 10px;color:#8b949e}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:40px}.stat{background:#161b22;border:1px solid #21262d;border-radius:12px;padding:20px}.val{font-size:32px;font-weight:700;font-family:monospace;line-height:1;color:#58a6ff}.val.g{color:#3fb950}.val.c{color:#79c0ff}.val.e{color:#f0883e}.lbl{font-size:12px;color:#8b949e;margin-top:8px}.ttl{font-size:13px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px}table{width:100%;border-collapse:collapse}thead th{font-size:11px;font-weight:600;color:#8b949e;text-transform:uppercase;letter-spacing:.6px;text-align:left;padding:8px 12px;border-bottom:1px solid #21262d}tbody tr{border-bottom:1px solid #161b22}tbody tr:hover{background:#161b22}tbody td{font-family:monospace;font-size:13px;padding:10px 12px;color:#c9d1d9}.dim{color:#8b949e}.empty{text-align:center;color:#8b949e;font-size:13px;padding:32px}footer{margin-top:40px;font-size:12px;color:#484f58;text-align:center}footer a{color:#58a6ff;text-decoration:none}</style>
        </head><body><div class="wrap">
        <header><h1>YourMemory</h1><span class="pill">Install Counter</span></header>
        <div class="stats">
          <div class="stat"><div class="val">${total?.n ?? 0}</div><div class="lbl">Total installs</div></div>
          <div class="stat"><div class="val g">${last7?.n ?? 0}</div><div class="lbl">Last 7 days</div></div>
          <div class="stat"><div class="val c">${last30?.n ?? 0}</div><div class="lbl">Last 30 days</div></div>
          <div class="stat"><div class="val e">${emailCount?.n ?? 0}</div><div class="lbl">Registered emails</div></div>
        </div>
        <p class="ttl">Recent installs</p>
        <table><thead><tr><th>Instance ID</th><th>Installed at (UTC)</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <footer><a href="https://github.com/sachitrafa/YourMemory">YourMemory</a> · anonymous install counter · <a href="/emails">view emails</a></footer>
        </div></body></html>`,
        { headers: { ...CORS, "Content-Type": "text/html; charset=utf-8" } },
      );
    }

    // ── POST /register — legacy (backwards compat)
    if (req.method === "POST" && url.pathname === "/register") {
      let email: string, instance_id: string;
      try {
        const body = await req.json() as { email?: string; instance_id?: string };
        email = body.email?.trim() ?? "";
        instance_id = body.instance_id?.trim() ?? "";
      } catch {
        return json({ ok: false, reason: "Invalid request body." }, 400);
      }

      const validation = isValidEmail(email);
      if (!validation.valid) return json({ ok: false, reason: validation.reason }, 422);

      if (instance_id) {
        const attempts = await env.DB.prepare(
          "SELECT COUNT(*) as n FROM register_attempts WHERE instance_id = ? AND created_at >= datetime('now', '-1 hour')",
        ).bind(instance_id).first<{ n: number }>();
        if ((attempts?.n ?? 0) >= 5)
          return json({ ok: false, reason: "Too many attempts. Please try again in an hour." }, 429);
        await env.DB.prepare("INSERT INTO register_attempts (instance_id) VALUES (?)").bind(instance_id).run();
      }

      const token = crypto.randomUUID();
      await env.DB.prepare(
        "INSERT INTO emails (email, instance_id, token) VALUES (?, ?, ?) ON CONFLICT(email) DO UPDATE SET token = excluded.token, instance_id = excluded.instance_id",
      ).bind(email, instance_id, token).run();

      sendEmail(env, email, "Your YourMemory access token", tokenEmailHtml(token)).catch(() => {});
      return json({ ok: true });
    }

    // ── POST / — record install
    if (req.method === "POST") {
      let instance_id: string;
      try {
        const body = await req.json() as { instance_id?: string };
        instance_id = body.instance_id ?? "";
        if (!instance_id || typeof instance_id !== "string" || instance_id.length > 64)
          throw new Error("invalid");
      } catch {
        return new Response("Bad request", { status: 400, headers: CORS });
      }

      await env.DB.prepare("INSERT OR IGNORE INTO installs (instance_id) VALUES (?)").bind(instance_id).run();
      return json({ ok: true });
    }

    return new Response("Method not allowed", { status: 405, headers: CORS });
  },
};
