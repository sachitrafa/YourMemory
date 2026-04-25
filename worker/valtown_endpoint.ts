// Val.town HTTP endpoint — paste this as a new HTTP val at val.town
// Name it: yourmemory_install_counter

import { sqlite } from "https://esm.town/v/std/sqlite";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

export default async function handler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return new Response(null, { headers: CORS });
  if (req.method !== "POST")
    return new Response("Method not allowed", { status: 405, headers: CORS });

  await sqlite.execute(`
    CREATE TABLE IF NOT EXISTS installs (
      instance_id TEXT PRIMARY KEY,
      created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
  `);

  let instance_id: string;
  try {
    ({ instance_id } = await req.json());
    if (!instance_id || typeof instance_id !== "string" || instance_id.length > 64)
      throw new Error("invalid");
  } catch {
    return new Response("Bad request", { status: 400, headers: CORS });
  }

  // INSERT OR IGNORE — duplicate instance_ids are silently dropped
  await sqlite.execute(
    `INSERT OR IGNORE INTO installs (instance_id) VALUES (?)`,
    [instance_id]
  );

  return new Response(JSON.stringify({ ok: true }), {
    headers: { ...CORS, "Content-Type": "application/json" },
  });
}
