import os
import sys
from dotenv import load_dotenv
from src.db.connection import get_backend, get_conn

load_dotenv()


def _add_columns(conn, backend: str) -> None:
    """Idempotent ALTER TABLE additions for new columns."""
    if backend == "sqlite":
        for col, defn in [
            ("context_paths", "TEXT DEFAULT NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {defn}")
            except Exception:
                pass  # column already exists
        for tbl in ["user_activity", "memory_history"]:
            pass  # created by schema.sql already

    elif backend == "duckdb":
        for col, defn in [
            ("context_paths", "VARCHAR DEFAULT NULL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {defn}")
            except Exception:
                pass

    elif backend == "postgres":
        cur = conn.cursor()
        for col, defn in [
            ("context_paths", "TEXT DEFAULT NULL"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='memories' AND column_name='{col}'
                    ) THEN
                        ALTER TABLE memories ADD COLUMN {col} {defn};
                    END IF;
                END $$;
            """)
        conn.commit()
        cur.close()


def _create_buffer_table(conn, backend: str) -> None:
    """Verbatim rolling conversation buffer for opt-in headless 'lean-window' mode.
    Distinct from `memories` (qwen-distilled): keeps the last N exchanges raw so a
    flushed window can be reconstructed losslessly. Idempotent across all backends."""
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_buffer (
                id             BIGSERIAL PRIMARY KEY,
                user_id        TEXT NOT NULL,
                user_text      TEXT,
                assistant_text TEXT,
                created_at     TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_convbuf_user ON conversation_buffer(user_id, id);
        """)
        conn.commit()
        cur.close()
    elif backend == "duckdb":
        try:
            conn.execute("CREATE SEQUENCE IF NOT EXISTS conversation_buffer_seq")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_buffer (
                    id             BIGINT DEFAULT nextval('conversation_buffer_seq'),
                    user_id        VARCHAR NOT NULL,
                    user_text      VARCHAR,
                    assistant_text VARCHAR,
                    created_at     TIMESTAMP DEFAULT now()
                )
            """)
        except Exception as exc:
            print(f"buffer table (duckdb) skipped: {exc}", file=sys.stderr)
    else:  # sqlite
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_buffer (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        TEXT NOT NULL,
                user_text      TEXT,
                assistant_text TEXT,
                created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_convbuf_user ON conversation_buffer(user_id, id);
        """)


def migrate():
    backend = get_backend()

    schema_map = {
        "postgres": "schema.sql",
        "sqlite":   "sqlite_schema.sql",
        "duckdb":   "duckdb_schema.sql",
    }
    schema_path = os.path.join(os.path.dirname(__file__), schema_map[backend])

    with open(schema_path, "r") as f:
        schema = f.read()

    conn = get_conn()

    if backend == "sqlite":
        conn.executescript(schema)
    elif backend == "duckdb":
        for stmt in schema.split(";"):
            # Strip comment lines, keep SQL lines
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            sql = "\n".join(lines).strip()
            if sql:
                conn.execute(sql)
    else:
        cur = conn.cursor()
        cur.execute(schema)
        conn.commit()
        cur.close()

    # ── Additive column migrations (safe to re-run) ───────────────────────
    _add_columns(conn, backend)

    # ── Verbatim conversation buffer (opt-in headless lean-window mode) ────
    _create_buffer_table(conn, backend)

    # ── Post-schema FTS setup ─────────────────────────────────────────────
    if backend == "sqlite":
        # Backfill any rows that existed before the FTS table was created.
        # Use INSERT OR REPLACE so updated content is re-indexed.
        conn.executescript("""
            INSERT OR REPLACE INTO memories_fts(rowid, content)
            SELECT id, content FROM memories;
        """)

    elif backend == "duckdb":
        # Install the FTS extension once (no-op if already installed).
        try:
            conn.execute("INSTALL fts; LOAD fts;")
        except Exception as exc:
            print(f"DuckDB FTS extension unavailable — keyword search disabled: {exc}",
                  file=sys.stderr)

    conn.close()
    print(f"Migration complete ({backend}).", file=sys.stderr)

    # Bootstrap the graph backend (creates indexes for Neo4j, touches pickle for NetworkX)
    try:
        from src.graph import get_graph_backend
        get_graph_backend()
        print("Graph backend initialised.", file=sys.stderr)
    except Exception as exc:
        print(f"Graph backend init skipped: {exc}", file=sys.stderr)


if __name__ == "__main__":
    migrate()
