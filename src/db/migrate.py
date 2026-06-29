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


def _create_audit_table(conn, backend: str) -> None:
    """Append-only, hash-chained audit log: every read/write/delete on a memory.
    Tamper-evident (each row chains the prior row's hash) and retained >=90 days.
    Never UPDATEd or DELETEd by the app except the controlled retention prune.
    Idempotent across all backends."""
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              BIGSERIAL PRIMARY KEY,
                ts              TEXT NOT NULL,
                actor_user_id   TEXT NOT NULL,
                actor_agent_id  TEXT,
                action          TEXT NOT NULL,
                operation       TEXT NOT NULL,
                target_id       BIGINT,
                detail          TEXT,
                source          TEXT NOT NULL DEFAULT 'http',
                prev_hash       TEXT NOT NULL,
                row_hash        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_log(ts);
            CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_user_id, ts);
        """)
        conn.commit()
        cur.close()
    elif backend == "duckdb":
        try:
            conn.execute("CREATE SEQUENCE IF NOT EXISTS audit_log_seq")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id              BIGINT DEFAULT nextval('audit_log_seq'),
                    ts              VARCHAR NOT NULL,
                    actor_user_id   VARCHAR NOT NULL,
                    actor_agent_id  VARCHAR,
                    action          VARCHAR NOT NULL,
                    operation       VARCHAR NOT NULL,
                    target_id       BIGINT,
                    detail          VARCHAR,
                    source          VARCHAR NOT NULL DEFAULT 'http',
                    prev_hash       VARCHAR NOT NULL,
                    row_hash        VARCHAR NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_user_id, ts)")
        except Exception as exc:
            print(f"audit table (duckdb) skipped: {exc}", file=sys.stderr)
    else:  # sqlite
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT NOT NULL,
                actor_user_id   TEXT NOT NULL,
                actor_agent_id  TEXT,
                action          TEXT NOT NULL,
                operation       TEXT NOT NULL,
                target_id       INTEGER,
                detail          TEXT,
                source          TEXT NOT NULL DEFAULT 'http',
                prev_hash       TEXT NOT NULL,
                row_hash        TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_ts    ON audit_log(ts);
            CREATE INDEX IF NOT EXISTS idx_audit_actor ON audit_log(actor_user_id, ts);
        """)


def _create_archive_table(conn, backend: str) -> None:
    """Holds originals that were compressed into a summary memory. Lets the live
    `memories` table stay lean (so recall stays fast and clean) while keeping the
    pre-compression facts for reversibility and audit. Idempotent across backends."""
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS memory_archive (
                orig_id     BIGINT,
                user_id     TEXT NOT NULL,
                content     TEXT NOT NULL,
                category    TEXT,
                importance  DOUBLE PRECISION,
                agent_id    TEXT,
                visibility  TEXT,
                created_at  TIMESTAMPTZ,
                archived_at TIMESTAMPTZ DEFAULT NOW(),
                summary_id  BIGINT
            );
            CREATE INDEX IF NOT EXISTS idx_archive_user ON memory_archive(user_id);
        """)
        conn.commit()
        cur.close()
    elif backend == "duckdb":
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_archive (
                    orig_id     BIGINT,
                    user_id     VARCHAR NOT NULL,
                    content     VARCHAR NOT NULL,
                    category    VARCHAR,
                    importance  DOUBLE,
                    agent_id    VARCHAR,
                    visibility  VARCHAR,
                    created_at  TIMESTAMP,
                    archived_at TIMESTAMP DEFAULT now(),
                    summary_id  BIGINT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_archive_user ON memory_archive(user_id)")
        except Exception as exc:
            print(f"archive table (duckdb) skipped: {exc}", file=sys.stderr)
    else:  # sqlite
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memory_archive (
                orig_id     INTEGER,
                user_id     TEXT NOT NULL,
                content     TEXT NOT NULL,
                category    TEXT,
                importance  REAL,
                agent_id    TEXT,
                visibility  TEXT,
                created_at  TIMESTAMP,
                archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                summary_id  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_archive_user ON memory_archive(user_id);
        """)


def _create_pool_tables(conn, backend: str) -> None:
    """Shared memory pools (team / institutional memory). A pool's memories live in the
    `memories` table under a namespaced user_id ('pool:<id>'), so they reuse all existing
    machinery (embedding, dedup, recall, decay, compaction, audit). These two tables hold
    the pool registry and role-based membership. Idempotent across backends."""
    if backend == "postgres":
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pools (
                pool_id    TEXT PRIMARY KEY,
                name       TEXT,
                owner      TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS pool_members (
                pool_id   TEXT NOT NULL,
                member_id TEXT NOT NULL,
                role      TEXT NOT NULL DEFAULT 'reader',
                can_read  BOOLEAN NOT NULL DEFAULT TRUE,
                can_write BOOLEAN NOT NULL DEFAULT FALSE,
                added_at  TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (pool_id, member_id)
            );
            CREATE INDEX IF NOT EXISTS idx_pool_member ON pool_members(member_id);
        """)
        conn.commit()
        cur.close()
    elif backend == "duckdb":
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pools (
                    pool_id    VARCHAR PRIMARY KEY,
                    name       VARCHAR,
                    owner      VARCHAR,
                    created_at TIMESTAMP DEFAULT now()
                )""")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pool_members (
                    pool_id   VARCHAR NOT NULL,
                    member_id VARCHAR NOT NULL,
                    role      VARCHAR NOT NULL DEFAULT 'reader',
                    can_read  BOOLEAN NOT NULL DEFAULT TRUE,
                    can_write BOOLEAN NOT NULL DEFAULT FALSE,
                    added_at  TIMESTAMP DEFAULT now(),
                    PRIMARY KEY (pool_id, member_id)
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pool_member ON pool_members(member_id)")
        except Exception as exc:
            print(f"pool tables (duckdb) skipped: {exc}", file=sys.stderr)
    else:  # sqlite
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pools (
                pool_id    TEXT PRIMARY KEY,
                name       TEXT,
                owner      TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS pool_members (
                pool_id   TEXT NOT NULL,
                member_id TEXT NOT NULL,
                role      TEXT NOT NULL DEFAULT 'reader',
                can_read  INTEGER NOT NULL DEFAULT 1,
                can_write INTEGER NOT NULL DEFAULT 0,
                added_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (pool_id, member_id)
            );
            CREATE INDEX IF NOT EXISTS idx_pool_member ON pool_members(member_id);
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

    # ── Append-only, hash-chained audit log (read/write/delete, 90-day+ retention) ──
    _create_audit_table(conn, backend)

    # ── Archive of originals compressed into summaries (memory compaction) ──
    _create_archive_table(conn, backend)

    # ── Shared memory pools (team / institutional memory) ──
    _create_pool_tables(conn, backend)

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
