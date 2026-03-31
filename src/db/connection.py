"""
Database connection factory.

Auto-detects backend from DATABASE_URL:
  postgresql:// or postgres://  → PostgreSQL + pgvector
  sqlite:///path                → SQLite at given path (legacy)
  (unset / anything else)       → DuckDB at ~/.yourmemory/memories.duckdb (default)

Usage:
    from src.db.connection import get_conn, get_backend

    conn = get_conn()
    backend = get_backend()   # "duckdb", "sqlite", or "postgres"
"""

import json
import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_backend() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("postgresql://") or url.startswith("postgres://"):
        return "postgres"
    if url.startswith("sqlite:///"):
        return "sqlite"
    return "duckdb"


def _duckdb_path() -> str:
    path = Path.home() / ".yourmemory" / "memories.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _sqlite_path() -> str:
    url = os.getenv("DATABASE_URL", "")
    if url.startswith("sqlite:///"):
        return url[10:]
    path = Path.home() / ".yourmemory" / "memories.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_conn():
    backend = get_backend()
    if backend == "postgres":
        import psycopg2
        return psycopg2.connect(os.getenv("DATABASE_URL"))
    if backend == "sqlite":
        path = _sqlite_path()
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    # DuckDB default
    import duckdb
    return duckdb.connect(_duckdb_path())


def emb_to_db(embedding: list, backend: str = None) -> object:
    """
    Serialize embedding for storage.
      DuckDB:   Python list (FLOAT[768] native)
      Postgres: '[0.1,0.2,...]' (pgvector wire format)
      SQLite:   JSON string (TEXT)
    """
    b = backend or get_backend()
    if b == "postgres":
        return f"[{','.join(str(x) for x in embedding)}]"
    if b == "sqlite":
        return json.dumps(embedding)
    return list(embedding)  # DuckDB: native list


def duckdb_rows(cur) -> list[dict]:
    """Convert DuckDB cursor results to list of dicts."""
    if cur.description is None:
        return []
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def duckdb_row(cur) -> dict | None:
    """Convert single DuckDB cursor result to dict."""
    if cur.description is None:
        return None
    row = cur.fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))
