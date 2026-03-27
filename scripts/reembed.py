"""
Re-embed all memories using the current embedding model.

Run this once after switching embedding models to ensure all stored
embeddings are compatible with the new model.

Usage:
    python scripts/reembed.py
    python scripts/reembed.py --dry-run   # preview without writing
    python scripts/reembed.py --batch 50  # custom batch size (default 100)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("DATABASE_URL", "postgresql://localhost:5432/yourmemory")

import psycopg2
from psycopg2.extras import RealDictCursor
from src.services.embed import embed


def reembed(dry_run: bool = False, batch_size: int = 100):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur  = conn.cursor(cursor_factory=RealDictCursor)

    cur.execute("SELECT COUNT(*) AS n FROM memories")
    total = cur.fetchone()["n"]
    print(f"Total memories to re-embed: {total}")

    if total == 0:
        print("Nothing to do.")
        cur.close()
        conn.close()
        return

    if dry_run:
        print("Dry run — no changes written.")
        cur.close()
        conn.close()
        return

    updated = 0
    offset  = 0

    while True:
        cur.execute(
            "SELECT id, content FROM memories ORDER BY id LIMIT %s OFFSET %s",
            (batch_size, offset),
        )
        rows = cur.fetchall()
        if not rows:
            break

        write_cur = conn.cursor()
        for row in rows:
            vec     = embed(row["content"])
            emb_str = f"[{','.join(str(x) for x in vec)}]"
            write_cur.execute(
                "UPDATE memories SET embedding = %s::vector WHERE id = %s",
                (emb_str, row["id"]),
            )
            updated += 1
            print(f"  [{updated}/{total}] id={row['id']}  {row['content'][:60]}")

        conn.commit()
        write_cur.close()
        offset += batch_size

    cur.close()
    conn.close()
    print(f"\nDone. Re-embedded {updated} memories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed all memories with current model.")
    parser.add_argument("--dry-run",  action="store_true", help="Preview without writing")
    parser.add_argument("--batch",    type=int, default=100, help="Batch size (default 100)")
    args = parser.parse_args()

    reembed(dry_run=args.dry_run, batch_size=args.batch)
