"""
LoCoMo Benchmark: YourMemory (full stack) vs Supermemory
---------------------------------------------------------
Uses the real LoCoMo dataset (locomo10.json, 10 conversation samples).

YourMemory  : full HTTP stack on localhost:8000
              POST /memories  — resolve + embed + graph index + decay
              POST /retrieve  — vector search + graph expansion + recall propagation
              DELETE /memories — per-id cleanup between samples

Supermemory : cloud API (supermemory.add / search.memories)

Metric: Recall@5 — correct answer in top-5 retrieved chunks?

Usage:
    # Start server first:
    python3 main.py &
    # Then run:
    python3 benchmarks/locomo_supermemory.py

Requires:
    - YourMemory server running on localhost:8000
    - Ollama running locally with nomic-embed-text pulled
    - SUPERMEMORY_API_KEY set (or hardcoded below)
    - LoCoMo dataset at ~/Desktop/locomo/data/locomo10.json
"""

import sys, os, json, time, uuid
import requests
import supermemory
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

LOCOMO_PATH    = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
YM_BASE_URL    = "http://localhost:8000"
SM_API_KEY     = os.getenv("SUPERMEMORY_API_KEY")
TOP_K          = 5
MAX_SAMPLES    = None   # None = all 10
MAX_QA         = None   # None = all QA pairs per sample
INDEX_WAIT_SEC = 10     # seconds for Supermemory to index after storing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(s: str) -> datetime:
    try:
        return dateparser.parse(s, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def answer_hit(answer: str, chunks: list) -> bool:
    al  = answer.lower().strip()
    ctx = " ".join(str(c) for c in chunks).lower()
    if al in ctx:
        return True
    toks = [t for t in al.split() if len(t) > 3]
    if not toks:
        return al in ctx
    return sum(1 for t in toks if t in ctx) / len(toks) >= 0.5


# ---------------------------------------------------------------------------
# YourMemory — full HTTP stack
# ---------------------------------------------------------------------------

class YourMemoryHTTP:
    """Calls the live YourMemory server. Full stack: resolve, graph, decay."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.user_id  = None
        self.session  = requests.Session()

    def new_user(self):
        self.user_id = f"locomo_{uuid.uuid4().hex[:10]}"

    def add(self, text: str):
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/memories",
                    json={"userId": self.user_id, "content": text, "importance": 0.7},
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [YM add failed: {e}]")

    def retrieve(self, query: str) -> list[str]:
        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/retrieve",
                    json={"userId": self.user_id, "query": query, "topK": TOP_K},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return [m["content"] for m in data.get("memories", [])]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [YM retrieve failed: {e}]")
                    return []

    def clear(self):
        """Delete all memories for current user_id."""
        try:
            resp = self.session.get(
                f"{self.base_url}/memories",
                params={"userId": self.user_id, "limit": 500},
                timeout=15,
            )
            memories = resp.json().get("memories", [])
            for m in memories:
                try:
                    self.session.delete(
                        f"{self.base_url}/memories/{m['id']}",
                        timeout=10,
                    )
                except Exception:
                    pass
        except Exception as e:
            print(f"    [YM clear failed: {e}]")


# ---------------------------------------------------------------------------
# Supermemory wrapper
# ---------------------------------------------------------------------------

class SupermemorySystem:
    def __init__(self, api_key: str):
        self.client        = supermemory.Supermemory(api_key=api_key)
        self.container_tag = None

    def new_container(self):
        self.container_tag = f"locomo_{uuid.uuid4().hex[:10]}"

    def add(self, text: str):
        for attempt in range(3):
            try:
                self.client.add(content=text, container_tag=self.container_tag)
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"    [SM add failed: {e}]")

    def search(self, query: str) -> list[str]:
        for attempt in range(3):
            try:
                resp = self.client.search.memories(
                    q=query,
                    container_tag=self.container_tag,
                    limit=TOP_K,
                    rerank=True,
                )
                chunks = []
                for item in resp.results or []:
                    content = getattr(item, "content", None)
                    if content:
                        chunks.append(content)
                    for chunk in getattr(item, "chunks", []) or []:
                        c = getattr(chunk, "content", None)
                        if c:
                            chunks.append(c)
                return chunks[:TOP_K]
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"    [SM search failed: {e}]")
                    return []

    def clear(self):
        if not self.container_tag:
            return
        try:
            self.client.documents.delete_bulk(container_tags=[self.container_tag])
        except Exception as e:
            print(f"    [SM cleanup failed: {e}]")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run():
    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    ym = YourMemoryHTTP(YM_BASE_URL)
    sm = SupermemorySystem(SM_API_KEY)

    samples = data[:MAX_SAMPLES] if MAX_SAMPLES else data

    total_qa = ym_total = sm_total = 0
    results  = []

    print()
    print("=" * 68)
    print("  LoCoMo BENCHMARK: YourMemory (full stack) vs Supermemory")
    print(f"  YM: localhost:8000 (vector + graph + decay + resolve)")
    print(f"  SM: cloud API (vector search, no temporal signal)")
    print(f"  Metric: Recall@{TOP_K} | Samples: {len(samples)} | QA/sample: {MAX_QA or 'all'}")
    print("=" * 68)

    for idx, sample in enumerate(samples):
        conv = sample["conversation"]
        sa   = conv.get("speaker_a", "A")
        sb   = conv.get("speaker_b", "B")

        qa_pairs = [
            q for q in sample["qa"]
            if q.get("category") in (1, 2, 3, 4)
            and isinstance(q.get("answer", ""), str)
            and q.get("answer", "").strip()
        ]
        if MAX_QA:
            qa_pairs = qa_pairs[:MAX_QA]

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1])
        )
        summaries = sample.get("session_summary", {})

        print(f"\n{'─'*68}")
        print(f"  Sample {idx+1}/{len(samples)}: {sa} & {sb}")
        print(f"  Sessions: {len(session_keys)}  |  QA pairs: {len(qa_pairs)}")

        # Reset both systems
        ym.new_user()
        sm.new_container()

        # Store session summaries
        stored = 0
        for sk in session_keys:
            summary = summaries.get(sk + "_summary", "")
            if not summary:
                continue
            ym.add(summary)
            sm.add(summary)
            stored += 1
            time.sleep(0.3)

        print(f"  Stored {stored} summaries in both systems.")
        print(f"  Waiting {INDEX_WAIT_SEC}s for Supermemory to index...")
        time.sleep(INDEX_WAIT_SEC)

        # Evaluate
        ym_hits = sm_hits = 0
        for qa in qa_pairs:
            q, a    = qa["question"], qa["answer"]
            ym_ctx  = ym.retrieve(q)
            sm_ctx  = sm.search(q)
            ym_hits += answer_hit(a, ym_ctx)
            sm_hits += answer_hit(a, sm_ctx)
            time.sleep(0.15)

        n      = len(qa_pairs)
        ym_pct = round(ym_hits / n * 100) if n else 0
        sm_pct = round(sm_hits / n * 100) if n else 0
        delta  = ym_pct - sm_pct

        print(f"  YourMemory (full stack): {ym_hits}/{n} ({ym_pct}%)")
        print(f"  Supermemory:             {sm_hits}/{n} ({sm_pct}%)")
        print(f"  Δ (YM−SM):               {'+' if delta >= 0 else ''}{delta}%")

        total_qa += n
        ym_total += ym_hits
        sm_total += sm_hits
        results.append({
            "sample":   idx + 1,
            "speakers": f"{sa} & {sb}",
            "qa":       n,
            "ym_hits":  ym_hits,
            "sm_hits":  sm_hits,
            "ym_pct":   ym_pct,
            "sm_pct":   sm_pct,
        })

        # Cleanup
        ym.clear()
        sm.clear()

    # Final summary
    overall_ym    = round(ym_total / total_qa * 100) if total_qa else 0
    overall_sm    = round(sm_total / total_qa * 100) if total_qa else 0
    overall_delta = overall_ym - overall_sm

    print(f"\n{'═'*68}")
    print("  FINAL RESULTS — LoCoMo Recall@5")
    print(f"{'═'*68}\n")
    print(f"  {'#':<4} {'Speakers':<28} {'YM (full)':>10} {'SM':>6} {'Δ':>7}")
    print(f"  {'─'*58}")
    for r in results:
        d    = r["ym_pct"] - r["sm_pct"]
        sign = "+" if d >= 0 else ""
        print(f"  {r['sample']:<4} {r['speakers']:<28} {r['ym_pct']:>9}% {r['sm_pct']:>5}% {sign+str(d)+'%':>7}")

    print(f"  {'─'*58}")
    sign = "+" if overall_delta >= 0 else ""
    print(f"  {'ALL':<4} {str(total_qa)+' QA pairs':<28} {overall_ym:>9}% {overall_sm:>5}% {sign+str(overall_delta)+'%':>7}")
    print(f"\n  {'─'*58}")
    print(f"  YourMemory (full stack) Recall@{TOP_K}: {overall_ym}%")
    print(f"  Supermemory             Recall@{TOP_K}: {overall_sm}%")
    winner = "YourMemory" if overall_ym > overall_sm else ("Supermemory" if overall_sm > overall_ym else "Tie")
    print(f"  Winner: {winner}  ({sign+str(overall_delta)}% gap)")
    print(f"\n  Stack diff from previous run (in-process vs full stack):")
    print(f"  Previous YM (vector+decay only): 47%")
    print(f"  Current  YM (+ graph expansion): {overall_ym}%")
    print(f"  Graph layer delta: {overall_ym - 47:+d}%")
    print(f"{'═'*68}\n")

    return {"ym_recall": overall_ym, "sm_recall": overall_sm, "total_qa": total_qa}


if __name__ == "__main__":
    run()
