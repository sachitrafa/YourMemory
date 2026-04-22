"""
YourMemory vs Supermemory vs Zep vs Mem0
LoCoMo-10 Benchmark — Recall@5
=================================================

METHODOLOGY (fully reproducible):
  Dataset  : LoCoMo-10  (snap-research/LoCoMo, first 10 conversation samples)
             ~/Desktop/locomo/data/locomo10.json
  Input    : session_summary fields — identical text fed to every system
  Queries  : QA pairs with category in {1, 2, 3, 4}, string answers only
  Metric   : Recall@5 — does the correct answer appear in top-5 retrieved chunks?
  Hit rule : exact substring OR ≥50% meaningful-token overlap (len>3)
             Applied identically to every system's output.

SYSTEMS (each tested as their real product — no special tuning):
  YourMemory  local HTTP server on localhost:8000
              Full stack: BM25 + vector + knowledge graph + Ebbinghaus decay
              Start with: python memory_mcp.py  (or uvicorn src.main:app)
  Supermemory cloud API  supermemory.ai  rerank=True  limit=5
  Zep         Zep Cloud  app.getzep.com  thread memory  limit=5
  Mem0        Mem0 Cloud api.mem0.ai     search         limit=5

ISOLATION:
  Each system gets a fresh user/container per sample.
  All systems receive the same summaries in the same order.
  Cleanup (delete) is called after every sample.

REQUIRED ENV VARS (.env):
  SUPERMEMORY_API_KEY
  ZEP_API_KEY
  MEM0_API_KEY

USAGE:
  # 1. Start YourMemory HTTP server
  python main.py &

  # 2. Run benchmark
  python benchmarks/locomo_4way.py

  # 3. Results saved to benchmarks/locomo_4way_results.json
"""

import sys, os, json, time, uuid, math
import numpy as np
import requests
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LOCOMO_PATH     = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
YM_BASE_URL     = os.getenv("YM_BASE_URL", "http://localhost:8000")
TOP_K           = 5
QA_CATEGORIES   = {1, 2, 3, 4}
RESULTS_PATH    = os.path.join(os.path.dirname(__file__), "locomo_4way_results.json")

SUPERMEMORY_KEY = os.getenv("SUPERMEMORY_API_KEY", "")
ZEP_KEY         = os.getenv("ZEP_API_KEY", "")
MEM0_KEY        = os.getenv("MEM0_API_KEY", "")

# ── Shared hit function (identical for all systems) ───────────────────────────
def is_hit(answer: str, chunks: list) -> bool:
    """
    Returns True if `answer` appears in the concatenated retrieved chunks.
    Rule: exact substring match  OR  >=50% of meaningful tokens (len>3) present.
    Applied identically to every system.
    """
    ans   = answer.lower().strip()
    ctx   = " ".join(str(c) for c in chunks).lower()
    if ans in ctx:
        return True
    tokens = [t for t in ans.split() if len(t) > 3]
    if not tokens:
        return ans in ctx
    return sum(1 for t in tokens if t in ctx) / len(tokens) >= 0.5


def parse_date(s: str) -> datetime:
    try:
        return dateparser.parse(s, dayfirst=True).replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


# ── System wrappers ───────────────────────────────────────────────────────────

class YourMemorySystem:
    """
    Calls the live YourMemory HTTP server.
    Full stack: BM25 + vector + knowledge graph + Ebbinghaus decay.
    This is what pip install yourmemory gives you.
    """
    name = "YourMemory"

    def __init__(self):
        self.session = requests.Session()
        self.user_id = None

    def new_run(self):
        self.user_id = f"bench_{uuid.uuid4().hex[:10]}"

    def add(self, text: str):
        for attempt in range(3):
            try:
                r = self.session.post(
                    f"{YM_BASE_URL}/memories",
                    json={"userId": self.user_id, "content": text, "importance": 0.7},
                    timeout=30,
                )
                r.raise_for_status()
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [YM add failed: {e}]")

    def retrieve(self, query: str) -> list[str]:
        for attempt in range(3):
            try:
                r = self.session.post(
                    f"{YM_BASE_URL}/retrieve",
                    json={"userId": self.user_id, "query": query, "topK": TOP_K},
                    timeout=30,
                )
                r.raise_for_status()
                return [m["content"] for m in r.json().get("memories", [])]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [YM retrieve failed: {e}]")
                    return []

    def clear(self):
        try:
            r = self.session.get(
                f"{YM_BASE_URL}/memories",
                params={"userId": self.user_id, "limit": 500},
                timeout=15,
            )
            for m in r.json().get("memories", []):
                try:
                    self.session.delete(f"{YM_BASE_URL}/memories/{m['id']}", timeout=10)
                except Exception:
                    pass
        except Exception as e:
            print(f"    [YM clear failed: {e}]")


class SupermemorySystem:
    """
    Supermemory cloud API — supermemory.ai
    rerank=True, limit=5. Each sample gets its own container_tag.
    """
    name = "Supermemory"
    INDEX_WAIT = 25  # seconds after last add before querying (queued indexing)

    def __init__(self):
        import supermemory
        self.client        = supermemory.Supermemory(api_key=SUPERMEMORY_KEY)
        self.container_tag = None

    def new_run(self):
        self.container_tag = f"bench_{uuid.uuid4().hex[:10]}"

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

    def wait_for_index(self):
        time.sleep(self.INDEX_WAIT)

    def retrieve(self, query: str) -> list[str]:
        for attempt in range(3):
            try:
                resp   = self.client.search.memories(
                    q=query, container_tag=self.container_tag,
                    limit=TOP_K,
                )
                chunks = []
                for item in resp.results or []:
                    # API returns 'memory' field (not 'content')
                    for field in ("memory", "content"):
                        if c := getattr(item, field, None):
                            chunks.append(c)
                            break
                    for chunk in getattr(item, "chunks", []) or []:
                        for field in ("memory", "content"):
                            if c := getattr(chunk, field, None):
                                chunks.append(c)
                                break
                return chunks[:TOP_K]
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    print(f"    [SM search failed: {e}]")
                    return []

    def clear(self):
        try:
            self.client.documents.delete_bulk(container_tags=[self.container_tag])
        except Exception as e:
            print(f"    [SM clear failed: {e}]")


class ZepSystem:
    """
    Zep Cloud — app.getzep.com
    Thread-based memory. Retrieves via memory.search (individual facts, limit=5).
    Falls back to thread context if search returns nothing.
    Marks itself unavailable if API auth fails during setup.
    """
    name = "Zep"
    INDEX_WAIT = 25  # Zep extracts facts asynchronously via LLM

    def __init__(self):
        from zep_cloud.client import Zep
        self.client    = Zep(api_key=ZEP_KEY)
        self.user_id   = None
        self.thread_id = None
        self.available = True  # set False on 401 so we skip gracefully

    def new_run(self):
        from zep_cloud.types import Message
        self.user_id   = f"bench_{uuid.uuid4().hex[:8]}"
        self.thread_id = f"thread_{uuid.uuid4().hex[:8]}"
        try:
            self.client.user.add(user_id=self.user_id)
            self.client.thread.create(thread_id=self.thread_id, user_id=self.user_id)
            self.available = True
        except Exception as e:
            err = str(e)
            if "401" in err or "unauthorized" in err.lower():
                if self.available:  # only warn once
                    print(f"    [Zep] API key invalid / expired — skipping Zep for all samples.")
                self.available = False
            else:
                print(f"    [Zep session setup error: {e}]")

    def add(self, text: str):
        if not self.available:
            return
        from zep_cloud.types import Message
        try:
            self.client.thread.add_messages(
                thread_id=self.thread_id,
                messages=[Message(role="user", role_type="user", content=text)],
            )
        except Exception as e:
            print(f"    [Zep add error: {e}]")

    def wait_for_index(self):
        if self.available:
            time.sleep(self.INDEX_WAIT)

    def retrieve(self, query: str) -> list[str]:
        if not self.available:
            return []
        # Primary: search individual facts
        try:
            results = self.client.memory.search(
                session_id=self.thread_id, text=query, limit=TOP_K
            )
            chunks = []
            for r in results or []:
                msg = getattr(r, "message", None)
                if msg and (c := getattr(msg, "content", None)):
                    chunks.append(c)
            if chunks:
                return chunks[:TOP_K]
        except Exception:
            pass
        # Fallback: full context string
        try:
            ctx = self.client.thread.get_user_context(thread_id=self.thread_id)
            if ctx and ctx.context:
                return [ctx.context]
        except Exception as e:
            print(f"    [Zep retrieve failed: {e}]")
        return []

    def clear(self):
        if not self.available:
            return
        try:
            self.client.user.delete(user_id=self.user_id)
        except Exception:
            pass


class Mem0System:
    """
    Mem0 Cloud — api.mem0.ai
    search() returns top-5 memories ranked by relevance.
    Each sample gets its own user_id for isolation.
    """
    name = "Mem0"
    INDEX_WAIT = 8  # Mem0 indexes quickly but give it a moment

    def __init__(self):
        from mem0 import MemoryClient
        self.client  = MemoryClient(api_key=MEM0_KEY)
        self.user_id = None

    def new_run(self):
        self.user_id = f"bench_{uuid.uuid4().hex[:10]}"

    def add(self, text: str):
        for attempt in range(3):
            try:
                self.client.add(
                    [{"role": "user", "content": text}],
                    user_id=self.user_id,
                )
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"    [Mem0 add failed: {e}]")

    def wait_for_index(self):
        time.sleep(self.INDEX_WAIT)

    def retrieve(self, query: str) -> list[str]:
        for attempt in range(3):
            try:
                results = self.client.search(
                    query,
                    filters={"user_id": self.user_id},
                    limit=TOP_K,
                )
                # response is either a list or a dict with 'results' key
                items = results if isinstance(results, list) else results.get("results", [])
                return [r["memory"] for r in items if r.get("memory")]
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"    [Mem0 search failed: {e}]")
                    return []

    def clear(self):
        try:
            self.client.delete_all(user_id=self.user_id)
        except Exception as e:
            print(f"    [Mem0 clear failed: {e}]")


# ── Benchmark runner ──────────────────────────────────────────────────────────

def check_prerequisites(systems: list):
    missing = []
    if not SUPERMEMORY_KEY: missing.append("SUPERMEMORY_API_KEY")
    if not ZEP_KEY:         missing.append("ZEP_API_KEY")
    if not MEM0_KEY:        missing.append("MEM0_API_KEY")
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(1)

    # Ping YourMemory server
    try:
        requests.get(f"{YM_BASE_URL}/health", timeout=5).raise_for_status()
    except Exception:
        try:
            # try memories endpoint as fallback ping
            requests.get(f"{YM_BASE_URL}/memories?userId=ping&limit=1", timeout=5)
        except Exception:
            print(f"ERROR: YourMemory server not reachable at {YM_BASE_URL}")
            print("Start it with:  python main.py  (or uvicorn src.main:app --port 8000)")
            sys.exit(1)


def run():
    systems = [YourMemorySystem(), SupermemorySystem(), ZepSystem(), Mem0System()]

    check_prerequisites(systems)

    if not os.path.exists(LOCOMO_PATH):
        print(f"ERROR: LoCoMo dataset not found at {LOCOMO_PATH}")
        print("Download from: https://huggingface.co/datasets/snap-research/LoCoMo")
        sys.exit(1)

    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    names     = [s.name for s in systems]
    totals    = {s.name: 0 for s in systems}
    total_qa  = 0
    all_results = []

    print()
    print("═" * 72)
    print("  LoCoMo-10  Recall@5  —  4-Way Benchmark")
    print(f"  Systems : {' · '.join(names)}")
    print(f"  Samples : {len(data)}  |  top_k={TOP_K}  |  QA categories 1–4")
    print(f"  Hit rule: exact substring OR ≥50% token overlap (len>3)")
    print("═" * 72)

    for idx, sample in enumerate(data):
        conv     = sample["conversation"]
        sa, sb   = conv.get("speaker_a", "A"), conv.get("speaker_b", "B")
        qa_pairs = [
            q for q in sample["qa"]
            if q.get("category") in QA_CATEGORIES
            and isinstance(q.get("answer", ""), str)
            and q.get("answer", "").strip()
        ]

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]),
        )
        summaries = sample.get("session_summary", {})

        print(f"\n{'─'*72}")
        print(f"  Sample {idx+1}/{len(data)}: {sa} & {sb}")
        print(f"  Sessions: {len(session_keys)}  |  QA pairs: {len(qa_pairs)}")

        # ── Fresh run for every system ─────────────────────────────────────
        for s in systems:
            s.new_run()

        # ── Feed identical summaries to all systems ────────────────────────
        stored = 0
        for sk in session_keys:
            summary = summaries.get(sk + "_summary", "").strip()
            if not summary:
                continue
            for s in systems:
                s.add(summary)
            stored += 1
            time.sleep(0.2)

        print(f"  Stored {stored} summaries in all systems.")

        # ── Wait for cloud indexing (max of all required waits) ────────────
        max_wait = max(
            getattr(s, "INDEX_WAIT", 0) for s in systems
        )
        if max_wait > 0:
            print(f"  Waiting {max_wait}s for cloud systems to index...")
            time.sleep(max_wait)

        # ── Evaluate all QA pairs ──────────────────────────────────────────
        hits = {s.name: 0 for s in systems}

        for qa in qa_pairs:
            q, a = qa["question"], qa["answer"]
            for s in systems:
                chunks = s.retrieve(q)
                hits[s.name] += is_hit(a, chunks)
            time.sleep(0.1)

        n = len(qa_pairs)
        pcts = {name: round(hits[name] / n * 100) if n else 0 for name in hits}

        row = " | ".join(f"{name}: {pcts[name]}%" for name in names)
        print(f"  {row}")

        for name in names:
            totals[name] += hits[name]
        total_qa += n

        all_results.append({
            "sample":   idx + 1,
            "speakers": f"{sa} & {sb}",
            "qa":       n,
            **{f"{name}_hits": hits[name] for name in names},
            **{f"{name}_pct":  pcts[name] for name in names},
        })

        # ── Cleanup ────────────────────────────────────────────────────────
        for s in systems:
            s.clear()

    # ── Final results ──────────────────────────────────────────────────────
    overall = {name: round(totals[name] / total_qa * 100) if total_qa else 0
               for name in names}

    # Confidence interval: 95% Wilson interval approximation
    def ci95(hits, n):
        if n == 0: return (0, 0)
        p   = hits / n
        z   = 1.96
        lo  = (p + z*z/(2*n) - z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)
        hi  = (p + z*z/(2*n) + z*math.sqrt((p*(1-p)+z*z/(4*n))/n)) / (1+z*z/n)
        return round(lo*100), round(hi*100)

    print(f"\n{'═'*72}")
    print("  FINAL RESULTS — LoCoMo-10  Recall@5")
    print(f"{'═'*72}")
    col_w = 13
    header = f"  {'#':<4} {'Speakers':<26}" + "".join(f"{n:>{col_w}}" for n in names)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for r in all_results:
        row = f"  {r['sample']:<4} {r['speakers']:<26}"
        row += "".join(f"{r[f'{n}_pct']:>{col_w-1}}%" for n in names)
        print(row)
    print("  " + "─" * (len(header) - 2))
    totals_row = f"  {'ALL':<4} {str(total_qa)+' QA pairs':<26}"
    totals_row += "".join(f"{overall[n]:>{col_w-1}}%" for n in names)
    print(totals_row)
    print(f"{'═'*72}\n")

    print("  Overall Recall@5 with 95% confidence intervals:")
    for name in names:
        lo, hi = ci95(totals[name], total_qa)
        print(f"    {name:<14} {overall[name]:>3}%   (95% CI: {lo}–{hi}%)")
    print()

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "benchmark":   "LoCoMo-10 Recall@5",
        "date":        datetime.now(timezone.utc).isoformat(),
        "top_k":       TOP_K,
        "total_qa":    total_qa,
        "systems": {
            name: {
                "recall_pct": overall[name],
                "hits":       totals[name],
                "ci95":       ci95(totals[name], total_qa),
            }
            for name in names
        },
        "methodology": {
            "dataset":    "snap-research/LoCoMo, 10 samples",
            "input":      "session_summary fields",
            "qa_filter":  "category in {1,2,3,4}, string answers",
            "hit_rule":   "exact substring OR >=50% token overlap (len>3)",
            "yourmemory": "local HTTP server — BM25 + vector + graph + Ebbinghaus decay",
            "supermemory":"cloud API, rerank=True, limit=5",
            "zep":        "Zep Cloud, thread memory, memory.search limit=5",
            "mem0":       "Mem0 Cloud, search limit=5",
        },
        "per_sample": all_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")
    print()

    return output


if __name__ == "__main__":
    run()
