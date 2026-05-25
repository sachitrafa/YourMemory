"""
BEAM Baseline vs YourMemory Full Stack vs Hindsight — LoCoMo-10
================================================================

Three-way ablation on LoCoMo-10 Recall@5:

  BEAM       — BM25 + Embedding only. No entity graph, no decay pruning,
               no temporal boost. Tests the base retrieval layer in isolation.

  YourMemory — Full stack: BM25 + vector + entity graph + Ebbinghaus decay
               + temporal boost. This is what pip install yourmemory gives.

  Hindsight  — All stored summaries returned (no retrieval cutoff). Oracle
               upper bound — measures how much signal is lost by retrieving
               top-5 instead of the full context.

Directly addresses the HN question: "do BEAM and compare to hindsight."
Shows the contribution of the graph layer over the base BM25+vector retrieval.

USAGE:
  python3 -m uvicorn src.app:app --port 8000 &
  python3 benchmarks/beam_baseline.py

RESULTS: benchmarks/results/beam_baseline.json
"""

import sys, os, json, time, uuid, math
import requests
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOCOMO_PATH  = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
YM_BASE_URL  = os.getenv("YM_BASE_URL", "http://localhost:8000")
TOP_K        = 5
QA_CATEGORIES = {1, 2, 3, 4}
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "beam_baseline.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Hit function (identical across all systems) ───────────────────────────────
def is_hit(answer: str, chunks: list) -> bool:
    ans = answer.lower().strip()
    ctx = " ".join(str(c) for c in chunks).lower()
    if ans in ctx:
        return True
    tokens = [t for t in ans.split() if len(t) > 3]
    if not tokens:
        return ans in ctx
    return sum(1 for t in tokens if t in ctx) / len(tokens) >= 0.5


# ── Shared YourMemory HTTP client ─────────────────────────────────────────────
class YMClient:
    def __init__(self, name: str, no_graph: bool = False, hindsight: bool = False):
        self.name      = name
        self.no_graph  = no_graph
        self.hindsight = hindsight
        self.session   = requests.Session()
        self.user_id   = None

    def new_run(self):
        self.user_id = f"bench_{uuid.uuid4().hex[:10]}"

    def add(self, text: str, created_at: str | None = None):
        payload = {"userId": self.user_id, "content": text, "importance": 0.7}
        if created_at:
            payload["createdAt"] = created_at
        for attempt in range(3):
            try:
                self.session.post(
                    f"{YM_BASE_URL}/memories",
                    json=payload, timeout=30,
                ).raise_for_status()
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [{self.name} add failed: {e}]")

    def retrieve(self, query: str) -> list[str]:
        top_k = 500 if self.hindsight else TOP_K
        for attempt in range(3):
            try:
                r = self.session.post(
                    f"{YM_BASE_URL}/retrieve",
                    json={"userId": self.user_id, "query": query,
                          "topK": top_k, "noGraph": self.no_graph},
                    timeout=30,
                )
                r.raise_for_status()
                return [m["content"] for m in r.json().get("memories", [])]
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return []

    def clear(self):
        try:
            r = self.session.get(
                f"{YM_BASE_URL}/memories",
                params={"userId": self.user_id, "limit": 500}, timeout=15,
            )
            for m in r.json().get("memories", []):
                try:
                    self.session.delete(f"{YM_BASE_URL}/memories/{m['id']}", timeout=10)
                except Exception:
                    pass
        except Exception:
            pass


# ── Benchmark runner ──────────────────────────────────────────────────────────
def run():
    if not os.path.exists(LOCOMO_PATH):
        print(f"ERROR: LoCoMo dataset not found at {LOCOMO_PATH}")
        sys.exit(1)
    try:
        requests.get(f"{YM_BASE_URL}/memories?userId=ping&limit=1", timeout=5)
    except Exception:
        print(f"ERROR: YourMemory server not reachable at {YM_BASE_URL}")
        sys.exit(1)

    with open(LOCOMO_PATH) as f:
        data = json.load(f)

    systems = [
        YMClient("BEAM",       no_graph=True,  hindsight=False),
        YMClient("YourMemory", no_graph=False, hindsight=False),
        YMClient("Hindsight",  no_graph=False, hindsight=True),
    ]
    names    = [s.name for s in systems]
    totals   = {s.name: 0 for s in systems}
    total_qa = 0
    all_results = []

    print()
    print("═" * 72)
    print("  BEAM vs YourMemory vs Hindsight  —  LoCoMo-10 Recall@5")
    print(f"  BEAM=BM25+vector only  |  YourMemory=full stack  |  Hindsight=oracle")
    print(f"  top_k={TOP_K} (Hindsight=all)  |  QA categories 1–4")
    print("═" * 72)

    for idx, sample in enumerate(data):
        conv   = sample["conversation"]
        sa, sb = conv.get("speaker_a", "A"), conv.get("speaker_b", "B")

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
        print(f"  Sample {idx+1}/{len(data)}: {sa} & {sb}  |  QA: {len(qa_pairs)}")

        for s in systems:
            s.new_run()

        # Store same summaries with historical timestamps for all systems
        stored = 0
        for sk in session_keys:
            summary = summaries.get(sk + "_summary", "").strip()
            if not summary:
                continue
            raw_date   = conv.get(f"{sk}_date_time", "")
            created_at = None
            if raw_date:
                try:
                    created_at = dateparser.parse(raw_date, dayfirst=True).replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            for s in systems:
                s.add(summary, created_at=created_at)
            stored += 1
            time.sleep(0.2)

        print(f"  Stored {stored} summaries in all systems.")

        hits = {s.name: 0 for s in systems}
        for qa in qa_pairs:
            q, a = qa["question"], qa["answer"]
            for s in systems:
                chunks = s.retrieve(q)
                hits[s.name] += is_hit(a, chunks)
            time.sleep(0.05)

        n    = len(qa_pairs)
        pcts = {name: round(hits[name] / n * 100) if n else 0 for name in hits}
        row  = " | ".join(f"{name}: {pcts[name]}%" for name in names)
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

        for s in systems:
            s.clear()

    # ── Final summary ──────────────────────────────────────────────────────
    overall = {name: round(totals[name] / total_qa * 100) if total_qa else 0
               for name in names}

    def ci95(h, n):
        if n == 0: return (0, 0)
        p = h / n; z = 1.96
        lo = (p + z*z/(2*n) - z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
        hi = (p + z*z/(2*n) + z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
        return round(lo*100), round(hi*100)

    print(f"\n{'═'*72}")
    print("  FINAL RESULTS — LoCoMo-10 Recall@5")
    print(f"{'═'*72}")
    col_w  = 14
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

    graph_gain     = overall["YourMemory"] - overall["BEAM"]
    hindsight_gap  = overall["Hindsight"]  - overall["YourMemory"]

    print(f"  Graph contribution  : +{graph_gain}pp  (YourMemory vs BEAM)")
    print(f"  Retrieval ceiling   : +{hindsight_gap}pp  remaining to Hindsight oracle")
    print()
    print("  Overall Recall@5 with 95% CI:")
    for name in names:
        lo, hi = ci95(totals[name], total_qa)
        print(f"    {name:<14} {overall[name]:>3}%   CI95: {lo}–{hi}%")
    print()

    output = {
        "benchmark":       "BEAM vs YourMemory vs Hindsight — LoCoMo-10",
        "date":            datetime.now(timezone.utc).isoformat(),
        "top_k":           TOP_K,
        "total_qa":        total_qa,
        "graph_gain_pp":   graph_gain,
        "hindsight_gap_pp": hindsight_gap,
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
            "input":      "session_summary fields with historical timestamps",
            "hit_rule":   "exact substring OR >=50% token overlap (len>3)",
            "beam":       "BM25 + vector only, no graph, no decay pruning",
            "yourmemory": "BM25 + vector + entity graph + decay + temporal boost",
            "hindsight":  "all memories returned (oracle upper bound)",
        },
        "per_sample": all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run()
