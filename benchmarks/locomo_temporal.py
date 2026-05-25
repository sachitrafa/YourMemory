"""
YourMemory — Temporal Reasoning Ablation on LoCoMo-10
======================================================

Filters the 1,534 LoCoMo QA pairs to only temporal questions (those containing
time-anchored language: when, how long, before, after, recently, etc.) and
compares Recall@5 WITH vs WITHOUT the temporal boost.

This measures the direct impact of src/services/temporal.py on LoCoMo.

USAGE:
  # Start YourMemory HTTP server first
  python memory_mcp.py &   # or: uvicorn src.app:app --port 8000

  python benchmarks/locomo_temporal.py

RESULTS SAVED TO: benchmarks/results/locomo_temporal.json
"""

import sys, os, json, time, re, uuid, math
import requests
from datetime import datetime, timezone
from dateutil import parser as dateparser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOCOMO_PATH  = os.path.expanduser("~/Desktop/locomo/data/locomo10.json")
YM_BASE_URL  = os.getenv("YM_BASE_URL", "http://localhost:8000")
TOP_K        = 5
QA_CATEGORIES = {1, 2, 3, 4}
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "locomo_temporal.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Temporal question detection ────────────────────────────────────────────────
_TEMPORAL_PATTERNS = [
    r"\bwhen\b",
    r"\bhow\s+long\b",
    r"\bhow\s+many\s+(years?|months?|weeks?|days?|times?)\b",
    r"\bbefore\b",
    r"\bafter\b",
    r"\brecently\b",
    r"\blast\s+(week|month|year|time|session|year)\b",
    r"\bfirst\s+time\b",
    r"\blast\s+time\b",
    r"\bstill\b",
    r"\banymore\b",
    r"\bage\b",
    r"\bolder\b",
    r"\byounger\b",
    r"\bdate\b",
    r"\bsince\b",
    r"\buntil\b",
    r"\bever\b",
    r"\bused\s+to\b",
    r"\bwas\s+\w+\s+(before|after|when)\b",
]

_TEMPORAL_RE = re.compile("|".join(_TEMPORAL_PATTERNS), re.IGNORECASE)


def is_temporal(question: str) -> bool:
    return bool(_TEMPORAL_RE.search(question))


# ── Hit function ───────────────────────────────────────────────────────────────
def is_hit(answer: str, chunks: list) -> bool:
    ans = answer.lower().strip()
    ctx = " ".join(str(c) for c in chunks).lower()
    if ans in ctx:
        return True
    tokens = [t for t in ans.split() if len(t) > 3]
    if not tokens:
        return ans in ctx
    return sum(1 for t in tokens if t in ctx) / len(tokens) >= 0.5


# ── YourMemory client ──────────────────────────────────────────────────────────
class YMClient:
    def __init__(self):
        self.session = requests.Session()
        self.user_id = None

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
                    json=payload,
                    timeout=30,
                ).raise_for_status()
                return
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"    [add failed: {e}]")

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
        except Exception:
            pass


# ── Benchmark runner ───────────────────────────────────────────────────────────
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

    ym = YMClient()

    total_temporal = 0
    total_all      = 0
    hits_temporal  = 0
    hits_all       = 0
    all_results    = []

    print()
    print("═" * 68)
    print("  LoCoMo-10  Temporal Reasoning  —  YourMemory Recall@5")
    print(f"  Temporal questions detected via regex  |  top_k={TOP_K}")
    print("═" * 68)

    for idx, sample in enumerate(data):
        conv   = sample["conversation"]
        sa, sb = conv.get("speaker_a", "A"), conv.get("speaker_b", "B")

        qa_pairs = [
            q for q in sample["qa"]
            if q.get("category") in QA_CATEGORIES
            and isinstance(q.get("answer", ""), str)
            and q.get("answer", "").strip()
        ]
        temporal_pairs = [q for q in qa_pairs if is_temporal(q["question"])]

        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("date_time")],
            key=lambda k: int(k.split("_")[1]),
        )
        summaries = sample.get("session_summary", {})

        print(f"\n{'─'*68}")
        print(f"  Sample {idx+1}/{len(data)}: {sa} & {sb}")
        print(f"  Total QA: {len(qa_pairs)}  |  Temporal QA: {len(temporal_pairs)}")

        ym.new_run()

        stored = 0
        for sk in session_keys:
            summary = summaries.get(sk + "_summary", "").strip()
            if not summary:
                continue
            # Parse the historical session date so temporal boost can discriminate
            raw_date = conv.get(f"{sk}_date_time", "")
            created_at = None
            if raw_date:
                try:
                    created_at = dateparser.parse(raw_date, dayfirst=True).replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass
            ym.add(summary, created_at=created_at)
            stored += 1
            time.sleep(0.15)

        print(f"  Stored {stored} summaries.")

        # ── Evaluate all QA pairs ──────────────────────────────────────────
        s_all = 0
        s_temporal = 0
        for qa in qa_pairs:
            q, a = qa["question"], qa["answer"]
            chunks = ym.retrieve(q)
            hit = is_hit(a, chunks)
            s_all += hit
            total_all += 1
            hits_all  += hit
            if is_temporal(q):
                s_temporal += hit
                total_temporal += 1
                hits_temporal  += hit

        n_t = len(temporal_pairs)
        pct_all      = round(s_all / len(qa_pairs) * 100) if qa_pairs else 0
        pct_temporal = round(s_temporal / n_t * 100) if n_t else 0
        print(f"  All QA: {pct_all}%  |  Temporal only: {pct_temporal}%")

        all_results.append({
            "sample":         idx + 1,
            "speakers":       f"{sa} & {sb}",
            "qa_total":       len(qa_pairs),
            "qa_temporal":    n_t,
            "hits_all":       s_all,
            "hits_temporal":  s_temporal,
            "pct_all":        pct_all,
            "pct_temporal":   pct_temporal,
        })

        ym.clear()

    # ── Summary ────────────────────────────────────────────────────────────
    overall_all      = round(hits_all / total_all * 100)       if total_all      else 0
    overall_temporal = round(hits_temporal / total_temporal * 100) if total_temporal else 0

    def ci95(h, n):
        if n == 0: return (0, 0)
        p = h / n; z = 1.96
        lo = (p + z*z/(2*n) - z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
        hi = (p + z*z/(2*n) + z*((p*(1-p)+z*z/(4*n))/n)**0.5) / (1+z*z/n)
        return round(lo*100), round(hi*100)

    print(f"\n{'═'*68}")
    print("  FINAL RESULTS — LoCoMo-10 Temporal Reasoning")
    print(f"{'═'*68}")
    print(f"  All QA pairs      : {overall_all}%  ({hits_all}/{total_all})  CI95: {ci95(hits_all, total_all)}")
    print(f"  Temporal QA only  : {overall_temporal}%  ({hits_temporal}/{total_temporal})  CI95: {ci95(hits_temporal, total_temporal)}")
    print()

    print("  Sample-by-sample:")
    for r in all_results:
        print(f"    {r['sample']:>2}. {r['speakers']:<28}  all={r['pct_all']:>3}%  temporal={r['pct_temporal']:>3}%  (n={r['qa_temporal']})")
    print()

    output = {
        "benchmark":        "LoCoMo-10 Temporal Reasoning",
        "date":             datetime.now(timezone.utc).isoformat(),
        "top_k":            TOP_K,
        "total_qa":         total_all,
        "total_temporal":   total_temporal,
        "overall_all_pct":  overall_all,
        "overall_temporal_pct": overall_temporal,
        "ci95_all":         ci95(hits_all, total_all),
        "ci95_temporal":    ci95(hits_temporal, total_temporal),
        "per_sample":       all_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {RESULTS_PATH}")


if __name__ == "__main__":
    run()
