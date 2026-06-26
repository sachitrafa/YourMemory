from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.services.retrieve import retrieve
from src.services.session import (
    recall_cached, recall_cache_set, session_track, start_watchdog,
)

router = APIRouter()
start_watchdog()   # idempotent — safe to call on import


class RetrieveRequest(BaseModel):
    userId:         str
    query:          str
    topK:           int                  = Field(5, ge=1, le=500)
    expandK:        int                  = Field(0, ge=0, le=50)   # extra graph neighbours (additive)
    scoreThreshold: Optional[float]      = None        # drop memories scoring below this
    scope:          Optional[List[str]]  = None        # hard filter by context_paths scope
    currentPath:    Optional[str]        = None        # spatial boost: current file/dir path
    noGraph:        bool                 = False       # ablation: skip graph expansion


@router.post("/retrieve")
def retrieve_memories(req: RetrieveRequest):
    user_id = req.userId.strip().lower()

    # ── Recall throttling ──────────────────────────────────────────────────
    cached = recall_cached(user_id, req.query)
    if cached is not None:
        return cached

    result = retrieve(user_id, req.query, req.topK, current_path=req.currentPath, no_graph=req.noGraph,
                      score_threshold=req.scoreThreshold, scope=req.scope, expand_k=req.expandK)

    # ── Session wrap-up tracking ───────────────────────────────────────────
    session_track(user_id, [m["id"] for m in result.get("memories", [])])

    # ── Cache result ───────────────────────────────────────────────────────
    recall_cache_set(user_id, req.query, result)

    # ── Record activity ────────────────────────────────────────────────────
    try:
        from src.services.decay import record_activity
        record_activity(user_id)
    except Exception:
        pass

    # ── Audit (read) ───────────────────────────────────────────────────────
    try:
        from src.services.audit import log_event
        mems = result.get("memories", [])
        log_event("read", "retrieve", user_id,
                  detail={"count": len(mems), "qlen": len(req.query or ""),
                          "ids": [m["id"] for m in mems][:10]})
    except Exception:
        pass

    return result


# ── Progressive disclosure: two-stage recall ────────────────────────────────
# Stage 1 (search) returns a CHEAP compact index — the model scans it and decides
# what it needs. Stage 2 (get) returns full content for one memory PLUS its 1-hop
# graph neighbourhood (the connected region). This lets the model pull depth on
# demand instead of re-reading source files, while keeping per-turn tokens low.

class SearchRequest(BaseModel):
    userId:  str
    query:   str
    topK:    int = Field(8, ge=1, le=50)
    expandK: int = Field(4, ge=0, le=20)


@router.post("/memory/search")
def memory_search(req: SearchRequest):
    """Compact index: [{id, summary, score, category, via_graph}]. ~cheap to inject."""
    user_id = req.userId.strip().lower()
    result = retrieve(user_id, req.query, req.topK, expand_k=req.expandK)
    hits = []
    for m in result.get("memories", []):
        c = m.get("content", "") or ""
        hits.append({
            "id":        m["id"],
            "summary":   (c[:90] + "…") if len(c) > 90 else c,
            "score":     round(m.get("score", 0), 3),
            "category":  m.get("category"),
            "via_graph": bool(m.get("via_graph", False)),
        })
    return {"results": hits, "count": len(hits)}


@router.get("/memory/get")
def memory_get(userId: str, id: int, neighbors: int = 5):
    """Full content for one memory + its 1-hop graph neighbourhood (the connected region)."""
    from src.services.retrieve import _fetch_by_ids
    from src.graph.graph_store import expand_with_graph
    from src.db.connection import get_backend
    user_id = userId.strip().lower()
    backend = get_backend()
    main = _fetch_by_ids([(id, 1.0)], user_id, backend)
    if not main:
        return {"memory": None, "neighbors": []}
    nb = [(nid, ew) for nid, ew in expand_with_graph([id], user_id, top_k=neighbors + 1)
          if nid != id][:neighbors]
    neighbors_out = _fetch_by_ids(nb, user_id, backend) if nb else []
    return {"memory": main[0], "neighbors": neighbors_out}
