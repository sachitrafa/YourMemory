"""
Semantic deduplication for POST /memories.

Detects near-duplicate memories via cosine similarity and applies one of:
  - reinforce : sim ≥ 0.95  — paraphrase, bump recall_count only
  - replace   : 0.85–0.95 + contradiction detected — overwrite with incoming
  - merge     : 0.85–0.95 + no contradiction — entity-append to existing
  - new       : sim < 0.85  — genuinely distinct, plain INSERT
"""

import math
from src.services.extract import _nlp

DEDUP_THRESHOLD     = 0.85   # below → always new memory
REINFORCE_THRESHOLD = 0.95   # at or above → reinforce (paraphrase)

# Polarity verb antonym pairs (spaCy lemma → antonym lemma)
_ANTONYMS = {
    "love":    "hate",    "hate":    "love",
    "like":    "dislike", "dislike": "like",
    "prefer":  "avoid",   "avoid":   "prefer",
    "use":     "stop",    "stop":    "use",
    "want":    "refuse",  "refuse":  "want",
    "enjoy":   "dislike",
    "start":   "stop",
}


def find_near_duplicate(user_id: str, embedding_str: str, conn) -> dict | None:
    """
    Return the closest existing memory if cosine similarity >= DEDUP_THRESHOLD,
    else None. Uses the caller's open connection.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, category, importance, recall_count,
               1 - (embedding <=> %s::vector) AS similarity
        FROM memories
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT 1
    """, (embedding_str, user_id, embedding_str))
    row = cur.fetchone()
    cur.close()

    if row is None:
        return None

    sim = row[5]
    if sim < DEDUP_THRESHOLD:
        return None

    return {
        "id":          row[0],
        "content":     row[1],
        "category":    row[2],
        "importance":  row[3],
        "recall_count": row[4],
        "similarity":  sim,
    }


def detect_contradiction(existing_text: str, incoming_text: str) -> bool:
    """
    Return True if the incoming text contradicts the existing one.
    Uses spaCy lemmas to find polarity verb antonym pairs.
    """
    existing_verbs = {tok.lemma_.lower() for tok in _nlp(existing_text) if tok.pos_ == "VERB"}
    incoming_verbs = {tok.lemma_.lower() for tok in _nlp(incoming_text) if tok.pos_ == "VERB"}

    for verb in existing_verbs:
        antonym = _ANTONYMS.get(verb)
        if antonym and antonym in incoming_verbs:
            return True
    return False


def merge_entities(existing_text: str, incoming_text: str) -> str:
    """
    Append entities/noun-chunks from incoming that are absent from existing.
    Returns the merged string, or existing_text unchanged if nothing new found.
    """
    existing_lower = existing_text.lower()

    incoming_doc = _nlp(incoming_text)

    # Layer 1: named entities
    candidates = [ent.text for ent in incoming_doc.ents]
    # Layer 2: noun chunks
    candidates += [chunk.text for chunk in incoming_doc.noun_chunks]
    # Layer 3: capitalised tokens (catches tech names like MongoDB, Spring, Vue)
    candidates += [
        tok.text for tok in incoming_doc
        if tok.text[0].isupper() and not tok.is_stop and len(tok.text) > 2
    ]

    new_terms = [
        t for t in candidates
        if t.lower() not in existing_lower and len(t.strip()) > 2
    ]

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in new_terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)

    if not deduped:
        return existing_text
    if len(deduped) == 1:
        return f"{existing_text} with {deduped[0]}"
    return f"{existing_text} with {', '.join(deduped[:-1])} and {deduped[-1]}"


def resolve(user_id: str, content: str, embedding: list, conn) -> dict:
    """
    Facade: decide what to do with an incoming memory.

    Returns:
        {
          "action":   "new" | "reinforce" | "replace" | "merge",
          "content":  str,          # final content to store/update
          "existing": dict | None,  # matched row if any
        }
    """
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    match = find_near_duplicate(user_id, embedding_str, conn)

    if match is None:
        return {"action": "new", "content": content, "existing": None}

    sim = match["similarity"]

    if sim >= REINFORCE_THRESHOLD:
        return {"action": "reinforce", "content": match["content"], "existing": match}

    # 0.85 ≤ sim < 0.95
    if detect_contradiction(match["content"], content):
        return {"action": "replace", "content": content, "existing": match}

    merged = merge_entities(match["content"], content)
    return {"action": "merge", "content": merged, "existing": match}
