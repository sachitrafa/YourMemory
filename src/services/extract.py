import json
import os
import re
import sys
import urllib.request

_QUESTION_WORDS = {"what", "who", "where", "when", "why", "how", "which", "whose", "whom"}

_IMPERATIVE_PATTERNS = [
    r'^(please|use|try|do|don\'t|make|create|add|remove|delete|update)',
    r'^(convert|transform|change|modify|fix|help|show|tell)',
    r'^(install|run|execute|start|stop|restart|configure)',
]

# Load spaCy if available — falls back to regex if model not installed yet
# Run `yourmemory-setup` once after pip install to download the model
_nlp = None
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    try:
        _nlp = spacy.load("en_core_web_md")
    except OSError:
        print(
            "YourMemory: spaCy model not found. Run `yourmemory-setup` once to install it.\n"
            "  Falling back to built-in regex categorization.",
            file=sys.stderr,
        )
except Exception:
    pass

# ── Rules-based extraction constants ─────────────────────────────────────────

_FILLER_STARTS = (
    "hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "sure", "yes", "no",
    "got it", "sounds good", "great", "awesome", "perfect", "absolutely", "certainly",
)

_HIGH_NER_LABELS = {"PERSON", "ORG", "GPE", "DATE", "PRODUCT", "LANGUAGE"}

_HIGH_RE = re.compile(
    r"\b(prefer|always|never|must|require|decided|going to|will use|"
    r"my name is|i am|i'm a|i work|i've been|i have|i live|i moved|i started|"
    r"deadline|due date|important|critical|key requirement)\b",
    re.IGNORECASE,
)

_FAILURE_RE = re.compile(
    r"\b(fail(ed|s)?|broke|broken|error|bug|crash|doesn't work|not working|"
    r"blocked|issue|problem|wrong|exception|traceback)\b",
    re.IGNORECASE,
)

_STRATEGY_RE = re.compile(
    r"\b(fix(ed)?|solution|solved|instead|alternative|workaround|"
    r"replace[sd]?|switch(ed)? to|migrat(ed|ing))\b",
    re.IGNORECASE,
)

_ASSUMPTION_RE = re.compile(
    r"\b(might|probably|maybe|perhaps|think|assume|seem|should|could|possibly)\b",
    re.IGNORECASE,
)

_SKIP_RE = re.compile(
    r"^(import |from |#|//|\*|--|==|```|\s*\{|\s*\[)",
)


def is_question(text: str) -> bool:
    """Return True if the text is a question — questions are not stored as memories."""
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    first_word = re.split(r"\s+", stripped.lower())[0]
    return first_word in _QUESTION_WORDS


def should_store_llm(content: str) -> bool:
    """Ask the local Ollama model whether this content is worth storing as a long-term memory.
    Fails open — returns True if Ollama is unreachable so storage is never silently blocked.
    """
    ollama_url   = os.getenv("YOURMEMORY_OLLAMA_URL",   "http://localhost:11434")
    ollama_model = os.getenv("YOURMEMORY_OLLAMA_MODEL", "qwen2.5:7b")

    prompt = (
        "Decide whether the following text is worth storing as a long-term memory for an AI assistant.\n\n"
        "STORE if it contains: a user preference, a project decision, a technical config value, "
        "a tool or library choice, a bug fix, a workflow rule, or any recurring fact.\n"
        "SKIP if it is: a greeting, a vague question with no new information, "
        "a one-time ephemeral action, or generic filler.\n\n"
        f"Text: {content}\n\n"
        "Reply with exactly one word: STORE or SKIP"
    )

    payload = json.dumps({
        "model":   ollama_model,
        "prompt":  prompt,
        "stream":  False,
        "keep_alive": os.getenv("YOURMEMORY_OLLAMA_KEEPALIVE", "30m"),
        "options": {"temperature": 0, "num_predict": 8},
    }).encode()

    try:
        req = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            answer = json.loads(resp.read()).get("response", "").strip().upper()
            return not answer.startswith("SKIP")
    except Exception:
        return True  # fail open


def categorize(text: str) -> str:
    """
    Classify text as fact or assumption.
    Uses spaCy dependency parse when available, regex heuristics otherwise.
    Run `yourmemory-setup` to enable spaCy.
    """
    if _nlp is not None:
        doc = _nlp(text)
        has_subject = any(tok.dep_ in ("nsubj", "nsubjpass") for tok in doc)
        return "fact" if has_subject else "assumption"

    text_lower = text.lower().strip()
    for pattern in _IMPERATIVE_PATTERNS:
        if re.match(pattern, text_lower):
            return "assumption"
    return "fact"


def extract_facts_rules(text: str) -> list[dict]:
    """Extract memorable facts without an LLM using spaCy NER + regex heuristics.
    Activated by YOURMEMORY_EXTRACT_BACKEND=rules — no Ollama required.
    """
    if not text or len(text.strip()) < 20:
        return []

    # Sentence splitting — spaCy sents if available, else punctuation split
    if _nlp is not None:
        doc = _nlp(text[:10000])
        sentences = [s.text.strip() for s in doc.sents]
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text[:10000])

    facts = []
    seen: set[str] = set()

    for sent in sentences:
        sent = sent.strip()

        # Skip junk
        if not sent or len(sent.split()) < 4:
            continue
        if is_question(sent):
            continue
        if _SKIP_RE.match(sent):
            continue
        lower = sent.lower()
        if any(lower.startswith(f) for f in _FILLER_STARTS):
            continue

        # Deduplicate
        key = lower[:80]
        if key in seen:
            continue
        seen.add(key)

        # Category — failure/strategy override dep-parse
        if _FAILURE_RE.search(sent):
            category = "failure"
        elif _STRATEGY_RE.search(sent):
            category = "strategy"
        elif _ASSUMPTION_RE.search(sent):
            category = "assumption"
        else:
            category = categorize(sent)

        # Importance — NER entity presence → HIGH
        has_entity = False
        if _nlp is not None:
            has_entity = any(ent.label_ in _HIGH_NER_LABELS for ent in _nlp(sent).ents)

        if has_entity or _HIGH_RE.search(sent):
            importance = "HIGH"
        elif len(sent.split()) >= 8:
            importance = "MED"
        else:
            importance = "LOW"

        facts.append({"fact": sent, "importance": importance, "category": category})

    return facts
