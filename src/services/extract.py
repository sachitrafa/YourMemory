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
    print(
        "YourMemory: spaCy model not found. Run `yourmemory-setup` once to install it.\n"
        "  Falling back to built-in regex categorization.",
        file=sys.stderr,
    )
except Exception:
    pass


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
