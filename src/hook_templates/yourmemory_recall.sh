#!/bin/bash
# Recall memories before every Claude Code prompt.
# Reads the user's prompt from stdin JSON, queries YourMemory, injects as additionalContext.

QUERY=$(jq -r '.prompt // ""' 2>/dev/null)
[ -z "$QUERY" ] && exit 0

# Skip injection on trivial prompts — saves Claude tokens on short replies
[ "${#QUERY}" -lt 30 ] && exit 0

# Resolve the user_id (env → ~/.yourmemory/user_id → system login name)
source "$(dirname "$0")/yourmemory_user.sh"
USER_ID=$(resolve_yourmemory_user)

RESULT=$(curl -sf --max-time 5 -X POST http://localhost:3033/retrieve \
  -H "Content-Type: application/json" \
  -d "{\"userId\":$(printf '%s' "$USER_ID" | jq -Rs .),\"query\":$(echo "$QUERY" | jq -Rs .), \"topK\":3}" 2>/dev/null)

[ -z "$RESULT" ] && exit 0

MEMORIES=$(echo "$RESULT" | jq -r '[.memories[]? | select(.score >= 0.5 and .similarity >= 0.50) | "- " + .content] | join("\n")' 2>/dev/null)
[ -z "$MEMORIES" ] && exit 0

CONTEXT="[YourMemory — recalled context]
$MEMORIES"

jq -n --arg ctx "$CONTEXT" '{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": $ctx
  }
}'
