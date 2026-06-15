#!/bin/bash
# Resolve the YourMemory user_id consistently across all hooks.
# Resolution order (first non-empty wins):
#   1. $YOURMEMORY_USER          — explicit override
#   2. ~/.yourmemory/user_id     — value chosen at setup
#   3. system login name         — automatic, per-machine (one identity per OS user)
# Output: the resolved id, lowercased, on stdout.
resolve_yourmemory_user() {
  local uid="${YOURMEMORY_USER:-}"
  if [ -z "$uid" ] && [ -s "$HOME/.yourmemory/user_id" ]; then
    uid=$(tr -d '[:space:]' < "$HOME/.yourmemory/user_id")
  fi
  [ -z "$uid" ] && uid=$(id -un 2>/dev/null)
  [ -z "$uid" ] && uid="user"
  printf '%s' "$uid" | tr '[:upper:]' '[:lower:]'
}
