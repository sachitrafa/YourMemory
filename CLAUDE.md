# YourMemory — Claude Integration Guide

## Memory Workflow

The memory system is fully automated via hooks. Do NOT call MCP tools explicitly.

### Recall — automatic
The `UserPromptSubmit` hook runs before every message and injects relevant memories as a `system-reminder`. Use that injected context directly — no manual recall call needed.

### Store — automatic
The `Stop` hook fires at session end and extracts facts from the last exchange via `/auto-store`. No explicit `store_memory` or `update_memory` calls needed.

### Your only job
Use the recalled context from `system-reminder` to inform your responses. Everything else is handled by the hooks.

---

## User
- Name: Sachit
- Default user_id: `"sachit"`
