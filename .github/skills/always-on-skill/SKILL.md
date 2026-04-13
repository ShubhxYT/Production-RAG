---
name: always-on-skill
description: "Always-on skill: provides persistent workspace guidance and lifecycle hooks. Loaded via workspace instruction to be available without explicit chat invocation."
---

# Always-On Skill

This skill is intended to be available in the workspace context continuously (loaded by a workspace instruction).

Use cases:
- Provide workspace-level guidance and helper commands
- Run deterministic lifecycle checks via hooks
- Inject small, non-invasive helpers into agent context

Files in this package:
- `.github/copilot-instructions.md` — workspace instruction that ensures this skill is loaded (`applyTo: "**"`).
- `.github/hooks/always_on_hook.json` — example hook that runs at lifecycle events.

Guidelines:
- Keep the `description` field short and include trigger keywords if you want it discoverable by search.
- Avoid heavy or blocking logic in hooks; prefer lightweight checks or logging.
