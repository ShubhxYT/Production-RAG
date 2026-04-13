---
name: always-on-workspace-instruction
applyTo: "**"
description: "Load the `always-on-skill` into workspace context on every interaction. Use when you want lightweight helpers and lifecycle hooks available without explicit chat invocation."
---

This workspace instruction ensures the `always-on-skill` assets are discoverable and included in context for the repository. It intentionally uses `applyTo: "**"` to make the instruction available for all files and interactions in the workspace.

Notes:
- Use this pattern sparingly: `applyTo: "**"` increases context loaded for every agent action.
- Keep instruction content minimal and non-sensitive.
