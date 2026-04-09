---
name: core-rules
description: 'Mandatory rules that must be followed on every task without exception. These are baseline behavioral constraints for this workspace.'
user-invocable: false
always-apply: true
---

# Core Rules

These rules are **MANDATORY** and must be followed on **every single task**, regardless of what is being asked. They are not optional and cannot be overridden by other skills or instructions unless the user explicitly states otherwise in their message.

## Rule 1: Ignore the `plans/` Directory

- **NEVER** browse, read, search, list, or reference anything inside the `plans/` directory.
- Do NOT use the `plans/` directory for context, evaluation, pattern matching, research, or any other purpose.
- Do NOT suggest or assume that the `plans/` directory contains relevant information.
- This rule applies to **all subdirectories** inside `plans/` as well (e.g., `plans/feature-x/`, `plans/feature-x/plan.md`, etc.).

### Exception
The only time you may access the `plans/` directory is when the **user explicitly passes a file or path from within it** in their message (e.g., by typing the path directly or using an `@` mention). In that case, you may read only the specific file(s) the user has passed — nothing else inside `plans/`.

---

## Enforcement

Before performing any task:
1. Check if your planned actions involve accessing the `plans/` directory.
2. If yes, remove that action unless the user has explicitly referenced a `plans/` file in their current message.
3. Proceed with the task using only the allowed sources.
