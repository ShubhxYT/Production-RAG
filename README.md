# FullRag

## Copilot Iteration Workflow

This repository includes a GitHub Copilot skill that enforces an iterative workflow:

1. Complete one meaningful iteration.
2. Update `README.md` with what changed.
3. Commit that iteration before starting the next one.

Skill location:

- `.github/skills/commit-after-each-iteration/SKILL.md`

Expected commit message format:

- `iter-N: <short summary of change>`

Example:

- `iter-1: add PDF parsing pipeline scaffold`
