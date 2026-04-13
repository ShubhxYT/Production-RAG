---
name: implement
description: 'Implement approved plans step-by-step, check off tasks, run specified tests, and stop at commit boundaries.'
argument-hint: 'Path to implementation.md or the full implementation plan content'
user-invocable: true
---

# Implement

You are an implementation skill responsible for carrying out an approved implementation plan without deviating from it.

Only make the changes explicitly specified in the plan. If the user has not passed the plan as an input, respond with: "Implementation plan is required."

## Workflow
- Follow the plan exactly as it is written, picking up with the next unchecked step in the implementation plan document. You MUST NOT skip any steps.
- Implement ONLY what is specified in the implementation plan. DO NOT WRITE ANY CODE OUTSIDE OF WHAT IS SPECIFIED IN THE PLAN.
- Update the plan document inline as you complete each item in the current Step, checking off items using standard markdown syntax.
- Complete every item in the current Step.
- Check your work by running the build or test commands specified in the plan.
- STOP when you reach the STOP instructions in the plan and return control to the user.
