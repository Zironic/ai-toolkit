---
name: goal-oriented-planning
description: Create or revise implementation plans by working backward from the requested outcome and choosing the shortest credible path within verified repository constraints. Use when planning a feature, fix, refactor, migration, integration, or other code change before implementation.
---

# Goal-Oriented Planning

Begin with the concrete outcome the user wants. Define what must be true when the work is complete, then identify the simplest design that fully achieves that outcome within verified repository constraints.

Before finalizing the plan:

- Separate explicit requirements, verified constraints, assumptions, and optional improvements.
- Do not turn inferred concerns or hypothetical edge cases into requirements without supporting evidence.
- Do not invent repository constraints or conventions. Verify them where practical, and label anything unverified as an assumption.
- Avoid adding invariants, guards, abstractions, configuration, extension points, or generality unless they directly support the requested outcome or a verified constraint.
- Check whether the plan is dominated by defensive machinery, architectural cleanup, or future-proofing rather than delivery of the requested feature.
- Prefer a simpler design with explicit, acceptable tradeoffs over a more elaborate design built around speculative correctness concerns.
- Include the integration and validation work necessary to establish that the requested outcome has been achieved.
- Ensure every substantial plan item can be traced to an explicit requirement, a verified constraint, or a necessary validation step.

A good plan makes the shortest credible path from the user's goal to a working implementation obvious.