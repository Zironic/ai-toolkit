---
name: handoff
description: Record a token-efficient handoff for the active implementation plan by updating its git-bug ticket with only the state another agent needs to continue. Use at a major plan-step boundary, before switching agents or sessions, or when asked to hand off current work.
---

# Plan Handoff

Create a compact continuation record for the active plan. The durable plan remains in `tasks/open/` or the relevant design document; mutable execution state belongs in that plan's git-bug ticket.

Follow the repository's planning principles: preserve only information needed to reach the requested outcome by the shortest credible path. Do not preserve speculative concerns, implementation history, or completed details that the repository can reveal cheaply. See the goal-oriented planning skill for the governing planning discipline.

## Resolve the handoff target

1. Identify the active plan from the current task, referenced plan file, current branch work, or explicit user instruction.
2. Identify the git-bug ticket associated with that plan. Prefer an explicit ticket pointer in the plan or current conversation. Otherwise inspect relevant open tickets and choose only when the association is clear.
3. If no ticket exists and current work has meaningful mutable state, create one with a short actionable title and a body pointing to the durable plan. Do not create a ticket for a trivial or completed task.
4. Never write to a generic shared handoff file. Each plan owns its own ticket, so concurrent agents and unrelated plans cannot overwrite one another.
5. Do not update another plan's ticket merely because it touches the same files. Record cross-plan dependencies by ticket or plan reference.

## Verify before recording

Inspect current repository state rather than relying only on conversation memory:

- `git status --short --branch`
- relevant diff or commits
- the active plan document
- the active ticket and its latest comments
- tests or commands actually run

Do not claim tests passed, code shipped, or blockers were resolved without evidence. Mark unknown state as unknown.

## Write one append-only ticket comment

Append a new comment to the active ticket. Do not rewrite the plan for ordinary progress and do not replace previous handoffs. Use this exact compact structure, omitting empty sections:

```text
HANDOFF <YYYY-MM-DD HH:MM local> | branch=<branch> | head=<short-sha>

Plan: <path>
Position: <completed plan step or current boundary> -> <next plan step>

Done:
- <only completed facts that constrain or enable later work>

Decisions:
- <decision>: <brief reason>; source=<plan/code/test/commit/ticket>

State:
- <relevant uncommitted or committed changes, named by file/symbol>
- <important runtime or repository state not recoverable from the plan alone>

Validation:
- PASS: <command or check>
- FAIL: <command or check> - <concise failure>
- NOT RUN: <required check and why>

Next:
1. <single concrete next action>
2. <following action only when already determined by the plan>

Blockers/Risks:
- <active blocker or verified risk only>

Refs:
- <other ticket, plan, commit, PR, log, or artifact needed to continue>
```

## Token discipline

- Target 250-700 tokens. Exceed 1,000 only when several verified blockers or independent workstreams must be transferred.
- Prefer paths, symbols, commit IDs, ticket IDs, and commands over prose explanations.
- Refer to the plan instead of restating its architecture, acceptance criteria, or remaining steps.
- Refer to the diff or commit instead of narrating edits already visible in git.
- Preserve failed approaches only when repeating them would be likely and expensive; state the failure and evidence in one line.
- Preserve measurements only when they determine the next decision; include units and test conditions.
- Exclude conversational chronology, command transcripts, resolved errors, broad repository summaries, and code explanations recoverable by inspection.
- Do not duplicate unchanged facts from the previous handoff. State only deltas plus the current next action.

## Concurrency rules

- One plan -> one durable plan document -> one git-bug ticket containing append-only handoffs.
- Multiple agents working the same plan append separate comments; include branch and HEAD so their states remain attributable.
- Multiple plans touching the same subsystem keep separate tickets and reference each other under `Refs` when coordination is required.
- Never use branch name alone as plan identity.
- Never move or close another plan's ticket during handoff.
- Close the ticket only when the plan's completion criteria are implemented and verified, not merely because one agent finished its assigned step.

## Finish

After writing the ticket comment, return only:

- ticket ID
- plan path
- branch and HEAD
- the exact next action

Do not paste the full handoff comment back into chat unless the user asks.