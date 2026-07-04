# Git Tickets

This repo uses `git-bug` for stateful local tickets. Keep durable design and architecture rationale in Markdown; keep live task state, validation requests, experiments, and follow-ups in tickets.

## Tool

Use `scripts/tickets.cmd` from the repo root. It bypasses local PowerShell script-signing policy for this repo-local helper. It prefers the repo-local `tools/git-bug.exe` when present and falls back to `git-bug` on `PATH`.

Examples:

```powershell
.\scripts\tickets.cmd list
.\scripts\tickets.cmd list-closed
.\scripts\tickets.cmd show ee6fa64
.\scripts\tickets.cmd new "Short actionable title" "Longer body"
.\scripts\tickets.cmd label ee6fa64 prefetch perf needs-run
.\scripts\tickets.cmd close ee6fa64
```

Direct `git-bug` works too:

Note: Codex sandbox sessions may need approval for ticket operations because
`git-bug` stores state under `.git/git-bug`. Normal terminals should not need
that approval.

```powershell
.\tools\git-bug.exe bug --format plain
.\tools\git-bug.exe bug show ee6fa64
.\tools\git-bug.exe bug comment new ee6fa64 --message "Run result..."
```

## Policy

- Markdown docs are for stable plans, architecture decisions, and runbooks.
- Tickets are for active TODOs, validation state, run requests, bugs, experiments, and "next agent should know" state.
- When a Markdown plan has an active follow-up, create or update a ticket and leave only a short pointer in the doc.
- Close tickets when the code change is implemented and verified, or when the idea is intentionally abandoned.

For the current ticket list, run `.\scripts\tickets.cmd list` (or `list-closed`) —
it is not mirrored here.
