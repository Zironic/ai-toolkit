# tasks/

Working docs for in-flight and completed engineering efforts (mostly the
memory-management / offload streaming work). Higher-level "why we chose X"
rationale lives in [`docs/decisions/`](../docs/decisions/).

- **`open/`** — plans/TODOs with work still pending. Each file is the durable
  spec for one effort.
- **`done/`** — move a doc here once its work is fully shipped (keep it as the
  record of what was built and why).

These `.md` files are the **primary planning/design docs** — stable, edited when
the *design* changes. **Current status** (what's done, in progress, blocked)
lives in the **git-bug** ticket for each effort, not here — so the specs don't
have to be rewritten as work proceeds. See `../CLAUDE.md` for git-bug usage.

List `open/` to see what's in flight; each doc's first paragraph states its
effort. (No index table here — it would just drift.)
