# GitHub Copilot / AI agent instructions for AI-Toolkit ✅

**Purpose:** Short, actionable guidance for AI coding agents working in this repository. Focus on safe commands, common patterns, and where to find the right places to change code.
## **CRITICAL: Always Use Serena First (#serena MCP server)**

**For ALL analysis, investigation, and code understanding tasks, use Serena semantic tools:**

### **Standard Serena Workflow**
1. **Start with Serena memories**: Use Serena to list memories and read relevant ones for context #serena
2. **Use semantic analysis**: Use Serena to find [symbols/functions/patterns] related to [issue] #serena
3. **Get symbol-level insights**: Use Serena to analyze [specific function] and show all referencing symbols #serena
4. **Create new memories**: Use Serena to write a memory about [findings] for future reference #serena

### **Serena-First Examples**

# Instead of: "Search the codebase for database queries"
# Use: "Use Serena to find all database query functions and analyze their performance patterns #serena"

# Instead of: "Find all admin functions"  
# Use: "Use Serena to get symbols overview of admin files and find capability-checking functions #serena"

# Instead of: "How do the three systems integrate?"
# Use: "Use Serena to read the system-integration-map memory and show cross-system dependencies #serena"


## Quick start (safe, non-destructive) ⚡
- Activate environment: `& venv\Scripts\Activate.ps1` (Windows) or `source venv/bin/activate` (POSIX).
- Install deps (human confirms GPU-specific installs like PyTorch with CUDA): `python -m pip install -r requirements.txt`.
- Fast checks (prefer these before proposing edits):
  - Run unit tests: `python -m pytest testing -q` or single test file: `python -m pytest testing/test_<name>.py -q`.
  - Run a targeted test: `python -m pytest testing -q -k <substring>`.
  - Build UI (fast): `./run-ui build` (POSIX) or `.\run-ui build` (PowerShell). Use the background start helpers (`.\run-ui-start-background.ps1` / `./run-ui-start-background.sh`) to start a production server. To skip the build when starting the background helper, pass `-NoBuild` on Windows (PowerShell) or set `NO_UI_BUILD=1` on POSIX.

## Safe UI helper commands (use these, don't run dev servers unattended) 🧭
- Start background production-like UI (non-interactive):
  - Windows: `.\run-ui-start-background.ps1`
  - POSIX: `./run-ui-start-background.sh`
- Stop the background UI:
  - Windows: `.\run-ui-stop-background.ps1`
  - POSIX: `./run-ui-stop-background.sh`
- Background helper writes: `ui/ui_start.pid`, `ui/ui_start.log`, `ui/ui_start.err`.
- Health check (prod-style): `curl -sfS http://localhost:8675` (returns non-zero if unreachable).
- Playwright/tests notes: always run Playwright from `ui/` or use `npm --prefix ui ...`; prefer production server for e2e to avoid the dev overlay; use `npm --prefix ui exec playwright test` when scripting.

## CLI utilities you will use often 🔧
- Dataset eval (safe with caps):
  `python tools/eval_dataset.py --dataset-path <dataset> --model <name> --batch-size 1 --out-dir <dataset> --sample-fraction 0.1 --max-samples 100`
  - **Always** cap `--max-samples` or `--sample-fraction` for automated runs.
- Worker API endpoints (used by UI): POST `/api/eval_dataset` to enqueue eval jobs; GET `/api/eval_dataset/{id}/result` to fetch JSON report.

## One-line policy: DO NOT start GPU training runs without explicit human approval ⚠️
- Examples of heavy commands to never run autonomously: `python run.py config/<train_config>.yml` or `python run_modal.py config/...`.
- If a task requires GPU runs, ask for resource details and human confirmation, then document the decision in `LEARNINGS.md`.

## Project structure & where to act 🗺️
- `config/` — job config templates (`config/examples/`). Edit here to add new job configs.
- `jobs/` — job **types** (`TrainJob.py`, `GenerateJob.py`, `ExtractJob.py`) and `jobs/process/` contains the per-step processes.
- `toolkit/` — core helpers used across jobs (use these rather than reimplementing):
  - `toolkit/dataloader_mixins.py`, `toolkit/config_modules.py`, `toolkit/stable_diffusion_model.py`, `toolkit/model_utils.py`, `toolkit/train_tools.py`, `toolkit/util/loss_utils.py`, `toolkit/paths.py`.
- `ui/` — Node.js frontend and worker code. Use `npm --prefix ui ...` when scripting.
- `scripts/` — small utility scripts (dataset repair, conversions, helpers).
- `testing/` — pytest test suite: add focused tests here when changing behavior.

## Conventions & patterns to follow 📏
- Jobs follow a **Job** and **Process** model: add new long-running work as a `Job` (in `jobs/`) and break steps into `process/` classes.
- Use existing helpers in `toolkit/` (encoding, model loading, path constants). Search for function names rather than re-implementing logic.
- Tests: prefer small, targeted tests covering the change (`testing/`) and run the subset locally before committing.
- Documentation: when adding or changing features, update `AGENTS.md` (short note) and `LEARNINGS.md` (why, tests, caveats).

## UI/E2E gotchas 👀
- Dev overlay (Next.js dev) can interfere with Playwright clicks. Prefer production `build+start` for E2E, or use the overlay-removal helper referenced in `LEARNINGS.md` (see `ui/tests/e2e/test_helpers.ts`).
- Always verify `Get-Location` / `pwd` when running UI tests from automation; if tests fail to find config, use `npm --prefix ui` to avoid nested path issues.

## When editing code (agent-specific tips) 💡
- Small, focused edits are preferred. Add or update a test in `testing/` that reproduces the intended behavior.
- If you change an interface used by multiple symbols, search for references (e.g., `rg "ClassName|def name"`) and update callers.
- Add brief notes to `AGENTS.md` and `LEARNINGS.md` describing the change and any test coverage decisions.

## Files to check for precedent and examples 📚
- `AGENTS.md` (root) — detailed, human-oriented guidance (read before making automation changes).
- `LEARNINGS.md` — past findings and gotchas (search here for prior failures).
- `jobs/` and `jobs/process/` — example job/process patterns; copy structure for new job types.
- `ui/run-ui-*.ps1` / `run-ui-start-background.sh` — how the repo expects the UI to be started/stopped.

---
If anything here is unclear or you want me to add examples for a particular area (UI tests, adding a new Job, dataset eval automation), tell me which section to expand and I will iterate. ✅
