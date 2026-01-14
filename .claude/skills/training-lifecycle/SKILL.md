---
name: training-lifecycle
description: Primary index for training-related code. Use when working on training jobs, loss functions (e.g., preservation_loss), LoRA adapters, checkpoints, dataset bucketing, pipeline/worker scheduling, or reproducibility.
---

# Overview

This skill is the authoritative, canonical starting point (the PRIMARY INDEX) for any work involving training jobs, model adapters (e.g., LoRA), checkpointing, datasets, scheduler/workers, or reproducibility. Agents MUST consult this skill and the canonical chapters under `docs/training_lifecycle/` first — before running broad repository searches or making code changes. The skill provides curated `CODE_MAP`s, condensed references, and task-specific playbooks that map documentation to the exact files and symbols you should inspect.

## When to use

- Always: for any request related to training jobs, model changes, adapters/LoRA, loss functions, checkpointing, datasets, scheduler/worker behavior, or reproducibility.
- You MUST consult this skill (and the referenced `docs/training_lifecycle/*` chapter(s)) before performing any broad repository searches or proposing code changes.
- Use `references/CODE_MAP.md` to find the canonical files/symbols to inspect, then follow the playbooks in `playbooks/` for step-by-step instructions. Update `CODE_MAP.md` when you add new persistent symbols or move implementations.

## Activation & triggers

Agents should consider activating this skill when user queries mention any of the following keywords or symbols: `training`, `job`, `run.py`, `JobLoader`, `checkpoint`, `pretraining`, `bucket`, `scheduler`, `worker`, `prisma`, `cron`, `pipeline`, `LoRA`, `dataset`, `auto_crop_to_bucket`.

When activated, the agent MUST follow this procedure (the docs are authoritative):

1. Read `docs/training_lifecycle/Training_Job_Lifecycle.md` (master index) and the specific chapter(s) referenced there that match the user's task (for example, `07-training-loop.md` for loss/optimizer changes or `06-pretraining-setup.md` for dataset/buckets).
2. Consult `references/CODE_MAP.md` to find the highest-value files and symbols to inspect next (this is a curated short-cut — the docs map to code locations you should inspect first).
3. Load `references/REFERENCE.md` for condensed guidance and playbooks in `playbooks/` for exact step-by-step instructions for common tasks.
4. Use subagents for targeted searching: prefer invoking a subagent (e.g., Raptor Mini) to run concise `rg`/`ripgrep` queries and return a short list of file:line matches — this minimizes context and token usage. Only after reviewing docs and playbooks and using a subagent should you run broader repository searches (`rg`/`ripgrep`) to look for additional implementation details or edge-case occurrences.

6. If broad searches are necessary: limit scope to relevant directories (e.g., `extensions_built_in/`, `toolkit/`, `jobs/`, `scripts/`, `ui/`), prefer filename or symbol-focused queries, and prefer `rg` patterns that include line numbers and file globs. Example commands:
   - `rg "preservation_loss|_compute_and_apply_preservation_loss" extensions_built_in -n`
   - `rg "apply_lora|assistant_lora_path|LoRA" toolkit extensions_built_in ui -n`
   - `rg "run.py|JobLoader|load_job_from_config" -n run.py jobs toolkit`
   - Use `rg -g '!ui/**'` or `rg -g '!tests/**'` to exclude noisy directories when appropriate.
   - When in doubt, consult `.git/blame`/commit history to understand recent changes before editing.

This ordering is designed so the agent starts with documentation-authoritative mappings and avoids unnecessary broad code searches.

This ordering is designed so the agent starts with documentation-authoritative mappings and avoids unnecessary broad code searches.

**Search hints (recommended commands)**

- Prefer using a subagent (e.g., Raptor Mini) to run these queries and return a concise set of `file:line` matches to minimize context/token usage.
- `rg "preservation_loss|_compute_and_apply_preservation_loss" -n`
- `rg "apply_lora|assistant_lora_path|LoRA" -n`
- `rg "run.py|JobLoader|load_job_from_config" -n`
- `rg "auto_crop_to_bucket|DatasetConfig|bucket_logic" -n`

**Agent procedure (MANDATORY)**

1. Read `docs/training_lifecycle/Training_Job_Lifecycle.md` and the chapter(s) most relevant to the user's task.
2. Consult `.claude/skills/training-lifecycle/references/CODE_MAP.md` for canonical file/symbol mappings.
3. Follow the relevant playbook in `.claude/skills/training-lifecycle/playbooks/` and inspect the exact files/line ranges it lists.
4. Run the playbook-specified tests (or the symbol existence checks in `testing/`) and run `python .claude/skills/training-lifecycle/scripts/validate_skill.py .claude/skills/training-lifecycle`.
5. Only after these steps run broad repo searches (e.g., `rg`) if necessary.

## Metadata

- compatibility: Designed for GitHub Copilot Skills and Claude Code
- keywords: training,job,run,checkpoint,pretraining,pipeline,scheduler,worker,dataset,index,primary
- allowed-tools: Read Python(*)

## Quick checklist (short)

- [ ] Define job in UI / config
- [ ] Validate dataset and transforms
- [ ] Verify scheduler & worker config
- [ ] Prepare pipeline & LoRA/model settings
- [ ] Configure checkpoints and push-to-hub steps
- [ ] Run a short smoke job and verify outputs

## Step-by-step (detailed)

1. Plan the job: See `references/REFERENCE.md#01-ui-job-creation` for UI and config guidance.
2. Prepare data and buckets: See `references/REFERENCE.md#06-pretraining-setup`.
3. Configure database and API access: See `references/REFERENCE.md#02-prisma-and-db`.
4. Configure worker scheduling: See `references/REFERENCE.md#03-worker-scheduling`.
5. Load/run pipeline: See `references/REFERENCE.md#04-run-py-and-job-loader` and `references/REFERENCE.md#05-pipeline-and-jobs`.
6. Monitor checkpoints & reproducibility: See `references/REFERENCE.md#08-checkpoints` and `references/REFERENCE.md#09-repro-troubleshoot`.

## Examples

- Example: Create a short smoke job with one epoch, verify checkpoint creation and logs.
- Example: Run failure repro steps from `09-repro-troubleshoot.md` when experiment is non-deterministic.

## How an agent should use this skill

1. Detect trigger words in the user query (see the Activation & triggers section).
2. Load `references/CODE_MAP.md` to find the most relevant files and symbols.
3. Read the short guidance in `references/REFERENCE.md` and then open the corresponding `docs/training_lifecycle/*` chapter for detail.
4. When suggesting code changes, reference the exact file paths and lines; add tests or smoke-configs if applicable, and run the skill validator script (`scripts/validate_skill.py`) before proposing a PR.

### Example user prompts that should activate this skill

- "Help me add a new training job that uses LoRA and checkpoints to S3"
- "Where is the logic that buckets images for pretraining?"
- "The job failed during training with NaNs — how to debug reproducibility?"

## Files & scripts

- `references/REFERENCE.md` — condensed chapter summaries and pointers.
- `scripts/generate_checklist.py` — helper to produce an actionable checklist from the references.

> Tip: Keep the main `SKILL.md` concise (under ~500 lines). Add detailed references in `references/` so agents load them only when needed.
