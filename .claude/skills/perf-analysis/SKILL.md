---
name: perf-analysis
description: How to read training performance logs and find the live job in this repo. Load when asked about step times, slowdowns, performance_log.jsonl, log.txt, anything under output/, or which job is currently running.
---

# Perf log analysis

## Where job output lives (never search for it)

Live and finished job folders are always at `output/<job name>/` in this repo
(`output/` is a Windows junction into the sibling checkout
`AI-Toolkit-Easy-Install\AI-Toolkit\output\`). Never find/grep the drive for
`performance_log.jsonl` or `log.txt`. Per job folder:

- `performance_log.jsonl` -- the live run's perf windows
- `logs/{N}_performance_log.jsonl`, `logs/{N}_log.txt` -- auto-archived from
  earlier restarts of the same job (numbered)
- `log.txt` -- full stdout of the current run
- `config.yaml` / `.job_config.json` -- the resolved job config

Because `output/` is a junction to real training results, recursive deletes
under it are guarded by a hook -- treat its contents as user data.

## Reading perf logs: use the digest, not the raw jsonl

`venv/Scripts/python.exe scripts/digest_perf_log.py` collapses the jsonl
windows into a readable summary. It accepts a bare job name, a job folder
path, or no argument (most-recently-updated run under `output/`). Key flags:
`--last N`, `--all`, `--archived` -- see `--help`. Parse the raw jsonl by
hand only when the digest provably lacks the field you need.

## Finding the currently running job

Check the live python process's command line instead of guessing:

```
powershell -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Select-Object ProcessId, CommandLine"
```

`run.py` is invoked with the job's `.job_config.json` and `--log log.txt`
paths, which confirms the exact output folder name.

## Interpreting slowdowns (the short version)

- Sudden 10-30x step time near full VRAM with no error: WDDM paged past the
  dedicated ceiling. Load the `wddm-memory` skill.
- A hard `cudaErrorMemoryAllocation` during pin/startup: DXGI shared budget
  exhausted -- also `wddm-memory`.
- Slow first sampling/step after a phase change: compile or prefetch state
  was destroyed (resizes destroy prefetch) -- see the `compile-offload`
  skill.
