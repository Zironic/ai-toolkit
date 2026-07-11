#!/usr/bin/env python3
"""
Agent hook policy bundle for Claude Code + OpenAI Codex.

Implements four practical policies:
  1. Block direct reads of large/generated files (bounded/ranged reads pass).
  2. Rewrite raw PowerShell-in-Bash invocations through a stable UTF-8/no-progress/plain-output wrapper.
  3. Cap likely-huge command output (Bash and PowerShell tools) by re-running the
     command through this wrapper in its original shell and saving raw logs under .agent/logs/.
  4. Report Ruff findings that intersect lines changed relative to HEAD.
  5. Silently ASCII-fy edit/write payloads for source files: typographic
     punctuation is transliterated (smart quotes, em dash, arrows, ellipsis)
     and emoji/symbols are dropped. Letters (accented, CJK) and Markdown
     files are left alone, and Edit old_string anchors are never touched.
  6. Repo safety guards (policy 0, checked before any rewrite): deny
     `git-bug webui` and PYTORCH_CUDA_ALLOC_CONF assignments; ask before
     full training runs (run.py) and recursive deletion touching output/.

This script is intentionally conservative: it blocks only mechanically obvious waste.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# ----------------------------- policy knobs -----------------------------

# Generic source files get a generous cap (Claude's Read self-truncates at 2000
# lines anyway); only mechanically obvious waste gets blocked. Generated/vendor
# files keep the strict cap.
LARGE_FILE_BYTES = int(os.environ.get("AGENT_HOOK_LARGE_FILE_BYTES", "300000"))
GENERATED_FILE_BYTES = int(os.environ.get("AGENT_HOOK_GENERATED_FILE_BYTES", "10000"))

# Tool-visible output cap for wrapped commands. Raw output is saved under .agent/logs/.
VISIBLE_HEAD_LINES = int(os.environ.get("AGENT_HOOK_HEAD_LINES", "120"))
VISIBLE_TAIL_LINES = int(os.environ.get("AGENT_HOOK_TAIL_LINES", "120"))
# Keep below Claude Code's own 30k-char Bash output truncation, or this never fires.
POST_TOOL_MAX_CHARS = int(os.environ.get("AGENT_HOOK_POST_TOOL_MAX_CHARS", "24000"))

WRAP_MARKER = "__AGENT_HOOK_WRAPPED__=1"
SKIP_CAP_MARKER = "AGENT_HOOK_NO_CAP=1"

GENERATED_PATTERNS = [
    "*.min.js", "*.min.css", "*.map", "*.lock",
    "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Cargo.lock",
    "*.generated.*", "*.g.cs", "*.Designer.cs", "*.designer.cs",
    "dist/*", "build/*", "out/*", "coverage/*", "node_modules/*",
    ".venv/*", "venv/*", "__pycache__/*", ".git/*",
]

TEXT_EXTENSIONS = {
    ".cs", ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".jsonc", ".css",
    ".scss", ".html", ".md", ".yml", ".yaml", ".rs", ".go", ".java",
    ".kt", ".kts", ".toml", ".xml", ".ps1", ".sh", ".sql", ".txt",
}

ERROR_LINE_RE = re.compile(
    r"(?i)(error|failed|failure|exception|traceback|panic|fatal|segmentation|assert|cannot|denied|not found|timeout)"
)

NOISY_COMMAND_RE = re.compile(
    r"(?ix)"
    r"(\bpytest\b|\bnpm\s+(run\s+)?test\b|\bpnpm\s+test\b|\byarn\s+test\b|"
    r"\bdotnet\s+test\b|\bcargo\s+test\b|\bgo\s+test\b|"
    r"\bgit\s+diff\b|\bgit\s+log\b|\bdocker\s+logs\b|"
    r"\btree\b|\bls\s+-R\b|\bdir\s+/s\b|"
    r"\bgrep\s+-R\b|\brg\s+['\"]?\.['\"]?\b|"
    r"Get-ChildItem\b.*-Recurse\b)"
)

# Repo safety guards: mechanical rules from CLAUDE.md that should never depend
# on the model remembering them. deny = hard rule, ask = user confirms intent.
SAFETY_DENY: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)git-bug(?:\.exe)?['\"]?\s+webui"),
     "git-bug webui holds git-bug's single-access lock for its whole lifetime and "
     "blocks all CLI ticket work. Read tickets via the app's /tickets page; use the "
     "CLI (scripts/tickets.cmd) only for writes."),
    (re.compile(r"(?i)PYTORCH_CUDA_ALLOC_CONF\s*="),
     "Setting PYTORCH_CUDA_ALLOC_CONF (max_split_size_mb/gc_threshold) caused ~30x "
     "slowdowns near full VRAM on this box, and expandable_segments is unsupported "
     "on Windows. Leave allocator defaults alone (see CLAUDE.md)."),
]
SAFETY_ASK: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\bpython[\w.]*(?:\.exe)?['\"]?\s+[^\s;|&]*run\.py\b"),
     "This looks like a full training run (minutes to hours, real datasets and "
     "checkpoints). CLAUDE.md says to launch these only when the user explicitly asks."),
    (re.compile(r"(?i)(\brm\s+-\w*[rf]\w*\s+[^;|&]*\boutput\b"
                r"|\bRemove-Item\b(?=[^;|&]*-(?:Recurse|Force))(?=[^;|&]*\boutput\b)"
                r"|\brmdir\s+/s\b[^;|&]*\boutput\b)"),
     "output/ is a junction into the sibling checkout's real training outputs; "
     "recursive deletion there destroys finished runs. Confirm this is intended."),
]

# Typographic characters with an obvious ASCII spelling. Anything non-ASCII not
# in this map is kept if it is a letter/digit/combining mark, turned into a
# plain space if it is a Unicode space, and dropped otherwise (emoji, symbols,
# format characters like ZWJ/variation selectors).
ASCII_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",    # smart single quotes
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',    # smart double quotes
    "\u2032": "'", "\u2033": '"',                                  # prime marks
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",    # hyphens / en dash
    "\u2014": "-", "\u2015": "-", "\u2212": "-",                   # em dash / bar / minus
    "\u2026": "...",                                               # ellipsis
    "\u2022": "-", "\u00b7": "-", "\u2043": "-",                   # bullets
    "\u2192": "->", "\u2190": "<-", "\u2194": "<->",               # arrows
    "\u21d2": "=>", "\u21d0": "<=", "\u21d4": "<=>",               # double arrows
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=", "\u2248": "~=",
    "\u00d7": "x", "\u00f7": "/", "\u00b1": "+/-",
    "\u00ab": '"', "\u00bb": '"', "\u2039": "'", "\u203a": "'",    # guillemets
}

# Emoji modifiers/joiners are combining marks or format chars; drop them
# explicitly so a stripped emoji does not leave orphaned modifiers behind.
EMOJI_MODIFIER_DROP = {0x200D, 0xFE0E, 0xFE0F, 0x20E3, *range(0xFE00, 0xFE10), *range(0x1F3FB, 0x1F400)}

SANITIZE_SKIP_EXTENSIONS = {".md", ".ipynb"}

POWERSHELL_START_RE = re.compile(r"^\s*(pwsh(?:\.exe)?|powershell(?:\.exe)?)\b", re.I)
POWERSHELL_CMDLET_RE = re.compile(
    r"^\s*(Get-ChildItem|Get-Content|Set-Content|Select-Object|Where-Object|ForEach-Object|"
    r"ConvertTo-Json|Invoke-WebRequest|Invoke-RestMethod|Measure-Object|Test-Path|Copy-Item|Move-Item|"
    r"Remove-Item|New-Item)\b",
    re.I,
)

# ----------------------------- JSON protocol -----------------------------

def read_event() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError as exc:
        # Non-blocking hook failure: malformed input should not break the agent.
        print(f"agent hook: invalid JSON on stdin: {exc}", file=sys.stderr)
        sys.exit(0)


def write_json(obj: dict[str, Any]) -> None:
    json.dump(obj, sys.stdout, ensure_ascii=False, separators=(",", ":"))
    sys.stdout.write("\n")


def event_name(event: dict[str, Any], default: str) -> str:
    return str(event.get("hook_event_name") or default)


def deny_pre_tool(event: dict[str, Any], reason: str) -> None:
    write_json({
        "hookSpecificOutput": {
            "hookEventName": event_name(event, "PreToolUse"),
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    })


def ask_pre_tool(event: dict[str, Any], reason: str) -> None:
    write_json({
        "hookSpecificOutput": {
            "hookEventName": event_name(event, "PreToolUse"),
            "permissionDecision": "ask",
            "permissionDecisionReason": reason,
        }
    })


def allow_updated_input(event: dict[str, Any], updated_input: dict[str, Any], reason: str | None = None) -> None:
    out: dict[str, Any] = {
        "hookSpecificOutput": {
            "hookEventName": event_name(event, "PreToolUse"),
            "permissionDecision": "allow",
            "updatedInput": updated_input,
        }
    }
    if reason:
        out["hookSpecificOutput"]["permissionDecisionReason"] = reason
        out["hookSpecificOutput"]["additionalContext"] = reason
    write_json(out)


def add_context(event: dict[str, Any], text: str) -> None:
    write_json({
        "hookSpecificOutput": {
            "hookEventName": event_name(event, "PostToolUse"),
            "additionalContext": text,
        }
    })

# ----------------------------- path/file utils -----------------------------

def cwd_from_event(event: dict[str, Any]) -> Path:
    return Path(str(event.get("cwd") or os.getcwd())).resolve()


def project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / ".git").exists():
            return p
    return cur


def resolve_path(p: str, cwd: Path) -> Path:
    s = os.path.expandvars(os.path.expanduser(str(p).strip().strip('"').strip("'")))
    path = Path(s)
    if not path.is_absolute():
        path = cwd / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def rel_for_match(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def looks_generated(path: Path, root: Path) -> bool:
    rel = rel_for_match(path, root)
    name = path.name
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in GENERATED_PATTERNS)


def is_text_like(path: Path) -> bool:
    return path.suffix.lower() in TEXT_EXTENSIONS or path.suffix == ""


def file_policy_violation(path: Path, root: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        size = path.stat().st_size
    except OSError:
        return None
    generated = looks_generated(path, root)
    if generated and size > GENERATED_FILE_BYTES:
        return f"generated/vendor/lock file is {size:,} bytes"
    if size > LARGE_FILE_BYTES:
        return f"file is {size:,} bytes (> {LARGE_FILE_BYTES:,})"
    return None


def bounded_read_hint(path: Path) -> str:
    return (
        f"Direct full-file read blocked for {path}. Use a bounded read/search instead, e.g. "
        f"a ranged Read (offset/limit), rg \"symbol_or_error\" {path}, or head/tail."
    )

# ----------------------------- ascii sanitizing -----------------------------

def asciify(text: str) -> str:
    """Transliterate typographic punctuation to ASCII and drop emoji/symbols.

    Letters, digits, and combining marks (accented names, CJK) pass through
    untouched; Unicode spaces become plain spaces.
    """
    if text.isascii():
        return text
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp < 128:
            out.append(ch)
        elif ch in ASCII_MAP:
            out.append(ASCII_MAP[ch])
        elif cp in EMOJI_MODIFIER_DROP:
            continue
        else:
            cat = unicodedata.category(ch)
            if cat.startswith(("L", "M", "N")):
                out.append(ch)
            elif cat == "Zs":
                out.append(" ")
            # else: emoji, other symbols, control/format chars -> dropped
    return "".join(out)


def sanitize_target_path(ti: dict[str, Any]) -> str:
    for key in ("file_path", "filepath", "path", "notebook_path"):
        val = ti.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def mode_pre_edit() -> int:
    event = read_event()
    ti = event.get("tool_input") or {}
    if not isinstance(ti, dict):
        return 0
    suffix = Path(sanitize_target_path(ti)).suffix.lower()
    if suffix in SANITIZE_SKIP_EXTENSIONS:
        return 0
    if suffix and suffix not in TEXT_EXTENSIONS:
        return 0

    updated = dict(ti)
    changed = False
    # Never touch old_string: it must keep matching existing file bytes.
    for key in ("content", "new_string", "new_source"):
        val = ti.get(key)
        if isinstance(val, str):
            fixed = asciify(val)
            if fixed != val:
                updated[key] = fixed
                changed = True

    edits = ti.get("edits")
    if isinstance(edits, list):
        new_edits = []
        for entry in edits:
            if isinstance(entry, dict) and isinstance(entry.get("new_string"), str):
                fixed = asciify(entry["new_string"])
                if fixed != entry["new_string"]:
                    entry = {**entry, "new_string": fixed}
                    changed = True
            new_edits.append(entry)
        if changed:
            updated["edits"] = new_edits

    if changed:
        # Silent by design: no reason text, just the sanitized input.
        allow_updated_input(event, updated)
    return 0

# ----------------------------- shell command analysis -----------------------------

def shell_quote(s: str) -> str:
    # Double-quote form works in POSIX shells, PowerShell, and cmd for simple paths/args.
    return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'


def b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def unb64(s: str) -> str:
    return base64.b64decode(s.encode("ascii")).decode("utf-8", errors="replace")


def ps_quote(s: str) -> str:
    # Escape backtick first; the later replacements insert backticks of their own.
    return '"' + s.replace('`', '``').replace('$', '`$').replace('"', '`"') + '"'


def one_line_preview(command: str, limit: int = 160) -> str:
    flat = " ".join(command.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def self_command(mode: str, payload: str, *, extra_args: str = "", ps: bool = False, original: str = "") -> str:
    """Build the rewritten command. The original command stays visible in a
    trailing comment so permission prompts are not an opaque base64 blob."""
    exe = sys.executable or "python"
    script = str(Path(__file__).resolve())
    if ps:
        core = f"& {ps_quote(exe)} {ps_quote(script)} {mode}{extra_args} --b64 {ps_quote(payload)}"
        return f"{core} # {WRAP_MARKER} original: {one_line_preview(original)}"
    core = f"{WRAP_MARKER} {shell_quote(exe)} {shell_quote(script)} {mode}{extra_args} --b64 {shell_quote(payload)}"
    return f"{core} # original: {one_line_preview(original)}"


def command_is_bounded_read(command: str) -> bool:
    c = command.lower()
    bounded_terms = [
        "| head", "| tail", "select-object -first", "select-object -last",
        "-totalcount", "-tail", "--max-count", "-n ", "--line-number",
    ]
    return any(t in c for t in bounded_terms)


def extract_candidate_read_paths(command: str, cwd: Path) -> list[Path]:
    """Best-effort detection of direct file reads likely to dump a whole file."""
    if command_is_bounded_read(command):
        return []

    candidates: list[str] = []

    # POSIX-ish: cat/type file
    try:
        parts = shlex_split(command)
        if parts and parts[0].lower() in {"cat", "type"}:
            for token in parts[1:]:
                if token.startswith("-"):
                    continue
                candidates.append(token)
    except Exception:
        pass

    # PowerShell-ish: Get-Content path / gc path
    m = re.search(r"(?i)\b(?:Get-Content|gc)\b\s+([^|;&]+)", command)
    if m:
        raw = m.group(1).strip()
        # remove common flags and keep likely path-ish tokens
        raw = re.sub(r"(?i)\s+-(Raw|Encoding|ReadCount|Wait)\b(?:\s+\S+)?", " ", raw)
        for token in re.findall(r"\"([^\"]+)\"|'([^']+)'|(\S+)", raw):
            s = next((x for x in token if x), "")
            if s and not s.startswith("-"):
                candidates.append(s)
                break

    paths: list[Path] = []
    for c in candidates:
        p = resolve_path(c, cwd)
        if p.exists() and p.is_file():
            paths.append(p)
    return paths


def shlex_split(command: str) -> list[str]:
    import shlex
    return shlex.split(command, posix=(os.name != "nt"))


def is_powershell_command(command: str) -> bool:
    return bool(POWERSHELL_START_RE.search(command) or POWERSHELL_CMDLET_RE.search(command))


def _first_shell_token(text: str) -> str:
    try:
        parts = shlex_split(text)
    except Exception:
        parts = []
    if parts:
        return parts[0]
    return text.strip().split(maxsplit=1)[0] if text.strip() else ""


def _decode_base64_utf16le(text: str) -> str | None:
    token = _first_shell_token(text)
    if not token:
        return None
    try:
        return base64.b64decode(token.encode("ascii"), validate=True).decode("utf-16le")
    except Exception:
        return None


def _decode_powershell_command_arg(text: str) -> str:
    """Decode the common shell quoting layer around powershell -Command.

    This is not a full parser for every shell. It handles the cases that were
    breaking the hook: outer single/double quotes, backslash-escaped quotes,
    PowerShell backtick-escaped quotes, and cmd-style doubled quotes.
    """
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'"}:
        quote = s[0]
        s = s[1:-1]
        if quote == '"':
            s = s.replace('\\"', '"')
            s = s.replace('`"', '"')
            s = re.sub(r'""([^"\r\n]+)""', r'"\1"', s)
        else:
            s = s.replace("'\\''", "'")
            s = s.replace("''", "'")
        return s

    try:
        parts = shlex_split(s)
    except Exception:
        return s
    return " ".join(parts) if parts else s


def extract_powershell_inner(command: str) -> tuple[str, str] | None:
    """Return (exe, inner_script) for PowerShell snippets this hook can wrap.

    Explicit powershell/pwsh invocations are only rewritten when they use
    -Command/-c or -EncodedCommand. -File invocations are left alone because
    translating a script-file call into an inline script changes argument
    binding semantics.
    """
    m = POWERSHELL_START_RE.search(command)
    if not m:
        return ("pwsh", command)

    exe = m.group(1)
    rest = command[m.end():].strip()

    m_enc = re.search(r"(?is)(?:^|\s)(?:-|/)(?:EncodedCommand|enc|ec|e)\s+(.+)$", rest)
    if m_enc:
        decoded = _decode_base64_utf16le(m_enc.group(1))
        return (exe, decoded) if decoded is not None else None

    m_cmd = re.search(r"(?is)(?:^|\s)(?:-|/)(?:Command|c)\s+(.+)$", rest)
    if m_cmd:
        inner = _decode_powershell_command_arg(m_cmd.group(1))
        return (exe, inner) if inner else None

    return None


def likely_noisy(command: str) -> bool:
    if SKIP_CAP_MARKER in command:
        return False
    if WRAP_MARKER in command:
        return False
    if command_is_bounded_read(command):
        return False
    if NOISY_COMMAND_RE.search(command):
        return True
    return False

# ----------------------------- hook modes -----------------------------

def mode_pre_read() -> int:
    event = read_event()
    cwd = cwd_from_event(event)
    root = project_root(cwd)
    ti = event.get("tool_input") or {}
    if not isinstance(ti, dict):
        return 0

    # Respect bounded/ranged reads.
    bounded_keys = {"offset", "limit", "start_line", "end_line", "line_start", "line_end", "head", "tail"}
    if any(k in ti and ti[k] not in (None, "") for k in bounded_keys):
        return 0

    path_keys = ["file_path", "filepath", "path", "filename", "file", "absolute_path"]
    for key in path_keys:
        value = ti.get(key)
        if isinstance(value, str) and value.strip():
            path = resolve_path(value, cwd)
            why = file_policy_violation(path, root)
            if why:
                deny_pre_tool(event, f"{bounded_read_hint(path)} Reason: {why}.")
                return 0
    return 0


def mode_pre_bash() -> int:
    event = read_event()
    cwd = cwd_from_event(event)
    root = project_root(cwd)
    ti = event.get("tool_input") or {}
    if not isinstance(ti, dict):
        return 0
    command = str(ti.get("command") or "")
    if not command or WRAP_MARKER in command:
        return 0

    # 0. Repo safety guards, before any rewrite can obscure the command.
    for pattern, why in SAFETY_DENY:
        if pattern.search(command):
            deny_pre_tool(event, why)
            return 0
    for pattern, why in SAFETY_ASK:
        if pattern.search(command):
            ask_pre_tool(event, why)
            return 0

    # 1. Block direct large/generated full-file reads.
    for path in extract_candidate_read_paths(command, cwd):
        why = file_policy_violation(path, root)
        if why:
            deny_pre_tool(event, f"{bounded_read_hint(path)} Reason: {why}.")
            return 0

    # Claude Code's dedicated PowerShell tool passes a native PS script; it only
    # needs the output cap, not the PS-in-Bash rewrite (and the rewritten command
    # must itself be valid PowerShell).
    tool_name = str(event.get("tool_name") or "")
    if tool_name.lower() == "powershell":
        if likely_noisy(command):
            payload = json.dumps({"exe": "powershell", "script": command}, ensure_ascii=False)
            updated = dict(ti)
            updated["command"] = self_command("run-ps-capped", b64(payload), ps=True, original=command)
            allow_updated_input(event, updated, "Wrapped likely-high-output PowerShell command; full raw output will be saved under .agent/logs/.")
        return 0

    # 2. Rewrite raw PowerShell-in-Bash through a stable formatting/output wrapper.
    if is_powershell_command(command):
        extracted = extract_powershell_inner(command)
        if extracted is not None:
            exe, inner = extracted
            payload = json.dumps({"exe": exe, "script": inner}, ensure_ascii=False)
            updated = dict(ti)
            updated["command"] = self_command("run-ps-capped", b64(payload), original=command)
            allow_updated_input(event, updated, "Rewrote PowerShell command through UTF-8/plain-output/capped wrapper.")
            return 0

    # 3. Wrap likely-high-output commands before they can dump huge logs into context.
    # --shell bash re-runs the command under bash so POSIX syntax survives on
    # Windows (shell=True there would mean cmd.exe).
    if likely_noisy(command):
        updated = dict(ti)
        updated["command"] = self_command("run-capped", b64(command), extra_args=" --shell bash", original=command)
        allow_updated_input(event, updated, "Wrapped likely-high-output command; full raw output will be saved under .agent/logs/.")
        return 0

    return 0


def mode_post_output(agent: str = "auto") -> int:
    event = read_event()
    resp = event.get("tool_response")
    text = tool_response_to_text(resp)
    if len(text) <= POST_TOOL_MAX_CHARS:
        return 0

    cwd = cwd_from_event(event)
    log_path = save_raw_log(cwd, "post_tool_output", "<post tool output>", text, "")
    summary = summarize_output(text, "", log_path=log_path, returncode=None)

    # Codex includes model/turn_id fields. Claude requires updatedToolOutput with the original schema.
    # Prefer the explicit --agent flag from the hook config; sniff only as fallback.
    if agent == "auto":
        is_codex = "model" in event or "turn_id" in event
    else:
        is_codex = agent == "codex"
    if is_codex:
        write_json({
            "decision": "block",
            "reason": "Tool output exceeded policy cap; replacing with bounded summary.",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": summary,
            },
        })
        return 0

    if isinstance(resp, dict):
        new_resp = dict(resp)
        if "stdout" in new_resp or "stderr" in new_resp:
            new_resp["stdout"] = summary
            new_resp["stderr"] = ""
        elif "content" in new_resp:
            new_resp["content"] = summary
        else:
            # Unknown shape: do not risk schema mismatch; just add context.
            add_context(event, summary)
            return 0
        write_json({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"Large output was capped. Raw log: {log_path}",
                "updatedToolOutput": new_resp,
            }
        })
    elif isinstance(resp, str):
        write_json({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": f"Large output was capped. Raw log: {log_path}",
                "updatedToolOutput": summary,
            }
        })
    else:
        add_context(event, summary)
    return 0


DIFF_HUNK_RE = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@"
)


def repo_relative_path(path: Path, root: Path) -> str | None:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def whole_file_line_ranges(path: Path) -> list[tuple[int, int]]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            line_count = sum(1 for _ in f)
    except OSError:
        return []

    if line_count == 0:
        return []

    return [(1, line_count)]


def merge_line_ranges(
    ranges: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []

    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))

    return merged


def git_changed_line_ranges(
    root: Path,
    path: Path,
) -> list[tuple[int, int]] | None:
    """Return current-file line ranges changed relative to HEAD.

    Returns:
        A list of inclusive line ranges.
        An empty list when the file has no changes.
        None when Git could not determine the ranges.
    """
    rel = repo_relative_path(path, root)
    if rel is None:
        return None

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=str(root),
        text=True,
        capture_output=True,
        errors="replace",
    )

    # New/untracked files have no HEAD version, so all their lines are new.
    if tracked.returncode != 0:
        return whole_file_line_ranges(path)

    diff = subprocess.run(
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--no-color",
            "--unified=0",
            "HEAD",
            "--",
            rel,
        ],
        cwd=str(root),
        text=True,
        capture_output=True,
        errors="replace",
    )

    if diff.returncode != 0:
        return None

    ranges: list[tuple[int, int]] = []

    for line in diff.stdout.splitlines():
        match = DIFF_HUNK_RE.match(line)
        if match is None:
            continue

        start = int(match.group(1))
        count = int(match.group(2) or "1")

        # A deletion-only hunk has no corresponding lines in the current file.
        if count > 0:
            ranges.append((start, start + count - 1))

    return merge_line_ranges(ranges)


def json_location_span(
    value: dict[str, Any],
) -> tuple[int, int] | None:
    location = value.get("location")
    end_location = value.get("end_location")

    if not isinstance(location, dict):
        return None

    start_row = location.get("row")
    if not isinstance(start_row, int) or start_row < 1:
        return None

    end_row = start_row
    end_column = None

    if isinstance(end_location, dict):
        candidate_row = end_location.get("row")
        if isinstance(candidate_row, int) and candidate_row >= start_row:
            end_row = candidate_row

        candidate_column = end_location.get("column")
        if isinstance(candidate_column, int):
            end_column = candidate_column

    # Ruff locations are end-exclusive. A range ending at column 1 of the
    # following row does not actually cover that following row.
    if end_row > start_row and end_column == 1:
        end_row -= 1

    return start_row, max(start_row, end_row)


def spans_overlap(
    left: tuple[int, int],
    right: tuple[int, int],
) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def diagnostic_touches_changed_lines(
    diagnostic: dict[str, Any],
    changed_ranges: list[tuple[int, int]],
) -> bool:
    spans: list[tuple[int, int]] = []

    primary_span = json_location_span(diagnostic)
    if primary_span is not None:
        spans.append(primary_span)

    # Import sorting and some structural diagnostics are located at the start
    # of a block. Their fix range often covers the actual edited lines.
    fix = diagnostic.get("fix")
    if isinstance(fix, dict):
        edits = fix.get("edits")
        if isinstance(edits, list):
            for edit in edits:
                if isinstance(edit, dict):
                    edit_span = json_location_span(edit)
                    if edit_span is not None:
                        spans.append(edit_span)

    return any(
        spans_overlap(span, changed)
        for span in spans
        for changed in changed_ranges
    )


def format_ruff_diagnostic(
    rel: str,
    diagnostic: dict[str, Any],
) -> str:
    location = diagnostic.get("location")
    if not isinstance(location, dict):
        location = {}

    row = location.get("row", 1)
    column = location.get("column", 1)
    code = diagnostic.get("code") or "unknown"
    message = diagnostic.get("message") or "Ruff diagnostic"

    return f"{rel}:{row}:{column}: {code} {message}"


def mode_format_after_edit() -> int:
    """Report Ruff findings that intersect lines changed relative to HEAD."""
    event = read_event()
    cwd = cwd_from_event(event)
    root = project_root(cwd)

    files = sorted({
        path
        for path in extract_changed_files(event, cwd)
        if path.exists() and path.is_file()
    })
    py_files = [path for path in files if path.suffix.lower() == ".py"]

    if not py_files or len(py_files) > 20:
        return 0

    exe_dir = Path(sys.executable).parent
    ruff = next(
        (
            str(candidate)
            for candidate in (exe_dir / "ruff.exe", exe_dir / "ruff")
            if candidate.exists()
        ),
        None,
    ) or shutil.which("ruff")

    if not ruff:
        return 0

    findings: list[str] = []
    failures: list[str] = []

    for path in py_files:
        rel = repo_relative_path(path, root)
        if rel is None:
            continue

        changed_ranges = git_changed_line_ranges(root, path)
        if changed_ranges is None:
            failures.append(f"{rel}: could not determine changed-line ranges")
            continue

        if not changed_ranges:
            continue

        proc = subprocess.run(
            [
                ruff,
                "check",
                "--output-format",
                "json",
                rel,
            ],
            cwd=str(root),
            text=True,
            capture_output=True,
            errors="replace",
        )

        # Ruff uses 1 for normal lint findings and 2 for execution/config errors.
        if proc.returncode not in (0, 1):
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            suffix = f": {detail[0]}" if detail else ""
            failures.append(f"{rel}: Ruff failed{suffix}")
            continue

        try:
            diagnostics = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError as exc:
            failures.append(f"{rel}: invalid Ruff JSON: {exc}")
            continue

        if not isinstance(diagnostics, list):
            failures.append(f"{rel}: unexpected Ruff JSON result")
            continue

        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue

            if diagnostic_touches_changed_lines(
                diagnostic,
                changed_ranges,
            ):
                findings.append(format_ruff_diagnostic(rel, diagnostic))

    output: list[str] = []

    if findings:
        shown = findings[:40]
        output.append(
            "ruff findings on changed lines:\n" + "\n".join(shown)
        )

        if len(findings) > 40:
            output.append(f"... {len(findings) - 40} more findings")

    if failures:
        output.append(
            "ruff hook errors:\n" + "\n".join(failures[:10])
        )

    if output:
        add_context(event, "\n".join(output))

    return 0

# ----------------------------- runner modes -----------------------------

def mode_run_capped(args: argparse.Namespace) -> int:
    command = unb64(args.b64)
    cwd = Path.cwd()
    bash_exe = shutil.which("bash") if getattr(args, "shell", "system") == "bash" else None
    if bash_exe:
        proc = subprocess.run([bash_exe, "-c", command], cwd=str(cwd), text=True, capture_output=True, errors="replace")
    else:
        proc = subprocess.run(command, shell=True, cwd=str(cwd), text=True, capture_output=True, errors="replace")
    log_path = save_raw_log(cwd, "command", command, proc.stdout, proc.stderr)
    print(summarize_output(proc.stdout, proc.stderr, log_path=log_path, returncode=proc.returncode))
    return proc.returncode


def mode_run_ps_capped(args: argparse.Namespace) -> int:
    payload = json.loads(unb64(args.b64))
    requested_exe = str(payload.get("exe") or "pwsh")
    script = str(payload.get("script") or "")
    cwd = Path.cwd()

    exe = resolve_powershell_exe(requested_exe)
    if not exe:
        # Fallback: run through shell, still capped, if PowerShell executable is not found.
        proc = subprocess.run(script, shell=True, cwd=str(cwd), text=True, capture_output=True, errors="replace")
        log_path = save_raw_log(cwd, "powershell_fallback", script, proc.stdout, proc.stderr)
        print(summarize_output(proc.stdout, proc.stderr, log_path=log_path, returncode=proc.returncode))
        return proc.returncode

    wrapped_script = build_powershell_wrapper(script)
    cmd = [exe, "-NoLogo", "-NoProfile", "-NonInteractive"]
    if os.name == "nt" and "powershell" in Path(exe).name.lower():
        cmd += ["-ExecutionPolicy", "Bypass"]

    encoded = base64.b64encode(wrapped_script.encode("utf-16le")).decode("ascii")
    encoded_cmd = [*cmd, "-EncodedCommand", encoded]

    # CreateProcess has a finite command-line length. For very large scripts,
    # fall back to a temp .ps1. That file is written with a BOM when needed,
    # so Windows PowerShell 5.1 does not read it as ANSI.
    if os.name == "nt" and sum(len(x) + 3 for x in encoded_cmd) > 30000:
        ps1 = build_temp_ps1(cwd, wrapped_script, exe=exe)
        run_cmd = [*cmd, "-File", str(ps1)]
    else:
        ps1 = None
        run_cmd = encoded_cmd

    proc = subprocess.run(
        run_cmd,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if ps1 is not None:
        try:
            ps1.unlink(missing_ok=True)
        except Exception:
            pass
    log_path = save_raw_log(cwd, "powershell", script, proc.stdout, proc.stderr)
    print(summarize_output(proc.stdout, proc.stderr, log_path=log_path, returncode=proc.returncode))
    return proc.returncode

# ----------------------------- output helpers -----------------------------

def tool_response_to_text(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        parts = []
        for key in ("stdout", "stderr", "content", "text", "output"):
            val = resp.get(key)
            if isinstance(val, str):
                parts.append(val)
        if parts:
            return "\n".join(parts)
        return json.dumps(resp, ensure_ascii=False, indent=2)
    return str(resp)


def save_raw_log(cwd: Path, kind: str, command: str, stdout: str, stderr: str) -> Path:
    root = project_root(cwd)
    log_dir = root / ".agent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gitignore = log_dir.parent / ".gitignore"
    if not gitignore.exists():
        try:
            gitignore.write_text("*\n", encoding="ascii")
        except OSError:
            pass
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = log_dir / f"{ts}-{kind}.log"
    with path.open("w", encoding="utf-8", errors="replace", newline="\n") as f:
        f.write(f"# kind: {kind}\n# cwd: {cwd}\n# command:\n{command}\n\n")
        f.write("# stdout\n")
        f.write(stdout or "")
        f.write("\n\n# stderr\n")
        f.write(stderr or "")
    try:
        logs = sorted(log_dir.glob("*.log"))
        for stale in logs[:-200]:
            stale.unlink()
    except OSError:
        pass
    return path


def first_last_lines(text: str, head: int, tail: int) -> tuple[list[str], list[str], int]:
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return lines, [], len(lines)
    return lines[:head], lines[-tail:], len(lines)


def summarize_output(stdout: str, stderr: str, *, log_path: Path, returncode: int | None) -> str:
    combined = ""
    if stdout:
        combined += stdout
    if stderr:
        combined += ("\n" if combined else "") + "[stderr]\n" + stderr

    head, tail, total = first_last_lines(combined, VISIBLE_HEAD_LINES, VISIBLE_TAIL_LINES)
    error_lines = [ln for ln in combined.splitlines() if ERROR_LINE_RE.search(ln)]
    # Keep unique-ish first 80 error lines.
    seen = set()
    compact_errors = []
    for ln in error_lines:
        key = ln.strip()[:300]
        if key and key not in seen:
            seen.add(key)
            compact_errors.append(ln)
        if len(compact_errors) >= 80:
            break

    out: list[str] = []
    out.append("[agent hook] Command output capped/summarized.")
    if returncode is not None:
        out.append(f"[agent hook] Exit code: {returncode}")
    out.append(f"[agent hook] Raw log: {log_path}")
    out.append(f"[agent hook] Total lines: {total}")
    if compact_errors:
        out.append("\n[agent hook] Detected error/failure lines:")
        out.extend(compact_errors)
    out.append("\n[agent hook] Head:")
    out.extend(head)
    if tail:
        out.append("\n[agent hook] ... output truncated ...")
        out.append("\n[agent hook] Tail:")
        out.extend(tail)
    return "\n".join(out)

# ----------------------------- PowerShell helpers -----------------------------

def resolve_powershell_exe(requested: str) -> str | None:
    names = []
    req_name = Path(requested).name.lower()
    if "pwsh" in req_name:
        names = [requested, "pwsh", "pwsh.exe", "powershell", "powershell.exe"]
    elif "powershell" in req_name:
        names = [requested, "powershell", "powershell.exe", "pwsh", "pwsh.exe"]
    else:
        names = ["pwsh", "pwsh.exe", "powershell", "powershell.exe"]
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def build_powershell_wrapper(script: str) -> str:
    prelude = "$ErrorActionPreference = 'Stop'\n$ProgressPreference = 'SilentlyContinue'\n$InformationPreference = 'Continue'\n$WarningPreference = 'Continue'\ntry {\n    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)\n    $OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n    if ($PSVersionTable.PSVersion.Major -ge 7) { $PSStyle.OutputRendering = 'PlainText' }\n} catch {}\ntry {\n    & {\n"
    postlude = '} *>&1 | Out-String -Width 4096\n    if ($global:LASTEXITCODE -is [int] -and $global:LASTEXITCODE -ne 0) { exit $global:LASTEXITCODE }\n} catch {\n    [Console]::Error.WriteLine(($_ | Out-String -Width 4096))\n    exit 1\n}\n'
    return prelude + script + "\n" + postlude


def _is_windows_powershell(exe: str) -> bool:
    name = Path(exe).name.lower()
    return name in {"powershell", "powershell.exe"}


def build_temp_ps1(cwd: Path, script: str, *, exe: str | None = None) -> Path:
    tmp_dir = project_root(cwd) / ".agent" / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path_s = tempfile.mkstemp(prefix="hook-", suffix=".ps1", dir=str(tmp_dir), text=False)
    path = Path(path_s)

    # Windows PowerShell 5.1 reads BOM-less script source as the active ANSI
    # code page. UTF-8 with BOM is the least disruptive Unicode file fallback.
    encoding = "utf-8-sig" if exe and _is_windows_powershell(exe) else "utf-8"
    with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
        f.write(script)
    return path

# ----------------------------- edited-file helpers -----------------------------

def extract_changed_files(event: dict[str, Any], cwd: Path) -> list[Path]:
    ti = event.get("tool_input") or {}
    if not isinstance(ti, dict):
        return []
    out: list[Path] = []

    for key in ("file_path", "filepath", "path"):
        val = ti.get(key)
        if isinstance(val, str) and val.strip():
            out.append(resolve_path(val, cwd))

    cmd = ti.get("command")
    if isinstance(cmd, str):
        # Codex apply_patch shape.
        for m in re.finditer(r"^\*\*\*\s+(?:Update|Add)\s+File:\s+(.+)$", cmd, re.M):
            out.append(resolve_path(m.group(1).strip(), cwd))
        for m in re.finditer(r"^\+\+\+\s+b/(.+)$", cmd, re.M):
            p = m.group(1).strip()
            if p != "/dev/null":
                out.append(resolve_path(p, cwd))

    # Preserve only text-like paths.
    dedup = []
    seen = set()
    for p in out:
        key = str(p).lower() if os.name == "nt" else str(p)
        if key not in seen and is_text_like(p):
            seen.add(key)
            dedup.append(p)
    return dedup


# ----------------------------- CLI -----------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("pre-read")
    sub.add_parser("pre-bash")
    sub.add_parser("pre-edit")
    p = sub.add_parser("post-output")
    p.add_argument("--agent", choices=["auto", "claude", "codex"], default="auto")
    sub.add_parser("format-after-edit")
    p = sub.add_parser("run-capped")
    p.add_argument("--b64", required=True)
    p.add_argument("--shell", choices=["system", "bash"], default="system")
    p = sub.add_parser("run-ps-capped")
    p.add_argument("--b64", required=True)
    args = parser.parse_args(argv)

    if args.mode == "pre-read":
        return mode_pre_read()
    if args.mode == "pre-bash":
        return mode_pre_bash()
    if args.mode == "pre-edit":
        return mode_pre_edit()
    if args.mode == "post-output":
        return mode_post_output(args.agent)
    if args.mode == "format-after-edit":
        return mode_format_after_edit()
    if args.mode == "run-capped":
        return mode_run_capped(args)
    if args.mode == "run-ps-capped":
        return mode_run_ps_capped(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
