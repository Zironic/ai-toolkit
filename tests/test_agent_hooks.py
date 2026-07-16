from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).parent.parent
HOOK_PATH = REPO_ROOT / ".agent-hooks" / "agent_hooks.py"
SPEC = importlib.util.spec_from_file_location("agent_hooks_under_test", HOOK_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load hook module from {HOOK_PATH}")
HOOKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOOKS)


def safety_ask_reason(command: str) -> str | None:
    for pattern, reason in HOOKS.SAFETY_ASK:
        if pattern.search(command):
            return reason
    return None


def run_pre_bash(command: str, cwd: Path = REPO_ROOT) -> dict[str, object] | None:
    event = {
        "cwd": str(cwd),
        "hook_event_name": "PreToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": command},
    }
    with (
        mock.patch.object(HOOKS, "read_event", return_value=event),
        mock.patch.object(HOOKS, "project_root", return_value=cwd),
        mock.patch.object(HOOKS, "write_json") as write_json,
    ):
        HOOKS.mode_pre_bash()
    if not write_json.called:
        return None
    return write_json.call_args.args[0]


class AsciiOnlyStream:
    encoding = "ascii"

    def __init__(self) -> None:
        self.parts: list[str] = []

    def write(self, value: str) -> int:
        value.encode(self.encoding)
        self.parts.append(value)
        return len(value)

    def text(self) -> str:
        return "".join(self.parts)


class SharedJunctionSafetyTests(unittest.TestCase):
    def test_recursive_delete_commands_require_confirmation(self) -> None:
        commands = (
            "rm datasets\\training-image.png",
            "Remove-Item datasets\\training-image.png",
            "cmd.exe /c del /q datasets\\training-image.png",
            "rm -rf datasets",
            "rm datasets -Recurse -Force",
            "ri -r datasets",
            "rd datasets -Recurse",
            "Remove-Item -LiteralPath datasets -Recurse -Force",
            "rmdir /s /q datasets",
            "cmd.exe /c rd /s /q datasets",
            "cmd.exe /c del /s /q datasets\\*",
            "rm -rf output",
            "Remove-Item -LiteralPath output -Recurse -Force",
            "rmdir /s /q output",
        )

        for command in commands:
            with self.subTest(command=command):
                reason = safety_ask_reason(command)
                self.assertIsNotNone(reason)
                self.assertIn("user data", reason)

    def test_training_launch_variants_require_confirmation(self) -> None:
        commands = (
            "python run.py config.yaml",
            "python -u run.py config.yaml",
            "py -3 run.py config.yaml",
            ".\\venv\\Scripts\\python.exe -u run.py config.yaml",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(safety_ask_reason(command))

    def test_direct_smoke_script_requires_confirmation(self) -> None:
        smoke_pattern, reason = HOOKS.SMOKE_ASK

        self.assertTrue(
            smoke_pattern.search(
                "venv\\Scripts\\python.exe scripts\\smoke_krea2_train_cuda.py"
            )
        )
        self.assertIn("smoke script", reason)

    def test_wrapper_marker_does_not_bypass_safety_rules(self) -> None:
        result = run_pre_bash(f"{HOOKS.WRAP_MARKER} rm -rf datasets")

        self.assertIsNotNone(result)
        decision = result["hookSpecificOutput"]
        self.assertEqual("ask", decision["permissionDecision"])

    def test_non_destructive_access_does_not_require_confirmation(self) -> None:
        self.assertIsNone(safety_ask_reason("Get-ChildItem -LiteralPath datasets"))
        self.assertIsNone(safety_ask_reason("Get-ChildItem -LiteralPath output"))
        self.assertIsNone(safety_ask_reason("rm output\\one-disposable-log.txt"))

    def test_lock_taking_ticket_reads_are_denied(self) -> None:
        commands = (
            ".\\tools\\git-bug.exe bug show c4e29f1",
            ".\\tools\\git-bug.exe bug --format plain",
            ".\\scripts\\tickets.cmd list",
            ".\\scripts\\tickets.cmd list-closed",
            ".\\scripts\\tickets.cmd show c4e29f1",
        )

        for command in commands:
            with self.subTest(command=command):
                result = run_pre_bash(command)
                self.assertIsNotNone(result)
                output = result["hookSpecificOutput"]
                self.assertEqual("deny", output["permissionDecision"])
                self.assertIn("single-access lock", output["permissionDecisionReason"])

    def test_ticket_writes_are_not_denied(self) -> None:
        commands = (
            ".\\tools\\git-bug.exe bug comment new c4e29f1 --message update",
            ".\\tools\\git-bug.exe bug status close c4e29f1",
            ".\\scripts\\tickets.cmd comment c4e29f1 update",
            ".\\scripts\\tickets.cmd close c4e29f1",
        )

        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(run_pre_bash(command))


class GpuSmokeToggleTests(unittest.TestCase):
    SMOKE_COMMAND = "venv\\Scripts\\python.exe scripts\\smoke_krea2_train_cuda.py"
    START_JOB_COMMAND = (
        "node -e \"require('./dist/cron/actions/startJob.js').default("
        "'31d4713e-46bd-4623-acec-8c21a60f160a').then(()=>console.log('STARTED'))\""
    )
    DB_UPDATE_COMMAND = (
        "venv\\Scripts\\python.exe -c \"import sqlite3; "
        "c=sqlite3.connect(r'aitk_db.db'); "
        "c.execute('update Job set job_config=? where id=?')\""
    )
    PROCESS_RESTART_COMMAND = (
        "$p = Get-Process -Id 36704 -ErrorAction SilentlyContinue; "
        "if ($p) { Stop-Process -Id 36704 -Force }"
    )

    @staticmethod
    def _decision(result: dict[str, object] | None) -> str | None:
        if result is None:
            return None
        output = result.get("hookSpecificOutput")
        if not isinstance(output, dict):
            return None
        return output.get("permissionDecision")

    def _write_toggle(
        self,
        root: Path,
        content: str | None,
        relpath: Path | None = None,
    ) -> None:
        toggle = root / (relpath or HOOKS.GPU_SMOKE_TOGGLE_RELPATHS[0])
        toggle.parent.mkdir(parents=True, exist_ok=True)
        toggle.write_text("" if content is None else content, encoding="ascii")

    def test_smoke_asks_when_toggle_file_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_pre_bash(self.SMOKE_COMMAND, cwd=Path(tmp))

        self.assertEqual("ask", self._decision(result))

    def test_empty_toggle_file_allows_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, None)

            result = run_pre_bash(self.SMOKE_COMMAND, cwd=root)

        self.assertEqual("allow", self._decision(result))

    def test_empty_markdown_toggle_file_allows_smokes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, None, Path(".agent/allow-gpu-smokes.md"))

            result = run_pre_bash(self.SMOKE_COMMAND, cwd=root)

        self.assertEqual("allow", self._decision(result))

    def test_toggle_file_saying_off_keeps_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, "# temporarily disabled\noff\n")

            result = run_pre_bash(self.SMOKE_COMMAND, cwd=root)

        self.assertEqual("ask", self._decision(result))

    def test_toggle_does_not_unlock_full_training_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, "on")

            result = run_pre_bash("python run.py config.yaml", cwd=root)

        self.assertEqual("ask", self._decision(result))

    def test_job_control_commands_ask_when_toggle_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for command in (
                self.START_JOB_COMMAND,
                self.DB_UPDATE_COMMAND,
                self.PROCESS_RESTART_COMMAND,
                self.DB_UPDATE_COMMAND + "; " + self.PROCESS_RESTART_COMMAND,
            ):
                with self.subTest(command=command):
                    result = run_pre_bash(command, cwd=root)
                    self.assertEqual("ask", self._decision(result))

    def test_job_control_commands_are_allowed_when_toggle_is_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, "on")

            for command in (
                self.START_JOB_COMMAND,
                self.DB_UPDATE_COMMAND,
                self.PROCESS_RESTART_COMMAND,
                self.DB_UPDATE_COMMAND + "; " + self.PROCESS_RESTART_COMMAND,
            ):
                with self.subTest(command=command):
                    result = run_pre_bash(command, cwd=root)
                    self.assertEqual("allow", self._decision(result))

    def test_gpu_grant_does_not_override_dataset_deletion_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_toggle(root, "on")

            result = run_pre_bash(
                self.START_JOB_COMMAND + "; Remove-Item datasets -Recurse -Force",
                cwd=root,
            )

        self.assertEqual("ask", self._decision(result))


class ReadPolicyTests(unittest.TestCase):
    def test_skip_then_first_is_a_bounded_read(self) -> None:
        command = (
            "Get-Content -LiteralPath file.txt | "
            "Select-Object -Skip 100 -First 20"
        )

        self.assertTrue(HOOKS.command_is_bounded_read(command))

    def test_line_number_flags_are_not_mistaken_for_limits(self) -> None:
        self.assertFalse(HOOKS.command_is_bounded_read("cat -n file.txt"))
        self.assertFalse(
            HOOKS.command_is_bounded_read("rg --line-number pattern file.txt")
        )

    def test_get_content_raw_keeps_the_following_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "large.txt"
            path.write_text("payload", encoding="ascii")
            command = f"Get-Content -Raw -LiteralPath '{path}'"

            candidates = HOOKS.extract_candidate_read_paths(command, REPO_ROOT)

        self.assertEqual([path.resolve()], candidates)

    def test_git_diff_check_is_not_wrapped_as_high_output(self) -> None:
        self.assertFalse(HOOKS.likely_noisy("git diff --check -- example.py"))
        self.assertTrue(HOOKS.likely_noisy("git diff -- example.py"))


class CommandTimeoutTests(unittest.TestCase):
    def test_test_commands_get_the_longer_timeout(self) -> None:
        self.assertEqual(
            HOOKS.TEST_TIMEOUT_SECONDS,
            HOOKS.command_timeout_seconds("python -m pytest tests -q"),
        )
        self.assertEqual(
            HOOKS.COMMAND_TIMEOUT_SECONDS,
            HOOKS.command_timeout_seconds("git diff -- README.md"),
        )

    def test_timeout_is_returned_as_exit_124_with_partial_output(self) -> None:
        expired = subprocess.TimeoutExpired(
            cmd=["slow-command"],
            timeout=1,
            output="partial output\n",
            stderr="partial error\n",
        )
        process = mock.Mock()
        process.pid = 1234
        process.returncode = -1
        process.communicate.side_effect = [
            expired,
            ("partial output\n", "partial error\n"),
        ]
        with (
            mock.patch.object(HOOKS.subprocess, "Popen", return_value=process),
            mock.patch.object(HOOKS, "_terminate_process_tree") as terminate,
        ):
            result = HOOKS.run_bounded_subprocess(
                ["slow-command"],
                timeout_seconds=1,
                text=True,
                capture_output=True,
            )

        terminate.assert_called_once_with(process)
        self.assertEqual(HOOKS.TIMEOUT_RETURN_CODE, result.returncode)
        self.assertEqual("partial output\n", result.stdout)
        self.assertIn("partial error", result.stderr)
        self.assertIn("timed out after 1 seconds", result.stderr)

    def test_real_child_is_stopped_at_the_timeout(self) -> None:
        started = time.monotonic()
        result = HOOKS.run_bounded_subprocess(
            [HOOKS.sys.executable, "-c", "import time; time.sleep(30)"],
            timeout_seconds=0.2,
            text=True,
            capture_output=True,
        )

        self.assertEqual(HOOKS.TIMEOUT_RETURN_CODE, result.returncode)
        self.assertLess(time.monotonic() - started, 5.0)


class UnicodeOutputTests(unittest.TestCase):
    def test_emit_output_escapes_text_unsupported_by_console_encoding(self) -> None:
        stream = AsciiOnlyStream()
        arrow = chr(0x2192)

        with mock.patch.object(HOOKS.sys, "stdout", stream):
            HOOKS.emit_output(f"before {arrow} after")

        self.assertEqual("before \\u2192 after\n", stream.text())

    def test_json_protocol_is_ascii_safe(self) -> None:
        stream = AsciiOnlyStream()
        arrow = chr(0x2192)

        with mock.patch.object(HOOKS.sys, "stdout", stream):
            HOOKS.write_json({"message": arrow})

        self.assertEqual('{"message":"\\u2192"}\n', stream.text())


class EditSanitizingTests(unittest.TestCase):
    def test_apply_patch_only_sanitizes_added_source_lines(self) -> None:
        arrow = chr(0x2192)
        patch = f"""*** Begin Patch
*** Update File: example.py
@@
 context with {arrow} anchor
-old {arrow} anchor
+new {arrow} source
*** Update File: README.md
@@
+keep {arrow} markdown
*** End Patch
"""

        fixed = HOOKS.asciify_apply_patch(patch)

        self.assertIn(f" context with {arrow} anchor", fixed)
        self.assertIn(f"-old {arrow} anchor", fixed)
        self.assertIn("+new -> source", fixed)
        self.assertIn(f"+keep {arrow} markdown", fixed)

    def test_pre_edit_updates_codex_apply_patch_input(self) -> None:
        arrow = chr(0x2192)
        patch = (
            "*** Begin Patch\n*** Add File: example.py\n"
            f"+value = '{arrow}'\n*** End Patch\n"
        )
        event = {
            "hook_event_name": "PreToolUse",
            "tool_input": {"command": patch},
        }
        with (
            mock.patch.object(HOOKS, "read_event", return_value=event),
            mock.patch.object(HOOKS, "write_json") as write_json,
        ):
            HOOKS.mode_pre_edit()

        result = write_json.call_args.args[0]
        updated = result["hookSpecificOutput"]["updatedInput"]
        self.assertIn("+value = '->'", updated["command"])


class OutputPolicyTests(unittest.TestCase):
    def test_short_successful_output_passes_through_unchanged(self) -> None:
        output = "test_error_reporting_is_safe ... ok\n"

        summary = HOOKS.summarize_output(
            output,
            "",
            log_path=REPO_ROOT / "unused.log",
            returncode=0,
        )

        self.assertEqual(output, summary)
        self.assertNotIn("Detected error", summary)

    def test_codex_post_output_does_not_block_a_successful_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event = {
                "cwd": tmp,
                "hook_event_name": "PostToolUse",
                "model": "test-model",
                "tool_response": "x" * (HOOKS.POST_TOOL_MAX_CHARS + 1),
            }
            with (
                mock.patch.object(HOOKS, "read_event", return_value=event),
                mock.patch.object(HOOKS, "write_json") as write_json,
            ):
                HOOKS.mode_post_output("codex")

        result = write_json.call_args.args[0]
        self.assertNotIn("decision", result)
        self.assertIn("systemMessage", result)

    def test_claude_post_output_preserves_the_tool_response_shape(self) -> None:
        response = {
            "stdout": "x" * (HOOKS.POST_TOOL_MAX_CHARS + 1),
            "stderr": "",
            "interrupted": False,
            "isImage": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            event = {
                "cwd": tmp,
                "hook_event_name": "PostToolUse",
                "tool_response": response,
            }
            with (
                mock.patch.object(HOOKS, "read_event", return_value=event),
                mock.patch.object(HOOKS, "write_json") as write_json,
            ):
                HOOKS.mode_post_output("claude")

        result = write_json.call_args.args[0]
        updated = result["hookSpecificOutput"]["updatedToolOutput"]
        self.assertEqual(False, updated["interrupted"])
        self.assertEqual(False, updated["isImage"])
        self.assertLessEqual(
            len(updated["stdout"]),
            HOOKS.POST_HOOK_SUMMARY_MAX_CHARS,
        )


class SkillMirrorTests(unittest.TestCase):
    def test_repository_skill_trees_are_byte_identical(self) -> None:
        self.assertEqual([], HOOKS.skill_tree_differences(REPO_ROOT))

    def test_skill_tree_drift_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / ".agents" / "skills" / "sample" / "SKILL.md"
            right = root / ".claude" / "skills" / "sample" / "SKILL.md"
            left.parent.mkdir(parents=True)
            right.parent.mkdir(parents=True)
            left.write_bytes(b"left")
            right.write_bytes(b"right")

            differences = HOOKS.skill_tree_differences(root)

        self.assertEqual(1, len(differences))
        self.assertIn("differs", differences[0])


class SilentTextNormalizationTests(unittest.TestCase):
    def test_normalizer_removes_blank_lines_and_applies_line_style(self) -> None:
        cases = (
            (b"alpha\nbeta\n\n", b"\r\n", b"alpha\r\nbeta\r\n"),
            (b"alpha\r\nbeta\r\n \t\r\n", b"\n", b"alpha\nbeta\n"),
            (b"alpha\nbeta", b"\r\n", b"alpha\r\nbeta\r\n"),
            (b"alpha  ", b"\n", b"alpha  \n"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            for index, (before, newline, expected) in enumerate(cases):
                with self.subTest(index=index):
                    path = Path(tmp) / f"case-{index}.txt"
                    path.write_bytes(before)

                    self.assertTrue(HOOKS.normalize_text_file(path, newline))
                    self.assertEqual(expected, path.read_bytes())

    def test_normalizer_ignores_binary_and_empty_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "binary.bin"
            empty = Path(tmp) / "empty.txt"
            binary.write_bytes(b"binary\x00payload\n\n")
            empty.write_bytes(b"")

            self.assertFalse(HOOKS.normalize_text_file(binary, b"\n"))
            self.assertFalse(HOOKS.normalize_text_file(empty, b"\n"))
            self.assertEqual(b"binary\x00payload\n\n", binary.read_bytes())
            self.assertEqual(b"", empty.read_bytes())

    def test_worktree_style_follows_core_autocrlf(self) -> None:
        for value, expected in (("true\n", b"\r\n"), ("input\n", b"\n")):
            with self.subTest(value=value.strip()):
                completed = subprocess.CompletedProcess(
                    ["git", "config"],
                    0,
                    stdout=value,
                    stderr="",
                )
                with mock.patch.object(
                    HOOKS,
                    "run_bounded_subprocess",
                    return_value=completed,
                ):
                    self.assertEqual(
                        expected,
                        HOOKS.preferred_worktree_newline(REPO_ROOT),
                    )

    def test_post_edit_cleanup_adds_no_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "notes.md"
            path.write_bytes(b"content\n\n")
            event = {"cwd": str(root), "hook_event_name": "PostToolUse"}

            with (
                mock.patch.object(HOOKS, "read_event", return_value=event),
                mock.patch.object(HOOKS, "project_root", return_value=root),
                mock.patch.object(
                    HOOKS,
                    "extract_changed_files",
                    return_value=[path],
                ),
                mock.patch.object(
                    HOOKS,
                    "preferred_worktree_newline",
                    return_value=b"\n",
                ),
                mock.patch.object(HOOKS, "add_context") as add_context,
            ):
                result = HOOKS.mode_format_after_edit()

            self.assertEqual(0, result)
            self.assertEqual(b"content\n", path.read_bytes())
            add_context.assert_not_called()


class HookConfigTests(unittest.TestCase):
    def test_codex_registers_pre_edit_and_avoids_post_output_blocking(self) -> None:
        config = json.loads((REPO_ROOT / ".codex" / "hooks.json").read_text())
        pre_groups = config["hooks"]["PreToolUse"]
        post_groups = config["hooks"]["PostToolUse"]
        format_hooks = [
            hook
            for group in post_groups
            for hook in group["hooks"]
            if "format-after-edit" in hook.get("command", "")
        ]

        self.assertTrue(
            any("apply_patch" in group.get("matcher", "") for group in pre_groups)
        )
        self.assertFalse(
            any(
                "post-output" in hook.get("command", "")
                for group in post_groups
                for hook in group["hooks"]
            )
        )
        self.assertTrue(format_hooks)
        self.assertTrue(
            all("statusMessage" not in hook for hook in format_hooks)
        )
        self.assertTrue(
            all(
                "venv\\Scripts\\python.exe" in hook["commandWindows"]
                for groups in config["hooks"].values()
                for group in groups
                for hook in group["hooks"]
            )
        )

    def test_claude_registers_edit_checks_and_output_replacement(self) -> None:
        config = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
        pre_groups = config["hooks"]["PreToolUse"]
        post_groups = config["hooks"]["PostToolUse"]
        format_hooks = [
            hook
            for group in post_groups
            for hook in group["hooks"]
            if "format-after-edit" in hook.get("command", "")
        ]

        self.assertTrue(
            any(
                "pre-edit" in hook.get("command", "")
                for group in pre_groups
                for hook in group["hooks"]
            )
        )
        self.assertTrue(format_hooks)
        self.assertTrue(
            all("statusMessage" not in hook for hook in format_hooks)
        )
        self.assertTrue(
            any(
                "post-output --agent claude" in hook.get("command", "")
                for group in post_groups
                for hook in group["hooks"]
            )
        )


@unittest.skipUnless(os.name == "nt", "PowerShell wrapper tests require Windows")
class PowerShellWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.executable = HOOKS.resolve_powershell_exe("powershell")
        if cls.executable is None:
            raise unittest.SkipTest("PowerShell is not installed")

    def run_wrapped(self, body: str, *, constrained: bool = False) -> subprocess.CompletedProcess[str]:
        wrapper = HOOKS.build_powershell_wrapper(body)
        if constrained:
            wrapper = (
                "$ExecutionContext.SessionState.LanguageMode = 'ConstrainedLanguage'\n"
                + wrapper
            )
        encoded = base64.b64encode(wrapper.encode("utf-16le")).decode("ascii")
        return subprocess.run(
            [
                self.executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-EncodedCommand",
                encoded,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def test_successful_cmdlet_exits_zero(self) -> None:
        result = self.run_wrapped("Write-Output 'wrapper-success'")

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("wrapper-success", result.stdout + result.stderr)

    def test_non_terminating_cmdlet_error_exits_nonzero(self) -> None:
        missing = "definitely_missing_agent_hook_test"
        result = self.run_wrapped(f"Get-Item -LiteralPath '.\\{missing}'")

        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(missing, result.stdout + result.stderr)

    def test_native_exit_code_is_preserved(self) -> None:
        result = self.run_wrapped(
            'cmd.exe /d /c "echo native-failure 1>&2 & exit /b 7"'
        )

        self.assertEqual(7, result.returncode, result.stdout + result.stderr)
        self.assertIn("native-failure", result.stdout + result.stderr)

    def test_native_stderr_does_not_fail_a_successful_command(self) -> None:
        result = self.run_wrapped(
            'cmd.exe /d /c "echo benign-native-stderr 1>&2 & exit /b 0"'
        )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn("benign-native-stderr", result.stdout + result.stderr)

    def test_error_reporting_works_in_constrained_language(self) -> None:
        missing = "definitely_missing_constrained_hook_test"
        result = self.run_wrapped(
            f"Get-Item -LiteralPath '.\\{missing}'",
            constrained=True,
        )

        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(missing, result.stdout + result.stderr)
        self.assertNotIn("Cannot invoke method", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
