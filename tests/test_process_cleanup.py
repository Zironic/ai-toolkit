from jobs import BaseJob
import toolkit.process_cleanup as process_cleanup

def test_force_exit_uses_taskkill_for_windows_process_tree():
    calls = []
    process_cleanup._force_exit_process_tree(1234, platform="win32", run=lambda *args, **kwargs: calls.append((args, kwargs)), exit_func=lambda code: calls.append(("exit", code)))
    assert calls[0][0][0] == ["taskkill", "/PID", "1234", "/T", "/F"]
    assert calls[1] == ("exit", 1)

def test_force_exit_falls_back_when_taskkill_cannot_start():
    exits = []
    def fail_to_start(*args, **kwargs):
        raise OSError("taskkill unavailable")
    process_cleanup._force_exit_process_tree(1234, platform="win32", run=fail_to_start, exit_func=exits.append)
    assert exits == [1]

def test_force_exit_uses_direct_exit_off_windows():
    exits = []
    process_cleanup._force_exit_process_tree(1234, platform="linux", run=lambda *args, **kwargs: None, exit_func=exits.append)
    assert exits == [1]
class _CleanupProcess:
    def __init__(self, name, calls, error=None):
        self.name = name
        self.calls = calls
        self.error = error
        self.job = object()

    def cleanup(self):
        self.calls.append(self.name)
        if self.error is not None:
            raise self.error


def test_base_job_cleanup_releases_every_process_after_failure():
    from jobs.BaseJob import BaseJob

    calls = []
    job = BaseJob.__new__(BaseJob)
    job.process = [
        _CleanupProcess("first", calls),
        _CleanupProcess("second", calls, RuntimeError("boom")),
    ]
    processes = list(job.process)

    import pytest
    with pytest.raises(RuntimeError, match="_CleanupProcess: boom"):
        job.cleanup()

    assert calls == ["second", "first"]
    assert job.process == []
    assert all(process.job is None for process in processes)

class _FlushRecorder:
    def __init__(self):
        self.flushed = False

    def flush(self):
        self.flushed = True


def test_successful_ui_worker_flushes_and_exits_zero():
    stdout = _FlushRecorder()
    stderr = _FlushRecorder()
    exits = []

    did_exit = process_cleanup.exit_ui_worker_successfully(
        environ={"IS_AI_TOOLKIT_UI": "1"},
        exit_func=exits.append,
        stdout=stdout,
        stderr=stderr,
    )

    assert did_exit
    assert stdout.flushed and stderr.flushed
    assert exits == [0]


def test_cli_success_keeps_natural_interpreter_shutdown():
    exits = []

    did_exit = process_cleanup.exit_ui_worker_successfully(
        environ={}, exit_func=exits.append
    )

    assert not did_exit
    assert exits == []


def test_ui_cleanup_exits_before_releasing_process_graph():
    calls = []
    process = _CleanupProcess("trainer", calls)
    job = BaseJob.__new__(BaseJob)
    job.process = [process]
    exits = []

    process_cleanup.cleanup_job_before_ui_exit(
        job,
        environ={"IS_AI_TOOLKIT_UI": "1"},
        exit_func=exits.append,
        stdout=_FlushRecorder(),
        stderr=_FlushRecorder(),
    )

    assert calls == ["trainer"]
    assert job.process == []
    assert exits == [0]


def test_cli_cleanup_does_not_retain_process_graph():
    calls = []
    job = BaseJob.__new__(BaseJob)
    job.process = [_CleanupProcess("trainer", calls)]

    process_cleanup.cleanup_job_before_ui_exit(job, environ={})

    assert calls == ["trainer"]
    assert job.process == []


def test_failed_ui_job_cleanup_does_not_report_success_exit():
    calls = []
    job = BaseJob.__new__(BaseJob)
    job.process = [_CleanupProcess("trainer", calls)]
    exits = []

    process_cleanup.cleanup_job_before_ui_exit(
        job,
        allow_ui_success_exit=False,
        environ={"IS_AI_TOOLKIT_UI": "1"},
        exit_func=exits.append,
    )

    assert calls == ["trainer"]
    assert job.process == []
    assert exits == []
