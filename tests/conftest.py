"""Shared fixtures for the memory-management test suite."""

import os
import subprocess
import sys
import time

import pytest
from _pytest.reports import TestReport

# ---------------------------------------------------------------------------
# 'process_isolated' marker: run the whole marked file in a disposable process.
#
# CUDA compilation, host registration, and torch's caching host allocator can
# leave process-global state that cannot be safely scrubbed between tests
# (git-bug ticket f2aceba). Files that exercise those mechanisms run in a
# disposable child process. Process termination is the cleanup boundary.
# pytest-forked/xdist --forked need os.fork and do not exist on Windows, so
# this is a subprocess-based equivalent. One child per marked file amortizes
# the ~10 s torch/CUDA startup; per-test outcomes are recovered from -q output.
# ---------------------------------------------------------------------------

_ISOLATED_ENV = "AI_TOOLKIT_PROCESS_ISOLATED"
_isolated_module_results = {}


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "process_isolated: exercises process-global CUDA/host-registration "
        "state; the whole file runs in a disposable subprocess (ticket f2aceba)",
    )


def _run_module_isolated(item):
    path = str(item.path)
    cached = _isolated_module_results.get(path)
    if cached is not None:
        return cached
    env = dict(os.environ)
    env[_ISOLATED_ENV] = "1"
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path, "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(item.config.rootpath),
    )
    duration = time.perf_counter() - started
    failed = set()
    for line in proc.stdout.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):
            failed.add(line.split(" ")[1])
    result = {
        "failed": failed,
        "returncode": proc.returncode,
        "output": proc.stdout + proc.stderr,
        "duration": duration,
    }
    _isolated_module_results[path] = result
    return result


def pytest_terminal_summary(terminalreporter):
    if not _isolated_module_results:
        return
    terminalreporter.section("isolated file durations")
    for path, result in sorted(
        _isolated_module_results.items(),
        key=lambda entry: entry[1]["duration"],
        reverse=True,
    ):
        terminalreporter.write_line(f'{result["duration"]:.2f}s {path}')


def _report(item, when, outcome, longrepr=None):
    report = TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords={item.name: 1},
        outcome=outcome,
        longrepr=longrepr,
        when=when,
        sections=[],
        duration=0.0,
        start=0.0,
        stop=0.0,
    )
    item.ihook.pytest_runtest_logreport(report=report)


def pytest_runtest_protocol(item, nextitem):
    if os.environ.get(_ISOLATED_ENV) or not item.get_closest_marker("process_isolated"):
        return None
    result = _run_module_isolated(item)
    # nodeids inside the subprocess are rootpath-relative with forward
    # slashes, same as item.nodeid when the suite runs from the repo root.
    failed = item.nodeid in result["failed"]
    if result["returncode"] != 0 and not result["failed"]:
        # Subprocess died before producing per-test results (crash, collect
        # error): fail every test in the file with the raw output.
        failed = True
    item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    _report(item, "setup", "passed")
    if failed:
        _report(
            item,
            "call",
            "failed",
            longrepr="failed in disposable subprocess (marker: process_isolated)\n\n"
            + result["output"],
        )
    else:
        _report(item, "call", "passed")
    _report(item, "teardown", "passed")
    item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True


@pytest.fixture(autouse=True)
def _release_leaked_pinned_arenas():
    """Release any PinnedWeightArena a test forgot to tear down.

    A leaked arena keeps its flats cudaHostRegister'd for the process
    lifetime; when the CPU allocator later recycles those pages for a fresh
    torch.empty, re-registering them raises 'resource already mapped' and
    every subsequent register-mechanism pin in the suite flips pageable
    (the 763bb75 collision, reproduced order-dependently across tests).
    Production owners release explicitly (_destroy_pinned_arena); this sweep
    is test hygiene only.
    """
    yield
    from toolkit.memory_management.pinned_arena import _LIVE_ARENAS

    # release() discards from _LIVE_ARENAS, so snapshot first.
    for arena in list(_LIVE_ARENAS):
        try:
            arena.release()
        except Exception:
            pass
    _LIVE_ARENAS.clear()
