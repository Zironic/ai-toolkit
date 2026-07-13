import os
import subprocess
import sys
import threading
from typing import Callable

ERROR_EXIT_GRACE_SECONDS = 15.0

def _force_exit_process_tree(pid: int, platform: str = sys.platform, run: Callable = subprocess.run, exit_func: Callable[[int], None] = os._exit) -> None:
    """Immediately end this process, including descendants on Windows."""
    if platform == "win32":
        try:
            run(["taskkill", "/PID", str(pid), "/T", "/F"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
        except Exception:
            pass
    exit_func(1)

def exit_ui_worker_successfully(
    *, environ=None, exit_func=os._exit, stdout=None, stderr=None
) -> bool:
    """Flush output and end a successfully cleaned-up detached UI worker."""
    environ = os.environ if environ is None else environ
    if environ.get("IS_AI_TOOLKIT_UI") != "1":
        return False
    stdout = sys.stdout if stdout is None else stdout
    stderr = sys.stderr if stderr is None else stderr
    stdout.flush()
    stderr.flush()
    exit_func(0)
    return True

def arm_error_exit_watchdog(grace_seconds: float = ERROR_EXIT_GRACE_SECONDS) -> threading.Event:
    """Prevent detached UI training from hanging forever in error cleanup."""
    disarmed = threading.Event()
    pid = os.getpid()
    def watch() -> None:
        if not disarmed.wait(grace_seconds):
            _force_exit_process_tree(pid)
    threading.Thread(target=watch, name="aitk-error-exit-watchdog", daemon=True).start()
    return disarmed