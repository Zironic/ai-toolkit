import sys
import os


def print_acc(*args, **kwargs):
    """Print only on local main process; import accelerator lazily to avoid
    heavyweight imports during test collection."""
    try:
        from toolkit.accelerator import get_accelerator
        if get_accelerator().is_local_main_process:
            print(*args, **kwargs)
    except Exception:
        # Fall back to printing to keep tests and diagnostics visible when
        # accelerator is not available during unit tests.
        print(*args, **kwargs)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def setup_log_to_file(filename):
    try:
        from toolkit.accelerator import get_accelerator
        if get_accelerator().is_local_main_process:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
    except Exception:
        # If accelerator not available during tests, still allow log setup to proceed
        pass
    sys.stdout = Logger(filename)
    sys.stderr = Logger(filename)
