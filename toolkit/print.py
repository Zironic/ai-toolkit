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
        # Keep original stdout/stderr stream as terminal, but open our logfile
        # with utf-8 and a safe error handler so non-encodable characters do not
        # crash the process when writing to disk.
        self.terminal = sys.stdout
        # Use utf-8 with replace to ensure writes never raise on encoding
        self.log = open(filename, 'a', encoding='utf-8', errors='replace')

    def write(self, message):
        # Write to terminal, but guard against terminals that cannot encode
        # certain unicode characters (Windows cp1252, etc.). If terminal.write
        # raises, attempt a safe fall-back where non-encodable characters are
        # replaced.
        safe = None
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            try:
                safe = message.encode('utf-8', errors='replace').decode('utf-8')
                self.terminal.write(safe)
            except Exception:
                try:
                    # Last resort: fall back to ascii with replacements
                    safe = message.encode('ascii', errors='replace').decode('ascii')
                    self.terminal.write(safe)
                except Exception:
                    # Give up quietly - writing to terminal failed
                    pass
        except Exception:
            # Some other I/O error writing to terminal - swallow to avoid hard crash
            try:
                self.terminal.write(repr(message))
            except Exception:
                pass

        # Always attempt to write to our log file (utf-8, replace). If that fails,
        # attempt to write a safe-replaced string.
        try:
            self.log.write(message)
            self.log.flush()  # Make sure it's written immediately
        except Exception:
            try:
                if safe is None:
                    safe = message.encode('utf-8', errors='replace').decode('utf-8')
                self.log.write(safe)
                self.log.flush()
            except Exception:
                # Give up silently to avoid raising in logging
                pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass


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
