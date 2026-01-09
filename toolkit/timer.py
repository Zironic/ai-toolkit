import time
from collections import OrderedDict, deque
import sys
import os

# check if is ui process will have IS_AI_TOOLKIT_UI in env
is_ui = os.environ.get("IS_AI_TOOLKIT_UI", "0") == "1"

class Timer:
    def __init__(self, name='Timer', max_buffer=10):
        self.name = name
        self.max_buffer = max_buffer
        self.timers = OrderedDict()
        self.active_timers = {}
        self._after_print_hooks = []
        # GPU timing is disabled by default to avoid extra overhead. Toggle at runtime with
        # `timer.gpu_timing_enabled = True` when running short diagnostics.
        self.gpu_timing_enabled = False
        self._active_cuda_events = {}

        # Pre-create a reusable context class (avoid recreating a class per __call__ invocation)
        class _TimerContext:
            def __init__(self, timer, name):
                self._timer = timer
                self._name = name
                self._started = False

            def __enter__(self):
                self._timer.start(self._name)
                self._started = True
                return self

            def __exit__(self, exc_type, exc_value, tb):
                # Only stop if the timer was started and is still active
                if not self._started:
                    return
                if self._name in self._timer.active_timers:
                    # Normal exit: stop the named timer
                    self._timer.stop(self._name)
                else:
                    # If an exception occurred or timer was canceled elsewhere, ensure it's removed
                    self._timer.cancel(self._name)

        # Store the context class so that __call__ can reuse it without recreating
        self._TimerContext = _TimerContext

    def start(self, timer_name):
        if timer_name not in self.timers:
            self.timers[timer_name] = deque(maxlen=self.max_buffer)
        self.active_timers[timer_name] = time.time()
        # Optionally record a GPU start event for more accurate GPU kernel timing.
        # Disabled by default to avoid overhead; enable with `timer.gpu_timing_enabled = True`.
        if getattr(self, 'gpu_timing_enabled', False):
            try:
                import torch
                if torch.cuda.is_available():
                    ev = torch.cuda.Event(enable_timing=True)
                    ev.record()
                    if not hasattr(self, '_active_cuda_events'):
                        self._active_cuda_events = {}
                    self._active_cuda_events[timer_name] = ev
            except Exception:
                # best-effort; do not crash timing on import/availability issues
                pass

    def cancel(self, timer_name):
        """Cancel an active timer."""
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]

    def stop(self, timer_name):
        # If the timer was not started, silently ignore (avoid raising in hot path)
        if timer_name not in self.active_timers:
            # also cleanup any cuda event if present
            if hasattr(self, '_active_cuda_events') and timer_name in self._active_cuda_events:
                del self._active_cuda_events[timer_name]
            return

        elapsed_time = time.time() - self.active_timers[timer_name]
        self.timers[timer_name].append(elapsed_time)

        # If GPU events were recorded for this timer, attempt to capture GPU elapsed time without global sync.
        if getattr(self, 'gpu_timing_enabled', False) and hasattr(self, '_active_cuda_events') and timer_name in self._active_cuda_events:
            try:
                import torch
                if torch.cuda.is_available():
                    stop_ev = torch.cuda.Event(enable_timing=True)
                    stop_ev.record()
                    start_ev = self._active_cuda_events.pop(timer_name, None)
                    if start_ev is not None:
                        gpu_ms = stop_ev.elapsed_time(start_ev)
                        gpu_secs = gpu_ms / 1000.0
                        gpu_timer_name = f"{timer_name}_gpu"
                        if gpu_timer_name not in self.timers:
                            self.timers[gpu_timer_name] = deque(maxlen=self.max_buffer)
                        self.timers[gpu_timer_name].append(gpu_secs)
            except Exception:
                # best-effort; ignore GPU timing failures and ensure cleanup
                try:
                    if timer_name in self._active_cuda_events:
                        del self._active_cuda_events[timer_name]
                except Exception:
                    pass

        # Clean up active timers
        del self.active_timers[timer_name]

    def add_after_print_hook(self, hook):
        self._after_print_hooks.append(hook)

    def print(self):
        # Consolidate printing to a single call to avoid heavy I/O cost when called frequently.
        timing_dict = {}
        # sort by longest at top
        lines = []
        lines.append(f"Timer '{self.name}':")
        for timer_name, timings in sorted(self.timers.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_time = sum(timings) / len(timings)
            lines.append(f" - {avg_time:.4f}s avg - {timer_name}, num = {len(timings)}")
            timing_dict[timer_name] = avg_time

        # Call hooks with the timing dict before printing the consolidated block so hooks can append or use it
        for hook in self._after_print_hooks:
            try:
                hook(timing_dict)
            except Exception:
                # best effort; don't crash printing on hook failures
                pass

        # Print the entire block in a single atomic call to reduce flush overhead
        if not is_ui:
            try:
                # Import locally to avoid cyclical import issues at module import time
                from toolkit.print import print_acc
                print_acc('\n' + '\n'.join(lines) + '\n')
            except Exception:
                # fallback to plain print if import fails
                print('\n' + '\n'.join(lines) + '\n')

    def reset(self):
        self.timers.clear()
        self.active_timers.clear()

    def __call__(self, timer_name):
        """Return a context manager instance for the named timer (re-uses the context class)."""
        return self._TimerContext(self, timer_name)

    # Backwards-compatible no-op enter/exit removed (we use the context object above)
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass
