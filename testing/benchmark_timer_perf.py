import os
import sys
import time
from collections import deque, OrderedDict

# Ensure repo root is on sys.path so we can import local packages when running from tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the current optimized Timer
from toolkit.timer import Timer as OptimizedTimer

# Define a legacy Timer implementation that mimics the pre-change behavior
class LegacyTimer:
    def __init__(self, name='Timer', max_buffer=10):
        self.name = name
        self.max_buffer = max_buffer
        self.timers = OrderedDict()
        self.active_timers = {}
        self.current_timer = None
        self._after_print_hooks = []

    def start(self, timer_name):
        if timer_name not in self.timers:
            self.timers[timer_name] = deque(maxlen=self.max_buffer)
        self.active_timers[timer_name] = time.time()
        self.current_timer = timer_name

    def cancel(self, timer_name):
        if timer_name in self.active_timers:
            del self.active_timers[timer_name]

    def stop(self, timer_name):
        if timer_name not in self.active_timers:
            raise ValueError(f"Timer '{timer_name}' was not started!")
        elapsed_time = time.time() - self.active_timers[timer_name]
        self.timers[timer_name].append(elapsed_time)
        del self.active_timers[timer_name]
        if len(self.timers[timer_name]) > self.max_buffer:
            self.timers[timer_name].popleft()

    def add_after_print_hook(self, hook):
        self._after_print_hooks.append(hook)

    def print(self):
        print(f"Timer '{self.name}':")
        for timer_name, timings in sorted(self.timers.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_time = sum(timings) / len(timings)
            print(f" - {avg_time:.6f}s avg - {timer_name}, num = {len(timings)}")
        for hook in self._after_print_hooks:
            hook({k: sum(v)/len(v) for k,v in self.timers.items()})

    def __call__(self, timer_name):
        self.current_timer = timer_name
        self.start(timer_name)
        return self

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.stop(self.current_timer)
        else:
            self.cancel(self.current_timer)


def benchmark_start_stop(timer_factory, iterations=20000, timers_per_iter=4):
    t = timer_factory('bench')
    names = [f't{i}' for i in range(timers_per_iter)]
    start = time.time()
    for _ in range(iterations):
        # simulate nested timing
        for n in names:
            ctx = t(n)
            # enter and immediately exit
            ctx.__enter__()
            ctx.__exit__(None, None, None)
    end = time.time()
    return end - start


def benchmark_print(timer_factory, populate_n=1000):
    t = timer_factory('bench_print', max_buffer=10)
    for i in range(populate_n):
        name = f't{i % 10}'
        t.start(name)
        t.stop(name)
    start = time.time()
    t.print()
    end = time.time()
    return end - start


def make_optimized_factory():
    def f(name, max_buffer=10):
        return OptimizedTimer(name=name, max_buffer=max_buffer)
    return f


def make_legacy_factory():
    def f(name, max_buffer=10):
        return LegacyTimer(name=name, max_buffer=max_buffer)
    return f

if __name__ == '__main__':
    iters = 5000
    tps = 6

    print('Running micro-benchmarks with iterations=%d, timers_per_iter=%d' % (iters, tps))

    legacy_start_stop = benchmark_start_stop(make_legacy_factory(), iterations=iters, timers_per_iter=tps)
    opt_start_stop = benchmark_start_stop(make_optimized_factory(), iterations=iters, timers_per_iter=tps)

    legacy_print = benchmark_print(make_legacy_factory(), populate_n=2000)
    opt_print = benchmark_print(make_optimized_factory(), populate_n=2000)

    # Now simulate an environment where stdout is a flushing Logger (log file writer)
    class FakeLogger:
        def __init__(self, path='tmp_bench_log.txt'):
            self.f = open(path, 'w')
        def write(self, s):
            # simulate blocking I/O per write (e.g., UI logger flush) with a small sleep
            self.f.write(s)
            self.f.flush()
            time.sleep(0.0005)
        def flush(self):
            self.f.flush()

    print('\nPlain console timing above (stdout), now simulate flushing-stdout to measure real effect')
    import sys
    real_stdout = sys.stdout
    sys.stdout = FakeLogger()
    try:
        legacy_print_flush = benchmark_print(make_legacy_factory(), populate_n=2000)
        opt_print_flush = benchmark_print(make_optimized_factory(), populate_n=2000)
    finally:
        sys.stdout.f.close()
        sys.stdout = real_stdout

    print('\nResults:')
    print(f'Legacy start/stop time: {legacy_start_stop:.6f}s')
    print(f'Optimized start/stop time: {opt_start_stop:.6f}s')
    print(f'Legacy print time (stdout): {legacy_print:.6f}s')
    print(f'Optimized print time (stdout+print_acc): {opt_print:.6f}s')
    print('\nWith flushing stdout (simulating Logger):')
    print(f'Legacy print time (flushing): {legacy_print_flush:.6f}s')
    print(f'Optimized print time (flushing + print_acc): {opt_print_flush:.6f}s')
    print('\nSpeedups (flushing stdout):')
    print(f'Print speedup: {legacy_print_flush / opt_print_flush:.2f}x')
