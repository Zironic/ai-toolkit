"""Time-driven selective-checkpointing autotuner.

The fastest safe keep_last is not necessarily the largest value that fits.
This controller hill-climbs measured forward/backward time per resolution
bucket, while treating the reserved-memory spill line as a hard ceiling.
"""

from typing import Callable, Optional


class _BucketState:
    __slots__ = (
        "keep_last", "settled", "last_peak", "block_bytes", "pending_grow",
        "time_samples", "peak_samples", "best_keep", "best_time",
        "_previous_peak",
        "warmup_pending",
    )

    def __init__(self):
        self.keep_last = 0
        self.settled = False
        self.last_peak = None
        self.block_bytes = 0
        self.pending_grow = False
        self.time_samples = []
        self.peak_samples = []
        self.best_keep = 0
        self.best_time = None
        self._previous_peak = None
        self.warmup_pending = False


class CheckpointKeepLastAutotuner:
    def __init__(
        self,
        device_total_bytes: int,
        max_keep_last: int,
        margin_bytes: int = 512 * 1024 ** 2,
        default_block_bytes: int = 512 * 1024 ** 2,
        set_keep_last: Optional[Callable[[int], None]] = None,
        samples_per_candidate: int = 2,
        min_improvement: float = 0.01,
    ):
        self.device_total_bytes = int(device_total_bytes)
        self.max_keep_last = max(0, int(max_keep_last))
        self.margin_bytes = int(margin_bytes)
        self.default_block_bytes = int(default_block_bytes)
        self._set_keep_last = set_keep_last
        self.samples_per_candidate = max(1, int(samples_per_candidate))
        self.min_improvement = max(0.0, float(min_improvement))
        self.buckets: dict = {}

    @property
    def limit_bytes(self) -> int:
        return self.device_total_bytes - self.margin_bytes

    def _state(self, bucket) -> _BucketState:
        return self.buckets.setdefault(bucket, _BucketState())

    def _apply(self, value: int) -> None:
        if self._set_keep_last is not None:
            self._set_keep_last(int(value))

    def recommend(self, bucket) -> int:
        """keep_last to use for the next step at this resolution bucket."""
        keep_last = self._state(bucket).keep_last
        self._apply(keep_last)
        return keep_last

    def is_settled(self, bucket) -> bool:
        return self._state(bucket).settled

    def _settle_at_best(self, state: _BucketState) -> None:
        state.keep_last = state.best_keep
        state.settled = True
        state.pending_grow = False
        state.time_samples.clear()
        state.peak_samples.clear()

    def observe(self, bucket, peak_bytes: int, step_time_s: float) -> bool:
        """Observe one completed accumulation at the current candidate.

        Memory is an immediate safety veto. Otherwise, average a small number
        of timings, keep climbing only while elapsed time improves, and revert
        to the fastest measured candidate on the first regression. Returns
        True when the hard memory ceiling was crossed so the caller can release
        allocator cache after backing off.
        """
        state = self._state(bucket)
        peak_bytes = int(peak_bytes)
        step_time_s = float(step_time_s)
        state.last_peak = peak_bytes

        if peak_bytes > self.limit_bytes:
            # Even a previously-safe value can spill for a larger aspect ratio
            # within the same bucket. Back off at least one block immediately.
            state.best_keep = min(state.best_keep, max(0, state.keep_last - 1))
            self._settle_at_best(state)
            return True

        if state.settled:
            return False

        # A keep_last change alters the forward/recompute/backward access trace.
        # The caller re-records it on the first step at the new candidate; do
        # not contaminate timing with that intentionally-cold prefetch step.
        if state.warmup_pending:
            state.warmup_pending = False
            state.time_samples.clear()
            state.peak_samples.clear()
            return False

        state.time_samples.append(step_time_s)
        state.peak_samples.append(peak_bytes)
        if len(state.time_samples) < self.samples_per_candidate:
            return False

        candidate_time = sum(state.time_samples) / len(state.time_samples)
        candidate_peak = max(state.peak_samples)
        state.time_samples.clear()
        state.peak_samples.clear()

        if state.pending_grow and state.last_peak is not None:
            previous_peak = state._previous_peak
            if previous_peak is not None:
                jump = candidate_peak - previous_peak
                if jump > 0:
                    state.block_bytes = max(state.block_bytes, jump)
        state.pending_grow = False

        improved = (
            state.best_time is None
            or candidate_time < state.best_time * (1.0 - self.min_improvement)
        )
        if not improved:
            self._settle_at_best(state)
            return False

        state.best_keep = state.keep_last
        state.best_time = candidate_time

        if state.keep_last >= self.max_keep_last:
            state.settled = True
            return False

        block = state.block_bytes if state.block_bytes > 0 else self.default_block_bytes
        if candidate_peak + block > self.limit_bytes:
            state.settled = True
            return False

        # Preserve the accepted candidate's peak so the next observation can
        # learn its actual per-block memory increment.
        state._previous_peak = candidate_peak
        state.keep_last += 1
        state.pending_grow = True
        state.warmup_pending = True
        return False

    def report(self) -> Optional[str]:
        if not self.buckets:
            return None
        gib = 1024 ** 3
        parts = []
        for bucket in sorted(self.buckets):
            state = self.buckets[bucket]
            best = "?" if state.best_time is None else f"{state.best_time:.2f}s"
            parts.append(
                f"{bucket}:keep_last={state.keep_last}"
                f"{'' if state.settled else '(tuning)'}"
                f" best={state.best_keep}@{best}"
                f" block={state.block_bytes / gib:.2f}GiB"
                f" peak={(state.last_peak or 0) / gib:.2f}GiB"
            )
        return "[CheckpointAutotune] limit={:.2f}GiB  {}".format(
            self.limit_bytes / gib, "  ".join(parts)
        )
