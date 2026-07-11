"""CPU-only tests for compile-neutral immutable residency publication."""

import unittest

from extensions_built_in.diffusion_models.krea2.src.immutable_arena import (
    KreaImmutableArenaError,
    KreaImmutablePlanExecutor,
    build_execution_fingerprint,
)
from toolkit.memory_management.residency import ResidencyDelta, ResidencyPlan


class _FakeArena:
    def immutable_signature(self):
        return (True, ("stable",))


class _FakeResidency:
    def __init__(self):
        self.arena = _FakeArena()
        self.plan = ResidencyPlan.build("train", ())
        self.device = None

    def reconcile(self, plan):
        old = self.plan
        self.plan = plan
        return ResidencyDelta(
            tuple(sorted(plan.resident_leaf_keys - old.resident_leaf_keys)),
            tuple(sorted(old.resident_leaf_keys - plan.resident_leaf_keys)),
            0,
        )

    def resident_bytes(self):
        return 0


class _EmptyModel:
    def __init__(self):
        self.blocks = []


class ImmutableRuntimeSourceTableTests(unittest.TestCase):
    def _executor(self):
        return KreaImmutablePlanExecutor(
            _EmptyModel(),
            _FakeResidency(),
            compile_blocks=False,
        )

    def test_residency_publication_preserves_program_objects(self):
        executor = self._executor()
        train = executor.program(executor.TRAIN)
        sample = executor.program(executor.SAMPLE)
        train_trunk = train.trunk
        sample_trunk = sample.trunk
        kernels = dict(executor._block_kernels)

        executor.set_residency_plan(ResidencyPlan.build("sample", ()))
        executor.set_residency_plan(ResidencyPlan.build("train", ()))

        self.assertIs(executor.program(executor.TRAIN), train)
        self.assertIs(executor.program(executor.SAMPLE), sample)
        self.assertIs(executor.program(executor.TRAIN).trunk, train_trunk)
        self.assertIs(executor.program(executor.SAMPLE).trunk, sample_trunk)
        self.assertEqual(executor._block_kernels, kernels)
        self.assertEqual(executor.stats["residency_transitions"], 2)

    def test_transition_is_rejected_while_execution_is_active(self):
        executor = self._executor()
        generation = executor.begin_execution()
        try:
            with self.assertRaisesRegex(
                KreaImmutableArenaError,
                "residency_transition_during_execution",
            ):
                executor.set_residency_plan(ResidencyPlan.build("sample", ()))
        finally:
            executor.end_execution(generation)

    def test_structural_fingerprint_ignores_residency(self):
        train = ResidencyPlan.build("train", ())
        sample = ResidencyPlan.build("sample", ())
        kwargs = {
            "mode": "train",
            "block_plans": (),
            "depth": 2,
            "checkpoint_mode": "full",
        }
        first = build_execution_fingerprint(
            residency_plan=train,
            **kwargs,
        )
        second = build_execution_fingerprint(
            residency_plan=sample,
            **kwargs,
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
