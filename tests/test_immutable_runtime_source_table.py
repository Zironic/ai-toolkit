"""CPU-only tests for compile-neutral immutable residency publication."""

import inspect
import unittest

import torch

from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter
from toolkit.memory_management.immutable_runtime import (
    ImmutableRuntimeError,
    ImmutableRuntimeSourceTable,
    ImmutableTransformerRuntime,
    build_program_fingerprint,
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
        # Two-phase lifecycle: construct (prepare), then finalize permanent
        # programs before any execution/publication. An empty model finalizes
        # with no LoRA state.
        executor = ImmutableTransformerRuntime(
            _EmptyModel(),
            _FakeResidency(),
            architecture_adapter=SingleStreamMMDiTAdapter(),
            block_operations=(),
            compile_blocks=False,
        )
        executor.finalize_execution()
        return executor

    def test_residency_publication_preserves_program_objects(self):
        executor = self._executor()
        self.assertIsInstance(executor._sources, ImmutableRuntimeSourceTable)
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

    def test_execution_context_rejects_overlap_and_releases_on_exception(self):
        executor = self._executor()

        with executor.execution(executor.TRAIN):
            self.assertEqual(executor.active_executions, 1)
            with self.assertRaisesRegex(
                ImmutableRuntimeError,
                "residency_transition_during_execution",
            ):
                executor.set_residency_plan(ResidencyPlan.build("sample", ()))
            with self.assertRaisesRegex(
                ImmutableRuntimeError,
                "immutable_execution_already_active:train",
            ):
                with executor.execution(executor.SAMPLE):
                    pass
            with self.assertRaisesRegex(
                ImmutableRuntimeError,
                "cannot_close_during_execution",
            ):
                executor.close()

        self.assertEqual(executor.active_executions, 0)
        with self.assertRaisesRegex(ValueError, "expected"):
            with executor.execution(executor.TRAIN):
                raise ValueError("expected")
        self.assertEqual(executor.active_executions, 0)

        executor.set_residency_plan(ResidencyPlan.build("sample", ()))
        with executor.execution(executor.SAMPLE):
            self.assertEqual(executor.active_executions, 1)

    def test_sampling_context_measures_success_and_clears_failure_state(self):
        executor = self._executor()
        calls = []

        def activate_sampling_image(**kwargs):
            calls.append(("activate", kwargs))
            executor._sampling_baseline = {"shape_key": kwargs["shape_key"]}

        def finish_sampling_image(*, shape_key):
            calls.append(("finish", shape_key))
            executor._sampling_baseline = None

        executor.activate_sampling_image = activate_sampling_image
        executor.finish_sampling_image = finish_sampling_image

        with executor.sampling(shape_key=(1, 2), cold_working_bytes=3):
            self.assertEqual(executor.active_executions, 1)
        self.assertEqual(
            calls,
            [
                (
                    "activate",
                    {
                        "shape_key": (1, 2),
                        "cold_working_bytes": 3,
                    },
                ),
                ("finish", (1, 2)),
            ],
        )

        with self.assertRaisesRegex(ValueError, "sample failed"):
            with executor.sampling(shape_key=(3, 4)):
                raise ValueError("sample failed")
        self.assertIsNone(executor._sampling_baseline)
        self.assertEqual(executor.active_executions, 0)

    def test_run_requires_matching_execution_context(self):
        executor = self._executor()

        with self.assertRaisesRegex(
            ImmutableRuntimeError,
            "immutable_execution_not_active",
        ):
            executor.run("hidden", {"conditioning": None})

        with executor.execution(executor.TRAIN):
            self.assertEqual(
                executor.run("hidden", {"conditioning": None}),
                "hidden",
            )

        # A no-grad forward inside a training step is legitimate (diff-output
        # preservation runs the prior prediction with the network detached) and
        # stays on the TRAIN program.
        with executor.execution(executor.TRAIN):
            with torch.no_grad():
                self.assertEqual(
                    executor.run("hidden", {"conditioning": None}),
                    "hidden",
                )

        # The reverse still fails closed: no grad graphs during sampling.
        executor.set_residency_plan(ResidencyPlan.build("sample", ()))
        with executor.execution(executor.SAMPLE):
            with self.assertRaisesRegex(
                ImmutableRuntimeError,
                "immutable_execution_mode_mismatch:active=sample:call=train",
            ):
                executor.run("hidden", {"conditioning": None})

    def test_structural_fingerprint_has_no_residency_input(self):
        parameters = inspect.signature(build_program_fingerprint).parameters
        self.assertNotIn("residency", parameters)
        self.assertNotIn("residency_plan", parameters)

        first = build_program_fingerprint(
            "train",
            (),
            architecture_key="single_stream_mmdit",
            depth=2,
            checkpoint_mode="full",
        )
        second = build_program_fingerprint(
            "train",
            (),
            architecture_key="single_stream_mmdit",
            depth=2,
            checkpoint_mode="full",
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
