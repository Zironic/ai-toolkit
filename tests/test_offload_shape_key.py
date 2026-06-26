import unittest

import torch

from toolkit.memory_management import MemoryManager


class OffloadShapeKeyTests(unittest.TestCase):
    def test_preservation_resolution_partitions_macro_step_traces(self):
        batch = {"latents": torch.zeros(1, 4, 64, 64)}

        plain_512 = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=False,
            dop_resolution=None,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        dop_256 = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        dop_full_res = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=None,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )

        self.assertNotEqual(plain_512, dop_256)
        self.assertNotEqual(dop_full_res, dop_256)

    def test_dop_cache_policy_partitions_macro_step_traces(self):
        batch = {"latents": torch.zeros(1, 4, 64, 64)}

        no_cache = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=False,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        cache_enabled = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=True,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )

        self.assertNotEqual(no_cache, cache_enabled)


if __name__ == "__main__":
    unittest.main()