"""raise_dynamo_recompile_limit lifts dynamo's per-code-object cache cap.

Regression for the step-101 crash: bucketed shapes + sampling-boundary trunk
rebuilds accumulated 9 cache entries on _ingraph_training_trunk's code object
and fullgraph=True turned the default limit of 8 into FailOnRecompileLimitHit.
"""

import unittest

import torch

from toolkit.memory_management.arena_offload.transfer import raise_dynamo_recompile_limit


class RaiseDynamoRecompileLimitTests(unittest.TestCase):
    def _snapshot(self):
        config = torch._dynamo.config
        return {
            attr: getattr(config, attr)
            for attr in ("recompile_limit", "cache_size_limit")
            if hasattr(config, attr)
        }

    def test_raises_low_limits(self):
        config = torch._dynamo.config
        saved = self._snapshot()
        self.assertTrue(saved, "torch._dynamo.config lost both limit attrs")
        try:
            for attr in saved:
                setattr(config, attr, 8)
            raise_dynamo_recompile_limit()
            for attr in saved:
                self.assertGreaterEqual(getattr(config, attr), 128, attr)
        finally:
            for attr, value in saved.items():
                setattr(config, attr, value)

    def test_never_lowers_a_higher_limit(self):
        config = torch._dynamo.config
        saved = self._snapshot()
        try:
            for attr in saved:
                setattr(config, attr, 512)
            raise_dynamo_recompile_limit()
            for attr in saved:
                self.assertEqual(getattr(config, attr), 512, attr)
        finally:
            for attr, value in saved.items():
                setattr(config, attr, value)


if __name__ == "__main__":
    unittest.main()
