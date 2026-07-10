import unittest

from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)


class CompileStreamedOffloadSliceTests(unittest.TestCase):
    def _model(self):
        return SingleStreamDiT(
            SingleMMDiTConfig(
                features=32,
                tdim=16,
                txtdim=32,
                heads=4,
                multiplier=1,
                layers=1,
                patch=1,
                channels=4,
                txtheads=4,
                txtkvheads=4,
            )
        )

    def test_compile_flags_default_off(self):
        cfg = ModelConfig(name_or_path="dummy")
        self.assertFalse(cfg.layer_offloading_compile_streamed)
        self.assertFalse(cfg.train_compile_blocks)

        cfg = ModelConfig(
            name_or_path="dummy",
            layer_offloading_compile_streamed=True,
            train_compile_blocks=True,
        )
        self.assertTrue(cfg.layer_offloading_compile_streamed)
        self.assertTrue(cfg.train_compile_blocks)

    def test_training_readiness_reports_trainable_base_weight(self):
        model = self._model()
        readiness = model.training_compile_readiness({"blocks.0"})
        status = readiness["statuses"][0]
        self.assertFalse(status["ready"])
        self.assertIn("trainable_base_weight", status["reasons"])

    def test_training_readiness_accepts_frozen_resident_block(self):
        model = self._model()
        for param in model.blocks[0].parameters():
            param.requires_grad_(False)
        readiness = model.training_compile_readiness({"blocks.0"})
        status = readiness["statuses"][0]
        self.assertTrue(status["ready"])
        self.assertEqual([], status["reasons"])

    def test_training_readiness_reports_streaming_hook(self):
        model = self._model()
        for param in model.blocks[0].parameters():
            param.requires_grad_(False)
        model.blocks[0].attn.wq._layer_memory_manager = object()
        try:
            readiness = model.training_compile_readiness({"blocks.0"})
        finally:
            del model.blocks[0].attn.wq._layer_memory_manager
        status = readiness["statuses"][0]
        self.assertFalse(status["ready"])
        self.assertIn("streaming_hook", status["reasons"])

    def test_sampling_readiness_reports_forward_hook(self):
        model = self._model()
        handle = model.blocks[0].attn.wq.register_forward_pre_hook(
            lambda _module, args: None
        )
        try:
            reasons = SingleStreamDiT._block_compile_reject_reasons(model.blocks[0])
        finally:
            handle.remove()
        self.assertIn("hook_present", reasons)

    def test_regional_sampling_compile_skips_ingraph_streamed_blocks(self):
        model = self._model()
        model._ingraph_sampling_plans = {0: object()}
        compiled_count, eager_count = model.enable_compiled_sampling()
        self.assertEqual(0, compiled_count)
        self.assertEqual(1, eager_count)
        self.assertIsNotNone(model._compiled_blocks)
        self.assertIsNone(model._compiled_blocks[0])


if __name__ == "__main__":
    unittest.main()
