"""Unit checks for the saved-tensor (autograd activation footprint) probe.

The probe is a debug-only diagnostic (AITK_SAVED_TENSOR_PROBE=<n>) that tallies
what the backward graph retains, so decisions like FP8-autograd or swapping the
attention backend are grounded in measured numbers. These tests guard the two
properties that make the number trustworthy:

  * storage dedup  -- saved tensors routinely alias one allocation; a naive
    numel*element_size sum over-counts wildly. Aliased saves must count once.
  * weight/activation split -- a Linear saves its weight for backward; those
    bytes are already accounted for in resident+ring and must not be reported
    as activations.

Requires CUDA (the pack hook intentionally only tallies device tensors). Skips
cleanly on CPU-only machines; run through the project venv to exercise it.
"""

import types
import unittest

import torch

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def _make_probe(unet):
    """A bare SDTrainer with only the attributes the probe methods touch."""
    self = SDTrainer.__new__(SDTrainer)
    self._saved_tensor_probe_active = True
    self._saved_tensor_probe_stats = {}
    self._saved_tensor_probe_param_ptrs = None
    self.sd = types.SimpleNamespace(unet=unet)
    self.device_torch = torch.device('cuda')
    self.model_config = types.SimpleNamespace(compile=False)
    return self


@unittest.skipUnless(torch.cuda.is_available(), "saved-tensor probe tallies CUDA tensors only")
class SavedTensorProbeTests(unittest.TestCase):
    def test_weight_activation_split(self):
        lin = torch.nn.Linear(128, 128).cuda()
        probe = _make_probe(lin)
        with probe._capture_saved_tensors('normal_forward'):
            x = torch.randn(16, 128, device='cuda', requires_grad=True)
            out = lin(x).relu().pow(2).sum()
        out.backward()

        b = probe._saved_tensor_probe_stats['normal_forward']
        # The weight (128*128*4 bytes) is saved for the matmul backward and must
        # land in the param bucket, not the activation bucket.
        self.assertEqual(b['param_bytes'], 128 * 128 * 4)
        self.assertGreaterEqual(b['param_count'], 1)
        # Activations (the [16,128] input + relu output) land in activations.
        self.assertGreater(b['act_bytes'], 0)
        self.assertIn(('float32', (16, 128)), b['by_shape'])

    def test_alias_dedup(self):
        lin = torch.nn.Linear(8, 8).cuda()  # unrelated; a below is not a param
        probe = _make_probe(lin)
        with probe._capture_saved_tensors('dedup'):
            a = torch.randn(1000, device='cuda', requires_grad=True)
            s = (a * a).sum()  # `a` is saved twice, same storage
        s.backward()

        d = probe._saved_tensor_probe_stats['dedup']
        self.assertEqual(d['act_count'], 1, "aliased saved tensors must dedup to one storage")
        self.assertEqual(d['act_bytes'], 1000 * 4)

    def test_inactive_is_noop(self):
        lin = torch.nn.Linear(8, 8).cuda()
        probe = _make_probe(lin)
        probe._saved_tensor_probe_active = False
        with probe._capture_saved_tensors('off'):
            out = lin(torch.randn(4, 8, device='cuda', requires_grad=True)).sum()
        out.backward()
        self.assertEqual(probe._saved_tensor_probe_stats, {})

    def test_step_count_parsing(self):
        import os
        prev = os.environ.get('AITK_SAVED_TENSOR_PROBE')
        try:
            os.environ['AITK_SAVED_TENSOR_PROBE'] = '5'
            self.assertEqual(SDTrainer._read_saved_tensor_probe_steps(), 5)
            os.environ['AITK_SAVED_TENSOR_PROBE'] = 'true'
            self.assertEqual(SDTrainer._read_saved_tensor_probe_steps(), 3)
            os.environ['AITK_SAVED_TENSOR_PROBE'] = '0'
            self.assertEqual(SDTrainer._read_saved_tensor_probe_steps(), 0)
            del os.environ['AITK_SAVED_TENSOR_PROBE']
            self.assertEqual(SDTrainer._read_saved_tensor_probe_steps(), 0)
        finally:
            if prev is None:
                os.environ.pop('AITK_SAVED_TENSOR_PROBE', None)
            else:
                os.environ['AITK_SAVED_TENSOR_PROBE'] = prev


if __name__ == '__main__':
    unittest.main()
