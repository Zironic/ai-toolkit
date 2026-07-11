import copy
import unittest

import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from extensions_built_in.diffusion_models.krea2.src.pipeline import predict_velocity


class Krea2ReferenceAttentionTests(unittest.TestCase):
    def _model(self, layers=2):
        torch.manual_seed(123)
        return SingleStreamDiT(
            SingleMMDiTConfig(
                features=32,
                tdim=16,
                txtdim=32,
                heads=4,
                kvheads=4,
                multiplier=1,
                layers=layers,
                patch=1,
                channels=4,
                txtlayers=1,
                txtheads=4,
                txtkvheads=4,
            )
        )

    def _inputs(self, batch=1, target_len=2, ref_mask=None):
        if ref_mask is None:
            ref_mask = torch.ones(batch, 1, dtype=torch.bool)
        ref_mask = ref_mask.bool()
        ref_len = ref_mask.shape[1]
        txt_len = 2

        torch.manual_seed(456)
        context = torch.randn(batch, txt_len, 1, 32)
        target = torch.randn(batch, target_len, 4)
        refs = torch.randn(batch, ref_len, 4)
        timestep = torch.linspace(0.2, 0.8, batch)

        live_len = txt_len + target_len
        pos = torch.zeros(batch, live_len + ref_len, 3)
        pos[:, txt_len:live_len, 1] = torch.arange(target_len)
        pos[:, live_len:, 0] = 1
        pos[:, live_len:, 1] = torch.arange(ref_len)

        live_mask = torch.ones(batch, live_len, dtype=torch.bool)
        capture_mask = torch.cat((live_mask, ref_mask), dim=1)
        return {
            "context": context,
            "target": target,
            "refs": refs,
            "timestep": timestep,
            "capture_pos": pos,
            "reuse_pos": pos[:, :live_len],
            "capture_mask": capture_mask,
            "live_mask": live_mask,
            "ref_mask": ref_mask,
        }

    def _assert_capture_reuse(self, inputs):
        model = self._model().eval()
        capture = []
        with torch.no_grad():
            captured = model(
                img=torch.cat((inputs["target"], inputs["refs"]), dim=1),
                context=inputs["context"],
                t=inputs["timestep"],
                pos=inputs["capture_pos"],
                mask=inputs["capture_mask"],
                reflen=inputs["refs"].shape[1],
                isolate_refs=True,
                ref_kv_capture=capture,
            )
            reused = model(
                img=inputs["target"],
                context=inputs["context"],
                t=inputs["timestep"],
                pos=inputs["reuse_pos"],
                mask=inputs["live_mask"],
                isolate_refs=True,
                ref_kv_cache=(capture, inputs["ref_mask"]),
            )

        self.assertEqual(len(capture), len(model.blocks))
        self.assertEqual(captured.shape, reused.shape)
        torch.testing.assert_close(captured, reused, rtol=1e-5, atol=1e-6)

    def test_all_true_mask_capture_and_reuse_are_equivalent(self):
        # Covers the all-valid path where _mask() returns None. Isolation and
        # cached-key concatenation must materialize their own dense query rows.
        self._assert_capture_reuse(self._inputs())

    def test_predict_velocity_cache_reuse_matches_capture_pass(self):
        model = self._model().eval()
        torch.manual_seed(789)
        latents = torch.randn(1, 4, 1, 2)
        context = torch.randn(1, 2, 32)
        text_mask = torch.ones(1, 2, dtype=torch.bool)
        ref_latents = [[torch.randn(4, 1, 1)]]
        cache = {"kv": None, "mask": None}

        with torch.no_grad():
            captured = predict_velocity(
                model,
                latents,
                torch.tensor([0.5]),
                context,
                text_mask,
                ref_latents=ref_latents,
                isolate_refs=True,
                ref_kv_cache=cache,
            )
            self.assertIsNotNone(cache["kv"])
            self.assertIsNotNone(cache["mask"])
            reused = predict_velocity(
                model,
                latents,
                torch.tensor([0.5]),
                context,
                text_mask,
                ref_latents=ref_latents,
                isolate_refs=True,
                ref_kv_cache=cache,
            )

        torch.testing.assert_close(captured, reused, rtol=1e-5, atol=1e-6)
    def test_padded_reference_capture_and_reuse_are_equivalent(self):
        # Sample 1 has a padded reference token. This checks per-sample cached
        # key masks as well as the all-live-token reuse mask.
        ref_mask = torch.tensor([[True, True], [True, False]])
        self._assert_capture_reuse(self._inputs(batch=2, ref_mask=ref_mask))

    def _backward(self, model, inputs, checkpointed, isolate_refs):
        model.train()
        if checkpointed:
            model.enable_gradient_checkpointing(keep_last=0)

        target = inputs["target"].detach().clone().requires_grad_(True)
        refs = inputs["refs"].detach().clone().requires_grad_(True)
        context = inputs["context"].detach().clone().requires_grad_(True)
        output = model(
            img=torch.cat((target, refs), dim=1),
            context=context,
            t=inputs["timestep"],
            pos=inputs["capture_pos"],
            mask=inputs["capture_mask"],
            reflen=refs.shape[1],
            isolate_refs=isolate_refs,
        )
        output.square().mean().backward()
        parameter_grads = {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        return (
            output.detach(),
            target.grad.detach(),
            refs.grad.detach(),
            context.grad.detach(),
            parameter_grads,
        )

    def test_reference_backward_checkpointing_matches_eager(self):
        inputs = self._inputs()
        for isolate_refs in (False, True):
            with self.subTest(isolate_refs=isolate_refs):
                eager_model = self._model()
                checkpointed_model = copy.deepcopy(eager_model)
                eager = self._backward(
                    eager_model,
                    inputs,
                    checkpointed=False,
                    isolate_refs=isolate_refs,
                )
                checkpointed = self._backward(
                    checkpointed_model,
                    inputs,
                    checkpointed=True,
                    isolate_refs=isolate_refs,
                )

                for eager_tensor, checkpointed_tensor in zip(
                    eager[:4], checkpointed[:4], strict=True
                ):
                    torch.testing.assert_close(
                        checkpointed_tensor, eager_tensor, rtol=1e-5, atol=1e-6
                    )

                eager_grads, checkpointed_grads = eager[4], checkpointed[4]
                self.assertEqual(eager_grads.keys(), checkpointed_grads.keys())
                for name in eager_grads:
                    eager_grad = eager_grads[name]
                    checkpointed_grad = checkpointed_grads[name]
                    self.assertEqual(
                        eager_grad is None,
                        checkpointed_grad is None,
                        msg=f"gradient presence differs for {name}",
                    )
                    if eager_grad is not None:
                        torch.testing.assert_close(
                            checkpointed_grad,
                            eager_grad,
                            rtol=1e-5,
                            atol=1e-6,
                            msg=lambda message, name=name: f"{name}: {message}",
                        )


if __name__ == "__main__":
    unittest.main()