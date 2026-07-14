"""Quantized Linear contract smoke (CUDA).

First-line diagnostic separating quantizer/runtime problems from model
discovery and model-forward problems. Runs a small synthetic block (attention
projection + MLP projections + norm parameter, positional and keyword args,
optional tuple output) through the storage/execution contract the arena
depends on:

    eager forward; compiled forward; compiled repeat (no new Dynamo frame);
    backward; non-reentrant checkpoint + backward; fully resident state;
    "streamed" state (canonical-byte round trip through host memory that
    never touches `.weight`).

Required qtypes: qfloat8 (Quanto QBytesTensor), convrot4 (the hard
OstrisLinear execution case), orbit4 (second Ostris backend, proving the
contract is OstrisLinear-generic). Other ConvRot qtypes get declaration-only
coverage via --declare-only.

Hard requirement is execution-path parity (resident vs streamed, eager vs
compiled, checkpoint vs plain, matching input/adapter gradients). Agreement
with the dense reference is REPORTED with backend-appropriate context, never
asserted.

Examples:

    venv\\Scripts\\python.exe scripts\\smoke_quantized_linear_cuda.py ^
      --qtype qfloat8 --adapter lora --compile --checkpoint
    venv\\Scripts\\python.exe scripts\\smoke_quantized_linear_cuda.py ^
      --qtype convrot4 --adapter dora --compile --checkpoint
    venv\\Scripts\\python.exe scripts\\smoke_quantized_linear_cuda.py ^
      --qtype convrot8 --declare-only
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.smoke_profiles import (  # noqa: E402
    assert_representation,
    audit_quantized_representation,
)

HIDDEN = 256
SEQ = 64


# ---------------------------------------------------------------------------
# Synthetic model
# ---------------------------------------------------------------------------

class SyntheticBlock(torch.nn.Module):
    """One attention-style projection, two MLP-style projections, a norm
    parameter, optional bias, positional + keyword arguments, and an optional
    tuple output -- the smallest module that exercises everything the arena
    block boundary must preserve."""

    def __init__(self, hidden: int = HIDDEN):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.ones(hidden))
        self.attn_proj = torch.nn.Linear(hidden, hidden, bias=True)
        self.mlp_in = torch.nn.Linear(hidden, hidden * 2, bias=False)
        self.mlp_out = torch.nn.Linear(hidden * 2, hidden, bias=True)

    def forward(self, hidden_states, shift, *, scale=1.0, return_tuple=False):
        x = hidden_states * self.norm + shift
        x = x + self.attn_proj(x) * scale
        x = x + self.mlp_out(torch.nn.functional.gelu(self.mlp_in(x)))
        if return_tuple:
            return (x, x.float().square().mean())
        return x


class SyntheticTransformer2DModel(torch.nn.Module):
    """Class name doubles as the LoRA target (target_lin_modules)."""

    def __init__(self, num_blocks: int = 2, hidden: int = HIDDEN):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            SyntheticBlock(hidden) for _ in range(num_blocks)
        )

    def forward(self, hidden_states, shift, *, scale=1.0, use_checkpoint=False):
        for block in self.blocks:
            if use_checkpoint:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block, hidden_states, shift, scale=scale, use_reentrant=False
                )
            else:
                hidden_states = block(hidden_states, shift, scale=scale)
        return hidden_states


class _SyntheticBaseModel:
    """Minimal base_model shim for LoRASpecialNetwork (arch dispatch, LoKr
    format flag, block-name lookup)."""

    arch = "synthetic"
    use_old_lokr_format = False

    def __init__(self, device, dtype):
        self.device_torch = torch.device(device)
        self.torch_dtype = dtype

    def get_transformer_block_names(self):
        return None


# ---------------------------------------------------------------------------
# Quantization + canonical state
# ---------------------------------------------------------------------------

def _quantize_synthetic(model: torch.nn.Module, qtype: str) -> None:
    from toolkit.util.quantize import aotype, get_qtype, ostristype

    resolved = get_qtype(qtype)
    if isinstance(resolved, aotype):
        raise SystemExit(
            f"qtype {qtype!r} resolves to torchao; this contract smoke covers "
            "Quanto and OstrisLinear backends (torchao is the existing Krea2 "
            "baseline, scripts/smoke_krea2_train_cuda.py)"
        )
    if isinstance(resolved, ostristype):
        from toolkit.util.ostris_quant import convert_linear_to_ostris

        converted = 0
        eligible = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                eligible += 1
                if convert_linear_to_ostris(module, resolved.quantizer):
                    converted += 1
        if converted != eligible:
            raise SystemExit(
                f"{qtype}: only {converted}/{eligible} eligible Linears were "
                "converted to OstrisLinear -- a partially-converted model "
                "would pass representation checks vacuously"
            )
        return
    # Quanto path (qfloat8 and friends).
    from optimum.quanto import freeze

    from toolkit.util.quantize import quantize

    quantize(model, weights=resolved)
    freeze(model)


def _canonical_state(model: torch.nn.Module) -> dict:
    """Canonical bytes: registered compressed buffers (+ bias) for Ostris,
    qdata/scale (+ bias) for Quanto. NEVER reads OstrisLinear.weight or calls
    dequantize_weight -- that is the contract under test."""
    from toolkit.util.ostris_quant import OstrisLinear

    state = {}
    for name, module in model.named_modules():
        if isinstance(module, OstrisLinear):
            assert "weight" not in module._parameters, (
                f"{name}: OstrisLinear still exposes a weight Parameter"
            )
            entry = {"kind": "ostris", "buffers": {}, "bias": None}
            for buf_name, buf in module.named_buffers(recurse=False):
                entry["buffers"][buf_name] = buf.detach().cpu().clone()
            if module.bias is not None:
                entry["bias"] = module.bias.detach().cpu().clone()
            if not entry["buffers"]:
                raise SystemExit(
                    f"{name}: OstrisLinear has no registered quantizer buffers"
                )
            state[name] = entry
        elif isinstance(module, torch.nn.Linear):
            weight = module._parameters.get("weight", None)
            if weight is None:
                continue
            data = weight.data
            qdata = getattr(data, "_data", None)
            scale = getattr(data, "_scale", None)
            if qdata is None or scale is None:
                continue  # non-quantized linear (e.g. adapter internals)
            entry = {
                "kind": "quanto",
                "qdata": qdata.detach().cpu().clone(),
                "scale": scale.detach().cpu().clone(),
                "size": tuple(data.size()),
                "stride": tuple(data.stride()),
                "bias": (
                    module.bias.detach().cpu().clone()
                    if module.bias is not None
                    else None
                ),
            }
            state[name] = entry
    if not state:
        raise SystemExit("no quantized linear state found to canonicalize")
    return state


def _restore_canonical_state(model, state, qtype, device) -> None:
    """Movement path: rebuild device state from the host canonical copy
    without reconstructing an ordinary weight."""
    modules = dict(model.named_modules())
    for name, entry in state.items():
        module = modules[name]
        if entry["kind"] == "ostris":
            for buf_name, host in entry["buffers"].items():
                module._buffers[buf_name] = host.to(device)
            if entry["bias"] is not None:
                module.bias.data = entry["bias"].to(device)
        else:
            from optimum.quanto.tensor.qbytes import QBytesTensor

            from toolkit.util.quantize import get_qtype

            wrapper = QBytesTensor(
                get_qtype(qtype),
                0,
                torch.Size(entry["size"]),
                entry["stride"],
                entry["qdata"].to(device),
                entry["scale"].to(device),
                requires_grad=False,
            )
            module.weight = torch.nn.Parameter(wrapper, requires_grad=False)
            if entry["bias"] is not None:
                module.bias.data = entry["bias"].to(device)


@contextlib.contextmanager
def _count_dequantize_weight():
    """Counter over OstrisLinear.dequantize_weight for the no-materialization
    assertions (preparation/movement must not touch it; execution goes through
    the quantizer forward, not the .weight property)."""
    from toolkit.util.ostris_quant import OstrisLinear

    calls = {"count": 0}
    original = OstrisLinear.dequantize_weight

    def counted(self):
        calls["count"] += 1
        return original(self)

    OstrisLinear.dequantize_weight = counted
    try:
        yield calls
    finally:
        OstrisLinear.dequantize_weight = original


@contextlib.contextmanager
def _count_quantizer_forward():
    from toolkit.util.ostris_quant import OstrisQuantizer

    calls = {"count": 0}

    def make_counted(fn):
        def counted(self, module, x):
            calls["count"] += 1
            return fn(self, module, x)

        return counted

    overridden = []
    for cls in [OstrisQuantizer, *_all_subclasses(OstrisQuantizer)]:
        if "forward" in cls.__dict__:
            cls_original = cls.__dict__["forward"]
            cls.forward = make_counted(cls_original)
            overridden.append((cls, cls_original))
    if not overridden:
        base_original = OstrisQuantizer.forward
        OstrisQuantizer.forward = make_counted(base_original)
        overridden.append((OstrisQuantizer, base_original))
    try:
        yield calls
    finally:
        for cls, fn in overridden:
            cls.forward = fn


def _all_subclasses(cls):
    out = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def _apply_adapter(
    model,
    device,
    dtype,
    variant,
    rank=8,
    alpha=8.0,
    *,
    target_lin_modules=None,
    full_if_contains=None,
):
    from toolkit.config_modules import NetworkConfig
    from toolkit.lora_special import LoRASpecialNetwork

    base_shim = _SyntheticBaseModel(device, dtype)
    network_type = "lora" if variant == "full" else variant
    network_config = NetworkConfig(
        type=network_type, linear=rank, linear_alpha=alpha, transformer_only=True
    )
    network = LoRASpecialNetwork(
        text_encoder=None,
        unet=model,
        lora_dim=rank,
        multiplier=1.0,
        alpha=alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        full_if_contains=(
            list(full_if_contains or ["blocks.0.attn_proj"])
            if variant == "full"
            else []
        ),
        transformer_only=True,
        is_transformer=True,
        target_lin_modules=list(
            target_lin_modules or ["SyntheticTransformer2DModel"]
        ),
        base_model=base_shim,
    )
    network.force_to(device, dtype=torch.float32)
    network._update_torch_multiplier()
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.can_merge_in = False
    network.prepare_grad_etc(None, model)
    return network


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _err(a: torch.Tensor, b: torch.Tensor) -> dict:
    a = a.detach().float()
    b = b.detach().float()
    diff = (a - b).abs()
    denom = b.abs().clamp_min(1e-6)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
    }


def _require_parity(label, a, b, atol, results, failures):
    err = _err(a, b)
    results[label] = err
    if err["max_abs"] > atol:
        failures.append(f"{label}: max_abs {err['max_abs']:.3e} > {atol:.1e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="CUDA contract smoke for Quanto / OstrisLinear quantized "
        "linears. Do not hook this up to pytest (process-global custom ops)."
    )
    parser.add_argument("--qtype", required=True)
    parser.add_argument(
        "--adapter",
        choices=("none", "lora", "lokr", "dora", "full"),
        default="lora",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument(
        "--declare-only",
        action="store_true",
        help="declaration/schema-only pass (representation + canonical bytes, "
        "no execution) for the remaining ConvRot qtypes",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument(
        "--compile-parity-atol",
        type=float,
        default=0.25,
        help="eager-vs-compiled tolerance. Inductor legitimately fuses and "
        "reorders bf16 math, so compiled outputs are numerically different, "
        "not bitwise-equal (observed max_abs ~0.094 for qfloat8 at the "
        "default sizes). Checkpoint and streamed comparisons stay exact.",
    )
    parser.add_argument("--output-json", default=None)
    from scripts.smoke_runtime import add_contention_args

    add_contention_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    from scripts.smoke_runtime import fail_if_vram_contended
    from toolkit.train_tools import get_torch_dtype

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("this smoke is CUDA-only; pass --device cuda")
    fail_if_vram_contended(device, ignore_contention=args.ignore_contention)
    dtype = get_torch_dtype(args.dtype)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    failures: list[str] = []
    results: dict = {
        "qtype": args.qtype,
        "adapter": args.adapter,
        "compile": bool(args.compile),
        "checkpoint": bool(args.checkpoint),
        "declare_only": bool(args.declare_only),
        "parity": {},
        "dense_reference": {},
    }

    # Dense reference first; the quantized model starts from cloned state.
    dense = SyntheticTransformer2DModel().to(device, dtype)
    dense.requires_grad_(False)
    quantized = copy.deepcopy(dense)

    with _count_dequantize_weight() as prep_dequant:
        _quantize_synthetic(quantized, args.qtype)
        counts = audit_quantized_representation(quantized)
        canonical = _canonical_state(quantized)
    results["representation"] = counts
    results["canonical_entries"] = len(canonical)
    results["dequantize_weight_calls_preparation"] = prep_dequant["count"]
    failures.extend(assert_representation(counts, args.qtype))
    if prep_dequant["count"] > 0:
        failures.append(
            "canonical-byte preparation called OstrisLinear.dequantize_weight "
            f"{prep_dequant['count']} times (must read registered buffers only)"
        )

    if args.declare_only:
        results["failures"] = failures
        print(json.dumps(results, indent=2, sort_keys=True, default=str))
        if failures:
            raise SystemExit("\n".join(["DECLARATION FAILURES:", *failures]))
        print(f"[smoke] {args.qtype}: declaration/schema checks passed")
        return

    network = None
    if args.adapter != "none":
        network = _apply_adapter(quantized, device, dtype, args.adapter)
        quantized.train()
    adapter_ctx = network if network is not None else contextlib.nullcontext()

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(args.batch_size, SEQ, HIDDEN, generator=gen).to(
        device, dtype
    )
    shift = torch.randn(args.batch_size, SEQ, HIDDEN, generator=gen).to(
        device, dtype
    )

    def run(model, *, use_checkpoint=False, need_grad=False, do_backward=False):
        # Backward MUST stay inside the network context: the LoRA multiplier
        # is zeroed on exit, and a checkpoint recompute outside the context
        # builds a different graph (CheckpointError: saved-tensor mismatch).
        x = hidden.detach().clone().requires_grad_(need_grad)
        with adapter_ctx:
            out = model(x, shift, scale=1.0, use_checkpoint=use_checkpoint)
            if do_backward:
                out.float().square().mean().backward()
        return x, out

    # Dense reference (report-only).
    with torch.no_grad():
        _, dense_out = run(dense)

    # --- eager forward + backward (resident) ---
    with _count_quantizer_forward() as qfwd, _count_dequantize_weight() as exec_deq:
        x_eager, out_eager = run(quantized, need_grad=True, do_backward=True)
    results["quantizer_forward_calls"] = qfwd["count"]
    results["dequantize_weight_calls_execution"] = exec_deq["count"]
    if args.qtype.startswith(("convrot", "orbit")) and qfwd["count"] == 0:
        failures.append("OstrisLinear quantizer forward was never exercised")
    if not torch.isfinite(out_eager.detach().float()).all():
        failures.append("eager output is non-finite")
    if x_eager.grad is None or not torch.isfinite(x_eager.grad.float()).all():
        failures.append("eager input gradient is missing or non-finite")
    eager_input_grad = (
        x_eager.grad.detach().float().cpu().clone() if x_eager.grad is not None else None
    )
    adapter_grads_eager = {}
    if network is not None:
        adapter_grads_eager = {
            n: p.grad.detach().float().cpu().clone()
            for n, p in network.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        if not adapter_grads_eager:
            failures.append("no adapter gradients produced on the eager path")
        for n, g in adapter_grads_eager.items():
            if not torch.isfinite(g).all():
                failures.append(f"adapter gradient {n} is non-finite")
        network.zero_grad(set_to_none=True)
    results["dense_reference"]["eager_vs_dense"] = _err(out_eager, dense_out)

    # Eager inference-branch reference: ConvRot/Orbit forwards branch on
    # x.requires_grad (STE activation fake-quant vs plain dequant matmul), so
    # no-grad runs (compiled, streamed) must be compared against a no-grad
    # eager baseline, not the training-branch output.
    with torch.no_grad():
        _, out_eager_infer = run(quantized)

    # --- non-reentrant checkpoint + backward ---
    if args.checkpoint:
        x_ckpt, out_ckpt = run(
            quantized, use_checkpoint=True, need_grad=True, do_backward=True
        )
        _require_parity(
            "checkpoint_vs_eager_output", out_ckpt, out_eager, args.parity_atol,
            results["parity"], failures,
        )
        if x_ckpt.grad is None:
            failures.append("checkpoint path produced no input gradient")
        elif eager_input_grad is not None:
            _require_parity(
                "checkpoint_vs_eager_input_grad",
                x_ckpt.grad,
                eager_input_grad.to(device),
                args.parity_atol,
                results["parity"],
                failures,
            )
        if network is not None:
            ckpt_grads = {
                n: p.grad.detach().float().cpu()
                for n, p in network.named_parameters()
                if p.requires_grad and p.grad is not None
            }
            for n, g in adapter_grads_eager.items():
                if n not in ckpt_grads:
                    failures.append(f"adapter grad {n} missing on checkpoint path")
                else:
                    _require_parity(
                        f"checkpoint_adapter_grad::{n}",
                        ckpt_grads[n],
                        g,
                        args.parity_atol,
                        results["parity"],
                        failures,
                    )
            network.zero_grad(set_to_none=True)

    # --- compiled forward + repeat ---
    if args.compile:
        compiled_calls = torch.compile(quantized, dynamic=False)
        counters = torch._dynamo.utils.counters
        with torch.no_grad():
            _, out_compiled = run(compiled_calls)
        frames_after_first = counters["frames"].get("total", 0)
        with torch.no_grad():
            _, out_repeat = run(compiled_calls)
        frames_after_repeat = counters["frames"].get("total", 0)
        results["compile_frames_first"] = frames_after_first
        results["compile_new_frames_on_repeat"] = (
            frames_after_repeat - frames_after_first
        )
        if frames_after_repeat != frames_after_first:
            failures.append(
                "compiled repeat created "
                f"{frames_after_repeat - frames_after_first} new Dynamo frames "
                "for an identical shape"
            )
        _require_parity(
            "compiled_vs_eager_output",
            out_compiled,
            out_eager_infer,
            args.compile_parity_atol,
            results["parity"],
            failures,
        )
        _require_parity(
            "compiled_repeat_vs_compiled",
            out_repeat,
            out_compiled,
            0.0,
            results["parity"],
            failures,
        )

    # --- streamed state: canonical round trip, then bitwise output parity ---
    with _count_dequantize_weight() as move_deq:
        _restore_canonical_state(quantized, canonical, args.qtype, device)
    results["dequantize_weight_calls_movement"] = move_deq["count"]
    if move_deq["count"] > 0:
        failures.append(
            "canonical restore (movement path) called dequantize_weight "
            f"{move_deq['count']} times"
        )
    with torch.no_grad():
        _, out_streamed = run(quantized)
    _require_parity(
        "streamed_vs_resident_output",
        out_streamed,
        out_eager_infer,
        0.0,
        results["parity"],
        failures,
    )

    results["failures"] = failures
    payload = json.dumps(results, indent=2, sort_keys=True, default=str)
    print(payload)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
        print(f"[smoke] wrote {out}")
    if failures:
        raise SystemExit("\n".join(["CONTRACT FAILURES:", *failures]))
    print(f"[smoke] {args.qtype} quantized-linear contract passed")


if __name__ == "__main__":
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_quantized_linear_cuda", main))
