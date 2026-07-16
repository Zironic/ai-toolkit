"""Phase 0 seam proof for the generic block dispatcher (ticket b1a13d2).

Proves (or kills) the core mechanism of the generic block dispatcher plan
(tasks/done/GENERIC_BLOCK_DISPATCHER_PLAN.md): a real Krea SingleStreamBlock
whose ordinary forward executes from explicitly substituted state via
OriginalBlockInvoker + torch.func.functional_call, eager and compiled, with
backward and model-owned non-reentrant checkpoint recomputation.

Fidelity choices:
  - The real SingleStreamBlock class, random weights (the plan allows this).
  - Production quantization: torchao Float8WeightOnlyConfig via quantize_,
    exactly what qtype "float8" does in toolkit/util/quantize.py.
  - Destructive-commit simulation: after storage capture the module's own
    weights are replaced with zero-payload wrappers, so a correct result is
    only possible if substitution actually reached the executed math.
  - Fresh storage tensor identities per call (simulating ring slots), so the
    compiled path is exercised the way the arena would drive it.
  - suppress_errors is forced False: quantized production runs set it True,
    which would silently mask a functional_call trace failure as an eager
    fallback and corrupt the go/no-go signal.

Usage (Phase 0 gate = the first two):
    venv\\Scripts\\python.exe scripts\\seam_proof_dispatcher.py --variant torchao --lora
    venv\\Scripts\\python.exe scripts\\seam_proof_dispatcher.py --variant convrot4
    venv\\Scripts\\python.exe scripts\\seam_proof_dispatcher.py --variant convrot8
    venv\\Scripts\\python.exe scripts\\seam_proof_dispatcher.py --variant dense
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from extensions_built_in.diffusion_models.krea2.src.mmdit import (  # noqa: E402
    PositionalEncoding,
    SingleStreamBlock,
)

GIB = 1024 ** 3


# --------------------------------------------------------------------------
# Proposed production shapes (plan: "Execution design"). These are the exact
# mechanisms under proof; do not simplify them into plain function calls.
# --------------------------------------------------------------------------


class OriginalBlockInvoker(torch.nn.Module):
    """Owns the block and calls its preserved (pre-dispatcher) forward."""

    def __init__(self, block, saved_forward):
        super().__init__()
        self.block = block
        self._saved_forward = saved_forward

    def forward(self, *args, **kwargs):
        return self._saved_forward(*args, **kwargs)


def capture_reconstruction(value):
    """Freeze everything needed to rebuild a tensor-subclass wrapper.

    Captured once at declaration time (quantization-owned in production);
    never rediscovered during execution.
    """
    value = value.data if isinstance(value, torch.nn.Parameter) else value
    names, ctx = value.__tensor_flatten__()
    return {
        "cls": type(value),
        "names": tuple(names),
        "ctx": ctx,
        "size": tuple(value.shape),
        "stride": tuple(value.stride()),
    }


def rebuild_wrapper(recon, inner_tensors):
    return recon["cls"].__tensor_unflatten__(
        dict(zip(recon["names"], inner_tensors, strict=True)),
        recon["ctx"],
        torch.Size(recon["size"]),
        recon["size"] and recon["stride"],
    )


LEAF_PATHS = (
    "attn.wq",
    "attn.wk",
    "attn.wv",
    "attn.gate",
    "attn.wo",
    "mlp.gate",
    "mlp.up",
    "mlp.down",
)


def _child(module, path):
    for part in path.split("."):
        module = getattr(module, part)
    return module


def declare_block_storage(block):
    """Capture per-leaf ordered storage + replacement plan (CPU 'arena').

    Returns a list of dicts:
      state_name: functional_call target under the invoker ("block." prefix)
      tensors:    ordered CPU storage leaves
      recon:      wrapper rebuild spec or None for plain tensors

    OstrisLinear leaves have no weight parameter at all: the quantized state
    lives in backend-registered buffers, and each buffer becomes its own
    functional_call substitution target.
    """
    from toolkit.quantization.storage import linear_storage_binding
    from toolkit.util.ostris_quant import OstrisLinear

    entries = []
    for path in LEAF_PATHS:
        linear = _child(block, path)
        if linear.bias is not None:
            raise SystemExit("Krea SingleStreamBlock linears are bias-free")
        if isinstance(linear, OstrisLinear):
            for name, buf in linear.named_buffers(recurse=False):
                entries.append(
                    {
                        "state_name": f"block.{path}.{name}",
                        "tensors": (buf.detach().to("cpu", copy=True),),
                        "recon": None,
                    }
                )
            continue
        binding = linear_storage_binding(linear.weight, linear.bias)
        weight_value = linear.weight.data
        is_wrapped = binding.weight_leaf_count > 1
        entries.append(
            {
                "state_name": f"block.{path}.weight",
                "tensors": tuple(
                    item.tensor.detach().to("cpu", copy=True)
                    for item in binding.tensors[: binding.weight_leaf_count]
                ),
                "recon": capture_reconstruction(weight_value) if is_wrapped else None,
            }
        )
    return entries


def corrupt_resident_weights(block, entries, device):
    """Simulate destructive canonical commit: module state becomes garbage.

    After this, only true state substitution can produce correct outputs.
    """
    for entry in entries:
        target = entry["state_name"][len("block."):]
        mod_path, _, attr = target.rpartition(".")
        module = _child(block, mod_path)
        if entry["recon"] is not None:
            garbage = rebuild_wrapper(
                entry["recon"],
                tuple(
                    torch.zeros_like(t, device=device) for t in entry["tensors"]
                ),
            )
            setattr(module, attr, torch.nn.Parameter(garbage, requires_grad=False))
        elif attr == "weight":
            garbage = torch.zeros_like(entry["tensors"][0], device=device)
            setattr(module, attr, torch.nn.Parameter(garbage, requires_grad=False))
        else:
            # registered buffer: plain attribute assignment routes to _buffers
            setattr(
                module, attr, torch.zeros_like(entry["tensors"][0], device=device)
            )


def stream_storage(entries, device):
    """Fresh GPU copies of every storage leaf (fresh identity per call)."""
    return [
        tuple(t.to(device, non_blocking=True) for t in entry["tensors"])
        for entry in entries
    ]


def build_replacement_state(entries, streamed):
    state = {}
    for entry, tensors in zip(entries, streamed, strict=True):
        if entry["recon"] is None:
            state[entry["state_name"]] = tensors[0]
        else:
            state[entry["state_name"]] = rebuild_wrapper(entry["recon"], tensors)
    return state


def make_kernel(invoker, entries):
    """The stateless block kernel: explicit storage in, original output out."""

    def kernel(streamed, args, kwargs):
        state = build_replacement_state(entries, streamed)
        return torch.func.functional_call(
            invoker,
            state,
            args,
            kwargs,
            strict=False,
            tie_weights=False,
        )

    return kernel


class Dispatcher:
    """Installed as block.forward; routes through the kernel with streamed state."""

    def __init__(self, kernel, entries, device):
        self.kernel = kernel
        self.entries = entries
        self.device = device
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        streamed = stream_storage(self.entries, self.device)
        return self.kernel(streamed, args, kwargs)


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def _frames():
    return torch._dynamo.utils.counters["frames"].get("total", 0)


def _graph_breaks():
    return sum(torch._dynamo.utils.counters["graph_break"].values())


def _diff(a, b):
    return (a.detach().float() - b.detach().float()).abs().max().item()


def build_block(args, device, dtype):
    torch.manual_seed(args.seed)
    block = SingleStreamBlock(
        features=args.features,
        heads=args.heads,
        multiplier=args.multiplier,
    )
    # Real init is checkpoint-loaded; random zeros-mean weights make the
    # residual stream degenerate, so give the linears a small nonzero init.
    for path in LEAF_PATHS:
        torch.nn.init.normal_(_child(block, path).weight, std=0.02)
    block.to(device=device, dtype=dtype)
    if args.variant == "torchao":
        from torchao.quantization.quant_api import quantize_ as torchao_quantize_

        from toolkit.util.quantize import torchao_qtypes

        torchao_quantize_(block, torchao_qtypes["float8"])
    elif args.variant in ("convrot4", "convrot8"):
        from toolkit.util.ostris_quant import (
            convert_linear_to_ostris,
            get_ostris_quantizer,
        )

        quantizer = get_ostris_quantizer(args.variant)
        for path in LEAF_PATHS:
            if not convert_linear_to_ostris(_child(block, path), quantizer):
                raise SystemExit(f"{args.variant} refused to quantize {path}")
    block.requires_grad_(False)
    block.train()
    return block


class _StubNetwork:
    """The attribute surface LoRAModule's production forward actually reads.

    The seam under proof is the monkeypatched module forward + functional_call
    state substitution; network-level bookkeeping (merging, LoRM, gating) is
    inert here and pinned to its production-default values.
    """

    network_type = "lora"
    is_lorm = False
    is_active = True
    is_merged_in = False
    _multiplier = 1.0
    vector_gates = None
    is_assistant_adapter = False

    def __init__(self, device):
        # production _update_torch_multiplier builds a 1-element tensor
        self.torch_multiplier = torch.tensor([1.0], device=device)


def apply_lora(block, device, rank, seed):
    """Install production LoRAModule wrappers on every leaf linear.

    Uses the real monkeypatch mechanism (org_forward preserved, module.forward
    replaced) and the real functional_forward math. lora params stay fp32 and
    OUTSIDE the invoker module tree, exactly like production network install.
    Both mats get nonzero init so adapter gradients are non-trivial.
    """
    from toolkit.lora_special import LoRAModule

    network = _StubNetwork(device)
    gen = torch.Generator().manual_seed(seed + 2)
    modules = {}
    for path in LEAF_PATHS:
        linear = _child(block, path)
        lora = LoRAModule(
            f"seam${path.replace('.', '_')}",
            linear,
            multiplier=1.0,
            lora_dim=rank,
            alpha=float(rank),
            network=network,
        )
        with torch.no_grad():
            for mat in (lora.lora_down.weight, lora.lora_up.weight):
                mat.copy_(torch.randn(mat.shape, generator=gen) * 0.02)
        lora.to(device=device, dtype=torch.float32)
        lora.requires_grad_(True)
        lora.apply_to()
        modules[path] = lora
    return network, modules


def adapter_grads(lora_modules):
    grads = {}
    for path, lora in lora_modules.items():
        for name, param in lora.named_parameters():
            grad = param.grad
            grads[f"{path}.{name}"] = None if grad is None else grad.detach().clone()
            param.grad = None
    return grads


def adapter_grad_diff(a, b):
    worst = 0.0
    for key, ga in a.items():
        gb = b.get(key)
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            return float("inf")
        worst = max(worst, _diff(ga, gb))
    return worst


def make_inputs(args, device, dtype, headdim):
    gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    x = torch.randn(args.batch, args.seq, args.features, generator=gen).to(
        device, dtype
    )
    vec = torch.randn(args.batch, 1, 6 * args.features, generator=gen).to(
        device, dtype
    )
    # Real rope frequencies through the model's own PositionalEncoding, on a
    # 2-axis grid split across the head dimension like the production config.
    ax = headdim // 2
    posemb = PositionalEncoding(dim=headdim, axdims=[ax, ax], theta=1e3)
    side = int(args.seq ** 0.5) + 1
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(side), torch.arange(side), indexing="ij"
        ),
        dim=-1,
    ).reshape(1, -1, 2)[:, : args.seq, :]
    freqs = posemb(coords.to(device=device, dtype=torch.float32))
    freqs = freqs.expand(args.batch, *freqs.shape[1:]) if freqs.shape[0] == 1 else freqs
    return x, vec, freqs


def run_case(name, fn, results):
    try:
        row = fn()
        row["case"] = name
        row["ok"] = bool(row.get("ok", True))
    except Exception as error:  # the whole point is catching seam failures
        row = {"case": name, "ok": False, "error": repr(error)}
    results.append(row)
    print(json.dumps(row, indent=2, sort_keys=True, default=str))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("dense", "torchao", "convrot4", "convrot8"),
        default="torchao",
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "install production LoRAModule wrappers on every leaf before the "
            "forward is saved (the Phase 0 primary case is --variant torchao "
            "--lora)"
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--eager-only",
        action="store_true",
        help=(
            "skip the compiled cases (for variants whose plain resident "
            "execution already fails under torch.compile on this hardware, "
            "e.g. convrot4 on sm_89 -- a pre-existing limitation orthogonal "
            "to the dispatcher seam)"
        ),
    )
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help=(
            "max-abs output tolerance vs the resident baseline; 0 demands "
            "bitwise identity (same wrapper class, same kernels)"
        ),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (CPU compile results silently lie here)")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    headdim = args.features // args.heads

    # Honest failures only (see module docstring).
    torch._dynamo.config.suppress_errors = False

    block = build_block(args, device, dtype)
    x, vec, freqs = make_inputs(args, device, dtype, headdim)

    # ---- declaration BEFORE network install (production prep order) -------
    entries = declare_block_storage(block)

    # ---- network install: saved forward must include installed adapters ---
    lora_modules = {}
    if args.lora:
        _network, lora_modules = apply_lora(block, device, args.lora_rank, args.seed)

    # ---- resident baseline, AFTER install, BEFORE corruption --------------
    def baseline_forward(inp):
        return block(inp, vec, freqs, None)

    x_base = x.clone().requires_grad_(True)
    out_base = baseline_forward(x_base)
    out_base.float().square().mean().backward()
    grad_base = x_base.grad.detach().clone()
    agrad_base = adapter_grads(lora_modules)

    x_ckpt = x.clone().requires_grad_(True)
    out_ckpt_base = checkpoint(block, x_ckpt, vec, freqs, None, use_reentrant=False)
    out_ckpt_base.float().square().mean().backward()
    grad_ckpt_base = x_ckpt.grad.detach().clone()
    agrad_ckpt_base = adapter_grads(lora_modules)

    # Compiled resident baseline: the compiled dispatcher arm must be compared
    # against compiled math, not eager math -- bf16 Inductor fusion changes
    # rounding order and an eager comparison measures that, not the seam.
    original_forward = block.forward
    if not args.eager_only:
        compiled_resident = torch.compile(
            lambda inp: original_forward(inp, vec, freqs, None),
            mode="default",
            fullgraph=False,
            dynamic=False,
        )
        x_cbase = x.clone().requires_grad_(True)
        out_cbase = compiled_resident(x_cbase)
        out_cbase.float().square().mean().backward()
        grad_cbase = x_cbase.grad.detach().clone()
        agrad_cbase = adapter_grads(lora_modules)
        out_cbase = out_cbase.detach()
        torch.cuda.synchronize(device)

    # ---- destructive commit, dispatcher install ---------------------------
    saved_forward = block.forward  # bound class method, pre-dispatcher
    corrupt_resident_weights(block, entries, device)

    # Corruption proof: the module's own state must now be unusable.
    with torch.no_grad():
        corrupted = baseline_forward(x)
    corruption_effective = _diff(corrupted, out_base) > 1e-3

    invoker = OriginalBlockInvoker(block, saved_forward)
    kernel = make_kernel(invoker, entries)
    dispatcher = Dispatcher(kernel, entries, device)
    block.forward = dispatcher

    results = []
    results.append(
        {
            "case": "setup",
            "ok": bool(corruption_effective),
            "variant": args.variant,
            "corruption_effective": bool(corruption_effective),
            "leaf_class": type(_child(block, "attn.wq")).__name__,
            "lora": bool(args.lora),
            "storage_entries": len(entries),
            "torch": torch.__version__,
        }
    )
    print(json.dumps(results[-1], indent=2, sort_keys=True))

    tol = args.tolerance

    # ---- case: eager forward + backward through the dispatcher -----------
    def case_eager():
        x_arm = x.clone().requires_grad_(True)
        out = block(x_arm, vec, freqs, None)  # dispatcher path
        out.float().square().mean().backward()
        agrad_diff = adapter_grad_diff(adapter_grads(lora_modules), agrad_base)
        return {
            "ok": (
                torch.isfinite(out).all().item()
                and _diff(out, out_base) <= tol
                and _diff(x_arm.grad, grad_base) <= tol
                and agrad_diff <= tol
            ),
            "out_max_abs_diff": _diff(out, out_base),
            "grad_max_abs_diff": _diff(x_arm.grad, grad_base),
            "adapter_grad_max_abs_diff": agrad_diff,
            "finite": torch.isfinite(out).all().item(),
        }

    run_case("eager_forward_backward", case_eager, results)

    # ---- case: model-owned non-reentrant checkpoint (eager kernel) -------
    def case_checkpoint_eager():
        calls_before = dispatcher.calls
        x_arm = x.clone().requires_grad_(True)
        out = checkpoint(block, x_arm, vec, freqs, None, use_reentrant=False)
        out.float().square().mean().backward()
        redispatched = dispatcher.calls - calls_before
        agrad_diff = adapter_grad_diff(adapter_grads(lora_modules), agrad_ckpt_base)
        return {
            "ok": (
                _diff(out, out_ckpt_base) <= tol
                and _diff(x_arm.grad, grad_ckpt_base) <= tol
                and agrad_diff <= tol
                and redispatched == 2
            ),
            "out_max_abs_diff": _diff(out, out_ckpt_base),
            "grad_max_abs_diff": _diff(x_arm.grad, grad_ckpt_base),
            "adapter_grad_max_abs_diff": agrad_diff,
            "dispatcher_calls": redispatched,  # forward + recompute
        }

    run_case("checkpoint_recompute_eager", case_checkpoint_eager, results)

    if args.eager_only:
        return _finish(args, results)

    # ---- case: compiled kernel, forward + backward ------------------------
    compiled_kernel = torch.compile(kernel, mode="default", fullgraph=False, dynamic=False)
    dispatcher.kernel = compiled_kernel

    def case_compiled():
        frames0, breaks0 = _frames(), _graph_breaks()
        x_arm = x.clone().requires_grad_(True)
        out = block(x_arm, vec, freqs, None)
        out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        warm_frames = _frames() - frames0
        agrad_diff = adapter_grad_diff(adapter_grads(lora_modules), agrad_cbase)
        # Repeated calls with fresh storage identities must not retrace.
        frames1 = _frames()
        for _ in range(3):
            x_rep = x.clone().requires_grad_(True)
            out = block(x_rep, vec, freqs, None)
            out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        steady_frames = _frames() - frames1
        adapter_grads(lora_modules)  # clear accumulation from the reps
        return {
            "ok": (
                _diff(out, out_cbase) <= max(tol, 5e-3)
                and _diff(x_rep.grad, grad_cbase) <= max(tol, 5e-3)
                and agrad_diff <= max(tol, 5e-3)
                and steady_frames == 0
            ),
            "out_max_abs_diff_vs_compiled": _diff(out, out_cbase),
            "grad_max_abs_diff_vs_compiled": _diff(x_rep.grad, grad_cbase),
            "adapter_grad_max_abs_diff_vs_compiled": agrad_diff,
            "out_max_abs_diff_vs_eager": _diff(out, out_base),
            "warm_compile_frames": warm_frames,
            "steady_new_frames": steady_frames,
            "graph_breaks_added": _graph_breaks() - breaks0,
        }

    run_case("compiled_forward_backward", case_compiled, results)

    # ---- case: checkpoint OUTSIDE, compiled kernel INSIDE (ordering flip) -
    def case_checkpoint_compiled():
        # warm already done above; measure recompute retraces specifically
        x_warm = x.clone().requires_grad_(True)
        out = checkpoint(block, x_warm, vec, freqs, None, use_reentrant=False)
        out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        adapter_grads(lora_modules)  # clear warm-call accumulation
        frames0 = _frames()
        calls0 = dispatcher.calls
        x_arm = x.clone().requires_grad_(True)
        out = checkpoint(block, x_arm, vec, freqs, None, use_reentrant=False)
        out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        agrad_diff = adapter_grad_diff(adapter_grads(lora_modules), agrad_cbase)
        return {
            "ok": (
                _diff(out, out_cbase) <= max(tol, 5e-3)
                and _diff(x_arm.grad, grad_cbase) <= max(tol, 5e-3)
                and agrad_diff <= max(tol, 5e-3)
                and (_frames() - frames0) == 0
                and (dispatcher.calls - calls0) == 2
            ),
            "out_max_abs_diff_vs_compiled": _diff(out, out_cbase),
            "grad_max_abs_diff_vs_compiled": _diff(x_arm.grad, grad_cbase),
            "adapter_grad_max_abs_diff_vs_compiled": agrad_diff,
            "steady_new_frames": _frames() - frames0,
            "dispatcher_calls": dispatcher.calls - calls0,
        }

    run_case("checkpoint_outside_compile_inside", case_checkpoint_compiled, results)

    return _finish(args, results)


def _finish(args, results):
    failed = [r["case"] for r in results if not r["ok"]]
    summary = {
        "case": "summary",
        "ok": not failed,
        "variant": args.variant,
        "lora": bool(args.lora),
        "eager_only": bool(args.eager_only),
        "failed_cases": failed,
        "verdict": "GO" if not failed else "NO-GO (see failed_cases)",
    }
    results.append(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print(f"[seam] wrote {out_path}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from smoke_runtime import run_locked

    sys.exit(run_locked("seam_proof_dispatcher", main))
