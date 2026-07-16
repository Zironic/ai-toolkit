r"""CUDA smoke for masked Krea2-shaped native cuDNN grouped-query SDPA.

This script deliberately does not import ``toolkit``: package initialization
installs the production KV-expansion wrapper, while this diagnostic must call
PyTorch's native SDPA operator with unexpanded K/V for variant B.

Example:

    venv\Scripts\python.exe scripts\smoke_cudnn_gqa.py
    venv\Scripts\python.exe scripts\smoke_cudnn_gqa.py --lengths 512 4608
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


VARIANTS = ("expanded_auto", "native_cudnn_gqa", "expanded_cudnn")
HEADS_Q = 48
HEADS_KV = 12
HEAD_DIM = 128
GROUPS = HEADS_Q // HEADS_KV


@dataclass
class Stats:
    variant: str
    length: int
    forced_backend: str
    eligible: bool | None = None
    eager_forward: bool = False
    eager_backward: bool = False
    compiled: bool = False
    finite_output: bool = False
    finite_gradients: bool = False
    time_ms: float | None = None
    peak_allocated_mib: float | None = None
    peak_reserved_mib: float | None = None
    output_error: dict | None = None
    q_grad_error: dict | None = None
    k_grad_error: dict | None = None
    v_grad_error: dict | None = None
    error: str | None = None


def _backend_context(variant: str, efficient_eligible: bool):
    if variant == "expanded_auto":
        if efficient_eligible:
            return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        return contextlib.nullcontext()
    return sdpa_kernel(SDPBackend.CUDNN_ATTENTION)


def _call(variant: str, q, k, v, mask, efficient_eligible: bool):
    if variant != "native_cudnn_gqa":
        k = k.repeat_interleave(GROUPS, dim=1)
        v = v.repeat_interleave(GROUPS, dim=1)
    with _backend_context(variant, efficient_eligible):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            enable_gqa=variant == "native_cudnn_gqa",
        )


def _params(q, k, v, mask, enable_gqa):
    return torch.backends.cuda.SDPAParams(
        q, k, v, mask, 0.0, False, enable_gqa
    )


def _eligibility(q, k, v, mask):
    bc = torch.backends.cuda
    kx = k.repeat_interleave(GROUPS, dim=1)
    vx = v.repeat_interleave(GROUPS, dim=1)
    return {
        "native_cudnn_gqa": bc.can_use_cudnn_attention(
            _params(q, k, v, mask, True), False
        ),
        "expanded_cudnn": bc.can_use_cudnn_attention(
            _params(q, kx, vx, mask, False), False
        ),
        "expanded_efficient": bc.can_use_efficient_attention(
            _params(q, kx, vx, mask, False), False
        ),
    }


def _inputs(length: int, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape_q = (1, HEADS_Q, length, HEAD_DIM)
    shape_kv = (1, HEADS_KV, length, HEAD_DIM)
    q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, generator=generator)
    k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator)
    v = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16, generator=generator)
    upstream = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16, generator=generator)
    mask = torch.ones((1, 1, 1, length), device="cuda", dtype=torch.bool)
    mask[..., -max(1, length // 16):] = False
    return q, k, v, mask, upstream


def _run_once(fn, tensors):
    q0, k0, v0, mask, upstream = tensors
    q = q0.detach().clone().requires_grad_(True)
    k = k0.detach().clone().requires_grad_(True)
    v = v0.detach().clone().requires_grad_(True)
    out = fn(q, k, v, mask)
    out.backward(upstream)
    torch.cuda.synchronize()
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def _finite(values):
    return all(bool(torch.isfinite(value).all()) for value in values)


def _error(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    denominator = expected.float().abs().clamp_min(1e-6)
    return {
        "max_abs": float(delta.max()),
        "max_rel": float((delta / denominator).max()),
        "mean_abs": float(delta.mean()),
    }


def _measure(fn, tensors, warmup: int, iterations: int):
    for _ in range(warmup):
        _run_once(fn, tensors)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        _run_once(fn, tensors)
    end.record()
    torch.cuda.synchronize()
    return (
        start.elapsed_time(end) / iterations,
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
    )


def _environment():
    bc = torch.backends.cuda
    props = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "cudnn_sdpa_enabled": bc.cudnn_sdp_enabled(),
        "flash_available": bc.is_flash_attention_available(),
        "sdpa_priority": list(torch._C._get_sdp_priority_order()),
    }


def _print_table(rows):
    print("\nvariant              L eager/bwd compile backend       ms     alloc/reserved MiB")
    for row in rows:
        eager = f"{int(row.eager_forward)}/{int(row.eager_backward)}"
        memory = (
            f"{row.peak_allocated_mib:.1f}/{row.peak_reserved_mib:.1f}"
            if row.peak_allocated_mib is not None else "-"
        )
        timing = f"{row.time_ms:.3f}" if row.time_ms is not None else "-"
        print(
            f"{row.variant:20} {row.length:5} {eager:9} {str(row.compiled):7} "
            f"{row.forced_backend:12} {timing:>8} {memory:>22}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[512, 4608],
        help="sequence lengths; 4608 represents 4096 image + 512 text tokens",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true", help="print raw JSON results")
    parser.add_argument("--compile-cache-dir", default="tmp/torch_compile_cache")
    parser.add_argument("--no-compile-cache", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable; no variants were run.")
        return 0
    if not torch.cuda.is_bf16_supported():
        print("SKIP: this GPU does not support BF16.")
        return 0

    environment = _environment()
    print(json.dumps(environment, indent=2))
    from toolkit.compile_cache import CompileCacheSession

    compile_cache = CompileCacheSession(
        args.compile_cache_dir,
        f"smoke_cudnn_gqa_{torch.__version__}",
        enabled=not args.no_compile_cache,
        logger=lambda message: print(f"[smoke] {message}"),
    )
    compile_cache.load()
    rows = []
    compiled_fns = {}
    compile_start_counts = dict(torch._dynamo.utils.counters["stats"])

    for index, length in enumerate(args.lengths):
        tensors = _inputs(length, args.seed + index)
        q, k, v, mask, _ = tensors
        eligibility = _eligibility(q, k, v, mask)
        print(f"L={length} eligibility: {json.dumps(eligibility, sort_keys=True)}")
        references = {}

        for variant in VARIANTS:
            efficient = eligibility["expanded_efficient"]
            forced = (
                "efficient" if variant == "expanded_auto" and efficient
                else "automatic" if variant == "expanded_auto"
                else "cudnn"
            )
            row = Stats(variant, length, forced)
            row.eligible = (
                efficient if variant == "expanded_auto" else eligibility[variant]
            )
            fn = lambda q, k, v, m, variant=variant, efficient=efficient: _call(
                variant, q, k, v, m, efficient
            )
            try:
                values = _run_once(fn, tensors)
                row.eager_forward = True
                row.eager_backward = True
                row.finite_output = _finite(values[:1])
                row.finite_gradients = _finite(values[1:])
                references[variant] = values
                if variant != "expanded_auto" and "expanded_auto" in references:
                    reference = references["expanded_auto"]
                    errors = [_error(a, b) for a, b in zip(values, reference, strict=True)]
                    row.output_error, row.q_grad_error, row.k_grad_error, row.v_grad_error = errors
                row.time_ms, row.peak_allocated_mib, row.peak_reserved_mib = _measure(
                    fn, tensors, args.warmup, args.iterations
                )
            except Exception as error:
                row.error = f"eager: {type(error).__name__}: {error}"

            if row.eager_backward:
                try:
                    compiled = compiled_fns.get(variant)
                    if compiled is None:
                        compiled = torch.compile(fn, dynamic=True)
                        compiled_fns[variant] = compiled
                    compiled_values = _run_once(compiled, tensors)
                    row.compiled = _finite(compiled_values)
                except Exception as error:
                    suffix = f"compiled: {type(error).__name__}: {error}"
                    row.error = f"{row.error}; {suffix}" if row.error else suffix
            rows.append(row)

        del tensors, q, k, v, mask, references
        torch.cuda.empty_cache()

    compile_end_counts = dict(torch._dynamo.utils.counters["stats"])
    compile_delta = {
        key: compile_end_counts.get(key, 0) - compile_start_counts.get(key, 0)
        for key in set(compile_start_counts) | set(compile_end_counts)
    }
    compile_cache.save(force=True)
    print(f"compile counters delta: {json.dumps(compile_delta, sort_keys=True)}")
    _print_table(rows)
    for row in rows:
        if row.error:
            print(f"ERROR {row.variant} L={row.length}: {row.error}")
        if row.output_error is not None:
            print(
                f"ERROR_STATS {row.variant} L={row.length}: "
                f"{json.dumps({'out': row.output_error, 'q': row.q_grad_error, 'k': row.k_grad_error, 'v': row.v_grad_error})}"
            )
    if args.json:
        print(json.dumps({"environment": environment, "rows": [asdict(r) for r in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    # Serialize GPU scripts: two 11+ GiB smokes on a 12 GB card do not
    # just measure badly, the second OOMs. See scripts/_gpu_lock.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("smoke_cudnn_gqa", main))
