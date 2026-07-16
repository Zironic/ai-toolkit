"""Shared compiler evidence helpers for controlled MegaCache smokes."""

from __future__ import annotations

import functools
from pathlib import Path


def counter_snapshot(torch) -> dict[str, int]:
    snapshot = {}
    for group, values in torch._dynamo.utils.counters.items():
        for name, value in values.items():
            if value:
                snapshot[f"{group}.{name}"] = int(value)
    return dict(sorted(snapshot.items()))


def numeric_delta(after: dict, before: dict) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(after) | set(before))
        if int(after.get(key, 0)) != int(before.get(key, 0))
    }


def cache_dir_stats(path) -> dict[str, int]:
    root = Path(path)
    if not root.exists():
        return {"files": 0, "bytes": 0}
    files = [item for item in root.rglob("*") if item.is_file()]
    return {
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def install_backend_instrumentation() -> dict[str, int]:
    """Count backend work that cache counters alone cannot prove absent."""
    import triton
    import triton.compiler.compiler
    from torch._inductor.graph import GraphLowering
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    calls = {
        "inductor_codegen": 0,
        "triton_compile": 0,
        "benchmark_all_configs": 0,
        "coordinate_descent_tuning": 0,
    }

    original_compile_to_module = GraphLowering.compile_to_module

    @functools.wraps(original_compile_to_module)
    def counted_compile_to_module(*args, **kwargs):
        calls["inductor_codegen"] += 1
        return original_compile_to_module(*args, **kwargs)

    GraphLowering.compile_to_module = counted_compile_to_module

    original_compile = triton.compiler.compiler.compile

    @functools.wraps(original_compile)
    def counted_compile(*args, **kwargs):
        calls["triton_compile"] += 1
        return original_compile(*args, **kwargs)

    triton.compiler.compiler.compile = counted_compile
    triton.compile = counted_compile

    for method_name in ("benchmark_all_configs", "coordinate_descent_tuning"):
        original = getattr(CachingAutotuner, method_name)

        @functools.wraps(original)
        def counted_method(
            self,
            *args,
            __name=method_name,
            __original=original,
            **kwargs,
        ):
            calls[__name] += 1
            return __original(self, *args, **kwargs)

        setattr(CachingAutotuner, method_name, counted_method)

    return calls


def cache_evidence(
    counters: dict[str, int], backend_calls: dict[str, int]
) -> dict[str, int | bool]:
    def get(name: str) -> int:
        return int(counters.get(name, 0))

    aot_hit = get("aot_autograd.autograd_cache_hit")
    aot_miss = get("aot_autograd.autograd_cache_miss")
    aot_bypass = get("aot_autograd.autograd_cache_bypass")
    fx_hit = get("inductor.fxgraph_cache_hit")
    fx_miss = get("inductor.fxgraph_cache_miss")
    fx_bypass = get("inductor.fxgraph_cache_bypass")
    inductor_codegen = int(backend_calls.get("inductor_codegen", 0))
    triton_compile = int(backend_calls.get("triton_compile", 0))
    autotune = int(backend_calls.get("benchmark_all_configs", 0))
    coordinate_descent = int(backend_calls.get("coordinate_descent_tuning", 0))
    graph_breaks = sum(
        int(value)
        for key, value in counters.items()
        if key.startswith("graph_break.")
    )
    return {
        "aot_hit": aot_hit,
        "aot_miss": aot_miss,
        "aot_bypass": aot_bypass,
        "unique_graphs": get("stats.unique_graphs"),
        "fx_hit": fx_hit,
        "fx_miss": fx_miss,
        "fx_bypass": fx_bypass,
        "async_compile_hit": get("inductor.async_compile_cache_hit"),
        "async_compile_lookup": get("inductor.async_compile_cache_miss"),
        "inductor_codegen_calls": inductor_codegen,
        "triton_compile_calls": triton_compile,
        "autotune_benchmark_calls": autotune,
        "coordinate_descent_calls": coordinate_descent,
        "graph_breaks": graph_breaks,
        "backend_codegen_observed": bool(
            fx_miss
            or inductor_codegen
            or triton_compile
            or get("inductor.triton_bundler_save_kernel")
        ),
    }
