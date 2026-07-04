"""Inductor post-grad pass: pin fetch/free order in compiled graphs.

Functionalized fetch ops carry only data deps; in the AOT backward graph a
checkpoint unit's re-fetch depends only on its saved boundary activation, so
the scheduler may hoist later re-fetches above earlier frees -- exceeding the
ring depth (fail-fast depth-guard raise). This pass threads a REAL data
dependency through the chain: every `mm::fetch_start_after` node's guard base
is rewritten to the updated base of the previous fetch-op node in graph
order, so scheduling must preserve eager order. See
INGRAPH_PHASE4A_TRAINING_PLAN.md (S1 findings) for why effect tokens were
rejected on torch 2.12.

Install (idempotent) with `install_ordering_pass()` before compiling an
ingraph trunk.
"""

from __future__ import annotations

import operator

import torch

_FETCH_OPS = (
    "mm.fetch_start",
    "mm.fetch_start_after",
    "mm.fetch_wait",
    "mm.fetch_free",
    "mm.fetch_free_after",
)


def _wrapped_fetch_op(node):
    if node.op != "call_function":
        return None
    target = node.target
    if target is torch.ops.higher_order.auto_functionalized_v2 or (
        getattr(target, "__name__", "") == "auto_functionalized_v2"
    ):
        inner = node.args[0] if node.args else None
        name = str(getattr(inner, "name", inner))
        for op_name in _FETCH_OPS:
            if op_name in name:
                return op_name
        return None
    name = str(getattr(target, "name", lambda: target)()) if hasattr(
        target, "name"
    ) else str(target)
    for op_name in _FETCH_OPS:
        if op_name in name.replace("::", "."):
            return op_name
    return None


def order_fetch_ops_pass(graph: torch.fx.Graph) -> None:
    """Rewrite fetch_start_after guard bases to chain after the previous
    fetch-op node, enforcing eager fetch/free order via data deps."""
    prev = None
    for node in list(graph.nodes):
        op_name = _wrapped_fetch_op(node)
        if op_name is None:
            continue
        if (
            op_name == "mm.fetch_start_after"
            and prev is not None
            and node.kwargs.get("_all_bases")
        ):
            with graph.inserting_after(prev):
                gate = graph.call_function(operator.getitem, (prev, 1))
            new_kwargs = dict(node.kwargs)
            new_kwargs["_all_bases"] = [gate]
            node.kwargs = new_kwargs
        prev = node
    graph.lint()


def install_ordering_pass() -> None:
    """Idempotently chain onto torch._inductor.config.post_grad_custom_post_pass."""
    existing = torch._inductor.config.post_grad_custom_post_pass
    if getattr(existing, "_ingraph_ordering_pass", False):
        return

    def _pass(graph):
        if existing is not None:
            existing(graph)
        order_fetch_ops_pass(graph)

    _pass._ingraph_ordering_pass = True
    torch._inductor.config.post_grad_custom_post_pass = _pass
