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
import sys

import torch

# Longest-first: several names are prefixes of others and matching is by
# substring (e.g. "mm.fetch_start" would shadow "mm.fetch_start_after").
_FETCH_OPS = (
    "mm.fetch_start_gated",
    "mm.fetch_start_after",
    "mm.fetch_free_after",
    "mm.fetch_start",
    "mm.fetch_wait",
    "mm.fetch_free",
)


def _wrapped_fetch_op(node):
    """(op_name, is_auto_functionalized) for fetch-op nodes, else None.

    Matching is string-based on purpose: at the post-grad stage the target is
    either the auto_functionalized_v2 HOP (whose first arg is the wrapped
    OpOverload) or a plain OpOverload; identity/__name__ checks on the HOP
    object proved unreliable across torch builds."""
    if node.op != "call_function":
        return None
    if "auto_functionalized" in str(node.target):
        name = str(node.args[0]) if node.args else ""
        for op_name in _FETCH_OPS:
            if op_name in name:
                return op_name, True
        return None
    name = str(node.target).replace("::", ".")
    for op_name in _FETCH_OPS:
        if op_name in name:
            return op_name, False
    return None


def order_fetch_ops_pass(graph: torch.fx.Graph) -> None:
    """Rewrite fetch_start_after guard bases to chain after the previous
    fetch-op node, enforcing eager fetch/free order via data deps."""
    prev_auto = None  # last auto_functionalized fetch node (has tuple output)
    rewired = 0
    for node in list(graph.nodes):
        matched = _wrapped_fetch_op(node)
        if matched is None:
            continue
        op_name, is_auto = matched
        if op_name == "mm.fetch_start_after" and is_auto and prev_auto is not None:
            # Replace the functionalized fetch_start_after with a direct
            # fetch_start_gated whose gate is the previous fetch-op's updated
            # base -- a real data input, not mutation bookkeeping.
            host = node.kwargs.get("host_flat")
            if host is None and node.args:
                host = node.args[1] if len(node.args) > 1 else None
            if host is not None:
                if prev_auto.target is torch.ops.mm.fetch_start_gated.default:
                    gate = prev_auto  # token tensor, already a plain output
                else:
                    # Gate on the wrapped op's RETURN (tuple element 0, the
                    # token clone) -- a real extern-kernel output. Element 1+
                    # are updated-base bookkeeping the reinplacer dissolves
                    # into the original buffers, which would silently drop
                    # the ordering edge to the free itself.
                    with graph.inserting_after(prev_auto):
                        gate = graph.call_function(
                            operator.getitem, (prev_auto, 0)
                        )
                with graph.inserting_after(node):
                    gated = graph.call_function(
                        torch.ops.mm.fetch_start_gated.default, (host, gate)
                    )
                # The old node's users are getitems: index 0 is the token,
                # index >=1 are updated guard bases (pass-through of the
                # original base).
                for user in list(node.users):
                    if (
                        user.op == "call_function"
                        and user.target is operator.getitem
                    ):
                        if user.args[1] == 0:
                            user.replace_all_uses_with(gated)
                            graph.erase_node(user)
                        else:
                            base_list = node.kwargs.get("_all_bases") or []
                            idx = user.args[1] - 1
                            if 0 <= idx < len(base_list):
                                user.replace_all_uses_with(base_list[idx])
                                graph.erase_node(user)
                graph.erase_node(node)
                node = gated
                rewired += 1
        if is_auto or node.target is torch.ops.mm.fetch_start_gated.default:
            prev_auto = node
    if rewired:
        graph.lint()


class _OrderingGraphPass:
    """CustomGraphPass-shaped wrapper with a STABLE uuid.

    Inductor hashes post_grad_custom_post_pass into every fxgraph cache key;
    a bare closure is unpicklable, so torch salts the key per process and
    every run cache-misses (observed as 'different pass paths each run').
    The uuid is a content hash of this module's source: same code -> same
    key -> warm caches; editing the pass correctly invalidates them."""

    _ingraph_ordering_pass = True

    def __init__(self, existing=None):
        self._existing = existing

    def __call__(self, graph):
        if self._existing is not None:
            self._existing(graph)
        order_fetch_ops_pass(graph)

    def uuid(self):
        import hashlib
        import inspect

        source = inspect.getsource(sys.modules[__name__])
        salt = b""
        existing_uuid = getattr(self._existing, "uuid", None)
        if callable(existing_uuid):
            salt = bytes(str(existing_uuid()), "utf-8")
        elif self._existing is not None:
            salt = bytes(repr(self._existing), "utf-8")
        return hashlib.sha256(source.encode("utf-8") + salt).digest()


def install_ordering_pass() -> None:
    """Idempotently chain onto torch._inductor.config.post_grad_custom_post_pass."""
    existing = torch._inductor.config.post_grad_custom_post_pass
    if getattr(existing, "_ingraph_ordering_pass", False):
        return
    torch._inductor.config.post_grad_custom_post_pass = _OrderingGraphPass(existing)
