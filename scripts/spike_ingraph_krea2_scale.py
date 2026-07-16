"""Phase 0b scale spike for tasks/done/INGRAPH_STREAM_PLAN.md (git-bug 3ca8a7b).

Runs the toy-spike machinery (tests/test_ingraph_spike.py) against the REAL
Krea2 fp8 transformer (28 blocks x 8 quantized Linears, ~12 GB qdata):

  1. Load the fp8-quantized SingleStreamDiT on CPU (quantized disk cache).
  2. Pack each block's qdata+scale leaves into ONE pinned flat host buffer
     (attach-time packing, plan Phase 1 shape); free the pageable originals.
  3. Compile the whole 28-block trunk fullgraph=True with split-phase
     in-graph fetch ops + in-graph dequant + functional_call on a single
     template block (one graph, weights as inputs).
  4. Forward: parity vs eager, compile time, steady-state ms, VRAM peak.
  5. Fwd+bwd with SAC MUST_RECOMPUTE on fetch ops: re-fetch at scale
     (expect 2 x n_blocks fetches), compile time, step ms, VRAM peak.

Usage (venv python, repo root):
  venv/Scripts/python.exe scripts/spike_ingraph_krea2_scale.py [--blocks N]
      [--seq 1024] [--skip-backward] [--depth 3]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkit.config_modules import ModelConfig  # noqa: E402
from toolkit.memory_management.manager_modules import (  # noqa: E402
    _dequantize_to,
    _flatten_leaves,
)

DEVICE = torch.device("cuda")
ALIGN = 256


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- fetch ops --

_STATE = {
    "next_id": 0,
    "tickets": {},
    "transfer_stream": None,
    "real_fetches": 0,
    "consumed": [],  # (event,) ring for host-side depth guard
    "depth": 3,
}


def _ts():
    if _STATE["transfer_stream"] is None:
        _STATE["transfer_stream"] = torch.cuda.Stream()
    return _STATE["transfer_stream"]


@torch.library.custom_op("mm_spike2::fetch_start", mutates_args=())
def fetch_start(host_flat: torch.Tensor) -> torch.Tensor:
    # Host-side depth guard: bound run-ahead so no more than `depth` fetched
    # block buffers can be live, whatever the schedule does.
    while len(_STATE["consumed"]) >= _STATE["depth"]:
        _STATE["consumed"].pop(0).synchronize()
    ts = _ts()
    _STATE["real_fetches"] += 1
    with torch.cuda.stream(ts):
        buf = host_flat.to(DEVICE, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(ts)
    tid = _STATE["next_id"]
    _STATE["next_id"] += 1
    _STATE["tickets"][tid] = (buf, ev)
    return torch.tensor([tid], dtype=torch.int64)


@fetch_start.register_fake
def _(host_flat):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm_spike2::fetch_wait", mutates_args=())
def fetch_wait(token: torch.Tensor, nbytes: int) -> torch.Tensor:
    tid = int(token[0].item())
    buf, ev = _STATE["tickets"].pop(tid)
    cs = torch.cuda.current_stream()
    cs.wait_event(ev)
    buf.record_stream(cs)
    done = torch.cuda.Event()
    done.record(cs)
    _STATE["consumed"].append(done)
    return buf


@fetch_wait.register_fake
def _(token, nbytes):
    return torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)


# ------------------------------------------------------------- packing ---


def pack_block(block):
    """One pinned flat buffer per block for the 8 big quantized Linears.
    Returns (host_flat, leaf_meta, small_params) where leaf_meta maps a
    param name -> [(offset, nbytes, dtype, shape), ...] in leaf order and
    small_params holds every non-quantized param (GPU copies, resident)."""
    metas = {}
    small = {}
    leaves = []
    for name, p in block.named_parameters():
        if hasattr(p.data, "qdata"):
            lv = _flatten_leaves(p.data)
            rec = []
            for leaf in lv:
                rec.append([None, leaf.numel() * leaf.element_size(),
                            leaf.dtype, tuple(leaf.shape), leaf])
            metas[name] = rec
            leaves.extend(rec)
        else:
            small[name] = p.data.to(DEVICE)
    total = 0
    for rec in leaves:
        total = (total + ALIGN - 1) // ALIGN * ALIGN
        rec[0] = total
        total += rec[1]
    host = torch.empty(total, dtype=torch.uint8, pin_memory=True)
    for off, nb, dt, shape, leaf in leaves:
        host[off:off + nb].view(dt).reshape(shape).copy_(leaf)
    for rec in leaves:
        rec.pop()  # drop the CPU leaf ref so its storage can be freed
    return host, metas, small


def dequant_from_flat(flat, rec_list, compute_dtype):
    """In-graph: slice qdata+scale views out of the flat buffer and dequant.
    Leaf order for torchao float8 is (qdata, scale)."""
    (o1, n1, d1, s1), (o2, n2, d2, s2) = rec_list
    qdata = flat[o1:o1 + n1].view(d1).reshape(s1)
    scale = flat[o2:o2 + n2].view(d2).reshape(s2)
    w = qdata.to(compute_dtype) * scale.reshape(-1, 1).to(compute_dtype)
    return w


def block_params(flat, metas, small, compute_dtype):
    params = dict(small)
    for name, rec in metas.items():
        params[name] = dequant_from_flat(flat, [r[:4] for r in rec], compute_dtype)
    return params


# ------------------------------------------------------------- trunk -----


def make_trunk(template, packs, use_sac):
    from torch.func import functional_call

    def run_block(x, tvec, freqs, host, metas, small):
        tok = torch.ops.mm_spike2.fetch_start(host)
        flat = torch.ops.mm_spike2.fetch_wait(tok, host.numel())
        params = block_params(flat, metas, small, x.dtype)
        return functional_call(template, params, (x, tvec, freqs, None))

    if not use_sac:
        def trunk(x, tvec, freqs):
            for host, metas, small in packs:
                x = run_block(x, tvec, freqs, host, metas, small)
            return x
        return trunk

    from torch.utils.checkpoint import (
        CheckpointPolicy,
        checkpoint,
        create_selective_checkpoint_contexts,
    )

    def policy(ctx, op, *args, **kwargs):
        if op in (
            torch.ops.mm_spike2.fetch_start.default,
            torch.ops.mm_spike2.fetch_wait.default,
        ):
            return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.PREFER_RECOMPUTE

    ctx_fn = lambda: create_selective_checkpoint_contexts(policy)  # noqa: E731

    def trunk_sac(x, tvec, freqs):
        for host, metas, small in packs:
            x = checkpoint(
                run_block, x, tvec, freqs, host, metas, small,
                use_reentrant=False, context_fn=ctx_fn,
            )
        return x

    return trunk_sac


# --------------------------------------------------------------- main ----


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="krea/Krea-2-Raw")
    ap.add_argument("--blocks", type=int, default=-1, help="-1 = all 28")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--skip-backward", action="store_true")
    args = ap.parse_args()
    _STATE["depth"] = args.depth

    from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model

    t0 = time.perf_counter()
    log("loading fp8 transformer (CPU) ...")
    model = Krea2Model(
        device="cuda:0",
        model_config=ModelConfig(
            name_or_path=args.model_path, arch="krea2", dtype="bf16",
            quantize=True, qtype="float8", low_vram=True,
        ),
        dtype="bf16",
    )
    transformer = model._load_transformer()
    log(f"loaded in {time.perf_counter() - t0:.1f}s")

    blocks = list(transformer.blocks)
    if args.blocks > 0:
        blocks = blocks[: args.blocks]
    n = len(blocks)
    features = transformer.config.features

    # --- verify in-graph dequant math against the toolkit's, on block 0
    b0 = blocks[0]
    name0, p0 = next(
        (nm, p) for nm, p in b0.named_parameters() if hasattr(p.data, "qdata")
    )
    lv = _flatten_leaves(p0.data)
    assert len(lv) == 2, f"expected (qdata, scale) leaves, got {len(lv)}"
    ref_w = _dequantize_to(p0.data.to(DEVICE), torch.bfloat16)
    my_w = (
        lv[0].to(DEVICE).to(torch.bfloat16)
        * lv[1].to(DEVICE).reshape(-1, 1).to(torch.bfloat16)
    )
    if not torch.equal(ref_w, my_w):
        diff = (ref_w.float() - my_w.float()).abs().max().item()
        assert diff < 1e-3, f"dequant math mismatch: max diff {diff}"
        log(f"dequant math: allclose (max diff {diff:.2e}, not bitwise)")
    else:
        log("dequant math: bitwise vs _dequantize_to")
    del ref_w, my_w, lv

    # --- pack + pin all blocks, freeing pageable originals as we go
    log(f"packing {n} blocks into pinned flat buffers ...")
    t0 = time.perf_counter()
    packs = []
    pinned_bytes = 0
    for i, blk in enumerate(blocks):
        host, metas, small = pack_block(blk)
        pinned_bytes += host.numel()
        packs.append((host, metas, small))
        # free the pageable originals (RAM headroom on a 32 GB box)
        for nm, p in list(blk.named_parameters()):
            if hasattr(p.data, "qdata"):
                mod = blk.get_submodule(nm.rsplit(".", 1)[0])
                setattr(mod, nm.rsplit(".", 1)[1], None)
    del transformer, model
    import gc
    gc.collect()
    log(
        f"packed+pinned {pinned_bytes / 2**30:.2f} GiB in "
        f"{time.perf_counter() - t0:.1f}s (this commits the DXGI shared budget)"
    )

    template = blocks[0]

    # --- inputs
    torch.manual_seed(0)
    x = torch.randn(1, args.seq, features, device=DEVICE, dtype=torch.bfloat16)
    tvec = torch.randn(1, 6 * features, device=DEVICE, dtype=torch.bfloat16)
    # freqs via a tiny standalone PositionalEncoding (no params)
    from extensions_built_in.diffusion_models.krea2.src.mmdit import (
        PositionalEncoding,
    )
    headdim = features // 48
    axes = [headdim - 12 * (headdim // 16), 6 * (headdim // 16), 6 * (headdim // 16)]
    pos = torch.zeros(1, args.seq, 3, device=DEVICE)
    side = int(args.seq ** 0.5) + 1
    pos[0, :, 1] = torch.arange(args.seq, device=DEVICE) // side
    pos[0, :, 2] = torch.arange(args.seq, device=DEVICE) % side
    freqs = PositionalEncoding(features, axes, theta=100.0)(pos)

    # --- forward: eager reference then compiled
    trunk = make_trunk(template, packs, use_sac=False)
    with torch.no_grad():
        log("eager streamed forward (reference) ...")
        t0 = time.perf_counter()
        ref = trunk(x, tvec, freqs)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1e3
        log(f"eager forward: {eager_ms:.0f} ms")

        log("compiling trunk fullgraph=True (unrolled, one graph) ...")
        compiled = torch.compile(trunk, fullgraph=True, dynamic=False)
        t0 = time.perf_counter()
        out = compiled(x, tvec, freqs)
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - t0
        breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        log(f"forward compile+first-run: {compile_s:.1f}s, graph breaks: {breaks}")

        d = (out.float() - ref.float()).abs()
        log(
            f"parity vs eager: max {d.max().item():.4e} mean {d.mean().item():.4e} "
            f"(out rms {ref.float().pow(2).mean().sqrt().item():.3f})"
        )

        torch.cuda.reset_peak_memory_stats()
        _STATE["real_fetches"] = 0
        t0 = time.perf_counter()
        iters = 3
        for _ in range(iters):
            compiled(x, tvec, freqs)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - t0) * 1e3 / iters
        log(
            f"steady-state compiled forward: {fwd_ms:.0f} ms/pass "
            f"(fetches/pass={_STATE['real_fetches'] // iters}, "
            f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)"
        )

    if args.skip_backward:
        log("done (backward skipped)")
        return

    # --- fwd+bwd with SAC re-fetch
    log("compiling trunk with SAC (fwd+bwd) ...")
    torch._dynamo.reset()
    trunk_sac = make_trunk(template, packs, use_sac=True)
    compiled_sac = torch.compile(trunk_sac, fullgraph=True, dynamic=False)
    xg = x.detach().clone().requires_grad_(True)
    t0 = time.perf_counter()
    compiled_sac(xg, tvec, freqs).sum().backward()
    torch.cuda.synchronize()
    log(f"fwd+bwd compile+first-run: {time.perf_counter() - t0:.1f}s")

    torch.cuda.reset_peak_memory_stats()
    _STATE["real_fetches"] = 0
    xg.grad = None
    t0 = time.perf_counter()
    compiled_sac(xg, tvec, freqs).sum().backward()
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) * 1e3
    fetches = _STATE["real_fetches"]
    breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
    log(
        f"steady-state fwd+bwd: {step_ms:.0f} ms, fetches={fetches} "
        f"(expect {2 * n} = 2x{n} blocks), graph breaks: {breaks}, "
        f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
    )
    assert fetches == 2 * n, "backward did not re-fetch every block"
    assert xg.grad is not None and torch.isfinite(xg.grad).all()
    log("grad finite; scale spike complete")


if __name__ == "__main__":
    main()
