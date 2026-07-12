"""S0 (git-bug 0c577ef): what is one more resident block actually worth?

The two-timescale residency FSM assumes promoting a block buys step time. After
the device-side fetch ring, Krea2 training is compute-bound (occupancy 87->96%),
so that assumption is a hypothesis, not a fact: if streaming is already fully
hidden behind compute, the 2.50 GiB of reclaimable VRAM buys *headroom*, not
throughput, and the controller's promote gate should be tuned conservatively (or
the climb dropped in favour of the cap lever alone).

This script answers it directly. It loads Krea2 exactly as the training smoke
does, then sweeps residency N, N+1, N+2, ... using the runtime's own promoter
(`increase_training_residency`), so every arm is a block the controller would
really have chosen -- not an idealized one.

Arms are INTERLEAVED (round-robin, several rounds) rather than run in one
monotone climb: the same clocks, thermals and allocator state, so run-to-run
variance cannot masquerade as a residency effect. The first step after every
residency switch is discarded (the switch itself builds sidecars and re-warms
the fetch ring).

It also records the per-step allocator peak, which is what S0's second question
needs: sampling's slack pad (0.21 GiB) is rock-steady, training's peak is noisy,
so the training pad must come from measured variance rather than a guess.

Example:

    venv\\Scripts\\python.exe scripts\\measure_marginal_resident_block.py ^
      --extra-blocks 3 --rounds 3 --steps-per-arm 4
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import smoke_krea2_train_cuda as smoke  # noqa: E402

from extensions_built_in.diffusion_models.krea2.krea2 import (  # noqa: E402
    DoubleSharedModulation,
    Krea2Model,
    SimpleModulation,
)
from toolkit.basic import flush  # noqa: E402
from toolkit.memory_management import ingraph_stream, vram_budget  # noqa: E402
from toolkit.memory_management.manager import MemoryManager  # noqa: E402
from toolkit.util.quantize import quantize_model  # noqa: E402

GIB = 1024 ** 3


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cond-cache", default=smoke.DEFAULT_COND_CACHE)
    parser.add_argument("--model-path", default="krea/Krea-2-Raw")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--extra-blocks", type=int, default=3,
        help="how many blocks above the planner's baseline residency to sweep",
    )
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="round-robin passes over the arms (interleaving beats drift)",
    )
    parser.add_argument(
        "--steps-per-arm", type=int, default=4,
        help="measured steps per arm per round (plus 1 discarded warmup step)",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--discard-rounds", type=int, default=1,
        help="drop the first N rounds from the summary. The card boosts its "
        "clocks early: round 0 measured ~1.88s where every later round settled "
        "at ~2.32s, which is 20x the effect being measured and lands entirely "
        "in whichever arm runs first.",
    )
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument(
        "--dop", action="store_true",
        help="emulate diff-output preservation: an extra no-grad prior forward "
        "with the network off, plus a second grad-enabled preservation "
        "forward+backward (SDTrainer's shape). ~2.5x the streaming per step and "
        "far more memory pressure, so the residency trade is not the same one.",
    )
    parser.add_argument("--dop-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--dop-single-backward", action="store_true",
        help="mirror train_config.dop_single_backward: combine both losses into "
        "ONE backward, which keeps both graphs live to the peak (default False, "
        "matching the config default: backward the main loss first)",
    )
    parser.add_argument("--fp8-training-forward", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--compile-cache-dir", default="tmp/torch_compile_cache")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _model_config(args):
    """Reuse the smoke's ModelConfig builder; it wants the smoke's arg surface."""
    return smoke._build_model_config(
        Namespace(
            max_text_length=512,
            allow_download=False,
            cache_dir=None,
            checkpoint_filename=None,
            model_path=args.model_path,
            dtype="bf16",
            qtype="float8",
            pinned_weight_gib=-1.0,
            working_reserve_gib="-1",
            wddm_margin_gib=1.0,
            wddm_hard_gib=1.0,
            spill_reserve_pct=0.20,
            checkpoint_keep_last=0,
            block_stream_only=False,
            fp8_training_forward=args.fp8_training_forward,
            pinned_arena=False,
            prefetch_depth=args.prefetch_depth,
            no_compile=args.no_compile,
        )
    )


def _build_arms(executor, extra_blocks):
    """Baseline plan, then the blocks the real promoter would add, one at a time.

    Using `increase_training_residency` rather than hand-picking blocks is the
    point: it applies the promoter's own candidate rule (non-protected,
    whole-block, smallest first), so arm k is exactly the k-th block the
    controller would promote. Its `plan` is snapshotted so any arm can be
    re-activated later for the interleaved rounds.
    """
    arms = [
        {
            "name": "N",
            "added_blocks": (),
            "plan": executor.residency.plan,
            "resident_bytes": int(executor.residency.resident_bytes()),
        }
    ]
    for index in range(1, int(extra_blocks) + 1):
        result = executor.increase_training_residency(1 << 62, max_blocks=1)
        if not result["added_blocks"]:
            print(f"[s0] no promotable block left after +{index - 1}; stopping sweep")
            break
        arms.append(
            {
                "name": f"N+{index}",
                "added_blocks": tuple(result["added_blocks"]),
                "plan": result["plan"],
                "resident_bytes": int(executor.residency.resident_bytes()),
                "block_bytes": int(result["actual_growth_bytes"]),
            }
        )
    return arms


def _run_step(model, transformer, network, embeds, optimizer, trainable, args, gen, device):
    lat_h, lat_w = args.height // 8, args.width // 8
    latents = torch.randn(
        args.batch_size, 16, lat_h, lat_w, generator=gen
    ).to(device, model.torch_dtype)
    noise = torch.randn(
        args.batch_size, 16, lat_h, lat_w, generator=gen
    ).to(device, model.torch_dtype)
    timestep = (torch.rand(args.batch_size, generator=gen) * 1000.0).to(device)
    t_frac = (timestep.float() / 1000.0).view(-1, 1, 1, 1).to(device)
    noisy = ((1.0 - t_frac) * latents.float() + t_frac * noise.float()).to(
        model.torch_dtype
    )
    target = (noise.float() - latents.float()).detach()

    executor = transformer._immutable_runtime
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    retries_before = torch.cuda.memory_stats(device).get("num_alloc_retries", 0)
    t0 = time.perf_counter()
    with executor.execution(executor.TRAIN), network:
        # Diff-output preservation, as SDTrainer runs it: a no-grad prior
        # prediction with the network switched OFF, then -- after the main
        # forward -- a SECOND grad-enabled forward whose output is pulled back
        # toward that prior. It roughly 2.5x's the streaming volume per step and,
        # in single-backward mode, keeps two graphs live at once. That changes
        # both terms of the residency trade, which is why it gets its own arm.
        prior_pred = None
        if args.dop:
            was_active = network.is_active
            network.is_active = False
            try:
                with torch.no_grad():
                    prior_pred = model.get_noise_prediction(
                        noisy, timestep, embeds
                    ).detach()
            finally:
                network.is_active = was_active

        pred = model.get_noise_prediction(noisy, timestep, embeds)
        loss = torch.nn.functional.mse_loss(pred.float(), target)

        if prior_pred is None:
            loss.backward()
        elif args.dop_single_backward:
            preservation_pred = model.get_noise_prediction(noisy, timestep, embeds)
            preservation_loss = torch.nn.functional.mse_loss(
                preservation_pred.float(), prior_pred.float()
            )
            (loss + args.dop_multiplier * preservation_loss).backward()
        else:
            # Two-pass (the default): the main graph is freed before the
            # preservation forward allocates its own, which is the whole point
            # -- it keeps the peak at one graph instead of two.
            loss.backward()
            preservation_pred = model.get_noise_prediction(noisy, timestep, embeds)
            preservation_loss = torch.nn.functional.mse_loss(
                preservation_pred.float(), prior_pred.float()
            )
            (args.dop_multiplier * preservation_loss).backward()
    grads = sum(1 for p in trainable if p.grad is not None)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    stats = torch.cuda.memory_stats(device)
    return {
        "seconds": elapsed,
        "loss": float(loss.item()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / GIB,
        "reserved_gib": torch.cuda.memory_reserved(device) / GIB,
        "alloc_retries": int(stats.get("num_alloc_retries", 0) - retries_before),
        "device_free_gib": vram_budget.device_free_bytes(device) / GIB,
        "grads": grads,
    }


def _summarize(arm_name, samples, fetch_stats):
    times = [s["seconds"] for s in samples]
    peaks = [s["peak_allocated_gib"] for s in samples]
    fetch_bytes = fetch_stats.get("bytes", 0)
    return {
        "arm": arm_name,
        "steps": len(times),
        "median_s": statistics.median(times),
        "mean_s": statistics.fmean(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_s": min(times),
        "peak_allocated_gib_mean": statistics.fmean(peaks),
        "peak_allocated_gib_max": max(peaks),
        "peak_allocated_gib_stdev": (
            statistics.stdev(peaks) if len(peaks) > 1 else 0.0
        ),
        "alloc_retries": sum(s["alloc_retries"] for s in samples),
        "device_free_gib_min": min(s["device_free_gib"] for s in samples),
        "fetch_gib_per_step": fetch_bytes / GIB / max(1, len(times)),
        "h2d_ms_per_step": fetch_stats.get("h2d_ms", 0.0) / max(1, len(times)),
        "fetches_per_step": fetch_stats.get("fetches", 0) / max(1, len(times)),
    }


def main():
    args = _parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("AI_TOOLKIT_MEMORY_DIAGNOSTICS", "1")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    embeds = smoke._load_prompt_cache(str(Path(args.cond_cache)))
    embeds.text_embeds = [embeds.text_embeds[0]] * args.batch_size

    config = _model_config(args)
    model = Krea2Model(device="cuda", model_config=config, dtype="bf16")
    model.skip_te = True

    print("[s0] loading Krea2 transformer")
    transformer = model._load_transformer()
    flush(garbage_collect=False)
    if config.quantize and not getattr(
        model, "_transformer_quantized_during_load", False
    ):
        print("[s0] quantizing")
        quantize_model(model, transformer)
        flush()

    print("[s0] attaching immutable training memory")
    ignore_modules = [
        module
        for module in transformer.modules()
        if isinstance(module, (SimpleModulation, DoubleSharedModulation))
    ]
    model._attach_immutable_training_memory(transformer, ignore_modules)
    if getattr(transformer, "_memory_manager", None) is not None:
        MemoryManager._attach_prefetch_pool(transformer, device)
    model.model = transformer

    network = smoke._apply_lora(
        model, transformer, device, args.lora_rank, args.lora_alpha
    )
    trainable = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        network.prepare_optimizer_params(
            text_encoder_lr=args.lr, unet_lr=args.lr, default_lr=args.lr
        ),
        lr=args.lr,
    )

    compile_cache_key = None
    if args.compile_cache_dir and not args.no_compile:
        from extensions_built_in.diffusion_models.krea2.krea2 import _train_compile_cache_key
        from toolkit.compile_cache import load_compile_cache

        compile_cache_key = _train_compile_cache_key(model)
        if load_compile_cache(args.compile_cache_dir, compile_cache_key):
            print(f"[s0] loaded compile mega-cache ({compile_cache_key})")

    transformer.finalize_immutable_runtime()
    executor = transformer._immutable_runtime
    baseline_plan = transformer._mm_immutable_training_plan
    executor.activate(executor.TRAIN, baseline_plan)

    total_blocks = len(executor.residency.arena.block_keys())
    full_resident_gib = executor.full_model_resident_bytes() / GIB
    print(
        f"[s0] blocks={total_blocks} baseline_resident="
        f"{executor.residency.resident_bytes() / GIB:.2f} GiB "
        f"full_model={full_resident_gib:.2f} GiB"
    )

    arms = _build_arms(executor, args.extra_blocks)
    for arm in arms:
        print(
            f"[s0] arm {arm['name']}: "
            f"resident={arm['resident_bytes'] / GIB:.3f} GiB "
            f"added={arm['added_blocks']}"
        )

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    # Step 0 anywhere is the trunk compile; pay it once, outside the arms, on
    # the baseline plan so no arm carries it.
    executor.set_residency_plan(arms[0]["plan"])
    print("[s0] compile/warm step (not measured)")
    warm = _run_step(
        model, transformer, network, embeds, optimizer, trainable, args, gen, device
    )
    print(f"[s0] warm step {warm['seconds']:.2f}s loss={warm['loss']:.4f}")
    if warm["grads"] == 0:
        raise SystemExit("no LoRA gradients -- training path is broken")
    if compile_cache_key is not None:
        from toolkit.compile_cache import save_compile_cache

        save_compile_cache(args.compile_cache_dir, compile_cache_key)

    samples = {arm["name"]: [] for arm in arms}
    fetch_rounds = {arm["name"]: [] for arm in arms}
    dynamo_frames_before = sum(torch._dynamo.utils.counters["frames"].values())

    for round_index in range(args.rounds):
        # Alternate the arm order every round. Within a round the clock drifts
        # (thermal recovery, desktop contention), and with a fixed order that
        # drift lands on the arms in a fixed sequence -- a monotone drift then
        # reads as a monotone residency effect. Reversing every other round
        # makes each arm run early as often as it runs late, so the drift
        # cancels in the pooled medians instead of impersonating a result.
        ordered = arms if round_index % 2 == 0 else list(reversed(arms))
        for arm in ordered:
            executor.set_residency_plan(arm["plan"])
            transformer._mm_immutable_training_plan = arm["plan"]
            for _ in range(args.warmup_steps):
                _run_step(
                    model, transformer, network, embeds, optimizer, trainable,
                    args, gen, device,
                )
            ingraph_stream.fetch_stats(reset=True)
            arm_times = []
            for _ in range(args.steps_per_arm):
                row = _run_step(
                    model, transformer, network, embeds, optimizer, trainable,
                    args, gen, device,
                )
                row["round"] = round_index
                samples[arm["name"]].append(row)
                arm_times.append(row["seconds"])
            stats = ingraph_stream.fetch_stats(reset=True)
            fetch_rounds[arm["name"]].append(stats)
            print(
                f"[s0] round {round_index} arm {arm['name']}: "
                f"median={statistics.median(arm_times):.3f}s "
                f"h2d={stats['h2d_ms'] / max(1, len(arm_times)):.1f}ms/step "
                f"fetch={stats['bytes'] / GIB / max(1, len(arm_times)):.3f} GiB/step"
            )

    dynamo_frames_after = sum(torch._dynamo.utils.counters["frames"].values())

    keep = int(args.discard_rounds)
    kept = {
        arm["name"]: [
            row for row in samples[arm["name"]] if row["round"] >= keep
        ]
        for arm in arms
    }
    if not all(kept.values()):
        raise SystemExit(
            f"--discard-rounds {keep} left no samples; run more rounds"
        )
    fetch_totals = {
        arm["name"]: {
            key: sum(stats.get(key, 0) for stats in fetch_rounds[arm["name"]][keep:])
            for key in ("bytes", "h2d_ms", "fetches", "wait_ms")
        }
        for arm in arms
    }
    summaries = [
        _summarize(arm["name"], kept[arm["name"]], fetch_totals[arm["name"]])
        for arm in arms
    ]
    base = summaries[0]["median_s"]
    for summary, arm in zip(summaries, arms, strict=True):
        summary["resident_gib"] = arm["resident_bytes"] / GIB
        summary["vs_baseline_pct"] = (summary["median_s"] - base) / base * 100.0

    # No per-round pairing. It looks like the rigorous thing to do, but the
    # baseline arm occupies a different slot in each round (the order
    # alternates), so a within-round drift contaminates the pair itself -- it
    # produced a confident -6% on a run whose pooled medians were flat. The
    # pooled median across alternated rounds is the honest estimator; the
    # step-time noise band below is what says whether a difference is real.
    if (args.rounds - keep) % 2:
        print(
            f"[s0] WARNING: {args.rounds - keep} kept rounds is odd, so the "
            "alternating arm order does not fully cancel within-round drift. "
            "Prefer an even number of kept rounds."
        )

    print(
        f"\n[s0] ==== marginal resident block "
        f"(rounds {keep}..{args.rounds - 1}) ===="
    )
    print(
        f"{'arm':>5} {'resident':>9} {'median_s':>9} {'stdev':>7} "
        f"{'vs_N':>7} {'h2d_ms':>8} {'fetch_GiB':>10} {'peak_GiB':>9} {'peak_sd':>8}"
    )
    for summary in summaries:
        print(
            f"{summary['arm']:>5} {summary['resident_gib']:>8.2f}G "
            f"{summary['median_s']:>9.3f} {summary['stdev_s']:>7.3f} "
            f"{summary['vs_baseline_pct']:>6.2f}% "
            f"{summary['h2d_ms_per_step']:>8.1f} "
            f"{summary['fetch_gib_per_step']:>10.3f} "
            f"{summary['peak_allocated_gib_mean']:>9.2f} "
            f"{summary['peak_allocated_gib_stdev']:>8.3f}"
        )

    # The verdict S0 exists to deliver. Read the MARGINAL block, not the total:
    # the first resident block is a special case (nothing precedes it, so its
    # fetch has no compute to hide behind), and a flat curve after it means
    # every further block is bought with VRAM and paid for in nothing.
    marginal = []
    for previous, current in itertools.pairwise(summaries):
        marginal.append(
            (previous["median_s"] - current["median_s"]) / previous["median_s"] * 100.0
        )
    if marginal:
        noise = max(
            summary["stdev_s"] / summary["median_s"] * 100.0 for summary in summaries
        )
        print(
            f"\n[s0] marginal gain per promoted block: "
            f"{', '.join(f'{g:+.2f}%' for g in marginal)}  "
            f"(step-time noise ~{noise:.2f}%)"
        )
        beyond_first = [g for g in marginal[1:] if g > noise]
        if not beyond_first:
            print(
                "[s0] VERDICT: compute-bound beyond the first block. Streaming "
                "is fully hidden behind compute, so residency past the head of "
                "the trunk buys VRAM, not speed. The FSM should bank headroom "
                "and guarantee safety -- do not build a speed climb."
            )
        else:
            print(
                "[s0] VERDICT: streaming is NOT fully hidden. The climb is worth "
                "building; size S3's promote gate against these numbers."
            )

    # S0's second question: the training slack pad.
    all_peaks = [
        row["peak_allocated_gib"] for rows in kept.values() for row in rows
    ]
    pad = {
        "peak_allocated_gib_mean": statistics.fmean(all_peaks),
        "peak_allocated_gib_stdev": statistics.stdev(all_peaks),
        "peak_allocated_gib_max": max(all_peaks),
        "peak_spread_gib": max(all_peaks) - min(all_peaks),
    }
    print(
        f"[s0] training peak: mean={pad['peak_allocated_gib_mean']:.3f} GiB "
        f"stdev={pad['peak_allocated_gib_stdev']:.3f} "
        f"max={pad['peak_allocated_gib_max']:.3f} "
        f"spread={pad['peak_spread_gib']:.3f} GiB  "
        f"(slack pad >= spread; sampling's is 0.21 GiB)"
    )
    print(
        f"[s0] dynamo frames during sweep: "
        f"{dynamo_frames_after - dynamo_frames_before} "
        f"(non-zero means residency switches are retracing -- distrust the times)"
    )

    payload = {
        "config": {
            "width": args.width,
            "height": args.height,
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "steps_per_arm": args.steps_per_arm,
            "prefetch_depth": args.prefetch_depth,
            "fp8_training_forward": bool(args.fp8_training_forward),
            "dop": bool(args.dop),
            "dop_single_backward": bool(args.dop_single_backward),
            "compiled": not args.no_compile,
            "total_blocks": total_blocks,
            "full_model_resident_gib": full_resident_gib,
        },
        "arms": summaries,
        "marginal_gain_pct": marginal,
        "training_peak": pad,
        "dynamo_frames": dynamo_frames_after - dynamo_frames_before,
        "samples": {name: rows for name, rows in samples.items()},
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[s0] wrote {out}")

    transformer.disable_immutable_runtime()


if __name__ == "__main__":
    main()
