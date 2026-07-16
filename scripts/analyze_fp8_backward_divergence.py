"""Analyzer for the FP8 backward divergence experiment (ticket fce0b45).

Consumes dump dirs written by scripts/smoke_krea2_train_cuda.py --dump-dir
(meta.json + horizon_NNNN.pt + loss_series.json) and emits, per horizon and
per test arm, the plan's metric table (tasks/done/FP8_BACKWARD_DIVERGENCE_PLAN.md):

- effective-update metrics on dW = (alpha/r) * up @ down per LoRA module
  (rel Frobenius, cosine, max abs diff, norm ratio), aggregate + per-block,
- gradient cosine similarity (per param + aggregate),
- optimizer-state relative error (exp_avg, exp_avg_sq),
- fixed-eval output relative L2 in both eval-forward modes,
- loss delta series,

judged against TWO floors printed in the same table: the bf16b-vs-bf16a CUDA
nondeterminism floor (--floor dir) and the bf16-checkpoint rounding floor
(baseline dW round-tripped fp32->bf16->fp32, computed from the baseline alone).

Usage:
    venv\\Scripts\\python.exe scripts\\analyze_fp8_backward_divergence.py ^
        --baseline dumps/bf16-a --floor dumps/bf16-b ^
        --test dumps/fp8-bwd dumps/fp8-fwd dumps/fp8-full ^
        --out-json dumps/report.json --out-md dumps/report.md
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch

DOWN_SUFFIX = ".lora_down.weight"
UP_SUFFIX = ".lora_up.weight"
# Matches the main transformer blocks in any of the naming styles seen in
# dumps ("transformer$$blocks$$0$$attn$$wq", "blocks.3.attn", "..._blocks_17_...")
# while excluding txtfusion refiner_blocks (front-end, not backward depth).
_BLOCK_RE = re.compile(r"(?:^|(?<!refiner)[_.$])blocks[_.$]+(\d+)")


# --------------------------------------------------------------------------
# pure metric math (unit-tested in tests/test_fp8_divergence_metrics.py)
# --------------------------------------------------------------------------

def pair_lora_modules(params):
    """{module_key: (down, up)} from a flat {name: tensor} state dict.

    Modules whose counterpart is missing (e.g. IdentityModule up at full rank)
    are skipped -- dW is only defined for genuine down/up pairs.
    """
    downs = {}
    ups = {}
    for name, tensor in params.items():
        if name.endswith(DOWN_SUFFIX):
            downs[name[: -len(DOWN_SUFFIX)]] = tensor
        elif name.endswith(UP_SUFFIX):
            ups[name[: -len(UP_SUFFIX)]] = tensor
    return {key: (downs[key], ups[key]) for key in downs.keys() & ups.keys()}


def effective_dw(down, up, alpha, rank):
    return (float(alpha) / float(rank)) * (up.float() @ down.float())


class MetricAccumulator:
    """Streaming aggregate of tensor_metrics over many (test, ref) chunks.

    Aggregate cos / rel_fro / norm_ratio over a virtual concatenation
    decompose into per-chunk sums (dot, sum-of-squares, diff-sq, max) -- the
    full concatenation must NEVER be materialized: for the dW comparison it
    is two fp32 copies of the whole transformer's delta (~100 GB).
    """

    def __init__(self):
        self.dot = 0.0
        self.sq_t = 0.0
        self.sq_r = 0.0
        self.sq_diff = 0.0
        self.max_abs = 0.0
        self.numel = 0

    def update(self, test, ref):
        t = test.float().flatten()
        r = ref.float().flatten()
        self.dot += torch.dot(t, r).item()
        self.sq_t += torch.dot(t, t).item()
        self.sq_r += torch.dot(r, r).item()
        d = t - r
        self.sq_diff += torch.dot(d, d).item()
        if d.numel():
            self.max_abs = max(self.max_abs, d.abs().max().item())
        self.numel += t.numel()

    def metrics(self):
        ref_norm = math.sqrt(self.sq_r)
        test_norm = math.sqrt(self.sq_t)
        diff_norm = math.sqrt(max(self.sq_diff, 0.0))
        cos = float("nan")
        if ref_norm > 0 and test_norm > 0:
            cos = self.dot / (test_norm * ref_norm)
        return {
            "rel_fro": diff_norm / ref_norm if ref_norm > 0 else (0.0 if diff_norm == 0 else float("inf")),
            "cos": cos,
            "max_abs": self.max_abs,
            "norm_ratio": test_norm / ref_norm if ref_norm > 0 else float("nan"),
        }


def tensor_metrics(test, ref):
    """rel_fro / cos / max_abs / norm_ratio of test vs ref (any shape)."""
    acc = MetricAccumulator()
    acc.update(test, ref)
    return acc.metrics()


def block_index(module_key):
    """Transformer block index from a LoRA module key, or None (tproj etc.)."""
    m = _BLOCK_RE.search(module_key)
    return int(m.group(1)) if m else None


def dw_metrics(params_test, params_ref, alpha, rank):
    """Per-module + aggregate + per-block dW metrics between two LoRA states."""
    pairs_t = pair_lora_modules(params_test)
    pairs_r = pair_lora_modules(params_ref)
    per_module = {}
    acc = MetricAccumulator()
    # One module's dense dW pair at a time -- never all of them at once.
    for key in sorted(pairs_r.keys() & pairs_t.keys()):
        dw_t = effective_dw(*pairs_t[key], alpha, rank)
        dw_r = effective_dw(*pairs_r[key], alpha, rank)
        per_module[key] = tensor_metrics(dw_t, dw_r)
        per_module[key]["block"] = block_index(key)
        acc.update(dw_t, dw_r)
        del dw_t, dw_r
    aggregate = acc.metrics() if acc.numel else {}
    per_block = {}
    for m in per_module.values():
        per_block.setdefault(m["block"], []).append(m["rel_fro"])
    per_block = {
        ("tproj/other" if b is None else b): sum(v) / len(v)
        for b, v in sorted(per_block.items(), key=lambda kv: (kv[0] is None, kv[0]))
    }
    return {"aggregate": aggregate, "per_module": per_module, "per_block_rel_fro": per_block}


def checkpoint_rounding_metrics(params_ref, alpha, rank):
    """The bf16-checkpoint rounding floor: baseline vs its own bf16 round-trip."""
    rounded = {n: t.to(torch.bfloat16).float() for n, t in params_ref.items()}
    return dw_metrics(rounded, params_ref, alpha, rank)["aggregate"]


def grad_cosines(grads_test, grads_ref):
    per_param = {}
    acc = MetricAccumulator()
    for name in sorted(grads_ref.keys() & grads_test.keys()):
        m = tensor_metrics(grads_test[name], grads_ref[name])
        per_param[name] = m["cos"]
        acc.update(grads_test[name], grads_ref[name])
    aggregate = acc.metrics()["cos"] if acc.numel else float("nan")
    return {"aggregate": aggregate, "per_param": per_param}


def optim_state_rel_error(optim_test, optim_ref):
    """Relative L2 over concatenated exp_avg / exp_avg_sq moments."""
    out = {}
    for moment in ("exp_avg", "exp_avg_sq"):
        acc = MetricAccumulator()
        state_t = optim_test.get("state", {})
        state_r = optim_ref.get("state", {})
        for idx in sorted(set(state_r.keys()) & set(state_t.keys())):
            if moment in state_r[idx] and moment in state_t[idx]:
                acc.update(state_t[idx][moment], state_r[idx][moment])
        out[moment] = acc.metrics()["rel_fro"] if acc.numel else float("nan")
    return out


def eval_rel_l2(eval_test, eval_ref):
    return {
        mode: tensor_metrics(eval_test[mode], eval_ref[mode])["rel_fro"]
        for mode in ("bf16", "fp8")
        if mode in eval_test and mode in eval_ref
    }


def classify_growth(horizon_rel_fro):
    """Rough growth-mode label from {step: rel_fro} (steps >= 1, finite)."""
    pts = [(s, v) for s, v in sorted(horizon_rel_fro.items()) if s >= 1 and v > 0]
    if len(pts) < 3:
        return "insufficient-data"
    logs = [(math.log(s), math.log(v)) for s, v in pts]
    n = len(logs)
    mx = sum(x for x, _ in logs) / n
    my = sum(y for _, y in logs) / n
    denom = sum((x - mx) ** 2 for x, _ in logs)
    slope = sum((x - mx) * (y - my) for x, y in logs) / denom if denom else 0.0
    if slope < 0.25:
        return f"fixed-offset (log-log slope {slope:.2f})"
    if slope < 0.75:
        return f"sqrt-diffusive (log-log slope {slope:.2f})"
    if slope < 1.5:
        return f"linear (log-log slope {slope:.2f})"
    return f"superlinear/unstable (log-log slope {slope:.2f})"


# --------------------------------------------------------------------------
# dump-dir IO + report assembly
# --------------------------------------------------------------------------

def load_arm(dump_dir):
    """Index a dump dir WITHOUT loading horizon payloads (each is ~1 GB;
    compare_arms loads one baseline/test pair at a time)."""
    dump_dir = Path(dump_dir)
    meta = json.loads((dump_dir / "meta.json").read_text(encoding="utf-8"))
    horizons = {
        int(path.stem.split("_")[1]): path
        for path in sorted(dump_dir.glob("horizon_*.pt"))
    }
    loss_path = dump_dir / "loss_series.json"
    losses = (
        json.loads(loss_path.read_text(encoding="utf-8")) if loss_path.exists() else []
    )
    return {"name": dump_dir.name, "meta": meta, "horizons": horizons, "losses": losses}


def load_horizon(arm, step):
    return torch.load(arm["horizons"][step], map_location="cpu", weights_only=True)


def compare_arms(test, base, alpha, rank):
    """Full per-horizon comparison of one arm against the baseline arm."""
    result = {}
    common = sorted(set(test["horizons"]) & set(base["horizons"]))
    for step in common:
        ht, hb = load_horizon(test, step), load_horizon(base, step)
        row = {"dw": dw_metrics(ht["lora"], hb["lora"], alpha, rank)}
        if ht["grads"] and hb["grads"]:
            row["grad_cos"] = grad_cosines(ht["grads"], hb["grads"])
        row["optim"] = optim_state_rel_error(ht["optim"], hb["optim"])
        row["eval_rel_l2"] = eval_rel_l2(ht["eval"], hb["eval"])
        result[step] = row
    loss_b = {r["step"]: r["loss"] for r in base["losses"]}
    result_losses = [
        {"step": r["step"], "loss_delta": r["loss"] - loss_b[r["step"]]}
        for r in test["losses"]
        if r["step"] in loss_b
    ]
    growth = classify_growth(
        {s: r["dw"]["aggregate"].get("rel_fro", 0.0) for s, r in result.items() if r["dw"]["aggregate"]}
    )
    return {"horizons": result, "loss_delta": result_losses, "growth_mode": growth}


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3e}" if (abs(v) < 1e-3 or abs(v) >= 1e4) and v != 0 else f"{v:.6f}"
    return str(v)


def render_markdown(report):
    lines = ["# FP8 backward divergence report", ""]
    ckpt = report.get("checkpoint_rounding_floor", {})
    lines.append(
        f"bf16-checkpoint rounding floor (baseline dW round-trip): "
        f"rel_fro={_fmt(ckpt.get('rel_fro'))} cos={_fmt(ckpt.get('cos'))}"
    )
    lines.append("")
    for arm_name, cmp_ in report["arms"].items():
        lines.append(f"## {arm_name}  (growth: {cmp_['growth_mode']})")
        lines.append("")
        lines.append(
            "| step | dW rel_fro | dW cos | dW max_abs | norm_ratio | grad cos "
            "| optim exp_avg | optim exp_avg_sq | eval bf16 relL2 | eval fp8 relL2 |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for step, row in sorted(cmp_["horizons"].items()):
            agg = row["dw"]["aggregate"]
            gc = row.get("grad_cos", {}).get("aggregate")
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    step,
                    _fmt(agg.get("rel_fro")), _fmt(agg.get("cos")),
                    _fmt(agg.get("max_abs")), _fmt(agg.get("norm_ratio")),
                    _fmt(gc),
                    _fmt(row["optim"].get("exp_avg")),
                    _fmt(row["optim"].get("exp_avg_sq")),
                    _fmt(row["eval_rel_l2"].get("bf16")),
                    _fmt(row["eval_rel_l2"].get("fp8")),
                )
            )
        lines.append("")
        last = max(cmp_["horizons"]) if cmp_["horizons"] else None
        if last is not None:
            per_block = cmp_["horizons"][last]["dw"]["per_block_rel_fro"]
            lines.append(f"Per-block dW rel_fro at step {last} (backward depth):")
            lines.append("")
            lines.append("| block | rel_fro |")
            lines.append("|---:|---:|")
            for block, v in per_block.items():
                lines.append(f"| {block} | {_fmt(v)} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", required=True, help="bf16-a dump dir")
    parser.add_argument(
        "--floor", default=None,
        help="bf16-b dump dir (same-seed rerun): CUDA nondeterminism floor, "
        "reported as just another arm",
    )
    parser.add_argument("--test", nargs="+", default=[], help="fp8 arm dump dirs")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    base = load_arm(args.baseline)
    alpha = base["meta"]["lora_alpha"]
    rank = base["meta"]["lora_rank"]

    last_step = max(base["horizons"])
    report = {
        "baseline": base["name"],
        "checkpoint_rounding_floor": checkpoint_rounding_metrics(
            load_horizon(base, last_step)["lora"], alpha, rank
        ),
        "arms": {},
    }
    arm_dirs = ([args.floor] if args.floor else []) + list(args.test)
    for arm_dir in arm_dirs:
        arm = load_arm(arm_dir)
        label = (
            f"{arm['name']} (nondeterminism floor)"
            if args.floor and Path(arm_dir) == Path(args.floor)
            else arm["name"]
        )
        report["arms"][label] = compare_arms(arm, base, alpha, rank)

    md = render_markdown(report)
    print(md)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    if args.out_json:
        # per_module tables are large; keep them in JSON only.
        Path(args.out_json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
