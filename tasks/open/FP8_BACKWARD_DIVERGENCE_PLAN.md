# FP8 Backward Divergence Experiment

Ticket: git-bug `fce0b45`. Status/progress live on the ticket; this doc is
the durable design.

## Question

How bad is FP8 grad-input for *training*, measured as compounded trajectory
divergence over N steps -- not as the local numerical error of one backward
pass (which the existing one-time `allclose(rtol=2e-2)` self-check in
`_fp8_grad_input` already loosely bounds).

## What exactly is under test

> **Any divergence measurement taken before the gate was made real must be
> redone.** `layer_offloading_fp8_grad_input` gated only the legacy
> `_BouncingLinearFn.backward` described below. The compiled/arena path
> (`_Fp8LinearTrainingFn.backward`, the one every offloaded run with
> `layer_offloading_fp8_forward` on actually executes) called the native fp8
> grad-input unconditionally and never read the flag, and
> `ArenaOffloadConfig.fp8_backward` was populated but consumed by nothing. Both
> arms of any such A/B therefore ran the *same* fp8 backward: the experiment
> compared the lossy path against itself and its numbers mean nothing. The gate
> is now honored in both paths (captured in the traced forward, so flipping it
> recompiles rather than baking in a stale branch), and the OFF arm dequantizes
> straight to the compute dtype instead of transiting fp32.

`layer_offloading_fp8_grad_input` gates ONE thing, in
`toolkit/memory_management/manager_modules.py` (`_BouncingLinearFn.backward`,
~line 2080):

- gate OFF (baseline): backward dequantizes the fp8 weight to bf16, then
  `grad_input = grad_out @ W_bf16`.
- gate ON: per-output-row weight scales are folded into `grad_out` (in fp32),
  `grad_out` is quantized to fp8 e4m3 with a single per-tensor dynamic scale
  (amax/448 -- the coarsest granularity; one outlier squashes the rest's
  mantissa), and `grad_input = _scaled_mm(..., use_fast_accum=True)`.

The gate is *partial* by construction: those two substitutions (grad_out
operand quantization + fast accumulation) in that one matmul are the ONLY
lossy changes per backward pass. Output dtype (bf16), the LoRA A/B gradient
matmuls, gradients through attention/norms/GELU/modulation, and optimizer
math are all arm-invariant; grad-weight is never computed for the frozen
base.

The LoRA network itself is always fp32 during training regardless of base
quant (Ostris convention -- `network.force_to(..., torch.float32)` in
BaseSDTrainProcess; the smoke harness mirrors it). Consequences for the
experiment: the working set (A/B params, their grads as integrated by the
optimizer, Adam moments) lives in fp32, so update application adds no bf16
rounding of its own -- the nondeterminism floor is cleaner and small fp8-
induced deltas are not masked by parameter-storage quantization. The
contamination path is unchanged: the *incoming* grad_out is still bf16 and
has passed through the downstream base-Linear grad-input hops before feeding
the fp32 LoRA grads. Dumps must save LoRA/optimizer tensors in native fp32,
never cast on save.

The base weights are the *same fp8 bits* in both arms (quantization error of
the weights is common-mode and cancels). Forward is identical in both arms --
the gate touches only backward weight materialization. So the experiment
isolates exactly: fp8 quantization of the activation gradient + fp8 matmul
accumulation, on the path gradients take through frozen base Linears to reach
LoRA down weights and earlier blocks.

Out of scope for phase 1: `_Fp8LinearTrainingFn` (the
`--fp8-training-forward` resident path, whose backward always uses the fp8
grad-input compute) and the in-graph streamed path. Both can reuse the same
analyzer later.

## Experiment design

Driver: extend `scripts/smoke_krea2_train_cuda.py` (already seeded and
dataset-free; fixed latents/noise/timesteps from `--seed`, fresh LoRA + AdamW).

Arms form a 2x2 factorial over the two independent fp8 gates, plus the floor
run. Forward-fp8 (`layer_offloading_fp8_forward`, harness
`--fp8-training-forward`: native fp8 forward, activation quantized instead of
weight dequantized) and grad-input-fp8 are separate flags that cross cleanly
on the eager streamed path.

```
identical seed => identical LoRA init, batch order, noise, timesteps
optimizer state starts empty (identical) in every arm
|- arm bf16-a   : fwd dequant-bf16, bwd bf16   (baseline)
|- arm bf16-b   : same, same seed              (nondeterminism floor)
|- arm fp8-bwd  : fwd dequant-bf16, bwd fp8    (pure backward-kernel error)
|- arm fp8-fwd  : fwd native fp8,  bwd bf16    (pure forward activation-quant
|                                               error: perturbed loss surface)
|- arm fp8-full : fwd native fp8,  bwd fp8     (max-speed operating point +
                                                interaction)
each run 100 steps, dump state at horizons 0,1,5,10,25,50,100
```

The bf16-a vs bf16-b split measures the CUDA-kernel nondeterminism floor;
every fp8 arm is judged against that floor, not against bitwise identity.
(Optionally also run fp8-full twice to check the gates do not *widen* the
floor.) The factorial separates two different error kinds: forward quant
perturbs the objective itself; grad-input quant adds noise to an
otherwise-correct gradient. It also shows whether they compound or partially
cancel.

Constraint: the crossing is only clean on the eager streamed path
(`_BouncingLinearFn`). The resident/compiled path (`_Fp8LinearTrainingFn`)
hardwires fp8 grad-input in its backward regardless of the gate, so the
fp8-fwd cell is not expressible there. All arms run without
`--train-compile-blocks` (the harness default).

### Dumped per horizon (to a `--dump-dir`)

- LoRA network state_dict (A and B tensors).
- Optimizer state_dict (exp_avg, exp_avg_sq, step counters).
- LoRA parameter grads at the horizon step (captured before optimizer.step).
- Fixed-eval model output: one no-grad forward on a *fixed* eval batch
  (latents/noise/timestep/embeds generated from a separate fixed seed,
  identical across arms). Eval always runs in *both* eval-forward modes
  (dequant-bf16 and native fp8) regardless of the arm's training mode: the
  bf16 eval isolates accumulated weight divergence (forward is then
  arm-invariant given the weights, zero sampler contamination); the fp8 eval
  answers the train/inference-consistency question -- a LoRA trained under
  fp8 forward may compensate for the quantization and score *better* under
  fp8 sampling than the bf16-trained one.
- Per-step loss series (all steps, not just horizons).

### Metrics (analyzer script)

Primary parameter-space metric is the effective update, not raw A/B (the
factorization is non-unique):

```
dW = (alpha / r) * B @ A          per LoRA-wrapped layer
rel_fro   = ||dW_fp8 - dW_bf16||_F / ||dW_bf16||_F
cos       = cosine(flatten(dW_fp8), flatten(dW_bf16))
max_abs   = max|dW_fp8 - dW_bf16|
norm_ratio= ||dW_fp8||_F / ||dW_bf16||_F
```

Plus, per horizon:

- gradient cosine similarity per LoRA param (and aggregate),
- optimizer-state relative error (exp_avg, exp_avg_sq),
- fixed-eval output relative L2: `||out_fp8 - out_bf16|| / ||out_bf16||`,
- loss delta series.

Every metric is reported against TWO reference floors in the same table:

1. bf16b-vs-bf16a: the CUDA-kernel nondeterminism floor.
2. the bf16-checkpoint rounding floor: dW of the baseline arm round-tripped
   fp32->bf16->fp32 vs the original (LoRAs train fp32 but `save_weights`
   casts to save_dtype, typically bf16, ~2^-9 relative per element -- every
   real checkpoint undergoes this). Divergence below this line does not even
   survive saving the LoRA: immediately "practically immaterial".

Dumps bypass `save_weights` entirely (`torch.save` of raw fp32 state dicts)
so the metrics themselves are exact. The horizon
curve classifies the growth mode: fixed offset / linear / sqrt(N) diffusive /
unstable amplification.

All per-layer metrics are additionally reported **resolved by block index**
(backward depth). Mechanism being tested: a layer's LoRA grads are computed
in bf16 from its incoming grad_out, but that grad_out has passed through the
fp8 grad-input matmuls of every *downstream* base Linear -- so the last
block's LoRA gradient is nearly clean while the first block's has accumulated
all the fp8 hops. Divergence rising monotonically toward earlier blocks
confirms the mechanism and directly motivates a depth-scoped mitigation
(bf16 grad-input for the first N blocks, fp8 for the rest) as a Phase 2
variant alongside the tproj/blocks scope split.

## Deliverables

1. `scripts/smoke_krea2_train_cuda.py`: add `--fp8-grad-input` (wires
   `MemoryManager.set_fp8_grad_input_enabled`, mirroring trainer wiring of
   `layer_offloading_fp8_grad_input`), `--dump-dir`, `--dump-horizons`,
   `--eval-seed`, and optional `--init-lora` / `--init-optim` (branch both
   arms from an existing trained checkpoint instead of fresh init).
2. `scripts/analyze_fp8_backward_divergence.py`: takes two (or three) dump
   dirs, emits the per-horizon metric table as JSON + readable markdown.
3. `tests/test_fp8_divergence_metrics.py`: CPU unit test of the analyzer math
   on synthetic tensors with known injected error (no GPU, no model).
4. Results summary posted to the ticket (not into this doc).

## Phases

- **Phase 1 -- fresh-init, global gates, 5 arms x 100 steps.** ~6-7 min/arm
  at 512px (~35 min GPU total). This answers the headline question and the
  full-fp8 (max-speed operating point) question in one batch.
- **Phase 2 (conditional on Phase 1 being interesting) -- scope matrix.** The
  gate is currently a module-global (`_FP8_GRAD_INPUT`); add a per-layer
  predicate (include/exclude on layer key, already threaded through
  `ctx.layer_key`) to run: bf16 everywhere / fp8 tproj-only / fp8 blocks-only /
  fp8 everywhere. tproj is the shared modulation projector feeding every
  block's modulation -- high leverage, negligible memory; the blocks are the
  memory-saving bulk. This tells whether the savings and the divergence come
  from the same place. Per-layer gating stays a harness/diagnostic knob unless
  results justify a real config flag.
- **Phase 3 (optional realism upgrades):** branch from a real trained LoRA +
  saved optimizer moments; drive with a small fixed set of *real* cached
  latents instead of random ones (random-latent gradients may have different
  statistics); longer horizon (500+) if 100 steps is still inside the floor.

## Known limitations (accepted for phase 1)

- Random latents + one cached TE embedding, not a real dataset; gradient
  statistics differ from real training. Phase 3 addresses this if needed.
- Fresh LoRA init means early steps have atypically structured gradients
  (B starts at zero). The warm-branch variant exists for this.
- Single arch (Krea2), single rank/lr. Fine: the question is about this
  fork's gate on this card.
- Verdict criterion, in order of decisiveness: (a) divergence at step 100
  below the bf16-checkpoint rounding floor => immaterial, safe to enable by
  default (a normal save discards more). (b) within ~2-3x of the bf16-vs-bf16
  nondeterminism floor AND fixed-eval output error at the same order => safe.
  (c) orders of magnitude above both floors or superlinear growth => keep
  default-off.
