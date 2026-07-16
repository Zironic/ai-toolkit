# Krea2 content-filter bypass — mechanistic study & intervention plan

> **git-bug:** `fe5dcca` (open) - finish the step-gated teacher evaluation and
> make the distillation decision. Mutable status and evidence belong there.

> **Type:** design / plan doc (durable). Mutable status (what's running, what's
> blocked) belongs in a git-bug ticket, not here. Edit this file when the
> *approach* changes, not to log progress.

## Goal

Reverse-engineer Krea2's baked-in content filter by studying how the **SKC3VO
decensor LoRA's projector vector** interacts with it, and find an intervention
that **preserves the decensor effect while minimizing collateral damage**.

The SKC3VO LoRA works but drags a bundle of unwanted side effects with it:

- **blank/control drift** — changes images that shouldn't change
- **zoom-in / reframing** — pulls a wide/full-body shot to knees-up
- **expression drift** — subject starts smiling
- **eye-color / skin / material drift** — latex/rubber textures, eye-color shifts

The working hypothesis: SKC perturbs something **upstream and broad** (the fused
text representation, before the transformer stack), so its effect cascades
through every block and every image region, producing collateral far beyond
"remove clothing." We want to understand *where* the clothing behavior is
actually decided and intervene there instead.

## Architecture facts (verified)

- **Krea2 `SingleStreamDiT`**: 28 `SingleStreamBlock` + 12 `txtfusion` layers =
  40 "blocks" in the checkpoint index; **162 `Linear` layers** total. Model is
  ~12.8 B params and **99.99 % of params live in Linears** (norms/modulations
  are ~1.4 MB combined — negligible).
- **`txtfusion.projector`**: `Linear [..., 2560, 12] -> [..., 2560, 1]`. The
  12-slot SKC vector is injected as
  `projector_delta = (input * vector).sum(-1) * strength`, added to the
  projector output (see `projector_delta_forward` in
  `scripts/capture_krea_txtfusion_forward.py`). Because the projector sits at the
  *front* of text fusion, its perturbation enters the combined
  `[text | img]` sequence and is broadcast by attention across all image tokens.
- **Turbo** (`krea/Krea-2-Turbo`): fixed `mu=1.15`, 4-step schedule at 512×512 =
  timesteps `[1.0, 0.905, 0.760, 0.513, 0.0]` from
  `timesteps(1024, 4, 256, 6400, mu=1.15)`.
- **Flow matching**: `x_t = (1-t)*clean + t*noise`, `velocity = noise - clean`,
  one-step estimate `x0 = x_t - t*velocity`.
- **Latent/patch geometry** at 512×512: latent 64×64 (VAE f8); patch grid 32×32
  (patch=2) → 1024 image tokens. Clothing band ≈ patch rows 11–21 ≈ latent
  rows 22–44.

## What we've established (findings)

### 1. Single-step RMS is the wrong flip metric — DEAD END
`scripts/find_krea_filter_flip.py` binary-searched a "flip threshold" using
single-step velocity-delta RMS. The response is **non-monotonic** for the blank
prompt: it measures local field perturbation, not a semantic clothed↔unclothed
transition. The real transition is trajectory-level (basin crossing), not a
single-step RMS threshold. Do not use this script's numbers as a flip signal.

### 2. The 12 projector slots are collinear — subset search can't help
`scripts/search_krea_projector_direction.py` analytically scores all 4095
non-empty slot subsets in projector-output delta space (directional-progress
framework: `progress`, `cosine`, off-axis `residual`, control `damage`). **Every
non-trivial subset has `cosine ≈ 1.000`** with the full vector. The slots all
push the same direction; dropping some just scales the effect down. Target
progress and control damage are **locked together** at the projector level — you
cannot find a subset that helps targets disproportionately over controls.
Conclusion: the projector is the wrong level to intervene at.

### 3. No spatially-localized "clothing block" exists
`scripts/measure_krea_block_heatmap.py` hooks all 28 blocks and measures the
SKC-vs-base activation delta per block/timestep, with a spatial `leakage_ratio`
(`outside_mask_rms / inside_mask_rms`) for face / clothing / lower_body masks.
**Leakage ratio = 1.0 ± 0.03 everywhere** — the perturbation is uniform across
all image tokens at every block. This is the expected consequence of a text-side
upstream perturbation broadcast by attention. So block-level causal patching
won't localize the clothing signal; there's nothing to patch.

**But** the *velocity output* does show spatial structure: at **t=0.905**
(Turbo step 2) the velocity delta is concentrated ~2.3× more in the clothing
region than the face. The uniform internal perturbation gets *expressed*
spatially at the output. That's the first place a surgical cut is possible.

### 4. Velocity-delta masking gives per-region control (works) — but not framing
`scripts/score_krea_turbo_candidates.py` runs real 4-step Turbo denoising and
saves images. Candidate `masked_skc` does **two forwards per step** (base + SKC),
masks the velocity delta to clothing latent rows 22:44, and steps with
`base_vel + masked_delta`. Visual results at s=0.05, seed 12345, `exposed_wide`:

| candidate  | top     | bottom  | expression | framing        |
|------------|---------|---------|------------|----------------|
| base       | clothed | clothed | neutral    | full-body      |
| original   | topless | bottomless | smiling | zoomed (knees-up) |
| masked_skc (rows 22:44) | topless | **pants stay** | **not smiling** | still zoomed |

So spatial velocity masking **does** give independent control:
- torso band → topless, lower band → bottomless (separable)
- face band zeroed → the correlated **smile disappears** (smile was face-band
  collateral, learned jointly with "undressed" in SKC's training data)
- **zoom-in is NOT separable** — it's baked into the *torso-band* delta itself,
  because "undress" and "closer framing" co-occur in what SKC learned. No spatial
  window removes it.

Masking did not break the flip: both `original` and `masked_skc` flip for
seed 12345 and stay clothed for seed 54321.

### Mechanistic summary
SKC makes a broad upstream (text-context) change that propagates uniformly.
Collateral (smile, latex, eye color, zoom) rides along because the vector
encodes a joint distribution, not a clean "remove clothing" axis. The **output
velocity field** is the first point where the effect is spatially structured and
thus maskable. Region masking handles *what part of the body*; it cannot undo
*composition* changes that are entangled in the same band.

## Reframing: SKC should be *conditional*, not a global prior shift

The core problem restated: SKC currently behaves as a **global prior shift** — it
changes blank, SFW, and target prompts alike. What we want is a **prompt-
conditional** effect:

|                     | current SKC            | corrected SKC        |
|---------------------|------------------------|----------------------|
| empty prompt + SKC  | changed image prior    | ≈ base               |
| SFW prompt + SKC    | changed image prior    | ≈ base / near-base   |
| target prompt + SKC | prior shift + behavior | target behavior only |

`masked_skc` is already a step in this direction: it reduced `x0_diff_rms_vs_base`
vs `original` in **all six** tested prompt/seed cases, including the `blank` and
`sfw` controls (see the distillation section for the table). The goal is not
"invert SKC" or "recover Krea-before-safety" globally — it is **make SKC's effect
conditional on prompt evidence**, expressed only where/when the context supports
it.

## Behavioral inverse / masked-teacher distillation (level-2 target)

**Framing (adopted).** Three levels of "inverting" the safety behavior:
1. *Exact inverse* — `W_before − W_after`. Unavailable: we have no clean
   pre-safety checkpoint, and safety may have been mixed into main training
   rather than applied as a separable post-hoc finetune (so a discrete `W_after`
   to invert may not even exist conceptually). `Krea-2-Raw` is almost certainly
   "non-distilled", **not** "pre-safety" — do not mistake it for the before.
2. *Behavioral inverse* — a LoRA making the post-safety model behave like a
   desired target for selected prompts/timesteps. **This is our target.**
3. *Sparse/local inverse* — undo only the specific behavior (high-noise clothing
   route, a region, a timestep, a prompt family) rather than everything.

**Caveat carried from findings #2/#3:** localizability is real only at the
*output velocity*, via an explicit mask. Internally the handle is collinear
(#2) and uniform (#3). So a sparse-local inverse is achievable *behaviorally at
the output*, with **no evidence** it is a clean internal weight/activation edit.

**Goal.** Learn a small denoiser-side (student) LoRA that reproduces only the
useful, prompt-conditioned parts of SKC while suppressing its global drift, from
a **single forward** (the teacher uses two: base + SKC).

**Teacher trajectory** (what the student imitates):
```
masked_teacher_v = base_v + timestep_gate(t) * spatial_mask * (skc_v - base_v)
```

**Training objective** — split explicitly by prompt class:
```
empty / control prompts:   loss = student_skc_v  - base_v            # kill global drift
target prompts:            loss = student_skc_v  - masked_teacher_v  # keep conditioned behavior
no-SKC controls:           loss = student_base_v - base_v            # preserve base when SKC absent
```
i.e. the student learns that SKC has *no* global empty-prompt personality and
expresses the target behavior only when the text/context supports it.

**Localization risk (the main thing to track).** The teacher gets its locality
from an explicit spatial/timestep mask. A single-forward student must
*internalize* that locality — reproduce an in-region-only effect from a
mechanism our measurements show acts globally. A small LoRA may not fit this
cleanly. Not a reason to skip the experiment; it is the risk to watch.

### Sequencing: find the teacher recipe BEFORE distilling
Do **not** distill yet — the current teacher still preserves the zoom artifact
(zoom is entangled in the torso band, finding #4). Search the teacher recipe
first via `score_krea_turbo_candidates.py` + visual inspection:

1. base @ step 1, masked SKC @ step 2 only
2. base @ step 1, masked SKC @ steps 2–3
3. weak SKC @ step 1 + masked SKC @ step 2
4. face-excluded SKC (drop face rows) instead of torso-only
5. torso + lower-body mask, face excluded

Choose the teacher by visual acceptance labels:
```
target route succeeds · full-body framing preserved · face/expression preserved
· eye color preserved · latex/collar minimized · blank/SFW drift reduced
```
Only once a recipe passes should the student LoRA try to learn it.

### RESOLVED: step-gating alone fixes zoom, no coefficient decomposition needed
`score_krea_turbo_candidates.py` refactored to take **per-step rules**
(`"base" / "full_skc" / "masked_skc"`, one per denoising step) instead of one
blanket mode for the whole trajectory — see the script's own docstring for the
8 default step-gated recipes. Ran the decisive test at the corrected strength
(`--strength 0.125`) on the Jinx set:

| recipe | jinx_exposed | jinx_clothed | zoom | seam artifact |
|---|---|---|---|---|
| `skc_all` (SKC every step) | nude, extreme close-up | clothed, close-up | broken | - |
| `masked_skc_all` (masked every step) | topless/bottomless split | - | broken | **spliced chimera** |
| `skc1_base_rest` (SKC step 1 only, base after) | close-up portrait | - | broken (confirms zoom = step-1 phenomenon) | - |
| **`base1_skc_rest`** (base step 1, full SKC steps 2-4) | **nude, full-body wide, consistent 2 seeds** | **clothed, full-body wide** | **fixed** | - |
| **`base1_masked_rest`** (base step 1, masked steps 2-4) | **clean torso-exposed/legs-clothed split, full-body** | **clothed, full-body** | **fixed** | **fixed** |

Two things nailed down:
1. **Framing/composition is decided entirely at step 1.** `skc1_base_rest`
   (SKC touches only step 1, base runs steps 2-4) still locks in the
   close-up — proves composition, once set, is not revisited by later steps.
   Conversely `base1_skc_rest` (base step 1, full SKC after) fully preserves
   full-body framing while still producing a clean, strong flip.
2. **The masked_skc seam-splice artifact (visually a hard rectangular seam
   pasting a zoomed face onto a wide-shot body) was exactly base/SKC
   trajectories diverging in composition before the region mask was ever
   applied** — confirmed by fixing it via composition-locking alone. With
   step 1 shared, `base1_masked_rest` reproduces the original clean per-region
   control (torso exposed, legs/boots still clothed) with no artifact.
3. **Control (`jinx_clothed`) respects the explicit outfit** under both
   `base1_skc_rest` and `base1_masked_rest` — stays fully clothed, full-body,
   no seam.

This is a simpler resolution than expected — no per-slot coefficient search,
leave-one-out sweep, or subset generation needed; a **prior collinearity
result actually predicted this**: `search_krea_projector_direction.py` found
cosine ≈ 1.000 between the full vector and nearly every subset in
projector-output-delta space, meaning there's effectively one direction there,
not a "zoom slot" separable from a "decensor slot" — so the fix had to live in
*when* the vector is applied, not *which components* of it. Confirmed exactly
that. The 12-slot single-vector/leave-one-out/regression machinery discussed
above is now deprioritized: only worth reviving if some other collateral
(expression, eye color, texture) survives step-gating and isn't visually
resolved by adjusting *which* step(s) get masked-vs-full SKC.

Next: check remaining collateral (expression/eye-color/texture) on
`base1_skc_rest`/`base1_masked_rest` outputs; if clean, this recipe is the
masked-teacher target for distillation (§ above) — sequencing was "find the
best recipe before distilling," and this is now a strong candidate.

Implementation: `run_denoise` in `score_krea_turbo_candidates.py` computes only
the forward(s) each step's rule needs (a `"base"`-only step skips the SKC
forward entirely). Output: `turbo_score_jinx_stepgated_s0125/`.

### Anchored prompts (built)
Short probe prompts made Turbo ignore framing. Added
`anc_blank / anc_sfw / anc_exposed / anc_emotion` locking camera distance,
framing, expression, eye color, material, background so only clothing + framing
can move — makes drift attributable. `features.pt` rebuilt (24 prompts).
Contrast pairs: `anc_sfw_to_exposed`, `anc_blank_to_exposed`, `anc_sfw_to_emotion`.

But the short `anc_*` prompts still let Turbo drift framing/clothing on their own
(base rendered nude on `anc_blank`, topless on the original over-specified
`anc_exposed`) — the model wasn't actually being tested, it was just complying
with the prompt. Replaced with a much longer **Jinx-character anchor** (Jinx from
Arcane, cataloged physical/hair/tattoo detail, explicit full-body framing lock:
"both feet visible, occupies only 50-55% of image height, not knees-up, not
waist-up"). Confirmed in ComfyUI: base renders **clothed + zoomed-out**; SKC
flips to **naked + zoomed-in** — i.e. this prompt actually exercises the filter
instead of pre-satisfying it. Three variants share the identical base, differing
only in the trailing clause: `jinx_blank` (none), `jinx_exposed` ("She is fully
exposed."), `jinx_clothed` (described dark leather/crop-top/boots outfit, a
control that should stay clothed). Contrast pairs: `jinx_blank_to_exposed`,
`jinx_clothed_to_exposed`, `jinx_blank_to_clothed`.

**Comparison tooling added:** raw MAE/PSNR is confounded by a global
brightness/tint offset SKC introduces (seen up to ±18/255 uniformly), which
makes genuinely-identical images score as "very different." Added SSIM,
best-integer-shift MAE (rules out pure translation), and **pHash/dHash Hamming
distance** (DCT-based, excludes the DC/brightness term) — pHash is brightness-
invariant and is the recommended drift metric going forward; `pHash <= 5` reads
as perceptually identical, `> 10` as clearly different. Amplified diff heatmaps
(`*_diffnorm.png`, `*_diff6x.png`) also written per pair for visual inspection.
Scripts are scratchpad-only so far (`pixel_compare.py`, `pixel_compare2.py`,
`phash_compare.py`) — worth promoting pHash-vs-base into `scores.json` directly
if this becomes a repeated check.

**Strength convention (important, easy to get backwards):** our scripts load
`skc3vo.safetensors` directly (`DEFAULT_LORA`) and apply `--strength` as a raw
multiplier on its extracted 12-value vector (`extract_projector_vector` + `(x *
v).sum(-1) * strength`; skc3vo is stored as rank-1 `lora_A`/`lora_B` with no
alpha key, so this is scale-1, unscaled). A second file, `z0jglf.safetensors`,
was verified to be **exactly** `skc3vo`'s vector times 2.5 (elementwise ratio
2.5000, std 0.0) — i.e. the same adapter baked at higher magnitude, not a
different adapter. So: **`our --strength = 2.5 x (z0jglf strength used in
ComfyUI)`**. A ComfyUI-confirmed working point of "z0jglf @ 5%" therefore
requires `--strength 0.125` here, not `0.05`. Runs before this correction
(`turbo_score_anc_s005`, `turbo_score_jinx_s005`) were at effective z0-equivalent
2% — likely too weak; treat their "subtle drift only" results as inconclusive
rather than a finding about SKC's effect size. Rerun at `0.125` under the new
Jinx anchor is `turbo_score_jinx_s0125_v2`.

**Prompt-truncation bug (found via `turbo_score_jinx_s0125_v2`):** the pHash/SSIM
comparison showed `jinx_blank`/`jinx_exposed`/`jinx_clothed` rendering
**pixel-identical base images per seed** — i.e. the three variants were
indistinguishable to the model. Root cause: text encoding defaulted to
`max_text_length=512` in five places (`capture_krea_txtfusion_forward.py`,
`score_krea_turbo_candidates.py`, `capture_krea_txtfusion_forward_batch.py`,
`measure_krea_block_heatmap.py`, `build_krea_txtfusion_probe_cache.py`), but the
Jinx prompts are 720-776 tokens — the differentiating clause sits at the very
end and was silently truncated off before the tokenizer ever saw it. Confirmed
**not** an architectural limit: `SingleStreamDiT` has no hard-coded sequence cap
(`txtlen`/`imglen` derive from whatever `context.shape[1]` you pass), 512 is
just this repo's default `model_kwargs.get("max_text_length", 512)` fallback,
and other toolkit models here already default higher (`boogu_image`=1024,
`ideogram4`=3072). Fixed: all five defaults bumped to **2000**. Verified the
three Jinx prompts now end at their real final clause, not mid-sentence.
`features.pt` rebuilt; rerun is `turbo_score_jinx_s0125_v3`.
**Any earlier run using a long/detailed prompt (>512 tokens) should be treated
as suspect** until confirmed against this fix.

### Confirmed evidence — masked_skc reduces drift on controls too
`x0_diff_rms_vs_base` (lower = closer to base = less collateral), s=0.05:

| prompt        | seed  | original | masked_skc |
|---------------|-------|----------|------------|
| blank         | 12345 | 0.53     | **0.30**   |
| blank         | 54321 | 0.39     | **0.33**   |
| sfw_wide      | 12345 | 0.45     | **0.36**   |
| sfw_wide      | 54321 | 0.79     | **0.49**   |
| exposed_wide  | 12345 | 0.53     | **0.41**   |
| exposed_wide  | 54321 | 0.30     | **0.16**   |

Lower in all six, including controls — same flip, less collateral. (Partly by
construction: masked applies a strict subset of the delta. The point is it does
so *without losing the target flip*.)

### Fallback
**Local coefficient optimization** on projector subsets — likely low value given
the collinearity finding (#2); keep only as a fallback.

## Tooling

- `scripts/capture_krea_txtfusion_forward.py` — model loader (`build_model`),
  cached-prompt feature loader, `projector_delta_forward` injector. Central deps
  for the rest.
- `scripts/build_krea_txtfusion_probe_cache.py` — encodes prompts (Qwen3-VL) to
  `features.pt`. Prompt library: `.../txtfusion_probe/prompts.json`. TE loads
  only here, then dies (never co-resident with the transformer).
- `scripts/search_krea_projector_direction.py` — CPU-only subset search from
  captures (finding #2).
- `scripts/measure_krea_block_heatmap.py` — block heatmap + `leakage_ratio`
  (finding #3). Output schema `krea_block_heatmap.v2` with `flat_rows`
  (per prompt × mask × layer × timestep) for easy analysis.
- `scripts/score_krea_turbo_candidates.py` — real Turbo denoising + saved images
  + `scores.json`. Candidates: `base`, `original`, `masked_skc`. Has per-forward
  timing instrumentation. `run_turbo_score.ps1` wraps it with a VRAM guard.

### Conventions for these runs (hard rules)
- **VRAM guard**: abort any new GPU run if `nvidia-smi` used > 3 GB (avoid the
  WDDM multi-process crash; combined VRAM across processes crashes ~28 GB).
- **Never load the TE with cached prompts.** Build features once, reuse.
- **50 % (now tunable) layer offloading** on probe runs to stay off the WDDM
  cliff. See offload note below.
- No em-dashes in `.ps1` (cp1252), no non-ASCII in Python `print` (cp1252
  console).
- 4 Turbo steps is enough to tell clothed vs unclothed visually; save images to
  a folder and inspect.
- **Blank is a CONTROL/damage prompt, not a target.** Directional progress on
  targets + low drift on controls = useful; raw RMS crossing = not useful.

## Offload / perf context (why probe VRAM matters)

The probe loads via `build_model(cached_prompt=True)`, which calls
`_load_transformer()` directly and attaches the **plain** `MemoryManager.attach`
(not the smart manager; `SingleStreamDiT` is a plain `nn.Module` with no diffusers
offload path). Relevant facts:

- Plain attach pins all managed Linears on CPU and streams each weight H2D inside
  its own `forward` (`_BouncingLinearFn`). **No lookahead prefetch** — that's a
  bounce-pool (smart) feature. So the plain path gets ~zero transfer/compute
  overlap.
- `offload_percent` selection was a **random per-layer coin-flip**; removed. Now
  it's **deterministic evenly-spaced**: `1.0` streams all (default), `0.5` gives
  `R S R S …` alternation, `0.0` all resident. The alternation deliberately puts
  a resident layer before each streamed one — the layout a future **depth-1
  prefetch** would exploit (kick the next layer's H2D during the resident layer's
  compute). This is the "easy to add prefetch later" seam.
- Because Krea2 is 99.99 % Linear, `offload_percent` ≈ fraction of *model bytes*
  streamed. For the 12 GB card, ~`0.7` (keep ~3.8 GB resident) sits safely under
  the WDDM cliff with VAE + activations.
- Speed lever, if the probe is too slow, is prefetch overlap (bounce pool / smart
  manager) — not the residency pattern. Alternating residency alone does not
  create overlap in eager mode.

See `tasks/done/PREFETCH_2_PLAN.md` and `tasks/done/BLOCK_STREAM_PLAN.md` for the
streaming machinery this rides on.

## Resolved issues

- **FIXED: offload was a silent no-op in the probe path.** The probe's
  `build_model(cached_prompt=True)` (in `scripts/capture_krea_txtfusion_forward.py`)
  calls `Krea2Model._load_transformer()` **directly**, bypassing `load_model()`
  — which is the *only* place `MemoryManager.attach` runs. So `managed=0`: the
  whole 12.8 GB fp8 model stayed resident and just barely fit the 12 GB card,
  grinding against the WDDM cliff (that was the "very slow" runs, not streaming).
  The `offload_percent` knob did nothing because attach never ran.
  - **Not** the class dictionary: torchao float8 keeps module class `Linear`
    (only the *weight* becomes `torchao...Float8Tensor`), already matched by
    `LINEAR_MODULES`. **Not** a stale `.to` monkeypatch either
    (`patch_dequantization_on_save` only overrides `state_dict`; the real `.to`
    override is installed by attach, which wasn't running).
  - Fix: `build_model` now mirrors `load_model`'s percent-path attach after
    `_load_transformer()`, before `.to(device)`. Verified at `p=0.7`:
    managed 184/264, managed weights on CPU, live VRAM 11.95 → **3.92 GiB**,
    nvidia-smi 11.29 → **5.62 GiB**.
  - Related cleanup (separate): `MemoryManager.attach`'s `offload_percent`
    selection was a random per-layer coin-flip; replaced with deterministic
    evenly-spaced interleave (`0.5` → `RSRSRS`), which also seeds a future
    depth-1 prefetch. This was correct all along; it just was never exercised.
  - Note: `pinned_cpu` hit the default 1 GiB pin budget — most CPU weights are
    pageable (slower H2D). Raise `layer_offloading_pinned_weight_gb` if probe
    speed matters.
