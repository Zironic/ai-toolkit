# What this fork adds to Ostris AI Toolkit

> **Audience:** people who already use `ostris/ai-toolkit` and want to know why
> they might run, study, or borrow from `Zironic/ai-toolkit:faster-dop`.
>
> **Snapshot:** 2026-07-14  
> **Fork commit:** `a2080b396b652dbd2a85d8281817486ebcb11873`  
> **Upstream commit:** `18da85153b51eecb4e1005bf6e8d4211712611e2`  
> **Branch status at snapshot:** 88 commits ahead, 1 commit behind

This is not a file-by-file changelog. It is a practical guide to the fork's
headline features: what they are for, who might want them, and which ideas are
useful outside this repository.

For implementation detail beyond this guide, read the code, the plans under
`tasks/`, the rationale docs under `docs/decisions/`, and
`git diff <upstream commit>...<fork commit>` between the two commits above.

---

## The short version

The fork is primarily an experiment in making large diffusion-transformer
training practical and diagnosable on constrained consumer GPUs, especially
12 GB Windows/WDDM systems.

The biggest things to investigate are:

1. **A block-native arena offload runtime** for models that do not fit
   comfortably in VRAM.
2. **Windows-aware memory safety** using NVML, DXGI, centralized pin accounting,
   and a hard allocator guard.
3. **Native FP8 execution and storage-neutral quantization** that can work with
   both Quanto and TorchAO layouts.
4. **A lower-memory Krea2 stack** with ranged loading, quantized-model caching,
   persistent compile caches, and arena integration.
5. **A throwaway text-encoder worker** that keeps large text encoders out of the
   trainer process.
6. **More capable Differential Output Preservation (DOP)** with per-caption
   prompt replacement, reduced-resolution priors, and disk caching.
7. **Crash-resistant checkpointing** with asynchronous atomic writes and
   frequent latest-wins recovery snapshots.
8. **A first-class Anima model integration.**
9. **A large set of CUDA smokes, memory probes, and performance-analysis tools**
   for developers working on similar systems.

---

## At a glance

| Feature | Why an Ostris user might care | Status |
| --- | --- | --- |
| Block-native arena offload | Train a large transformer with only part of its frozen weights resident in VRAM | Implemented; Krea2 is the main production integration |
| NVML/DXGI memory governance | Avoid Windows paging cliffs and pinned-memory allocation crashes | Implemented; useful independently of the arena |
| Native FP8 execution | Reduce weight bandwidth and avoid unnecessary BF16 materialization | Implemented with hardware/layout qualification |
| Block-level `torch.compile` | Compile stable block math instead of a mutation-heavy whole-model wrapper | Implemented in the arena runtime |
| Persistent compile cache | Reuse Inductor/AOTAutograd/Triton artifacts across process restarts | Implemented where the installed PyTorch supports it |
| Krea2 ranged loading and quant cache | Lower startup RAM, avoid repeating expensive quantization | Implemented |
| Text-encoder worker | Prevent a large text encoder and transformer from coexisting in one process | Implemented for compatible model integrations |
| Extended DOP workflow | Faster and more flexible preservation training | Implemented; some parts are local workflow features |
| Async atomic saves | Reduce save stalls and avoid torn checkpoints | Implemented |
| Recovery LoRAs | Lose fewer steps after crashes | Implemented |
| Anima support | Train and preview Anima through the normal `BaseModel` extension system | Implemented |
| Krea projector/vector research tools | Explore low-dimensional text-fusion interventions | Implemented, specialized and experimental |

---

# 1. Block-native arena offload

## Why you might try it

Use this if a transformer's frozen base weights are too large to keep fully in
VRAM, but ordinary per-linear offload is too slow, too mutation-heavy, or too
hard to combine with `torch.compile`.

The arena backend is aimed at this workload:

- a large frozen diffusion transformer;
- a comparatively small trainable adapter;
- enough host RAM for the compressed base model;
- a GPU that can hold activations, optimizer state, a subset of weights, and a
  small transfer ring, but not the complete model;
- repeated transformer blocks with similar execution structure.

## What is different from upstream's ordinary offloader

Upstream's established `MemoryManager` works at the module or linear-layer
level. The fork keeps that backend, but adds a second architecture under:

```text
toolkit/memory_management/arena_offload/
```

The new backend:

- stores each selected block in one canonical pinned host allocation;
- repoints frozen base parameters into that storage once;
- never moves or replaces those parameters again during normal execution;
- keeps selected weights resident as GPU sidecars;
- transfers only the nonresident ranges needed for the current block;
- uses a reusable GPU-side transfer ring;
- invokes the model's original installed block forward through
  `torch.func.functional_call`;
- optionally compiles the functional block kernel;
- owns training/sampling residency transitions and cleanup.

The important design choice is that **residency changes do not mutate the model's
parameter topology**. Host storage stays stable while the runtime changes only
which GPU-side copies are available.

## What a user gets

Potential benefits include:

- a larger model fitting on a smaller card;
- fewer tiny transfer requests than per-linear offload;
- a stable block-level compile boundary;
- explicit accounting for the transfer ring and transient working set;
- sampling and training using the same owned runtime;
- fewer hidden parameter moves at phase boundaries;
- clearer failure behavior when an explicitly requested arena cannot be built.

## Minimal starting point for Krea2

A representative starting configuration is:

```yaml
model:
  layer_offloading: true
  layer_offloading_smart: true
  quantize: true
  qtype: qfloat8

train:
  gradient_checkpointing: true
```

Optional additions include:

```yaml
model:
  layer_offloading_fp8_forward: true
  compile: true
  compile_cache_dir: /path/to/compile-cache
```

More aggressive controls such as native FP8 grad-input, simulated-card sizing,
manual reserves, and sampling FP8 should be treated as tuning or validation
features rather than default recommendations.

## Limits

- Krea2 is the main complete integration and validation target.
- The core dispatcher and discovery code are generic, but that does not mean
  every existing ai-toolkit model is production-ready on the arena backend.
- Arena mode expects frozen base weights and model-owned gradient checkpointing.
- Explicit arena selection is fail-loud: it should not silently drop back to a
  different backend when construction is unsafe or unsupported.
- The best results depend on the GPU, PCIe path, host pin budget, model
  quantization, and workload shapes.

## What is worth copying

Developers building another trainer may want to study:

- transactional final-storage construction;
- immutable parameter identity after setup;
- GPU residency represented by sidecars rather than parameter moves;
- functional saved-forward dispatch;
- static compact transfer plans;
- transfer-ticket lifetime across checkpoint recomputation;
- a runtime facade that owns setup, phase changes, and teardown.

---

# 2. Windows/WDDM memory safety

## Why you might try it

This is relevant even when you do not use the arena backend.

Windows exposes two different failure modes:

1. **Dedicated-VRAM paging:** crossing the physical VRAM cliff can make a job
   silently become dramatically slower instead of throwing an OOM.
2. **Pinned-host/shared-budget exhaustion:** too much page-locked host memory can
   cause a hard CUDA allocation failure even when dedicated VRAM appears free.

The fork treats those as different budgets with different sensors and policies.

## What the fork adds

The host-memory layer includes:

- NVML-backed physical free-VRAM sensing;
- DXGI LOCAL and NON_LOCAL budget queries on Windows;
- a centralized pinned-memory ledger;
- exact-size `cudaHostRegister` accounting;
- managed pinned allocations;
- separate reserves for static weight storage and dynamic staging pools;
- classification of external GPU pressure;
- a hard per-process allocator cap below the WDDM paging cliff;
- simulated smaller-card validation.

The fork deliberately does not trust `torch.cuda.mem_get_info()` by itself for
physical availability on WDDM, because the driver may report what the process
could obtain by paging other tenants rather than what is physically free now.

## What a user gets

- fewer unexplained `cudaErrorMemoryAllocation` failures caused by host pinning;
- less risk of a run silently falling over the Windows paging cliff;
- warnings when another process is materially reducing model residency;
- bounded pinning shared by weights, transfer pools, and checkpoint staging;
- explicit OOM behavior instead of uncontrolled paging when the allocator cap
  is enabled.

## What is worth copying

This is one of the cleanest independently useful parts of the fork.

A project can adopt:

```text
toolkit/memory_management/pin_manager.py
toolkit/memory_management/vram_budget.py
toolkit/memory_management/nvml_meminfo.py
toolkit/memory_management/dxgi_meminfo.py
toolkit/memory_management/allocator_cap.py
```

without adopting the full arena execution model.

---

# 3. Native FP8 and storage-neutral quantization

## Why you might try it

Quantized weights save memory, but a trainer can give much of that benefit back
if every linear layer first materializes a BF16 copy.

The fork tries to keep quantized storage and execution separate so a memory
manager can move the physical tensors without knowing every quantizer's internal
class structure.

## What the fork adds

### A quantization storage declaration layer

`toolkit/quantization/storage.py` describes:

- which physical tensors back a logical weight;
- how those tensors map back to module state;
- how tensor subclasses are reconstructed;
- an execution-format key;
- temporary materialization requirements.

This is used by the arena, the functional dispatcher, and quantized checkpoint
loading.

### A neutral FP8 linear implementation

`toolkit/quantization/fp8_linear.py` normalizes supported FP8 layouts from:

- TorchAO `Float8Tensor`;
- Quanto `QBytesTensor`.

It can then choose between:

- native `torch._scaled_mm` execution when the GPU, shape, scale layout, and
  dtype qualify;
- a correct materializing fallback when they do not.

Forward, sampling, and grad-input policy are separate controls.

### Additional ConvRot formats

Upstream already has the base ConvRot4/ConvRot8 work. The fork adds or wires
additional variants such as:

- `convrotbitnet`;
- `convrotcomfyw4a4`;
- `convrotint2` through `convrotint8`.

## What a user gets

- lower resident weight memory;
- native FP8 GEMMs on supported hardware;
- fewer unnecessary full-precision weight copies;
- consistent handling of Quanto and TorchAO FP8 storage;
- better compatibility between quantized bases, LoRA/LoKr/DoRA, offload, and
  compilation.

## Limits

- Native speedups are hardware- and layout-dependent.
- Unsupported shapes and formats deliberately fall back.
- FP8 grad-input remains workload-sensitive. In the completed 100-step Krea2
  synthetic campaign on the reference RTX 4070, its dW and fixed-evaluation
  divergence stayed below identical-seed BF16 rerun nondeterminism and appeared
  as a fixed offset rather than a compounding trajectory. Native FP8 forward
  was about 18% faster in that campaign and was also non-compounding. This is
  evidence from one architecture, card, and random-latent workload, so validate
  the exact model and training objective before relying on it.
- Numerical behavior should be validated for the exact model and training goal.

## What is worth copying

The most reusable idea is not a specific quantizer. It is the division between:

1. **physical storage declaration**;
2. **residency and movement**;
3. **execution policy**.

That separation makes it easier to add new quantizers without teaching every
memory-management component about each tensor subclass.

---

# 4. Krea2: lower-memory loading, caching, and runtime integration

Krea2 receives the deepest model-specific work in the fork.

## Why you might try it

Use the fork's Krea2 path if you want to:

- train Krea2 on a constrained card;
- avoid loading the entire checkpoint into RAM at once;
- avoid requantizing the transformer on every launch;
- use arena offload and block compilation;
- persist compile artifacts across restarts;
- experiment with prompt-length policy or bounded noise bands.

## Headline Krea2 additions

### Ranged safetensors loading

Instead of loading the complete transformer state dictionary into memory, the
fork can:

- parse the safetensors header;
- read individual tensor ranges;
- validate names and shapes;
- materialize and quantize bounded submodules one at a time;
- release source storage as it proceeds;
- optionally populate final arena destinations.

This targets startup host-RAM spikes as well as runtime VRAM.

### Quantized-transformer cache

The fork can persist the already-quantized Krea2 transformer. The cache identity
covers the checkpoint, dtype, qtype, model configuration, and PyTorch version.

A cache hit avoids repeating the most expensive load-and-quantize work.

### Arena integration

Krea2's smart-offload path prepares canonical storage, estimates an initial
working reserve from configured resolutions, and lets the arena own block
execution and train/sample phase changes.

Percentage-based offload still uses the legacy manager.

### Persistent compile caches

Krea2 uses separate cache identities for training and sampling. Training keys
include dynamic-shape settings so incompatible graph strategies do not share a
cache accidentally.

### Explicit prompt overflow behavior

Instead of silently truncating at the old cap, Krea2 supports:

- `unlimited` as the default policy;
- `error` for strict rejection above `max_text_length`;
- cache versioning so embeddings created by the old truncating behavior are not
  reused.

Example:

```yaml
model:
  model_kwargs:
    prompt_overflow_policy: unlimited
    max_text_length: 512
```

### Bounded noise-band training

Krea2 can restrict the noise-fraction interval used for training:

```yaml
model:
  model_kwargs:
    noise_band_min: 0.0
    noise_band_max: 0.9
```

The default `[0.0, 1.0]` preserves the ordinary schedule.

### SDPA/GQA backend controls

The integration can select and validate grouped-query-attention behavior rather
than relying on one implicit backend choice.

## Specialized Krea research features

The fork also contains more experimental Krea-specific work:

- fixed SKC projector-vector injection;
- text-fusion forward capture and replay;
- projector direction/subset searches;
- block heatmaps;
- emotion and filter probes;
- candidate scoring and latent decoding tools.

These are useful to researchers but are not the first reason most users should
switch forks.

---

# 5. Persistent block compilation

## Why you might try it

`torch.compile` is difficult to combine with a runtime that mutates weights,
closures, or execution wrappers at every layer.

The fork moves the compile boundary to stable functional block kernels.

## What it adds

- one functional kernel per selected repeated block;
- optional block-granular `torch.compile`;
- dynamic sequence-shape bounds derived before the first call;
- compile identities that include model and dynamic-shape configuration;
- persistence through `torch.compiler.save_cache_artifacts()` and
  `load_cache_artifacts()`;
- diagnostics for recompiles, graph breaks, and transfer behavior;
- explicit prevention of a second whole-model compile wrapper when the memory
  runtime already owns compilation.

## What a user gets

- a better chance that compiled graphs survive weight streaming;
- fewer unnecessary shape specializations;
- reduced cold-start cost on later processes when cache artifacts are reusable;
- clearer attribution when performance changes after sampling or a resolution
  transition.

## What is worth copying

Two relatively independent components are useful elsewhere:

```text
toolkit/compile_cache.py
toolkit/compile_shape_bounds.py
```

The broader block dispatcher is more powerful, but also more coupled to the
arena's state-substitution and lifecycle model.

---

# 6. Throwaway text-encoder worker

## Why you might try it

Modern diffusion models often pair a large transformer with a large text or
vision-language encoder. Even if embeddings are cacheable, loading both into one
trainer process can create unnecessary VRAM and host-RAM pressure.

## How the fork changes the workflow

The fork can run a separate `te_only` process that:

1. loads the tokenizers and text components;
2. skips the denoiser and VAE;
3. writes dataset and auxiliary embeddings to disk;
4. exits completely;
5. lets the trainer start in `skip_te` mode with lightweight placeholders.

The auxiliary cache covers embeddings that ordinary dataset-caption caching did
not necessarily persist:

- blank/unconditional embeddings;
- trigger embeddings;
- inline sample conditional/unconditional pairs;
- DOP-transformed prompts.

Cache validity can include hashes of the actual resolved text-encoder and
tokenizer files, not just a model identifier.

## What a user gets

- no co-residency of the heavy text encoder and transformer;
- lower startup memory pressure;
- cleaner reuse of prompt embeddings across resumed jobs;
- better invalidation when local text-encoder weights change without changing
  their path or shape.

## What is worth copying

This worker/cache handoff is broadly useful for any pipeline where one expensive
frozen component produces reusable intermediate tensors.

The strongest parts are:

- separate-process lifetime as the cleanup mechanism;
- manifest-last cache completion;
- atomic per-file writes;
- content-aware fingerprinting;
- explicit fake-component placeholders in the consumer process.

---

# 7. Extended Differential Output Preservation

## Why you might try it

Use this if you already train subject or concept adapters with Differential
Output Preservation and want more control over its cost and prompt behavior.

## What the fork adds

- multiple trigger-to-class replacement pairs;
- per-caption DOP prompts instead of one global class prompt;
- on-disk embedding caches for transformed DOP captions;
- text-worker generation of those embeddings;
- optional reduced-resolution DOP passes;
- per-image prior-prediction caches;
- configurable prior sample count;
- a single-backward option;
- UI controls and warnings for the new combinations.

A representative configuration shape is:

```yaml
train:
  diff_output_preservation: true
  diff_output_preservation_resolution: 256
  dop_single_backward: true
  dop_prior_cache: true
  dop_prior_cache_samples: 12
```

## What a user gets

- lower DOP compute when a reduced prior resolution is acceptable;
- reusable prior predictions across later steps or resumes;
- better behavior for datasets containing more than one trigger/class mapping;
- less repeated text-encoder work.

## Limits

Some DOP behavior is intentionally local workflow code and is not part of the
planned memory-management upstream contribution.

---

# 8. Safer and less disruptive checkpointing

## Why you might try it

Use this if periodic LoRA saves visibly stall training or if long jobs lose too
much work after a crash.

## What the fork adds

### Atomic writes

Checkpoint files are written to a temporary path, optionally flushed, and then
moved into place atomically. Readers should see either the previous complete
checkpoint or the new complete checkpoint, not a partially written file.

### Asynchronous CPU-side saving

The training thread creates a consistent CPU snapshot. Serialization, hashing,
and disk I/O then run on one ordered background worker.

Errors from the worker are surfaced back on the training thread rather than
being silently ignored.

### Pinned snapshot staging

A bounded reusable pinned buffer stages GPU tensors in chunks. This reduces the
number of CUDA synchronizations compared with copying every tensor separately.

### Frequent recovery snapshots

A separate latest-wins recovery LoRA can be written more frequently than normal
numbered checkpoints:

```yaml
save:
  save_every: 1000
  recovery_every: 100
  snapshot_buffer_mb: 64
```

Pending older recovery writes can be coalesced because only the newest recovery
state matters.

## What is worth copying

These components are comparatively self-contained:

```text
toolkit/async_save.py
PinnedStager
save.recovery_every
```

The atomic writer and coalescing queue are useful well beyond diffusion
training.

---

# 9. First-class Anima integration

## Why you might try it

Upstream does not currently provide the same first-class Anima training path.
The fork registers Anima through the normal model-extension system:

```yaml
model:
  arch: anima
```

## What it includes

- `CosmosTransformer3DModel`;
- Qwen3 text encoding;
- `AnimaTextConditioner` with T5 token inputs;
- Qwen-Image/Cosmos VAE handling;
- 4D trainer-facing single-frame latents with 5D model calls;
- correct `0..1000` to `0..1` timestep conversion;
- padding-mask handling;
- quantization exclusions for the 5D patch projection;
- `AdvancedPromptEmbeds` caching;
- TE-worker and `skip_te` support;
- Anima LoRA save/load key conversion;
- embeds-only batched-CFG previews;
- registration, shape, and key-conversion tests;
- a 24 GB example configuration.

## Why developers may care even without using Anima

The integration is an example of moving a nonstandard architecture into a
self-contained `BaseModel` extension rather than scattering architecture checks
through trainer core code.

---

# 10. Operations, UI, and diagnostics

These are not as central as the memory runtime, but they improve day-to-day use.

## UI additions

The fork's guided UI understands additional fields for:

- smart offload;
- WDDM reserves and margins;
- FP8 execution controls;
- compile controls;
- recovery saves;
- DOP resolution and prior caching;
- batch CFG.

It also adds a canvas preview of the selected timestep distribution and
resolution-dependent shift behavior.

## Stale-job reconciliation

The UI can detect a database job still marked `running` after its process has
exited and reconcile it to queued, stopped, or error state.

## Process cleanup

Detached UI workers close job-owned runtime resources before interpreter
shutdown. A Windows watchdog can terminate the process tree if failed cleanup
hangs indefinitely.

## Local ticket viewer

The fork adds a read-only Tickets page that reconstructs git-bug tickets from
Git refs without taking git-bug's store lock.

This is repository-maintenance infrastructure rather than an ai-toolkit training
feature, but it may be useful to teams that keep issue state inside Git.

## Performance and CUDA tooling

The fork includes focused tools for:

- parsing structured performance logs;
- replaying prefetch traces;
- inspecting DXGI budgets and pin behavior;
- simulating working-reserve policy;
- measuring marginal resident-block value;
- benchmarking transfer and checkpoint paths;
- validating FP8 and quantization behavior;
- running small synthetic CUDA smokes;
- guarding shared GPU access with a lock file.

For developers, the testing methodology may be as valuable as the production
code: assert mechanisms and counters on small real-GPU models before spending
hours on a complete training run.

---

# What should you try first?

## You train Krea2 on a 12 GB Windows GPU

Start with:

- smart arena offload;
- qfloat8/FP8 weights;
- gradient checkpointing;
- block compilation only after the noncompiled path is stable;
- a persistent compile-cache directory;
- the default automatic memory reserves before manual tuning.

The most relevant code is:

```text
toolkit/memory_management/arena_offload/
extensions_built_in/diffusion_models/krea2/
toolkit/memory_management/pin_manager.py
toolkit/memory_management/vram_budget.py
```

## Your jobs fail or crawl unpredictably on Windows

Investigate the host/WDDM safety layer first, even if you keep the legacy
per-linear offloader.

Look for:

- external GPU tenants;
- shared-budget pin exhaustion;
- physical VRAM crossing the WDDM paging cliff;
- unbounded pinned allocations;
- allocator caps that are missing or based on an optimistic free-memory signal.

## Your text encoder is nearly as large as your denoiser

Try the TE worker and disk handoff before rewriting the model loader. Process
exit is a reliable way to reclaim every text-encoder allocation.

## Your saves interrupt training

Copy the atomic asynchronous saver and pinned staging buffer. Add recovery
snapshots if the main concern is lost work rather than archival checkpoint
frequency.

## You maintain a custom quantizer

Study the storage-declaration API. It offers a way for offload and functional
execution to understand physical tensor leaves without hard-coding the
quantizer's Python class throughout the runtime.

## You are adding a new model architecture

Study the Anima extension and the arena's saved-forward dispatcher. The useful
pattern is to keep complete model math inside the model integration while
letting shared infrastructure own storage, movement, and lifecycle.

---

# Best features to copy independently

| Subsystem | Value outside this fork | Coupling |
| --- | --- | --- |
| Pin manager + NVML/DXGI sensing | Prevent unbounded host pinning and Windows memory-cliff failures | Low to medium |
| Allocator cap | Convert silent WDDM paging into bounded allocation/OOM behavior | Low |
| Async atomic saver | Faster, crash-resistant checkpoint writes | Low |
| Pinned checkpoint stager | Fewer D2H synchronization points during snapshots | Low |
| TE worker + manifest cache | Avoid co-residency of expensive frozen components | Medium |
| Text-encoder content fingerprint | Correct invalidation after local weight/tokenizer changes | Low |
| Compile artifact cache | Reuse PyTorch compiler artifacts across processes | Low |
| Dynamic sequence-bound derivation | Reduce avoidable recompiles across buckets | Medium |
| Quantization storage declarations | Decouple tensor movement from quantizer semantics | Medium |
| Transactional final-storage loading | Avoid duplicate model representations during startup | Medium to high |
| Functional saved-forward dispatcher | Preserve model-owned block math while substituting runtime state | High |
| Arena residency/transfer runtime | Train models larger than practical resident VRAM | High |
| Process cleanup watchdog | Prevent detached Windows workers from hanging after failure | Low to medium |

---

# What is experimental or specialized?

Treat these as research or advanced tuning rather than general defaults:

- native FP8 grad-input;
- simulated smaller-card mode;
- manual WDDM and working-reserve tuning;
- Krea SKC projector injection;
- Krea text-fusion vector searches;
- specialized ConvRot variants;
- heavy profiling and benchmark scripts;
- local git-bug and agent workflow files.

The branch contains substantially more research and maintenance tooling than an
upstream feature PR would normally include.

---

# What is not unique to this fork anymore?

Some work appeared in the fork's history and later reached upstream by another
path. Current examples that should not be used as reasons to choose the fork
include:

- Krea2 edit conditioning;
- Krea2 reference-token KV caching;
- the base ConvRot4 and ConvRot8 implementations;
- OrbitQuant;
- the Z-Image v2 transformer wrapper and key conversion;
- several general WAN, Qwen Image, LTX, UI, and captioner fixes.

The meaningful comparison is the current net source difference, not every idea
that once landed in the fork first.

---

# Current caveats

- The fork is an active experimental branch, not a conservative drop-in release.
- The block-native arena is explicit and off by default.
- Krea2 is the first and best-validated arena integration; generic internals do
  not imply universal model support.
- Hardware-specific fast paths must qualify at runtime and may fall back.
- Many scripts and workflow files are local research infrastructure, not product
  features intended for upstream.
- At this snapshot the fork is one upstream commit behind: upstream disables the
  Hugging Face xet backend by default because of reported hangs.
- Real gains depend on model size, quantization, GPU architecture, PCIe/host
  memory behavior, resolutions, and training configuration.

---

# Bottom line

For an existing Ostris AI Toolkit user, the strongest reasons to investigate
this fork are:

1. **You need to train Krea2 or another large frozen-base transformer on a
   memory-constrained GPU.**
2. **You run on Windows and want explicit protection against WDDM paging and
   pinned-memory crashes.**
3. **You want native FP8, block compilation, and weight streaming to coexist in
   one owned runtime.**
4. **You want to keep large text encoders out of the trainer process.**
5. **You need more resilient checkpoints or a more capable DOP workflow.**
6. **You want a reference implementation for Anima or for integrating unusual
   model architectures cleanly.**

For a developer, the most transferable lessons are the separation of storage,
residency, and execution; the use of final-destination transactional loading;
and the decision to treat Windows host pinning and physical VRAM as first-class
budgets rather than incidental implementation details.
