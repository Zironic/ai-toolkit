# OstrisLinear Arena Quantization Support Plan

> **Completed 2026-07-16:** Arena storage and execution now support every
> current `OstrisLinear` generically through registered buffers, bias, and the
> module's own quantizer. Ticket `c9ee48d` is closed. Hardware-specific fast
> paths remain the responsibility of each quantizer and device.

> **git-bug:** `c9ee48d` ("Add compressed OstrisLinear support to smart arena
> offload"). Mutable implementation status, validation results, and handoff
> notes belong in that ticket.

## Outcome

Smart arena offload must preserve the compressed representation and execution
semantics of every `OstrisLinear` backend used by Krea2. A completed
implementation has all of these properties:

- Krea cache hits cannot silently replace an Ostris quantized module with an
  ordinary full-precision Linear.
- Host accounting, canonical arena storage, and H2D transfers operate on the
  declared ordered tensor storage, including bias, never on
  `OstrisLinear.weight`.
- Streamed and resident compiled blocks call the existing Ostris quantizer
  semantics with explicit tensor state through an operation bound during
  immutable runtime finalization. ConvRot activation transforms, native
  kernels, fallbacks, and training backward remain owned by the quantizer.
- LoRA, LoKr, DoRA, and Linear FullModule continue to work through the existing
  functional adapter seam.
- A later cache format can restore compressed Ostris state directly into final
  canonical destinations without a transient model-sized dequantized payload.
- Native row-wise FP8 uses the same storage/execution boundary: the arena moves
  an opaque ordered tensor tuple, while a bound FP8 operation owns native
  qualification, `_scaled_mm`, autograd, materialization, and fallback.

The central ownership rule is:

> The memory manager treats every layer payload as opaque typed tensor storage.
> It owns packing, pinning, transfer, residency, and lifetime. The layer or
> quantizer owns reconstruction, execution, fallbacks, autograd,
> requantization, and serialization semantics.

This plan adds the storage declaration and execution binding seams missing from
the arena. It does not introduce another quantization backend hierarchy or
reimplement Orbit, OrbitVQ, or ConvRot algorithms.

## Verified Current Constraints

The following are verified against the repository as of 2026-07-14:

- `OstrisLinear._save_to_state_dict()` exports a dequantized `weight` and bias,
  while Krea's `krea2_quantized_transformer_cache.v1` saves
  `transformer.state_dict()`.
- Krea's cache restore understands ordinary tensors and the existing
  Quanto/TorchAO tensor-subclass leaves, but has no Ostris module/state restore
  path.
- FP8 kernel qualification, activation quantization, `_scaled_mm`, autograd,
  grad-input policy, materialization, and fallback selection have been
  extracted to `toolkit/quantization/fp8_linear.py`. Ordered physical leaf
  discovery is in `toolkit/quantization/storage.py`.
- `LinearSpec` and block packs now carry ordered tensor leaves and an execution
  key, but remain Linear-specific through weight templates, weight leaf counts,
  role accessors, and requires-grad fields.
- Memory management still imports FP8 binding APIs in the supported legacy
  manager and current Arena FP8/runtime paths. These are execution-awareness
  seams rather than a reason to preserve the retired in-graph compatibility
  layer.
- Canonical layout discovery reads `module.weight`. For `OstrisLinear` that is
  a dequantizing property, so current arena construction cannot preserve the
  compressed representation.
- The eager `OstrisLinearLayerMemoryManager` already stages registered buffers
  and invokes the quantizer's forward. It proves the required semantics, but it
  mutates module buffers per call and is compiler-disabled, so it is not the
  arena execution design.
- Auto pin sizing in `MemoryManager._desired_pin_bytes_for_offload_ids()` only
  counts `Parameter` values named `weight` and `bias`; it misses Ostris buffers.
- `FunctionalLinear` and the immutable Krea path currently expose explicit
  weight, bias, and scale. The active generic-adapter work already establishes
  the behavioral requirements needed here: callable base, bias access, and
  original-basis weight materialization.
- Arena setup has a transactional pre-commit boundary and a fail-closed
  post-commit boundary. Ostris publication and rollback must preserve that
  lifecycle contract.
- CUDA block compilation is the supported compile validation path. Do not add
  CPU compilation or an MSVC requirement.

## Requirements, Assumptions, And Non-Goals

### Requirements

1. Fix unsafe cache behavior before enabling Ostris arena execution.
2. Use one ordered description of physical Linear storage for byte accounting,
   arena packing, transfer ABI construction, and compressed cache export.
3. Preserve existing plain and row-wise FP8 behavior while migrating FP8 to
   the same opaque storage and bound-operation contract required by Ostris.
4. Keep residency at whole-logical-Linear granularity even though an Ostris
   payload contains multiple leaves.
5. Include ordered tensor names, shapes, dtypes, and an opaque immutable
   execution key in the ABI/fingerprint.
6. Never swap `module._buffers` inside a compiled block.
7. Preserve existing quantizer-owned fallback and backward behavior.
8. Expose real-job behavior through existing job configuration; environment
   variables may be used only by focused diagnostics.
9. Select and bind the functional operation while constructing the immutable
   block program. Compiled invocations receive tensors and ordinary
   scalar/static arguments, never Python callable arguments, dynamic backend
   lookup, or module inspection.
10. Adding an `OstrisQuantizer` backed by ordinary dense tensors must require no
    modification under `toolkit/memory_management/`.
11. Changing FP8 execution policy must require no modification to arena
    packing, transfer, or residency code.
12. Native FP8 eligibility is an execution-binding decision, not a storage
    property. It must not remain in `LinearSpec`, block packs, transfer plans,
    residency metadata, or compile-call arguments.

### Assumptions To Prove Early

- `torch.func.functional_call` may be a valid implementation detail inside a
  quantizer-owned functional operation. Phase 4 will test it through CUDA
  `torch.compile`, checkpoint recomputation, and ConvRot custom-op backward,
  but memory management must not depend on it.
- Registered buffer order is not a sufficient persistent ABI by itself. The
  state descriptor will use stable names and an explicit execution key.
- All tensor-shaped backend state continues to obey the existing
  `OstrisQuantizer` rule that it is registered on the module as a buffer.

### Non-Goals

- A general `QuantBackend`, quantizer registry, or plugin framework.
- Reimplementing the individual Orbit/OrbitVQ/ConvRot kernels.
- Promoting or demoting individual buffers independently.
- Supporting non-tensor external storage, device handles, or genuinely new
  movement primitives. Those would require a separate memory-runtime design.
- Starting a full training job as part of implementation. The user launches
  real jobs; focused synthetic CUDA smokes are the default validation path.

## Design

### Ownership

The quantized layer or quantizer owns:

- explicit declaration, ordering, and names of canonical movable tensors;
- static execution identity;
- functional forward from explicit storage tensors;
- optional original-basis materialization for adapters;
- hardware dispatch, fallbacks, autograd, requantization, and serialization
  semantics.

For native row-wise FP8 this includes activation quantization, `_scaled_mm`,
scale and bias application, device/shape capability qualification, training
autograd, native or materialized `grad_input`, dequantized fallback, and
original-basis materialization for DoRA or LoKr.

The model execution adapter owns:

- associating a logical model Linear with its construction-time bound
  operation;
- capturing tensor indices for that operation;
- applying LoRA, LoKr, DoRA, or Linear FullModule behavior;
- building stable compiled block programs;
- selecting the storage view supplied by the finalized resident/streamed source
  plan without interpreting how memory management produced its tensors.

The memory manager owns:

- canonical host layout, alignment, and packing;
- pin registration and transfer readiness;
- transfer plans and device residency;
- tensor lifetime and stream recording;
- immutable storage ABI validation and fingerprinting;
- returning storage tuples in declared order.

The memory manager may compare or hash the opaque execution key. It does not
select an operation from it or interpret any tensor's quantization meaning.

### 1. Quantizer-owned canonical storage declaration

Each Linear implementation or quantizer explicitly declares its canonical
movable tensors once, in stable order. Bias participates in the same sequence;
it is semantically special to Linear execution but not to storage movement.

```python
@dataclass(frozen=True)
class TensorStorageSpec:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class LayerStorageSpec:
    tensors: tuple[TensorStorageSpec, ...]
    execution_key: tuple  # opaque and immutable to memory management


@dataclass(frozen=True)
class LayerStorageView:
    tensors: tuple[torch.Tensor, ...]
```

Examples of declared sequences are:

```text
plain:       ("weight", "bias")
quanto FP8:  ("qdata", "scale", "bias")
torchao FP8: ("qdata", "scale", "bias")
convrotint4: ("crn_qdata", "crn_scales", "crn_gratio", "bias")
```

The quantizer declaration, not generic enumeration of every module buffer,
defines which tensors are canonical and movable. This avoids accidentally
capturing derived or non-movable buffers. Registered buffer order is not an
ABI. Names and ordering are explicit.

The arena snapshots this declaration during construction. After construction,
memory management never calls back into `storage_tensors()`, enumerates module
buffers again, or asks the quantizer to interpret state. Auto pin sizing,
profiling, packing, transfer planning, residency, and cache destination layout
consume only the frozen specification and tensor bindings.

### 2. Opaque canonical arena ABI

Generalize `LinearSpec` from fixed weight, optional scale, and bias fields to a
logical layer name plus `LayerStorageSpec` and ordered `LeafSpec` bindings.
There is no representation-specific arena kind and no separate bias field in
the canonical movement ABI.

The memory manager may compare or hash `execution_key` to reject incompatible
layouts and form compile-cache identity. It must never branch on values such as
`"convrot4"`, `"orbitvq4"`, or `"fp8_rowwise"`, and must not use the key to
select execution.

The same FP8 qdata/scale/bias tuple may bind a native `_scaled_mm` operation on
one target and a materializing BF16 fallback on another. Its host layout and
transferred bytes are identical. Consequently native eligibility is absent
from the storage ABI and transfer metadata.

Compatibility accessors may temporarily reconstruct today's weight/scale/bias
views while existing consumers migrate, but the ordered tensor tuple is the
only canonical source of truth.

During canonical publication:

- every declared tensor, including bias, is repointed to its canonical host
  view using its declaration binding;
- the module retains its quantizer object and non-tensor static metadata;
- rollback restores every original parameter, buffer, attribute, and class
  affected before commit;
- ranged loading releases source tensors to meta after their bytes have been
  copied, so no second compressed CPU payload survives.

`OstrisLinear.weight` must never be evaluated during declaration capture,
packing, publication, release, transfer planning, or runtime execution.

FP8 backend wrappers additionally normalize into an execution-side semantic
specification. The current native variant is a 2D symmetric weight with one
scale per output row, no zero point, `out_in` layout, E4M3FN weight and dynamic
E4M3FN activation quantization. TorchAO `Float8Tensor` and Quanto
`QBytesTensor` have separate construction adapters that validate their own
metadata and produce the same semantic operation where applicable. Wrapper
leaf names and class identity are not runtime dispatch inputs. Different FP8
dtypes or scale granularities bind a materializing operation or fail closed
until that exact format has a verified native implementation.

### 3. Payload-agnostic transfer and residency

Build compact transfer ranges by iterating each frozen tensor sequence.
Resident sidecars and streamed transfer views return the same
`LayerStorageView.tensors` tuple in declared order. Promotion and demotion
remain atomic for a logical layer and do not mutate canonical module storage.

Fingerprints include ordered names, shapes, dtypes, and the opaque execution
key. Host pointer identity and residency choice remain outside the structural
compile key where current invariants require that.

Memory management owns canonical host layout, alignment, packing, pin
registration, transfer plans, residency, tensor lifetime, stream recording,
ABI validation, and readiness. It does not own tensor interpretation,
materialization, hardware dispatch, fallback, or autograd.

### 4. Construction-time functional operation binding

The quantized layer or quantizer owns functional execution from its declared
tuple and optional original-basis materialization. A narrow contract is:

```python
storage_tensors(module)
execution_key(module)
functional_forward(module, x, storage_tensors)
functional_materialize(module, storage_tensors, dtype)
```

The operation captured by the adapter is conceptually:

```python
@dataclass(frozen=True)
class BoundLinearOperation:
    forward_sample: Callable
    forward_train: Callable
    materialize: Callable | None
```

This is a construction-time description, not a callable object passed through
the compiled ABI. The adapter specializes its block program around the chosen
methods and tensor indices before Dynamo sees the program.

Bias is one entry in `storage_tensors`; its tuple index is known only to the
bound operation. `functional_materialize` is optional when an adapter does not
require an effective original-basis base weight.

During immutable runtime finalization, the model execution adapter associates
each logical Linear with its module and opaque execution key, selects the
quantizer-owned operation, captures its tensor indices, and builds a stable
block program:

```text
runtime finalization:
    module + execution_key
        -> bind functional operation
        -> build block program

compiled invocation:
    activations + storage tensor tuple + adapter tensors
        -> output
```

The operation is selected outside Dynamo and captured structurally by the
block program. It is not passed to the compiled function as a runtime argument.
Compiled invocation performs no dynamic backend lookup or module inspection.

The bound operation may internally use `torch.func.functional_call` if the
Phase 4 proof shows that it compiles and recomputes cleanly. If it does not,
implement explicit tuple-based methods on `OstrisQuantizer`. Either way, that
choice remains outside memory management.

Row-wise FP8 binding performs capability, dtype, dimension, and scale-shape
qualification here. Supported signatures bind the native operation;
unsupported signatures bind the materializing operation. Both consume the
same declared storage tuple.

The model execution adapter also owns applying LoRA, LoKr, DoRA, and Linear
FullModule behavior and requesting original-basis materialization when needed.
Generalize the existing functional adapter primitive only as far as needed to
wrap a bound operation; do not add a parallel adapter architecture.

### 5. Immutable construction and requantization lifecycle

The lifecycle is:

```text
load and quantize module
        -> quantizer declares canonical storage
        -> arena snapshots LayerStorageSpec
        -> arena repoints declared tensors to canonical host views
        -> LoRA/network attachment
        -> runtime finalization binds execution operations
        -> compiled train/sample programs are built
```

After arena construction, the declaration and bindings are immutable.
`requantize_()` may overwrite existing canonical views only when tensor names,
ordering, shapes, and dtypes remain identical. A requantization that changes
the storage schema requires runtime reconstruction; it must not silently
replace buffers or invalidate arena bindings.

### 6. Compressed cache schema

Until compressed restore exists, Krea must not read or write a quantized cache
for any `ostristype`. Change the schema/cache identity so existing v1 entries
cannot be selected, and validate after every supported cache restore that each
expected custom-quantized module is an `OstrisLinear` with the configured
qtype.

After arena execution is stable, add a new compressed schema containing, per
Ostris module:

- module path and qtype;
- original dtype;
- execution metadata;
- the complete ordered named tensor sequence, including bias when present,
  with shapes and dtypes.

Restore by installing `OstrisLinear` and meta placeholders on the meta model,
preparing final canonical destinations, and copying cached buffers directly
into those destinations. Do not call ordinary `state_dict()` for Ostris cache
payloads and do not materialize a full-model BF16 intermediate.

## Implementation Sequence

The neutral FP8 implementation and ordered storage discovery are the reference
baseline for this migration. Phases 1-3 finish separating their integration
from memory movement and must prove that changing FP8 execution policy or
adding a tensor-only quantizer does not require edits to arena packing,
transfer, or residency. Only then does Phase 4 bind Ostris operations to the
established contract.

### Phase 0 - Fail safe and account physical storage

1. Detect `ostristype` before Krea cache lookup/save.
2. Change cache identity and reject/ignore old Ostris cache entries.
3. Disable Ostris cache reads and writes until Phase 6.
4. Add post-load representation validation for supported quantized caches.
5. Add the quantizer-owned ordered storage declaration, with bias in the same
   tensor sequence.
6. Route auto pin sizing and relevant profiling byte counts through it.

Acceptance:

- An Ostris configuration cannot load or write a dequantized v1 cache.
- A deliberately mismatched cache/module representation fails closed.
- Ostris auto pin sizing reports compressed buffers plus bias, not zero and not
  logical BF16 bytes.
- Existing Quanto/TorchAO and plain accounting tests remain unchanged.

Phase 0 is independently releasable and should land before the larger arena
work.

### Phase 1 - Finish binding the extracted FP8 reference operation

1. Define `Fp8LinearSpec` with weight and activation dtype, scale granularity
   and dtype, zero-point presence, weight layout, and execution variant.
2. Add construction adapters for TorchAO `Float8Tensor` and Quanto
   `QBytesTensor`; both explicitly declare `(qdata, scale, bias)` and normalize
   the current rowwise E4M3FN payload without exposing backend identity to the
   operation.
3. Use native and specification-driven materializing FP8 operations in
   `toolkit/quantization/fp8_linear.py`. Native qualification dispatches on the
   semantic specification, never tensor count or wrapper leaf names.
4. Bind the operation in the architecture/model execution adapter during
   immutable runtime finalization from the module,
   target device, and runtime settings. Do not pass a callable or a native
   eligibility flag through the compiled ABI.
5. Remove operation binding and FP8 execution imports from arena layout,
   construction, transfer, and residency code. Temporary legacy-manager
   compatibility wrappers may delegate to `toolkit.quantization`, but the
   immutable arena must receive only a frozen storage declaration and return
   only ordered tensor views.
6. Make activation quantization and materialization consume the bound spec.
   Only rowwise symmetric E4M3FN is native initially; E5M2, E4M3FNUZ, and
   genuinely different scaling formats remain materialized or unsupported
   until their exact CUDA and backward behavior is verified.
7. Keep the existing physical arena layout temporarily while proving execution
   parity; do not combine this slice with the generic `LinearSpec` rewrite.
8. Exercise sampling, training forward/backward, checkpoint recomputation,
   LoRA, native `grad_input`, materialized fallback, and CUDA block compile.

Acceptance:

- FP8 execution behavior and existing numerical tolerances are unchanged.
- No FP8 qualification, operation binding, kernel, autograd, grad-input, or
  dequantization implementation remains in immutable arena packing, layout,
  transfer, or residency modules.
- Any retained legacy-manager FP8 wrapper is an isolated compatibility caller
  of `toolkit.quantization` and is not imported by the immutable arena.
- The same explicit tensor tuple selects native execution on a supported
  signature and materializing execution on an unsupported signature.
- An arbitrary two-leaf FP8 tuple is never inferred to be rowwise symmetric;
  it requires an explicit FP8 construction adapter and semantic spec.
- Adding another backend for the same rowwise E4M3FN payload requires only a
  construction adapter. A new numerical/scaling format may add an execution
  operation but requires no memory-manager change.
- Compiled calls accept tensors and ordinary scalar/static arguments only;
  operation selection and binding happen before Dynamo tracing.

### Phase 2 - Generalize canonical storage and migrate FP8 payloads

1. Convert `LinearSpec`, canonical block records, destination keys, typed views,
   and construction paths to frozen ordered tensor sequences.
2. Declare plain and row-wise FP8 payloads as ordered opaque tensor sequences,
   with bias participating in the same sequence.
3. Remove representation-specific arena kinds and separate bias movement from
   the canonical ABI.
4. Add declared-tensor publication, source-to-meta release, and transactional
   rollback.
5. Extend ranged checkpoint quantization so `add_block()` consumes compressed
   buffers directly.

Acceptance:

- Monkeypatching `OstrisLinear.weight` to raise does not break construction.
- Committed bytes equal aligned compressed payload leaves plus bias.
- No full-sized original-basis weight or duplicate compressed block payload
  survives commit.
- Injected pre-commit failures restore exact module state; post-commit behavior
  retains the established fail-closed lifecycle classification.
- The arena treats a synthetic future dense-tensor quantizer as an opaque
  ordered sequence without adding a representation branch.
- Plain and FP8 construction, packing, and publication use the same generic
  iteration path.

### Phase 3 - Generalize transfer and residency

1. Make transfer plans enumerate frozen tensor sequences rather than fixed
   weight/bias/scale roles.
2. Make source snapshots and resident sidecars assemble the same ordered
   `LayerStorageView.tensors` tuple.
3. Include ordered names, shapes, dtypes, and the opaque execution key in
   structural fingerprints.
4. Preserve deterministic coalescing and whole-Linear promotion/demotion.
5. Delete `kind`, `weight_scale`, `fp8_qualifies`, and `fp8_flags` from arena,
   transfer, residency, and compiled-call data structures. Remove transitional
   weight/scale/bias compatibility accessors once all consumers use tuples.

Acceptance:

- Fully streamed and fully resident execution receive identical logical state.
- H2D statistics equal compressed payload bytes.
- Promotion/demotion leaves canonical storage and module buffer identities
  stable.
- Residency changes alone do not create a new compiled graph.
- Transfer and residency code contains no qtype-specific dispatch.
- Native versus materialized FP8 selection changes no host layout, transfer
  range, H2D byte count, or residency decision.

### Phase 4 - Prove the Ostris functional state contract

1. Define stable tensor names, ordering, schema, opaque execution key, and
   materialization behavior for Ostris state.
2. Define quantizer-owned functional execution from the explicit ordered tuple.
3. Test whether the bound operation can internally use `functional_call` on
   `orbit4`, `orbitvq4`, `convrot4`, `convrot8`, `convrotint4`, and
   `convrotbitnet`.
4. Compare resident module execution with explicit copied-tuple execution for
   output and `grad_input`.
5. Exercise CUDA compile, checkpoint recomputation, and ConvRot custom-op
   backward; count graphs/calls rather than relying on wall time.
6. If necessary, implement explicit tuple-based quantizer methods and repeat
   the proof. Do not add memory-manager execution callbacks.
7. Run declaration/schema coverage for every other registered Ostris qtype.

Acceptance:

- Representative output and input gradients match within backend-appropriate
  tolerances.
- No per-call module buffer mutation is required.
- No unexpected Dynamo graph break is introduced.
- Execution keys distinguish layouts with different interpretation.
- Adding Ostris execution requires no representation branch under
  `toolkit/memory_management/`; it consumes the contract already proven by
  plain and FP8 payloads.

### Phase 5 - Integrate Krea execution and adapters

1. Bind each module's functional operation while constructing the immutable
   Krea block program.
2. Capture tensor indices and optional materialization in a stable bound
   operation outside Dynamo.
3. Make compiled calls accept only activations, ordered storage tensors,
   adapter tensors, and ordinary scalar/static arguments.
4. Extend the callable/materializable base operation used by functional
   adapters without exposing interpretation to memory management.
5. Validate LoRA, LoKr, DoRA, and Linear FullModule with streamed and resident
   Ostris bases.
6. Verify train, sample, and train-to-sample-to-train artifact reuse.

Acceptance:

- ConvRot uses its native custom-op forward and registered backward.
- Orbit/OrbitVQ preserve activation transforms and original-basis
  materialization.
- Hardware fallback decisions remain inside upstream quantizers.
- Adapter output, input gradients, and trainable gradients match resident
  execution and remain finite.
- Phase transitions do not recompile solely because residency changed.
- Dynamo performs no dynamic backend lookup or module inspection.

### Phase 6 - Add compressed Ostris cache v2

1. Implement explicit compressed export metadata and payload serialization.
2. Restore module class, qtype, meta placeholders, and canonical destinations
   without quantizing a materialized full weight.
3. Copy cached leaves directly into the final arena.
4. Validate module path, class, qtype, state names, shapes, dtypes, original
   dtype, and execution key before commit.
5. Re-enable cache reads/writes for `ostristype` only after the full cache
   acceptance matrix passes.
6. Enforce the immutable requantization rule: in-place overwrite is allowed
   only for an identical frozen tensor schema; schema changes require runtime
   reconstruction.

Acceptance:

- Cache miss and cache hit preserve classes, qtypes, compressed byte counts,
  outputs, and `grad_input`.
- A corrupt or incompatible entry fails before canonical commit and rebuilds
  from the checkpoint.
- Cache restore does not create a transient full-model BF16 payload.
- Requantization cannot silently replace a declared canonical tensor or change
  its name, order, shape, or dtype.

### Phase 7 - Close the validation matrix

Run focused CPU/schema coverage for all registered qtypes. Run synthetic real
CUDA coverage for one representative per execution family:

- `qfloat8`
- `float8`
- `orbit4`
- `orbitvq4`
- `convrot4`
- `convrot8`
- `convrotint4`
- `convrotbitnet`

For each applicable representative cover cold load, cache hit, training
forward/backward, checkpoint recomputation, sampling, two train/sample phase
boundaries, mixed residency, fully streamed execution, compiled blocks, and
adapter attachment before runtime finalization.

Required assertions:

- row-wise FP8 native and materializing bindings consume the same storage tuple
  and preserve sampling, training, fallback, LoRA, and `grad_input` behavior;
- no FP8 qualification, execution, dequantization, or autograd implementation
  remains under `toolkit/memory_management/`;
- arena, transfer, and residency data contain no representation kind or native
  FP8 eligibility flag;
- zero synthetic `OstrisLinear.weight` accesses in arena paths;
- no dequantized canonical storage;
- H2D bytes match compressed payload;
- streamed/resident output and `grad_input` parity;
- finite nonzero adapter gradients;
- no residency-only compile-count change;
- compiled programs receive no callable runtime arguments and perform no
  dynamic quantizer lookup;
- cache identity and state preservation;
- pin ledger and runtime ownership return to baseline after teardown;
- a second sequential runtime can prepare and close successfully;
- a synthetic new dense-tensor `OstrisQuantizer` exercises generic packing,
  transfer, and residency without changes under `toolkit/memory_management/`.

Use focused CUDA scripts and tests first. A full Krea training run remains a
separate user-launched acceptance step and must be recorded on the ticket.

## Dependencies And Coordination

- Coordinate canonical layout/lifecycle changes with umbrella ticket `553ffec`
  and acceptance ticket `02fe5b3`.
- Build on the immutable arena implementation tracked by `628b0cb`; do not
  recreate legacy in-graph streaming machinery.
- Build adapter integration on ticket `6dd9ba6` and
  `tasks/done/GENERIC_ADAPTER_IMMUTABLE_RUNTIME_PLAN.md`. The current worktree
  contains uncommitted adapter/runtime changes, so implementation must rebase
  its assumptions on the settled protocol before editing overlapping files.
- Preserve the arena package's separation from the legacy manager.
- If a new test exposes the known order-dependent global-state leak, follow the
  two-run confirmation policy and record it on ticket `f2aceba`; do not bisect
  the suite further.

## Definition Of Complete

The proposal is complete only when:

1. Native FP8 execution is bound outside memory management, and its current
   training, sampling, fallback, LoRA, and native `grad_input` behavior passes
   focused parity tests.
2. Arena packing, transfer, residency, and compiled-call metadata contain no
   FP8 representation kind, interpreted scale role, or native eligibility.
3. Unsafe Ostris cache reads/writes are fixed in released code.
4. Auto pin sizing and diagnostics count compressed physical storage.
5. Canonical storage, transfer, and residency never materialize or transfer an
   Ostris full-precision weight.
6. Compiled Krea train/sample execution preserves each quantizer's existing
   functional semantics and backward behavior.
7. Functional operation selection occurs during immutable program construction;
   compiled invocations contain no callable arguments, dynamic backend lookup,
   or module inspection.
8. All four supported adapter families pass functional parity on an Ostris
   base.
9. Requantization cannot change a frozen tensor schema without explicit runtime
   reconstruction.
10. Compressed cache miss/hit parity passes without a full-model BF16
   intermediate.
11. A new ordinary dense-tensor Ostris backend requires no edit under
   `toolkit/memory_management/`.
12. Focused CPU and CUDA validation passes, teardown returns ownership/ledger
   state to baseline, and the user-launched full-model smoke result is recorded
   on the git-bug ticket.
