# LoKr and DoRA Support in the Immutable Runtime

> **Superseded 2026-07-16:** The functional adapter served as an oracle; the
> execution adapter was deleted after the generic block dispatcher passed its
> acceptance gates.

## Decision

Deliver this work in two independently useful tiers.

Tier 0 replaces the current uncaught internal finalization error with a
setup-time diagnostic naming the unsupported adapter and target module. It
lands and ships independently.

Tier 1 adds the explicit functional-Linear seam required for standard LoRA,
LoKr, DoRA, and Linear `FullModule` execution in the immutable runtime. It is
additive to eager behavior: each supported adapter gets a
`functional_forward()` method and its existing `forward()` delegates to it.
The runtime substitutes a device-valid explicit base operation for
`org_forward`.

The immutable runtime is fork-only. Upstream compatibility here means keeping
edits to shared adapter files small, behavior-preserving, and independently
mergeable; it does not constrain fixing the fork runtime.

## Verified failure

`LokrModule` has `lokr_w1`/`lokr_w1_a`, not `lora_down`. During finalization:

1. Krea2 `_lora_owners_on()` ignores the LoKr forward owner.
2. `_collect_lora_entry()` returns `None`.
3. `_has_foreign_forward_hijack()` sees the bound owner without `lora_down`.
4. `_collect_block_loras()` raises
   `CompileRegionError(["unknown_forward_hijack"])`.
5. Nothing catches `CompileRegionError`, so the internal error escapes as
   training starts.

The later `DoRAModule`/`LokrModule` class-name check in
`_collect_lora_entry()` is unreachable for LoKr and is a different guard.

## Scope

Included:

- a clean setup failure for every unsupported adapter/forward hijack;
- a neutral functional leaf protocol;
- standard LoRA, LoKr, DoRA, and Linear `FullModule`;
- parity with current active/merged, multiplier, dropout, rank, gate, and
  assistant-scaling behavior relevant to those adapters; and
- the minimum fork-local Krea2/runtime integration needed to use the explicit
  fetched or resident base weight.

Excluded:

- per-block compiler fallback;
- arbitrary future/synthetic adapter support as an acceptance criterion;
- general LyCORIS support or a general adapter plugin framework;
- non-Linear `FullModule` execution in the immutable runtime; and
- coupling this work to arena extraction or an upstream PR.

## Tier 0: actionable setup failure

### Outcome

An unsupported adapter fails during immutable runtime setup with an actionable
message such as:

```text
arena offload supports standard LoRA adapters; found LokrModule on
blocks.3.attn.to_q
```

### Implementation

In the fork-local Krea2 discovery path, identify the bound forward owner that
makes a canonical leaf unsupported. Raise a public setup/configuration error
containing the feature name, concrete adapter class, complete canonical target
path including block index, and currently supported adapter family.

Do not expose `CompileRegionError` or `unknown_forward_hijack` for this case.
Do not silently disable arena offload or switch memory backends; that can change
job feasibility and must remain a user decision. Preserve compile-region errors
for genuinely internal invariants. Raise at finalization, before the first
model forward.

### Validation and acceptance

- A LoKr-like bound owner without `lora_down` reports `LokrModule` and its full
  target path.
- A second unknown-owner test proves the diagnostic is the floor for all
  unsupported adapters.
- Standard LoRA still finalizes unchanged.
- LoKr no longer exposes `unknown_forward_hijack` or an uncaught
  `CompileRegionError`.

## Tier 1: explicit functional adapter execution

### 1. Neutral functional leaf module

Add one neutral module below the adapter and runtime layers. Select its final
location after checking import direction. It must not import
`network_mixins`, LoRA network classes, Krea2, or arena/offload orchestration.

It defines the functional adapter protocol, a small explicit Linear callable
holding weight, bias, quantization scale, FP8 eligibility, and mode, and only
the composition helper needed by immutable leaf execution. The base call
delegates to `streamed_linear_tensors()` and permits explicit weight/bias
overrides for weight-space adapters. Private sentinels distinguish "use base"
from explicit `None`. It must not retain or copy arena tensors beyond the
current block call.

### 2. Behavior-preserving adapter methods

Add this shape to each named adapter:

```python
def functional_forward(self, inner, x, *args, **kwargs):
    ...

def forward(self, x, *args, **kwargs):
    return self.functional_forward(self.org_forward, x, *args, **kwargs)
```

`inner` replaces the current `self.org_forward` call exactly. This makes eager
execution use the same per-class semantics as immutable execution without
duplicating mathematics.

Implement for:

- standard LoRA in `ToolkitModuleMixin`;
- `LokrModule`, retaining its factorized update;
- `DoRAModule`, obtaining the explicit base weight instead of reading the
  host-backed canonical Parameter; and
- Linear `FullModule`, passing effective explicit weight/bias instead of
  swapping `_parameters` in immutable execution while preserving wider eager
  support.

Keep shared-file changes to the neutral module and small method/delegation edits
in `network_mixins.py`, `models/lokr.py`, `models/DoRA.py`, and the file that
defines `FullModule`. Fork runtime policy must not move into those files.

### 3. Bounded fork-local integration

Replace the fixed LoRA `A`/`B`/scale leaf call with the neutral functional
operation for the four supported constructions. Pass the fetched or resident
device tensor already selected for the leaf.

Use the smallest mechanism that reproduces installed forward ordering. First
test whether the ordered `org_forward` chain can be discovered reliably at
finalization. Add a persistent registry only if those tests show discovery
cannot preserve nesting. Do not build a general lifecycle framework without
that evidence.

Unsupported owners continue through Tier 0. The fork runtime may explicitly
recognize the four supported protocol implementations; construction-independent
extensibility is not this plan's goal. Remove only the fixed tuple/introspection
code superseded by this seam.

### 4. Preserve explicit-weight and FP8 behavior

The immutable block cannot call an adapter's ordinary forward directly because
`org_forward` reaches the canonical Linear whose Parameter points to host arena
storage, not the selected device tensor.

The innermost functional operation must retain the existing explicit
streamed/resident weight path, grad-safe FP8 training autograd operation,
compile-clean native FP8 sampling operation, and ordinary floating-point path.
DoRA and FullModule may materialize an effective weight only when their
mathematics require it. They must not add a second base-weight H2D fetch or
mutate the canonical Parameter.

## Tier 1 implementation order

1. Add per-class eager output/gradient parity tests around
   `functional_forward(self.org_forward, ...)`.
2. Add the neutral explicit Linear and prove no-adapter output and input-gradient
   parity with `streamed_linear_tensors()`.
3. Integrate standard LoRA and retain existing immutable LoRA tests.
4. Integrate LoKr and turn Tier 0's rejection case into successful train/sample
   execution.
5. Integrate DoRA, locking its base-weight and detach semantics to current eager
   behavior.
6. Integrate Linear FullModule without narrowing non-Linear eager behavior.
7. Remove only dead fixed tuple/introspection code after caller searches.
8. Run focused tests and reduced CUDA/FP8 validation, then user-launched
   `scripts/smoke_krea2_train_cuda.py` and
   `scripts/smoke_krea2_inference_cuda.py` runs with each supported
   `--adapter-variant`.

## Tier 1 validation

For all four constructions, compare ordinary eager execution with delegated
functional execution using identical state and controlled RNG. Assert output,
input-gradient, and every adapter-parameter gradient match; frozen base
parameters remain gradient-free; and live state retains current behavior.

Add focused immutable tests using mixed resident/streamed leaves. Assert
canonical Parameter identities and arena storage pointers do not change.

Follow the CUDA test policy with a reduced block and verify backward reaches
adapter parameters and block input, training still enters
`_fp8_linear_training`, sampling still enters `_fp8_linear_compiled`, and no
second H2D fetch or canonical weight copy/mutation occurs. Run focused tests
before the full suite and follow ticket `f2aceba` for known suite-only leaks.
The user launches the final real job through the UI.

## Tier 1 acceptance criteria

- Standard LoRA, LoKr, DoRA, and Linear `full_if_contains` match eager output
  and gradients and complete immutable train/sample execution.
- Eager and immutable calls share each adapter's `functional_forward()`.
- The immutable inner call consumes the selected device weight and never the
  host-backed canonical Linear Parameter.
- Existing FP8 kernels, residency/fetch behavior, arena identity, and block
  compilation remain intact.
- Unsupported adapters receive Tier 0's named setup error.
- No compiler fallback, LyCORIS-general machinery, or arbitrary-adapter
  acceptance requirement is added.

## Risks to verify, not pre-solve

- Chain discovery may be sufficient; test it before adding registry state.
- DoRA deliberately detaches part of its norm calculation. Existing eager
  output and gradients define the contract.
- FullModule supports arbitrary eager module types; only canonical Linear use
  belongs in immutable execution.
- Some functional adapter code may be rejected by `torch.compile`. That remains
  a visible bounded-support issue, not a trigger for automatic fallback.

## Tickets

- Tier 0 is tracked by git-bug ticket `19ca31c` and may land independently.
- Tier 1 is tracked by git-bug ticket `6dd9ba6`.

Tickets own implementation status, validation results, and handoff notes. This
document owns stable design and acceptance criteria.
