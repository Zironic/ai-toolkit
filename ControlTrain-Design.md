\# ControlTrain — Design \& Specification



\## Scope



This document defines the \*\*authoritative design, requirements, and acceptance criteria\*\* for ControlTrain. It is intentionally stable and forward-looking.



---



\## Functional Requirements



\### Control Signal Generation



\* Support control generation via:



&nbsp; \* Precomputed controls (preferred)

&nbsp; \* On-the-fly generation (after augmentation)

\* Current control type:



&nbsp; \* `canny`

\* Future control types must follow the same contract.



\### Dataloader Contract



`DataLoaderBatchDTO` MUST support:



\* `control\_tensor: torch.Tensor | None`

\* `control\_tensor\_list: list\[torch.Tensor] | None`

\* `control\_residuals: tuple\[list\[Tensor]] | None`



Rules:



\* Control tensors must be device- and dtype-aligned with training tensors

\* When CFG duplication is active, control tensors must be duplicated identically



\### Training Pipeline



\* Control tensors flow through:

&nbsp; dataloader → `BaseSDTrainProcess.process\_general\_training\_batch` → model forward

\* Unconditional CFG passes must receive \*\*zeroed control inputs\*\*



---



\## Adapter \& ControlNet Policy



\### Defaults



\* ControlNet adapters are \*\*frozen by default\*\* (`adapter.train = false`)

\* Finetuning must be explicitly enabled via config



\### Projection / Shim Policy



\* Shims are \*\*explicit modules\*\*, never implicit reshapes

\* Channel shims: `nn.Conv2d(1×1)`, zero-initialized

\* Width shims: `nn.Linear`, zero or small-xavier init

\* **Per-consumer application:** do **not** apply the adapter projection globally before calling the adapter. Projection must be applied only at consumer sites which explicitly expect the `adapter_in` width (e.g., where `linear.in_features == adapter_in`). This avoids corrupting modules that expect the original `encoder_dim` and prevents matmul shape mismatches.

* Auto-shims are opt-in via config and fail-fast when disabled



\### Save / Load Semantics



\* Shim metadata MUST be persisted with adapter checkpoints

\* Baked adapters MUST load without runtime shim insertion



---



\## Precompute-First Policy



\### Canonical Control Storage



\* Precompute writes one \*\*full-size canonical control image\*\* per source image

\* Resolution-specific variants are optional and explicit



\### Augmentation Rules



\* If `precompute\_control = true`:



&nbsp; \* Geometric augmentations MUST be disabled

\* If geometric augmentations are required:



&nbsp; \* `generate\_on\_the\_fly = true` MUST be used



---



\## Memory \& Offload Requirements



\### Offload Strategies



\* `accelerate` (primary, production, DDP-safe)

\* `memory\_manager` (optional)

\* `manual\_swap` (CPU-only dev/testing)

\* `none`



\### Residual Handling



\* Control residuals must be:



&nbsp; \* Detached

&nbsp; \* Stored on GPU or CPU-pinned memory per config

&nbsp; \* Shape-compatible with UNet per-block expectations



---



\## Configuration Surface (Authoritative)



Key groups:



\* `controlnet.\*`

\* `train.controlnet\_\*`

\* `dataset.control\_\*`



All config keys must:



\* Be validated at startup

\* Fail fast on incompatible combinations



---



\## Acceptance Criteria



\* ControlNet can be enabled/disabled purely via config

\* Control tensors are correctly shaped, typed, and aligned

\* Precompute is idempotent and manifest-driven

\* Frozen ControlNet params are excluded from optimizer groups

\* Shims are explicit, persisted, and test-covered
* Optional encoder hidden-state reduction/projection (`controlnet_encoder_reduce`) is supported and test-covered (e.g., `reduce='mean'`)

\* No silent shape coercion occurs at runtime



---



\## Non-Goals



\* Implicit adapter retraining

\* Silent fallback behaviors

\* GPU CI coverage (manual GPU QA only)

---

## Residual buffering (new)

Residuals produced by adapters should be **buffered** (by default to CPU pinned memory) and only moved to UNet's device at the time of UNet `predict_noise` via the helper `apply_buffered_residuals(pred_kwargs, unet, dtype)`.

* Config knobs: `controlnet_buffer_residuals` (bool, default true); `controlnet_residual_storage` (`'gpu'|'cpu_pinned'`, default `'cpu_pinned'`).
* Rationale: avoids premature device transfers, centralizes the move point, and prevents "tensors on different devices" errors when UNet and adapters are offloaded and brought back at different times.


