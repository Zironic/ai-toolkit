\# ControlTrain



\## Overview



ControlTrain adds optional **ControlNet-based conditioning** to the training pipeline, starting with **OpenPose pose control** (default) and designed to extend to other modalities (e.g. Canny, depth).



The goal is to provide \*\*stable, reproducible spatial conditioning\*\* while keeping the core training loop modular, memory-safe, and testable. The default posture is conservative: \*\*precompute-first\*\*, \*\*frozen ControlNet adapters\*\*, and \*\*explicit fail-fast validation\*\*.



\## What ControlTrain Does



\* Generates or loads control signals (currently OpenPose pose maps) aligned with training images

\* Feeds control tensors through the dataloader → training process → model

\* Supports optional ControlNet finetuning, adapter shims, and memory offload strategies

\* Integrates with existing dataset, job, UI, and save/load flows



\## Core Design Principles



\* \*\*Precompute-first\*\* for scale and determinism (with on-the-fly fallback)

\* \*\*Frozen ControlNet by default\*\*; train LoRA/LoKr unless explicitly enabled

\* \*\*Explicit shims, zero-initialized\*\*, never silent reshaping

\* \*\*Fail fast\*\* on mismatches with actionable error messages

\* \*\*Verbose diagnostics\*\* in risky or non-obvious code paths



\## Documentation Map



\* \*\*Design \& Specification\*\* → \[ControlTrain-Design.md](ControlTrain-Design.md)

\* \*\*Current Status \& Roadmap\*\* → \[ControlTrain-Status.md](ControlTrain-Status.md)

\* \*\*References \& Rationale\*\* → \[ControlTrain-Reference.md](ControlTrain-Reference.md)



\## Supported / Planned Control Types



\* ✅ OpenPose (pose / keypoint conditioning)

\* ⬜ OpenPose (pose heatmaps / skeletons)

\* ⬜ Depth / other structural controls



For detailed requirements, configs, and acceptance criteria, see \*\*ControlTrain-Design.md\*\*.
