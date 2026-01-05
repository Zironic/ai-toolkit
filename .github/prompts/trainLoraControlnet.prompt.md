---
name: trainLoraControlnet
description: Implement LoRA training with a frozen ControlNet adapter.
argument-hint: "<model>, <controlnet>, <referenceRepo>, <planDocs>, <implementationDoc>, <memoryDoc>"
---
You are tasked with implementing LoRA training for a diffusion model that uses a frozen ControlNet adapter. 
- <model>: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- <controlnet>: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- <referenceRepo>: https://github.com/aigc-apps/VideoX-Fun
- <planDocs>: "C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan.md"
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan2.md"
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan3.md"
- <implementationDoc>: "C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Implementation.md"
- <memoryDoc>: "C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Reference.md"

Task checklist:
1. Review the provided plan docs and the reference repo to understand the training flow and expected design patterns. The implementation doc should be used to track progress. The [] boxes below represent the steps to be completed. [X] indicates completed steps.
2. Implement LoRA adapter integration so that:
   - The ControlNet model is loaded and **frozen** (no gradients)
   - LoRA parameters are attached and trained (gradients enabled)
   - The training loop, config, and saved artifacts are compatible with the project's conventions
3. Add fail-fast runtime checks that raise clear errors for missing adapters, invalid training data shapes, or misconfiguration that would corrupt training.
4. Add small, fast unit tests that assert:
   - Adapter loads and attaches properly
   - ControlNet parameters are frozen and LoRA parameters are trainable
   - A single CPU training step with synthetic data updates only LoRA params
5. Update `<implementationDoc>` with what is implemented and completion criteria; update `<memoryDoc>` with findings, caveats, and instructions for future work.
6. Use HF APIs and the reference repo for guidance; cite sources in the implementation docs.
7. Deliverables: code changes, unit tests, updated implementation & memory docs

Constraints & best practices:
- Keep changes small, testable, and well-documented.
- Prefer existing helpers and testing patterns from the project rather than reimplementing utilities.
- Fail-fast: prefer explicit runtime errors and tests that detect misconfiguration early.
- Document any design decisions and unresolved issues in `<memoryDoc>`.
- All required configurations should have sensible defaults, this goes double for any configuration that can not be set by jobs/new simple jobs UI.
- Do not implement fallbacks that are not 1:1 identical with expected behavior. This is a numerical training task and silent fallbacks can corrupt training and make debugging near impossible.

If you're ready, apply the changes in small commits and run the test suite incrementally.
