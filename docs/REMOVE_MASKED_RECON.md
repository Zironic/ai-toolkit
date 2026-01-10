# Removal plan: Masked Reconstruction — exact code deletions

Summary
-------
This document enumerates the exact files and code blocks that must be removed (or updated) to delete the masked reconstruction feature from the codebase.
Follow the sequence below (deprecate → remove preferred) and make edits in a single PR with tests updated/removed in the same change.

Important: the listings below include 3–5 lines of context so reviewers can easily locate the change. Where a whole file is dedicated to masked recon, mark the file for deletion.

---

1) Remove the toolkit masked-recon implementation (delete file)
---------------------------------------------------------------
File: `toolkit/masked_recon.py`
Action: Delete the entire file.
Rationale: All masked recon logic is in this file (mask generation, previews, and the `apply_masked_recon_loss` entry point).

Snippet (top-of-file context):

```python
from toolkit.losses import masked_mse, luminance_mask_from_images
...

def build_control_mask(ctrl, train_config, target_size, device_torch):
    """Build a processed mask from control tensor(s) that matches the masked-recon pipeline.
    ...
```

Delete everything in `toolkit/masked_recon.py`.

---

2) Remove masked recontruction config fields
--------------------------------------------
File: `toolkit/config_modules.py`
Action: Remove masked_recon config attributes and mask preview options that are used only by masked_recon.
Rationale: These fields are unused once the feature is removed; avoid keeping dead config surface.

Exact block to remove (remove these lines):

```python
        # Masked reconstruction (LumiCtrl-style) — opt-in auxiliary loss
        self.masked_recon_weight: float = kwargs.get('masked_recon_weight', 0.5)  # default enabled; set >0 to enable masked reconstruction
        self.masked_recon_type: str = kwargs.get('masked_recon_type', 'illum')  # 'illum'|'edge'|'custom'
        self.masked_recon_mask_key: Optional[str] = kwargs.get('masked_recon_mask_key', None)  # dataset-provided mask attribute name
        # Control-derived mask tuning
        self.masked_recon_control_threshold: float = kwargs.get('masked_recon_control_threshold', 0.05)
        self.masked_recon_control_dilate: int = kwargs.get('masked_recon_control_dilate', 7)
        self.masked_recon_control_blur: int = kwargs.get('masked_recon_control_blur', 9)
        # Reference resolution used to create masks (build mask at this size then downsample to target)
        self.masked_recon_control_ref_max_size: int = kwargs.get('masked_recon_control_ref_max_size', 1024)
        # Auto-scale dilate based on detected control bbox fraction (useful for full-image controls)
        self.masked_recon_control_dilate_auto: bool = kwargs.get('masked_recon_control_dilate_auto', True)
        self.masked_recon_control_dilate_scale_factor: float = kwargs.get('masked_recon_control_dilate_scale_factor', 4.0)
        # Mask preview/debugging options
        self.mask_preview_enabled: bool = kwargs.get('mask_preview_enabled', False)
        # `mask_preview_max_steps` and `mask_preview_samples_per_step` removed — preview now runs once per job
        self.mask_preview_save_path: str = kwargs.get('mask_preview_save_path', 'output/{job_name}/masks')
        self.mask_preview_overwrite: bool = kwargs.get('mask_preview_overwrite', False)
        # whether to create an overlay image (mask blended over the source image) — default True
        self.mask_preview_overlay: bool = kwargs.get('mask_preview_overlay', True)
```

Additionally update the `allowed_aux` set (remove `'masked_recon'`):

```python
-    allowed_aux = {'none', 'edge', 'masked_recon'}
+    allowed_aux = {'none', 'edge'}
```

---

3) Remove masked recon usage in the trainer
-------------------------------------------
File: `extensions_built_in/sd_trainer/SDTrainer.py`
Action: Remove helper methods, calls, and logging related to masked recon.

Exact edits (remove these sections):

- Remove the `generate_mask_previews_if_enabled` method and its caller:
  - Caller site (in run/init area):

```py
                # Generate mask previews if requested (delegated to helper for testability)
                try:
                    self.generate_mask_previews_if_enabled()
                except Exception:
                    pass
```
  - Method to delete entirely (snippet to remove):

```py
    def generate_mask_previews_if_enabled(self):
        """Run the mask preview generation once per job when enabled.
        ...
            from toolkit.masked_recon import save_mask_previews
            ...
            save_mask_previews(datasets_for_preview, self.train_config, self.sd, save_path, overwrite=overwrite, overlay=overlay)
```

- Remove the local helper `_apply_masked_recon_loss_local` and `masked_recon_logged` init:

```py
        # Helper: compute and optionally apply masked reconstruction loss
        def _apply_masked_recon_loss_local(current_loss):
            try:
                loss_out, mloss = self._compute_and_apply_masked_recon_loss(current_loss, noisy_latents, imgs, batch, dtype)
                return loss_out, mloss
            except Exception as e:
                try:
                    print_acc(f"[MASKED_RECON] failed: {e}")
                except Exception:
                    print(f"[MASKED_RECON] failed: {e}")
                return current_loss, None

        # initialize masked recon logger
        masked_recon_logged = None
```

- Remove the masked recon timer block in the training step (the block that calls helper and sets `masked_recon_logged`):

```py
                # apply masked reconstruction if configured (best-effort, post-loss computation)
                with self.timer('masked_recon'):
                    loss, mloss = _apply_masked_recon_loss_local(loss)
                if mloss is not None:
                    masked_recon_logged = float(mloss.detach())
```

- Remove the wrapper method `_compute_and_apply_masked_recon_loss` which imports and delegates to `toolkit.masked_recon.apply_masked_recon_loss`:

```py
    def _compute_and_apply_masked_recon_loss(self, current_loss, noisy_latents, imgs, batch, dtype):
        """Thin wrapper: delegate masked reconstruction to `toolkit.masked_recon.apply_masked_recon_loss`.
        Returns (loss, mloss_tensor_or_None)
        """
        try:
            from toolkit.masked_recon import apply_masked_recon_loss
        except Exception:
            return current_loss, None

        try:
            return apply_masked_recon_loss(current_loss, self.train_config, self.sd, noisy_latents, imgs, batch, dtype, self.device_torch)
        except Exception as e:
            try:
                print_acc(f"[MASKED_RECON] helper failure: {e}")
            except Exception:
                print(f"[MASKED_RECON] helper failure: {e}")
            return current_loss, None
```

Notes: Removing these sections means masked recon will no longer be invoked during training; also remove any tests that assert masked recon logged values are present.

---

4) Update Base training process (noisy_latents grad preservation)
-----------------------------------------------------------------
File: `jobs/process/BaseSDTrainProcess.py`
Action: Remove the `mr_weight` logic and always detach `noisy_latents` (since masked recon no longer requires preserved grads).

Exact block to replace (remove the mr_weight detection and branch):

```py
            try:
                mr_weight = float(getattr(self.train_config, 'masked_recon_weight', 0.0))
            except Exception:
                mr_weight = 0.0

            if mr_weight == 0.0:
                noisy_latents.requires_grad = False
                noisy_latents = noisy_latents.detach()
            else:
                # preserve grad tracking so auxiliary losses (e.g., masked reconstruction) can backprop
                noisy_latents.requires_grad = True
                try:
                    print_acc(f"[DEBUG-FIX] preserving noisy_latents.grad because masked_recon_weight={mr_weight}")
                except Exception:
                    pass
```

Replace with a deterministic detach:

```py
            noisy_latents.requires_grad = False
            noisy_latents = noisy_latents.detach()
```

---

5) Remove or update scripts that used masked_recon helpers
----------------------------------------------------------
Files: (choose remove or adjust to not import masked_recon)
- `scripts/preview_mask.py` (imports `generate_control_mask`)
- `scripts/debug_mask_dense.py` (imports `generate_control_mask`)
- `scripts/debug_inspect_mask.py` (imports `generate_control_mask`)
- `scripts/debug_combined_case.py` (imports `generate_control_mask`)

Action: Remove masked_recon imports and related code or delete these scripts if they are exclusively for masked recon debugging.

Snippet to remove (example import at top):

```py
from toolkit.masked_recon import generate_control_mask
```

And any calls to `generate_control_mask` / `save_mask_previews` inside those scripts.

---

6) Remove or update UI elements and defaults
--------------------------------------------
Files to update:
- `ui/src/app/jobs/new/jobConfig.ts` — remove masked recon defaults and mask preview fields.

Remove the keys from defaults (delete these lines):

```ts
          // Masked reconstruction defaults (enabled by default)
          masked_recon_weight: 0.5,
          masked_recon_type: 'illum',
          masked_recon_mask_key: null,
          // Mask preview/debug options
          mask_preview_enabled: false,
          mask_preview_save_path: 'output/{job_name}/masks',
          mask_preview_overwrite: false,
          mask_preview_overlay: true,
```

- `ui/src/app/jobs/new/SimpleJob.tsx` — remove the entire "Masked Reconstruction" FormGroup (NumberInput, SelectInput, TextInput) and the "Mask Preview (debug)" section. Example removal snippet (remove these JSX elements):

```tsx
                <FormGroup label="Masked Reconstruction" className="pt-2">
                  <NumberInput ... />
                  <SelectInput ... />
                  <TextInput ... />
                </FormGroup>
                ...
                <FormGroup label="Mask Preview (debug)">
                  <Checkbox ... />
                  <TextInput ... />
                  <Checkbox ... />
                </FormGroup>
```

Note: Also remove any client-side migration/clean-up code that deals with removed keys (e.g., code that deletes `mask_preview_max_steps` or similar) in `jobConfig.ts`.

---

7) Remove related tests
------------------------
Files to delete or rewrite to assert deprecation/warning instead of functionality:
- `testing/test_masked_recon_default.py`
- `testing/test_masked_recon_exception_logging.py`
- `testing/test_masked_recon_helper.py`
- `testing/test_masked_recon_no_mutation.py`
- `testing/test_mask_preview.py`
- `testing/test_masked_reconstruction_integration.py`
- `testing/test_masked_recon_control.py`
- `testing/test_sdtrainer_mask_preview_integration.py`

Action: Delete these tests (if the functionality is removed) or update them to check for a deprecation warning / config validation behaviour.

Example change: replace positive assertion tests with tests that assert the training loop does NOT raise and that the config keys are ignored with logged warning.

---

8) Remove docs and references
------------------------------
Files to update/remove:
- `docs/MASKED_RECONSTRUCTION_IMPLEMENTATION.md` — delete or move to an `archived/` folder and add a short note explaining the feature removal and migration path.
- Any README / FAQ / ControlTrain* docs that mention `masked_recon` (search for `masked_recon` and remove references or add migration notes).

Example lines to remove from docs file (such as the config snippet):

```yaml
  masked_recon_weight: 0.5
  masked_recon_type: 'illum'
```

Add an entry to `LEARNINGS.md` documenting why the feature was removed and the PR reference.

---

9) Config examples and smoke tests
----------------------------------
Search and remove `masked_recon*` keys from: `config/examples/*` and `config/*_smoke_test.*` YAML/JSON. Replace or remove sample keys.

Recommended pattern:
- run `rg "masked_recon|mask_preview" -n config/` and remove any matched keys from example configs.

---

10) Post-change tasks (tests, PR notes)
---------------------------------------
- Add tests that verify old config keys are ignored / produce a deprecation warning (if choosing phased removal), or that they error cleanly if we remove immediately.
- Update `LEARNINGS.md` and project changelog with migration instructions (example: set masked_recon_weight to 0 before this release is cut to avoid surprises). Note whether manual GPU checks were done.
- Mark PR `manual-testing-required` if any GPU/manual verification is required.

---

Appendix: quick repo search patterns
----------------------------------
- rg "masked_recon|mask_preview|masked-reconstruction|mask_recon|reconstruction_mask" -n
- Search UI: rg "mask_preview_|masked_recon_" ui/ -n

---

If you want, I can (1) produce a patch that removes these sections and updates tests (phased removal: emit warnings first), or (2) run a safe automated PR that deletes the above files and updates tests. Which would you prefer?