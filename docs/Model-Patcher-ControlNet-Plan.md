# Model Patcher ControlNet Integration Plan

**Date:** 2026-01-10  
**Status:** PLANNING  
**Goal:** Add a configuration option to integrate ControlNet via model patching instead of the current two-pass (Z-Image + ControlNet residuals) approach.

---

## Executive Summary

Currently, Z-Image + ControlNet training uses a **two-pass workflow**:
1. **Pass 1 (ControlNet):** Run frozen controlnet on control images → compute residuals
2. **Pass 2 (Main model):** Load residuals and inject them into the main transformer during training

This plan proposes a **configuration option** to switch to a **single-pass integrated approach** using a **model patcher** that:
- Patches the main transformer to accept control inputs directly (similar to how `ControlLoraAdapter` and `SubpixelAdapter` extend models)
- Trains a single forward pass with controls integrated into the main model
- Keeps the same controlnet and base model, but loads and wires them differently

---

## Current Implementation Overview

### Current Two-Pass Flow

**Config example (current):**
```yaml
model:
  name_or_path: path/to/zimage
  controlnet_name_or_path: path/to/controlnet
  controlnet_file: transformer_controlnet.safetensors  # Optional
  is_controlnet_enabled: true

dataset:
  control_residuals_path: datasets/my_dataset/residuals/  # Pre-computed or runtime
  controlnet_mode: zimage
```

**Training loop (simplified):**
1. `setup_controlnet_training()` loads frozen controlnet into `sd.controlnet`
2. `compute_control_residuals()` runs controlnet forward in eval mode → returns detached residuals (stored in `control_residuals_list`)
3. Main model forward pass: residuals are injected via `down_block_additional_residuals` kwarg in transformer forward
4. Loss computed on main model output only (controlnet is frozen, not part of optimizer)

**Key files:**
- [BaseSDTrainProcess.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\jobs\process\BaseSDTrainProcess.py) — `setup_controlnet_training()` (line 287+)
- [controlnet_offload.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\controlnet_offload.py) — `compute_control_residuals()` (line 143+)
- [z_image.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\extensions_built_in\diffusion_models\z_image\z_image.py) — residual injection into transformer (line 1714+)
- [dataloader_mixins.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\dataloader_mixins.py) — residual loading from disk (line 1134+)

---

## Proposed Integrated Model Patcher Approach

### High-Level Design

Instead of two passes, we:
1. **Patch the main transformer** at model load time to accept concatenated control latents (similar to `ControlLoraAdapter` / `I2VAdapter` pattern)
2. **Load controlnet weights into the patcher** (frozen or trainable depending on config)
3. **Single forward pass:** control images → control latents → concat with noisy latents → main transformer → loss

### Architecture Pattern

We follow the existing adapter pattern from:
- [control_lora_adapter.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\models\control_lora_adapter.py) — `ImgEmbedder.from_model()` hijacks `x_embedder.forward()`
- [i2v_adapter.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\models\i2v_adapter.py) — `FrameEmbedder.from_model()` hijacks `patch_embedding.forward()`

**Key insight:** Both adapters **hijack the input embedding layer** to extend input channels and apply a learned projection before passing to the transformer backbone.

### Proposed Flow

```
┌─────────────────────┐
│ Control Images      │
│ (batch preprocessing)│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│ ControlNet Forward      │
│ (frozen or trainable)   │
│ → control_latents       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Concat:                 │
│ [noisy_latents,         │
│  control_latents]       │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Patched Input Embedder  │
│ (extended channels)     │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Main Transformer        │
│ (single forward pass)   │
└──────────┬──────────────┘
           │
           ▼
        [Loss]
```

---

## Implementation Plan

### Phase 1: Configuration & Mode Switching

**Goal:** Add config flag to switch between `residuals` (current) and `integrated` (new) mode.

**Config changes:**
```yaml
model:
  controlnet_integration_mode: residuals  # or 'integrated'
  # When 'integrated':
  controlnet_trainable: false  # default; set true to fine-tune controlnet during training
```

**Files to modify:**
- `toolkit/config_modules.py` — add `controlnet_integration_mode` to `ModelConfig` (default `residuals`)
- `toolkit/config_modules.py` — add `controlnet_trainable: bool = False` to `ModelConfig`

**Validation:**
- If `controlnet_integration_mode == 'integrated'` and `control_residuals_path` is set → warn/error (residuals path is not used in integrated mode)
- If `controlnet_integration_mode == 'residuals'` and `controlnet_trainable == True` → error (cannot train frozen controlnet in residuals mode)

---

### Phase 2: Integrated ControlNet Patcher

**Goal:** Implement a patcher that loads controlnet into the main model and extends input channels.

**New file:** `toolkit/models/integrated_controlnet_adapter.py`

**Key components:**

#### 2.1 `ControlNetInputEmbedder` class
Similar to `ImgEmbedder` and `FrameEmbedder`, this class:
- Wraps the original input embedding layer (`x_embedder` for Flux, `patch_embedding` for Wan)
- Extends input channels to accept `[noisy_latents, control_latents]`
- Hijacks the original layer's `forward()` method

**Pseudocode:**
```python
class ControlNetInputEmbedder(torch.nn.Module):
    def __init__(
        self,
        orig_layer: Union[torch.nn.Linear, torch.nn.Conv3d],
        control_channels: int,
        adapter_ref: weakref.ref,
    ):
        self.orig_layer = orig_layer
        self.control_channels = control_channels
        self.adapter_ref = adapter_ref
        
        # Create a new projection layer with extended input channels
        if isinstance(orig_layer, torch.nn.Linear):
            # Flux-style (x_embedder is Linear)
            orig_in = orig_layer.in_features
            new_in = orig_in + control_channels
            self.extended_layer = torch.nn.Linear(new_in, orig_layer.out_features, bias=orig_layer.bias is not None)
            # Copy original weights for non-control channels
            with torch.no_grad():
                self.extended_layer.weight[:, :orig_in] = orig_layer.weight.clone()
                if orig_layer.bias is not None:
                    self.extended_layer.bias[:] = orig_layer.bias.clone()
                # Initialize control channel weights to zero (or small random)
                self.extended_layer.weight[:, orig_in:] = 0.0
        
        elif isinstance(orig_layer, torch.nn.Conv3d):
            # Wan-style (patch_embedding is Conv3d)
            orig_in = orig_layer.in_channels
            new_in = orig_in + control_channels
            self.extended_layer = torch.nn.Conv3d(
                new_in, orig_layer.out_channels,
                kernel_size=orig_layer.kernel_size,
                stride=orig_layer.stride,
                padding=orig_layer.padding,
                bias=orig_layer.bias is not None
            )
            # Copy original weights
            with torch.no_grad():
                self.extended_layer.weight[:, :orig_in, ...] = orig_layer.weight.clone()
                if orig_layer.bias is not None:
                    self.extended_layer.bias[:] = orig_layer.bias.clone()
                # Zero-init control channels
                self.extended_layer.weight[:, orig_in:, ...] = 0.0
    
    def forward(self, x):
        # x shape: [batch, extended_channels, ...] (already concatenated)
        return self.extended_layer(x)
    
    @classmethod
    def from_model(
        cls,
        model: Union[FluxTransformer2DModel, WanTransformer3DModel],
        adapter: 'IntegratedControlNetAdapter',
        control_channels: int,
    ):
        """Factory method to hijack the model's input embedder."""
        if hasattr(model, 'x_embedder') and isinstance(model.x_embedder, torch.nn.Linear):
            # Flux-style
            orig_layer = model.x_embedder
            embedder = cls(orig_layer, control_channels, weakref.ref(adapter))
            # Hijack forward
            orig_layer._orig_integrated_controlnet_forward = orig_layer.forward
            orig_layer.forward = embedder.forward
            # Update config
            model.config.in_channels = model.config.in_channels + control_channels
            return embedder
        
        elif hasattr(model, 'patch_embedding') and isinstance(model.patch_embedding, torch.nn.Conv3d):
            # Wan-style
            orig_layer = model.patch_embedding
            embedder = cls(orig_layer, control_channels, weakref.ref(adapter))
            orig_layer._orig_integrated_controlnet_forward = orig_layer.forward
            orig_layer.forward = embedder.forward
            # Update config
            model.config.in_channels = model.config.in_channels + control_channels
            return embedder
        
        else:
            raise ValueError(f"Model type {model.__class__.__name__} not supported for integrated controlnet")
```

#### 2.2 `IntegratedControlNetAdapter` class
Main adapter class that:
- Holds reference to controlnet
- Holds reference to patched input embedder
- Provides `get_params()` for optimizer (if controlnet is trainable)
- Provides `forward()` that runs controlnet + concat

**Pseudocode:**
```python
class IntegratedControlNetAdapter(torch.nn.Module):
    def __init__(
        self,
        controlnet: torch.nn.Module,
        sd: 'StableDiffusion',
        model_config: 'ModelConfig',
        train_config: 'TrainConfig',
    ):
        super().__init__()
        self.controlnet = controlnet
        self.sd_ref = weakref.ref(sd)
        self.model_config = model_config
        self.train_config = train_config
        self.device_torch = sd.device_torch
        self.torch_dtype = sd.torch_dtype
        
        # Freeze or unfreeze controlnet
        self.controlnet_trainable = getattr(model_config, 'controlnet_trainable', False)
        if self.controlnet_trainable:
            self.controlnet.requires_grad_(True)
            self.controlnet.train()
        else:
            self.controlnet.requires_grad_(False)
            self.controlnet.eval()
        
        # Determine control output channels
        # For Z-Image controlnets, output is typically a tuple of residuals
        # We need to flatten/concat them into a single tensor
        self.control_channels = self._infer_control_channels()
        
        # Patch the main model's input embedder
        if hasattr(sd, 'transformer'):
            self.input_embedder = ControlNetInputEmbedder.from_model(
                sd.transformer,
                self,
                self.control_channels
            )
        elif hasattr(sd, 'unet'):
            # For SD1.5/SDXL style models (future expansion)
            raise NotImplementedError("UNet-based models not yet supported for integrated controlnet")
        else:
            raise ValueError("No transformer or unet found on StableDiffusion model")
    
    def _infer_control_channels(self) -> int:
        """Run a dummy forward to infer control output shape."""
        # Create dummy inputs matching expected shapes
        dummy_control = torch.zeros(1, 3, 64, 64, device=self.device_torch, dtype=self.torch_dtype)
        dummy_latents = torch.zeros(1, 16, 64, 64, device=self.device_torch, dtype=self.torch_dtype)
        dummy_timesteps = torch.zeros(1, device=self.device_torch, dtype=self.torch_dtype)
        
        with torch.no_grad():
            out = self.controlnet(dummy_control, dummy_latents, dummy_timesteps)
        
        # Handle tuple of residuals (Z-Image style)
        if isinstance(out, (tuple, list)):
            # Flatten and sum channels
            # This is model-specific; Z-Image returns multiple scales
            # For now, we take the first (largest) scale
            # TODO: implement proper multi-scale aggregation
            control_tensor = out[0]
        else:
            control_tensor = out
        
        return control_tensor.shape[1]  # channels dimension
    
    def get_control_latents(
        self,
        control_images: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Run controlnet and return control latents ready to concat."""
        if self.controlnet_trainable:
            # Training mode: keep gradients
            out = self.controlnet(control_images, noisy_latents, timesteps)
        else:
            # Frozen mode: no gradients
            with torch.no_grad():
                out = self.controlnet(control_images, noisy_latents, timesteps)
        
        # Handle tuple of residuals
        if isinstance(out, (tuple, list)):
            # Take first scale for now
            # TODO: implement multi-scale fusion strategy
            control_latents = out[0]
        else:
            control_latents = out
        
        return control_latents
    
    def get_params(self):
        """Return parameters for optimizer."""
        params = []
        
        # Input embedder is always trainable (new control projection)
        params += list(self.input_embedder.extended_layer.parameters())
        
        # Add controlnet params if trainable
        if self.controlnet_trainable:
            params += list(self.controlnet.parameters())
        
        return params
```

---

### Phase 3: Training Loop Integration

**Goal:** Modify training loop to use integrated adapter when `controlnet_integration_mode == 'integrated'`.

**Files to modify:**

#### 3.1 `BaseSDTrainProcess.py`
Add branch in `setup_controlnet_training()`:

```python
def setup_controlnet_training(self):
    if not getattr(self.sd, 'is_controlnet_enabled', False):
        return
    
    integration_mode = getattr(self.model_config, 'controlnet_integration_mode', 'residuals')
    
    if integration_mode == 'residuals':
        # Current implementation (unchanged)
        self._setup_controlnet_residuals_mode()
    
    elif integration_mode == 'integrated':
        # New integrated mode
        self._setup_controlnet_integrated_mode()
    
    else:
        raise ValueError(f"Unknown controlnet_integration_mode: {integration_mode}")

def _setup_controlnet_residuals_mode(self):
    """Current implementation (extract existing code into this method)."""
    # ... existing setup code ...
    pass

def _setup_controlnet_integrated_mode(self):
    """New integrated mode setup."""
    from toolkit.models.integrated_controlnet_adapter import IntegratedControlNetAdapter
    
    # Load controlnet if not already loaded
    if self.sd.controlnet is None:
        # Use existing lazy load logic
        # ... (similar to current implementation) ...
        pass
    
    # Create integrated adapter
    self.integrated_adapter = IntegratedControlNetAdapter(
        controlnet=self.sd.controlnet,
        sd=self.sd,
        model_config=self.model_config,
        train_config=self.train_config,
    )
    
    # Assign to process-level adapter for compatibility
    self.adapter = self.integrated_adapter
    
    # Verify integration
    self.print_and_status_update(f"[CONTROLNET] Integrated mode enabled (trainable={self.integrated_adapter.controlnet_trainable})")
```

#### 3.2 Training loop modification
In the main training loop (varies by process type, but generally in `hook_train_loop`):

**Current (residuals mode):**
```python
# Load residuals from batch or compute on-the-fly
control_residuals = batch.control_residuals_list  # or compute_control_residuals(...)

# Inject into transformer kwargs
transformer_kwargs['down_block_additional_residuals'] = control_residuals

# Forward pass
noise_pred = self.sd.predict_noise(
    latent_model_input=noisy_latents,
    text_embeddings=text_embeddings,
    timestep=timesteps,
    **transformer_kwargs
)
```

**New (integrated mode):**
```python
# Get control latents from adapter
control_latents = self.integrated_adapter.get_control_latents(
    control_images=batch.control_tensor,
    noisy_latents=noisy_latents,
    timesteps=timesteps,
)

# Concat with noisy latents
latent_model_input = torch.cat([noisy_latents, control_latents], dim=1)

# Forward pass (no additional residuals kwarg)
noise_pred = self.sd.predict_noise(
    latent_model_input=latent_model_input,
    text_embeddings=text_embeddings,
    timestep=timesteps,
    # No down_block_additional_residuals in integrated mode
)
```

**Detection logic:**
```python
integration_mode = getattr(self.model_config, 'controlnet_integration_mode', 'residuals')
if integration_mode == 'integrated' and hasattr(self, 'integrated_adapter'):
    # Use integrated mode logic
    pass
elif integration_mode == 'residuals':
    # Use current residuals mode logic
    pass
```

---

### Phase 4: Optimizer & Parameter Management

**Goal:** Ensure optimizer includes integrated adapter parameters when in integrated mode.

**Files to modify:**

#### 4.1 `BaseSDTrainProcess.py` — optimizer setup
In `hook_before_train_loop()` or wherever optimizer is created:

```python
def _get_trainable_params(self):
    """Collect all trainable parameters based on training mode."""
    params = []
    
    # Main model params (existing logic)
    if self.train_config.train_unet:
        params += list(self.sd.unet.parameters())
    # ... other main model params ...
    
    # Add integrated adapter params if present
    integration_mode = getattr(self.model_config, 'controlnet_integration_mode', 'residuals')
    if integration_mode == 'integrated' and hasattr(self, 'integrated_adapter'):
        adapter_params = self.integrated_adapter.get_params()
        params += adapter_params
        self.print_and_status_update(f"[CONTROLNET] Added {len(adapter_params)} parameters from integrated adapter")
    
    return params
```

#### 4.2 Validation
- If `controlnet_trainable == True`, verify controlnet params are in optimizer
- If `controlnet_trainable == False`, verify controlnet params are NOT in optimizer (only input embedder)

---

### Phase 5: Testing & Validation

**Goal:** Ensure integrated mode works correctly and produces comparable results to residuals mode.

#### 5.1 Unit Tests
Create `testing/test_integrated_controlnet.py`:

**Test cases:**
1. **Model patching:** Verify input embedder is correctly extended
2. **Forward pass:** Run dummy forward and verify shapes
3. **Optimizer params:** Verify correct params are included based on `controlnet_trainable`
4. **Frozen controlnet:** Verify gradients are disabled when `controlnet_trainable == False`
5. **Trainable controlnet:** Verify gradients flow through controlnet when `controlnet_trainable == True`

**Example test:**
```python
def test_integrated_controlnet_forward():
    # Create dummy model and controlnet
    # ... setup ...
    
    adapter = IntegratedControlNetAdapter(controlnet, sd, model_config, train_config)
    
    # Run forward
    control_latents = adapter.get_control_latents(
        control_images=torch.zeros(2, 3, 64, 64),
        noisy_latents=torch.zeros(2, 16, 64, 64),
        timesteps=torch.tensor([0.5, 0.5]),
    )
    
    # Verify shape
    assert control_latents.shape[0] == 2  # batch
    assert control_latents.ndim == 4  # [B, C, H, W]
```

#### 5.2 Smoke Test Config
Create `config/test_integrated_controlnet.yaml`:

```yaml
job: train
config:
  name: test_integrated_controlnet
  process:
    - type: sd_trainer
      training_folder: output
      device: cuda:0
      
      model:
        name_or_path: path/to/zimage
        controlnet_name_or_path: path/to/controlnet
        is_controlnet_enabled: true
        controlnet_integration_mode: integrated  # NEW
        controlnet_trainable: false  # NEW (default)
      
      network:
        type: lora
        linear: 16
        linear_alpha: 16
      
      train:
        batch_size: 1
        steps: 10
        lr: 1e-4
      
      datasets:
        - folder_path: datasets/jinx_cropped_tiny
          control_path: datasets/jinx_cropped_tiny  # Control images
          # NOTE: control_residuals_path is NOT used in integrated mode
```

**Validation:**
- Run config for 10 steps
- Verify no errors
- Check loss is reasonable (compare to residuals mode baseline)
- Verify checkpoint saves correctly

#### 5.3 Comparison Test
Run **identical config** in both modes and compare:
1. Loss curves (should be similar, not identical due to initialization differences)
2. GPU memory usage (integrated mode may use slightly more during forward pass)
3. Training speed (integrated mode may be faster due to single pass)
4. Final output quality (generate samples and compare)

---

### Phase 6: Documentation

**Goal:** Document new config options and usage patterns.

#### 6.1 User Documentation
Create `docs/ControlNet_Integration_Modes.md`:

**Contents:**
- Overview of residuals vs integrated mode
- When to use each mode
- Config examples
- Performance characteristics
- Troubleshooting

#### 6.2 Code Comments
Add docstrings to all new classes and methods explaining:
- Purpose and design
- Input/output shapes
- Relationship to existing adapter patterns

#### 6.3 Update `AGENTS.md`
Add entry to quick reference:
```markdown
### ControlNet integration patterns
- `toolkit/models/integrated_controlnet_adapter.py` — Single-pass integrated controlnet via model patching
- `toolkit/controlnet_offload.py` — Two-pass residuals mode (existing)
- `jobs/process/BaseSDTrainProcess.py` — `setup_controlnet_training()` branching logic
```

---

## Configuration API

### Model Config Options

```python
class ModelConfig:
    # Existing
    controlnet_name_or_path: Optional[str] = None
    controlnet_file: Optional[str] = None
    is_controlnet_enabled: bool = False
    
    # NEW
    controlnet_integration_mode: str = 'residuals'  # 'residuals' | 'integrated'
    controlnet_trainable: bool = False  # Only applies when integration_mode == 'integrated'
```

### Dataset Config Options

**Residuals mode (current):**
```yaml
datasets:
  - folder_path: path/to/images
    control_path: path/to/control_images
    control_residuals_path: path/to/residuals  # Pre-computed or computed on-the-fly
```

**Integrated mode (new):**
```yaml
datasets:
  - folder_path: path/to/images
    control_path: path/to/control_images
    # control_residuals_path is ignored in integrated mode
```

---

## Migration Path for Existing Users

### Backwards Compatibility

**Default behavior:** `controlnet_integration_mode = 'residuals'` (no change for existing users)

**Opt-in:** Users must explicitly set `controlnet_integration_mode: integrated` to use new mode

### Validation & Warnings

```python
# In config validation
if model_config.controlnet_integration_mode == 'integrated':
    if dataset_config.control_residuals_path is not None:
        warnings.warn(
            "control_residuals_path is set but controlnet_integration_mode='integrated'. "
            "Residuals will not be used. Set controlnet_integration_mode='residuals' to use pre-computed residuals."
        )
    
    if model_config.controlnet_trainable and not train_config.train_unet:
        raise ValueError(
            "controlnet_trainable=True requires train_unet=True (main model must be trainable)"
        )

if model_config.controlnet_integration_mode == 'residuals':
    if model_config.controlnet_trainable:
        raise ValueError(
            "controlnet_trainable=True is only supported in 'integrated' mode. "
            "Set controlnet_integration_mode='integrated' to train controlnet."
        )
```

---

## Open Questions & Design Decisions

### 1. Multi-Scale Residual Fusion

**Problem:** Z-Image controlnets return multiple scales of residuals (tuple). In residuals mode, these are injected at different transformer layers. In integrated mode, we concat at input — how to handle multiple scales?

**Options:**
- **A) Use only first (largest) scale** — Simplest, may lose information
- **B) Flatten and concat all scales** — Increases input channels significantly
- **C) Learn a fusion layer** — Adds trainable params, more flexible
- **D) Hierarchical patching** — Patch multiple layers (complex)

**Recommendation:** Start with **A** for Phase 1. Add **C** as follow-up if quality degrades.

### 2. Controlnet Trainability

**Question:** Should we allow fine-tuning the controlnet in integrated mode?

**Considerations:**
- **Pro:** More flexibility, allows adapting controlnet to specific styles
- **Con:** Risk of catastrophic forgetting if not careful with LR/regularization
- **Con:** Increases memory usage significantly

**Recommendation:** No

### 3. Inference Compatibility

**Question:** How to save and load integrated checkpoints for inference?

**Options:**
- **A) Merge controlnet into main model weights** — Simplest for inference, but large checkpoint
- **B) Save separate controlnet + patcher weights** — Smaller, requires loading both at inference
- **C) Save as LoRA of input embedder only** — Lightest, but requires controlnet at inference

**Recommendation:** Start with **B** (separate saves). Add **A** as an export option later.

### 4. Performance & Memory

**Question:** What are the memory/speed tradeoffs?

**Initial hypothesis:**
- **Memory:** Integrated mode uses slightly more during forward (larger input tensor), but saves memory by not storing intermediate residuals
- **Speed:** Integrated mode may be faster (single pass vs two passes), but depends on controlnet size
- **Training stability:** Unknown — may have different gradient flow characteristics

**Action:** Benchmark both modes with identical configs and document findings in `LEARNINGS.md`.

---

## Implementation Checklist

### Phase 1: Config & Mode Switching
- [ ] Add `controlnet_integration_mode` to `ModelConfig`
- [ ] Add `controlnet_trainable` to `ModelConfig`
- [ ] Add validation logic for invalid combinations
- [ ] Add warnings for unused config options
- [ ] Update config schema documentation

### Phase 2: Integrated Adapter Implementation
- [ ] Create `toolkit/models/integrated_controlnet_adapter.py`
- [ ] Implement `ControlNetInputEmbedder` class
  - [ ] Support Flux-style `x_embedder` (Linear)
  - [ ] Support Wan-style `patch_embedding` (Conv3d)
- [ ] Implement `IntegratedControlNetAdapter` class
  - [ ] Controlnet forward + concat logic
  - [ ] Multi-scale residual handling (initial: first scale only)
  - [ ] `get_params()` for optimizer
- [ ] Add initialization strategy (zero-init control channels)

### Phase 3: Training Loop Integration
- [ ] Refactor `setup_controlnet_training()` into mode-specific methods
- [ ] Add `_setup_controlnet_integrated_mode()`
- [ ] Modify training loop to detect integration mode
- [ ] Add control latent computation in integrated mode
- [ ] Update dataloader to skip residual loading in integrated mode

### Phase 4: Optimizer & Parameters
- [ ] Update optimizer setup to include integrated adapter params
- [ ] Add param count logging
- [ ] Verify frozen controlnet excludes gradients correctly
- [ ] Verify trainable controlnet includes gradients correctly

### Phase 5: Testing
- [ ] Create `testing/test_integrated_controlnet.py`
  - [ ] Test input embedder patching
  - [ ] Test forward pass shapes
  - [ ] Test optimizer param inclusion
  - [ ] Test frozen vs trainable modes
- [ ] Create smoke test config `config/test_integrated_controlnet.yaml`
- [ ] Run comparison test (residuals vs integrated)
- [ ] Document performance characteristics in `LEARNINGS.md`

### Phase 6: Documentation
- [ ] Create `docs/ControlNet_Integration_Modes.md`
- [ ] Add docstrings to all new classes/methods
- [ ] Update `AGENTS.md` quick reference
- [ ] Add examples to existing ControlNet docs
- [ ] Update FAQ with common issues

---

## Success Criteria

1. **Functionality:** Both modes produce training runs that converge without errors
2. **Quality:** Integrated mode produces comparable output quality to residuals mode
3. **Flexibility:** Users can switch between modes with a single config change
4. **Performance:** Integrated mode shows measurable speed or memory improvement (or documented tradeoff)
5. **Stability:** No regressions in existing residuals mode behavior
6. **Testing:** All unit tests pass, smoke test runs successfully
7. **Documentation:** Users can follow docs to configure and use integrated mode

---

## References

### Existing Adapter Patterns
- [control_lora_adapter.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\models\control_lora_adapter.py) — ControlLoRA pattern with `ImgEmbedder`
- [subpixel_adapter.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\models\subpixel_adapter.py) — Subpixel adapter with similar input hijacking
- [i2v_adapter.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\models\i2v_adapter.py) — I2V adapter with `FrameEmbedder`

### Current ControlNet Implementation
- [BaseSDTrainProcess.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\jobs\process\BaseSDTrainProcess.py) — Current setup and validation
- [controlnet_offload.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\toolkit\controlnet_offload.py) — Residual computation utilities
- [z_image.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\extensions_built_in\diffusion_models\z_image\z_image.py) — Residual injection into Z-Image transformer

### Testing Examples
- [test_controlnet_residuals.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\testing\test_controlnet_residuals.py) — Existing residuals tests
- [test_controlnet_offload.py](c:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\testing\test_controlnet_offload.py) — Offload and storage tests

---

## Notes

- This plan assumes Z-Image / Flux-style transformers as primary target. UNet-based models (SD1.5/SDXL) can be added as follow-up.
- Multi-scale fusion strategy is left as a follow-up — initial implementation uses first scale only.
- Checkpoint saving/loading format for integrated mode will be determined during Phase 4 testing.
- Performance benchmarks should be added to `LEARNINGS.md` once both modes are implemented.

---

**Next Steps:**
1. Review this plan with team/maintainers
2. Create GitHub issue/PR with this plan attached
3. Begin Phase 1 implementation (config changes)
4. Iterate through phases with testing at each step
