# ControlTrain Critical Implementation Issues

**IMPORTANT**: This document addresses critical implementation challenges that MUST be handled correctly for the ControlTrain implementation to work.

These issues were identified after the initial 3-part plan was written and require specific attention.

---

## Issue 1: Tile System for Z-Image Tensors ❌ CRITICAL

### Problem
Each individual image sample is converted into a batch of **16 tensors** using a tile system. Control images MUST be tiled in exactly the same way for tensors to match during training.

### Current Plan Status
**NOT ADDRESSED** - The plan does not mention tiling at all.

### Required Implementation

**Location**: `toolkit/dataloader_mixins.py` and control encoding

**1. Understand Z-Image Tiling**:

```python
# Z-Image divides images into tiles for processing
# Typical configuration: 4x4 grid = 16 tiles
# Each tile is processed as a separate item in the batch

def tile_image_zimage(
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64
) -> torch.Tensor:
    """
    Tile image into 16 patches for Z-Image processing.
    
    Args:
        image: Image tensor [C, H, W]
        tile_size: Size of each tile
        overlap: Overlap between tiles
        
    Returns:
        Tiled tensor [16, C, tile_H, tile_W]
    """
    # Implementation depends on Z-Image's specific tiling strategy
    # May use sliding window with overlap
    # Or fixed grid division
    pass
```

**2. Apply SAME Tiling to Control Images**:

```python
class AiToolkitDataset:
    def __getitem__(self, idx):
        # Load images
        image = self.load_image(file_item)
        control_image = self._get_control_image(file_item)
        
        # Apply synchronized transforms (existing)
        image, control_image = self._apply_synchronized_transforms(image, control_image, file_item)
        
        # NEW: Apply SAME tiling to both
        if self.model_uses_tiling():  # Check if Z-Image
            image_tiles = self.tile_image(image)
            control_tiles = self.tile_image(control_image)  # MUST use identical tiling
            
            return {
                'image_tiles': image_tiles,  # [16, C, H, W]
                'control_tiles': control_tiles,  # [16, C, H, W]
                'file_item': file_item,
            }
```

**3. Verify Tile Correspondence**:

```python
def verify_tile_alignment(image_tiles, control_tiles):
    """Verify tiles match spatially"""
    assert image_tiles.shape == control_tiles.shape, \
        f"Tile shape mismatch: {image_tiles.shape} vs {control_tiles.shape}"
    
    # Verify each tile corresponds to same spatial region
    # (Implementation depends on tiling strategy)
```

**CRITICAL QUESTIONS TO ANSWER**:
- What is Z-Image's exact tiling strategy? (4x4 grid? Sliding window?)
- What is the tile size? (512x512? 1024x1024?)
- Is there overlap between tiles?
- How are tiles reassembled after processing?

**ACTION REQUIRED**: 
1. Investigate Z-Image's tiling implementation in VideoX-Fun
2. Find the exact tiling function used
3. Ensure control images use IDENTICAL tiling parameters

---

## Issue 2: VideoX Transformer Architecture ⚠️ NEEDS CLARIFICATION

### Problem
The ControlNet needs the **VideoX transformer to be copied** for controls to be accepted properly.

### Current Plan Status
**PARTIALLY ADDRESSED** - Plan loads ControlNet but doesn't explicitly copy transformer structure.

### Required Implementation

**Location**: `extensions_built_in/diffusion_models/z_image/z_image_model.py`

**Question**: What does "VideoX transformer needs to be copied" mean?

**Interpretation A: Copy Base Transformer Config**:

```python
def load_controlnet_transformer(self, controlnet_path, controlnet_file, **kwargs):
    """Load ControlNet with copied transformer architecture"""
    
    # Copy the base transformer's architecture
    base_transformer_config = self.transformer.config.to_dict()
    
    # Generate ControlNet config based on base transformer
    controlnet_config = generate_controlnet_config(
        base_config=base_transformer_config,  # Use base as template
        controlnet_path=controlnet_path,
        controlnet_file=controlnet_file
    )
    
    # Load ControlNet with matching architecture
    self.controlnet = ZImageControlTransformer2DModel(**controlnet_config)
```

**Interpretation B: Share Transformer Components**:

```python
def load_controlnet_transformer(self, controlnet_path, controlnet_file, **kwargs):
    """Load ControlNet sharing some transformer components"""
    
    # Load ControlNet
    self.controlnet = ZImageControlTransformer2DModel(...)
    
    # Share certain layers/embeddings from base transformer?
    # (Need clarification on what should be shared)
    self.controlnet.pos_embed = self.transformer.pos_embed  # Example
```

**Interpretation C: Load VideoX-Specific Architecture**:

```python
# Ensure we're using VideoX-Fun's specific transformer variant
from videox_fun.models import ZImageControlTransformer2DModel
# Not standard diffusers version
```

**ACTION REQUIRED**:
1. Clarify what "VideoX transformer needs to be copied" means
2. Check VideoX-Fun implementation for how they handle this
3. Determine if we need custom model classes or just config matching

---

## Issue 3: Streaming Weights (Memory Explosion) ❌ CRITICAL

### Problem
Loading entire ControlNet state dict into a single Python dict will cause **system RAM to explode**. Weights need to be **streamed** during loading.

### Current Plan Status
**NOT ADDRESSED** - Uses `load_file()` which loads everything into RAM at once.

### Current Problematic Code (from Plan)

```python
# BAD - Loads entire file into RAM
from safetensors.torch import load_file
state_dict = load_file(safetensors_path)  # 💥 RAM explosion
self.controlnet.load_state_dict(state_dict, strict=False)
```

### Required Implementation

**Location**: `extensions_built_in/diffusion_models/z_image/z_image_model.py`

**Solution: Stream Weights from Disk**:

```python
def load_controlnet_transformer(
    self,
    controlnet_path: str,
    controlnet_file: str,
    **kwargs
):
    """Load ControlNet with streamed weight loading"""
    
    from safetensors import safe_open
    import torch
    
    safetensors_path = os.path.join(controlnet_path, controlnet_file)
    
    # Step 1: Initialize empty model
    config = generate_controlnet_config(...)
    self.controlnet = ZImageControlTransformer2DModel(**config)
    
    # Step 2: Stream weights directly into model
    # Do NOT load into intermediate dict
    
    print(f"Streaming ControlNet weights from {safetensors_path}")
    
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        # Get all tensor keys
        tensor_keys = f.keys()
        
        # Load weights one-by-one or in small chunks
        model_state_dict = self.controlnet.state_dict()
        
        for key in tensor_keys:
            if key in model_state_dict:
                # Load tensor directly from file
                tensor = f.get_tensor(key)
                
                # Set directly into model (avoid intermediate dict)
                # Use parameter assignment instead of load_state_dict
                set_nested_parameter(self.controlnet, key, tensor)
                
                # Free memory immediately
                del tensor
            else:
                print(f"Skipping unexpected key: {key}")
        
        print("Weight streaming complete")


def set_nested_parameter(model, key, tensor):
    """
    Set parameter directly in model by key path.
    Avoids creating intermediate dictionaries.
    """
    parts = key.split('.')
    module = model
    
    # Navigate to parent module
    for part in parts[:-1]:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    
    # Set the parameter
    param_name = parts[-1]
    if isinstance(module, torch.nn.Module):
        if hasattr(module, param_name):
            param = getattr(module, param_name)
            param.data = tensor.to(param.device, dtype=param.dtype)
        else:
            # Handle buffers
            module.register_buffer(param_name, tensor)
```

**Alternative: Use Accelerate's Loading**:

```python
from accelerate import load_checkpoint_in_model

def load_controlnet_transformer(self, controlnet_path, controlnet_file, **kwargs):
    """Load ControlNet using Accelerate's memory-efficient loading"""
    
    # Initialize model
    config = generate_controlnet_config(...)
    self.controlnet = ZImageControlTransformer2DModel(**config)
    
    # Use Accelerate to load weights efficiently
    safetensors_path = os.path.join(controlnet_path, controlnet_file)
    
    load_checkpoint_in_model(
        model=self.controlnet,
        checkpoint=safetensors_path,
        device_map="cpu",  # Start on CPU
        offload_folder=None,
        offload_buffers=False,
    )
    
    print("ControlNet loaded with memory-efficient streaming")
```

**CRITICAL**: Test with actual ControlNet file size. Z-Image ControlNet is likely **several GB**.

---

## Issue 4: ControlNet Doesn't Need Caption Embeds ⚠️ CORRECTION

### Problem
The ControlNet does **not particularly need** the caption prompt embeds, but the current plan passes them.

### Current Plan Status
**INCORRECTLY HANDLED** - Plan passes `encoder_hidden_states=text_embeddings` to ControlNet.

### Incorrect Code (from Part 3)

```python
# WRONG - Passes prompt embeds to ControlNet
control_context = self.controlnet(
    control_latents,
    timestep,
    encoder_hidden_states=text_embeddings,  # ❌ Not needed
    return_dict=False
)[0]
```

### Corrected Implementation

**Location**: Part 3 training forward pass

```python
def forward_pass(
    self,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    control_latents: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Forward pass through the denoising model."""
    
    # Prepare control kwargs
    control_kwargs = {}
    if control_latents is not None and self.sd.is_controlnet_enabled:
        # Process control through ControlNet
        # ControlNet only needs: control_latents and timestep
        control_context = self.sd.controlnet(
            control_latents,
            timesteps,
            # DO NOT pass encoder_hidden_states
            return_dict=False
        )[0]
        
        # Add to main transformer forward pass kwargs
        control_kwargs['control_context'] = control_context
        control_kwargs['controlnet_conditioning_scale'] = self.sd.controlnet_guidance_scale
    
    # Main transformer gets both prompt embeds AND control context
    noise_pred = self.sd.transformer(  # Note: transformer, not unet
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,  # Transformer needs this
        **control_kwargs,  # Control context passed here
        **kwargs
    ).sample
    
    return noise_pred
```

**Key Changes**:
- ControlNet forward: `controlnet(control_latents, timesteps)` - no text
- Main transformer forward: Gets text embeddings + control context

---

## Issue 5: Flux2 vs UNet Codepath ⚠️ CRITICAL TERMINOLOGY

### Problem
Need to ensure we're using the **Flux2-specific codepath** for Z-Image and not accidentally using the UNet codepath.

### Current Plan Status
**TERMINOLOGY ISSUE** - Plan uses `self.sd.unet` which is wrong for Flux2.

### Incorrect Terminology Throughout Plans

```python
# WRONG - "unet" is SD1.5/SDXL terminology
self.sd.unet.eval()
noise_pred = self.sd.unet(...)
for param in self.sd.unet.parameters():
```

### Corrected Terminology

**Z-Image uses a TRANSFORMER, not a UNet**:

```python
# CORRECT - Use "transformer" for Flux2/Z-Image
self.sd.transformer.eval()
noise_pred = self.sd.transformer(...)
for param in self.sd.transformer.parameters():
```

### Required Code Audit

**Files to Update**:
1. Part 2: `z_image_model.py` - Change `self.unet` to `self.transformer`
2. Part 3: Training loop - Change all `unet` references
3. Part 3: Pipeline loading - Ensure using `ZImagePipeline` with transformer

**Corrected Model Loading (Part 2)**:

```python
def load_model_with_controlnet(model_config: 'ModelConfig'):
    """Load Z-Image model with ControlNet"""
    
    model = ZImageModel(...)
    model.load_model()
    
    # Z-Image has a TRANSFORMER, not a UNet
    assert hasattr(model, 'transformer'), "Z-Image should have transformer attribute"
    
    # Load LoRA on transformer
    if model_config.network_config:
        network = LoRANetwork(
            transformer=model.transformer,  # NOT unet
            config=model_config.network_config
        )
        network.apply_to_model()
    
    # Load ControlNet
    if model_config.controlnet_enabled:
        model.load_controlnet_transformer(...)
    
    # Freeze base model (transformer)
    for param in model.transformer.parameters():
        param.requires_grad = False
    
    return model
```

**Corrected Training Step (Part 3)**:

```python
def train_step(self, step, batch):
    """Training step using Flux2 transformer"""
    
    # ... encoding ...
    
    # Forward through TRANSFORMER (not UNet)
    model_pred = self.sd.transformer(
        noisy_latents,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        **control_kwargs
    ).sample
    
    # ... loss computation ...
```

**Corrected Sampling (Part 3)**:

```python
def sample(self, step, sample_config):
    """Sampling with Flux2 transformer"""
    
    self.sd.transformer.eval()  # NOT unet.eval()
    
    # ... sampling code ...
    
    self.sd.transformer.train()  # NOT unet.train()
```

**ACTION REQUIRED**: 
1. Global search-replace `unet` → `transformer` in all ControlTrain implementation
2. Verify Z-Image model actually uses `transformer` attribute name
3. Check VideoX-Fun codebase for correct attribute names

---

## Issue 6: Abort Training if ControlNet Not Attached ⚠️ VERIFICATION

### Problem
The design needs to ensure that the job **actually aborts training** if the ControlNet is not properly attached to training.

### Current Plan Status
**PARTIALLY ADDRESSED** - Has validation but may not verify the connection during training.

### Current Validation (from Part 3)

```python
# Existing validation checks config, but doesn't verify actual connection
def validate_controlnet_training_config(model_config, train_config, dataset_configs):
    if not model_config.controlnet_enabled:
        return
    
    # Verifies config is set
    # Does NOT verify controlnet is actually connected
```

### Required: Runtime Connection Verification

**Location**: Part 3 training setup

**Add Connection Verification**:

```python
def setup_controlnet_training(self):
    """Setup ControlNet-specific training configuration"""
    
    if not self.sd.is_controlnet_enabled:
        return
    
    self.print("Setting up ControlNet training mode")
    
    # Existing setup...
    setup_controlnet_gradient_management(self.sd, freeze_controlnet=True)
    
    # NEW: Verify ControlNet is actually connected
    self.verify_controlnet_connection()
    
    self.print("ControlNet training setup complete")


def verify_controlnet_connection(self):
    """
    Verify ControlNet is properly connected and will be used in training.
    ABORT if not properly connected.
    """
    errors = []
    
    # 1. Check ControlNet model exists
    if self.sd.controlnet is None:
        errors.append("ControlNet model is None - not loaded")
    
    # 2. Check ControlNet parameters are frozen
    if self.sd.controlnet is not None:
        trainable_params = sum(
            p.numel() for p in self.sd.controlnet.parameters() if p.requires_grad
        )
        if trainable_params > 0:
            errors.append(
                f"ControlNet has {trainable_params:,} trainable parameters "
                "(should be 0 for LoRA training)"
            )
    
    # 3. Check at least one dataset has control images
    has_control_data = False
    for dataset in self.train_datasets:
        if hasattr(dataset, 'dataset_config'):
            if dataset.dataset_config.control_type is not None:
                has_control_data = True
                break
    
    if not has_control_data:
        errors.append("No dataset has control_type configured")
    
    # 4. Test forward pass with control
    try:
        self.test_controlnet_forward_pass()
    except Exception as e:
        errors.append(f"ControlNet forward pass test failed: {e}")
    
    # ABORT if any errors
    if errors:
        error_msg = "ControlNet connection verification FAILED:\n" + "\n".join(f"  - {e}" for e in errors)
        self.print(error_msg)
        raise RuntimeError(error_msg)
    
    self.print("✓ ControlNet connection verified")


def test_controlnet_forward_pass(self):
    """
    Test that ControlNet forward pass works with dummy data.
    This ensures the connection is actually functional.
    """
    batch_size = 1
    channels = 4  # VAE latent channels
    height = 64  # Latent space height
    width = 64
    
    # Create dummy inputs
    control_latents = torch.randn(
        batch_size, channels, 1, height, width,
        device=self.device_torch,
        dtype=self.train_dtype
    )
    timesteps = torch.tensor([100], device=self.device_torch)
    
    # Test ControlNet forward pass
    with torch.no_grad():
        control_context = self.sd.controlnet(
            control_latents,
            timesteps,
            return_dict=False
        )[0]
    
    # Verify output shape is reasonable
    assert control_context is not None, "ControlNet returned None"
    assert control_context.shape[0] == batch_size, "ControlNet output batch size mismatch"
    
    self.print(f"  ControlNet test pass successful: output shape {control_context.shape}")
```

**Add Runtime Check in Training Step**:

```python
def train_step(self, step, batch):
    """Training step with runtime verification"""
    
    # ... existing code ...
    
    # NEW: Verify control is being used (first 100 steps)
    if step < 100 and self.sd.is_controlnet_enabled:
        if control_latents is None:
            self.print(f"WARNING: Step {step} - ControlNet enabled but no control_latents in batch")
    
    # ... rest of training step ...
```

---

## Summary of Critical Issues

| Issue | Status in Current Plan | Severity | Action Required |
|-------|----------------------|----------|-----------------|
| 1. Tile System | ❌ Not Addressed | CRITICAL | Investigate Z-Image tiling, implement synchronized tiling |
| 2. VideoX Transformer Copy | ⚠️ Unclear | HIGH | Clarify what "copy" means, verify architecture |
| 3. Streaming Weights | ❌ Not Addressed | CRITICAL | Replace `load_file()` with streaming loader |
| 4. Caption Embeds | ⚠️ Incorrect | MEDIUM | Remove embeds from ControlNet forward pass |
| 5. Flux2 vs UNet | ⚠️ Wrong Terms | HIGH | Replace all `unet` → `transformer` |
| 6. Abort if Not Attached | ⚠️ Partial | MEDIUM | Add connection verification with abort |

## Required Actions Before Implementation

1. **Research Z-Image Tiling** (Issue #1)
   - Study VideoX-Fun implementation
   - Find exact tiling function
   - Document tiling parameters

2. **Research VideoX Transformer** (Issue #2)
   - Clarify "transformer copying" requirement
   - Check VideoX-Fun model architecture
   - Determine if custom classes needed

3. **Implement Streaming Loader** (Issue #3)
   - Replace all `load_file()` calls
   - Test with actual ControlNet file
   - Measure RAM usage

4. **Audit All Code** (Issues #4, #5)
   - Remove caption embeds from ControlNet
   - Replace `unet` with `transformer`
   - Verify attribute names in actual code

5. **Add Verification** (Issue #6)
   - Implement connection tests
   - Add abort conditions
   - Test failure scenarios

---

**NEXT STEP**: Before implementing Parts 1-3, these critical issues MUST be researched and resolved.
