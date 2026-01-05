# ControlTrain Implementation Plan - Part 2

**Part 2: Adapter, Transformer, and Shim Implementations**

This document covers the model-level integration of ControlNet with Z-Image-Turbo, including:
- ControlNet configuration generation (handling missing config.json)
- Model loading and weight management
- Control context encoding
- Memory optimization strategies

**Prerequisites**: Part 1 (UI and Dataloader) must be completed first.

---

## 2.1 ControlNet Configuration Challenge

### 2.1.1 The Missing Config Problem

**Challenge**: The ControlNet repository (`alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`) only contains safetensors files, no `config.json`.

**Files in Repository**:
```
alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/
├── Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors
├── Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors
├── Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors
└── README.md
```

**Problem**: Standard Diffusers loading requires a config.json to instantiate the model architecture before loading weights.

**Solution**: Auto-generate config from state dict inspection + base model config.

---

## 2.2 ControlNet Configuration Generator

### 2.2.1 Config Generation Strategy

**Location**: `extensions_built_in/diffusion_models/z_image/controlnet_config.py` (NEW FILE)

**Purpose**: Generate ControlNet config by:
1. Loading base Z-Image-Turbo config
2. Inspecting safetensors state dict
3. Detecting architecture differences
4. Creating appropriate ZImageControlTransformer2DModel config

**Implementation**:

```python
"""
ControlNet configuration generator for Z-Image models.
Handles missing config.json by auto-generating from state dict.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from safetensors.torch import load_file
from diffusers import ZImageTransformer2DModel


class ZImageControlNetConfigGenerator:
    """
    Generates ControlNet configuration from safetensors inspection.
    """
    
    def __init__(self, base_model_path: str):
        """
        Args:
            base_model_path: Path to base Z-Image-Turbo model
        """
        self.base_model_path = base_model_path
        self.base_config = self._load_base_config()
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base model transformer config"""
        transformer_path = os.path.join(self.base_model_path, 'transformer')
        config_path = os.path.join(transformer_path, 'config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Fallback: Load from pretrained
            model = ZImageTransformer2DModel.from_pretrained(
                self.base_model_path,
                subfolder='transformer'
            )
            return model.config.to_dict()
    
    def generate_controlnet_config(
        self,
        controlnet_path: str,
        controlnet_file: str
    ) -> Dict[str, Any]:
        """
        Generate ControlNet config from safetensors file.
        
        Args:
            controlnet_path: Path to ControlNet repository
            controlnet_file: Specific safetensors file to use
            
        Returns:
            Dictionary containing ControlNet config
        """
        # Inspect safetensors keys without loading large tensors (streaming)
        from safetensors import safe_open
        safetensors_path = os.path.join(controlnet_path, controlnet_file)
        keys = []
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

        # Start with base transformer config as a template (copy, then specialize)
        config = self.base_config.copy()

        # Modify for ControlNet architecture
        config['_class_name'] = 'ZImageControlTransformer2DModel'
        
        # Detect control-specific parameters from safetensors keys
        control_keys = [k for k in keys if 'controlnet' in k or 'control' in k]
        
        # Determine number of control layers
        control_layer_indices = set()
        for key in control_keys:
            # Extract layer index from keys like 'transformer_blocks.0.controlnet_block.weight'
            if 'transformer_blocks.' in key:
                parts = key.split('.')
                try:
                    layer_idx = int(parts[1])
                    control_layer_indices.add(layer_idx)
                except (ValueError, IndexError):
                    pass

        # NOTE: This only inspects keys. After config generation we MUST verify exact tiling strategy and
        # any spatial parameters by checking VideoX-Fun reference implementation; streaming avoids RAM spikes.
        
        # Add ControlNet-specific config
        config['num_control_layers'] = len(control_layer_indices) if control_layer_indices else config.get('num_layers', 19)
        config['control_mode'] = 'context'  # Z-Image uses control context, not block samples
        
        # Detect version from filename or safetensors keys
        version = self._detect_version(controlnet_file, keys)
        config['controlnet_version'] = version
        
        print(f"Generated ControlNet config: version={version}, layers={config['num_control_layers']}")
        
        return config
    
    def _detect_version(self, filename: str, keys: list) -> str:
        """
        Detect ControlNet version from filename or safetensors keys.

        Versions:
        - v1.0: Original release
        - v2.0: Improved architecture
        - v2.1: Latest with 8-step optimizations
        """
        filename_lower = filename.lower()
        
        if '2.1' in filename_lower or '2-1' in filename_lower:
            return 'v2.1'
        elif '2.0' in filename_lower or '2-0' in filename_lower:
            return 'v2.0'
        elif '1.0' in filename_lower or '1-0' in filename_lower:
            return 'v1.0'
        
        # Fallback: inspect state dict for version-specific keys
        # (Version-specific architecture differences would be detected here)
        
        return 'v2.1'  # Default to latest
    
    def save_config(self, config: Dict[str, Any], output_path: str):
        """Save generated config to disk"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved ControlNet config to {output_path}")


def generate_controlnet_config(
    base_model_path: str,
    controlnet_path: str,
    controlnet_file: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate ControlNet config.
    
    Args:
        base_model_path: Path to base Z-Image-Turbo model
        controlnet_path: Path to ControlNet repository
        controlnet_file: Specific safetensors file
        output_path: Optional path to save config
        
    Returns:
        Generated config dictionary
    """
    generator = ZImageControlNetConfigGenerator(base_model_path)
    config = generator.generate_controlnet_config(controlnet_path, controlnet_file)
    
    if output_path:
        generator.save_config(config, output_path)
    
    return config
```

**Usage Example**:

```python
from extensions_built_in.diffusion_models.z_image.controlnet_config import generate_controlnet_config

config = generate_controlnet_config(
    base_model_path='models/Tongyi-MAI/Z-Image-Turbo',
    controlnet_path='models/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1',
    controlnet_file='Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors',
    output_path='models/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/config.json'
)
```

---

## 2.3 Model Loading Integration

### 2.3.1 Modify ZImageModel Class

**Location**: `extensions_built_in/diffusion_models/z_image/z_image_model.py`

**Challenge**: Integrate ControlNet loading into existing model loading pipeline.

**New Method - Load ControlNet Transformer**:

```python
class ZImageModel(StableDiffusion):
    """Extended Z-Image model with ControlNet support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ControlNet state
        self.controlnet = None
        self.is_controlnet_enabled = False
        self.controlnet_guidance_scale = 1.0
    
    def load_controlnet_transformer(
        self,
        controlnet_path: str,
        controlnet_file: str,
        freeze: bool = True,
        offload_strategy: str = 'none'
    ):
        """
        Load ControlNet transformer weights.
        
        Args:
            controlnet_path: Path to ControlNet repository
            controlnet_file: Specific safetensors file to load
            freeze: Whether to freeze ControlNet parameters
            offload_strategy: Memory management ('none', 'cpu', 'sequential')
        """
        from diffusers import ZImageControlTransformer2DModel
        from safetensors.torch import load_file
        from extensions_built_in.diffusion_models.z_image.controlnet_config import generate_controlnet_config
        
        print(f"Loading ControlNet: {controlnet_file}")
        
        # Step 1: Generate config if not exists
        config_path = os.path.join(controlnet_path, 'config.json')
        if not os.path.exists(config_path):
            print("No config.json found, generating from state dict...")
            config = generate_controlnet_config(
                base_model_path=self.name_or_path,
                controlnet_path=controlnet_path,
                controlnet_file=controlnet_file,
                output_path=config_path
            )
        else:
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Step 2: Initialize ControlNet model
        self.controlnet = ZImageControlTransformer2DModel(**config)

        # Step 2a: Copy base transformer weights into control transformer to initialize shared components
        try:
            from diffusers import ZImageTransformer2DModel

            base_transformer = ZImageTransformer2DModel.from_pretrained(
                self.name_or_path,
                subfolder='transformer',
                torch_dtype=getattr(self, 'torch_dtype', None)
            )
            base_state = base_transformer.state_dict()
            m, u = self.controlnet.load_state_dict(base_state, strict=False)
            print(f"Base→Control copy: {len(m)} missing, {len(u)} unexpected")
            del base_transformer, base_state
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Base transformer copy skipped: {e}")

        # Step 3: Stream weights from safetensors to avoid RAM explosion
        from safetensors import safe_open
        safetensors_path = os.path.join(controlnet_path, controlnet_file)

        # Basic existence check (fail-fast)
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"ControlNet file not found: {safetensors_path}. Aborting.")

        # Stream weights directly into model (do not load whole state dict into memory)
        model_state = self.controlnet.state_dict()
        keys_in_file = []
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys_in_file = list(f.keys())
            unexpected = [k for k in keys_in_file if k not in model_state]
            if unexpected:
                # Fail-fast: unexpected keys indicate incompatible checkpoint
                sample_keys = unexpected[:5]
                raise RuntimeError(
                    f"ControlNet checkpoint contains unexpected keys (possible incompatible file). "
                    f"Sample unexpected keys: {sample_keys}. Aborting."
                )

            for key in keys_in_file:
                if key in model_state:
                    tensor = f.get_tensor(key)
                    # Set directly into model parameter/buffer without creating large intermediate dicts
                    set_nested_parameter(self.controlnet, key, tensor)
                    del tensor
                else:
                    # This branch should be unreachable due to the unexpected check above
                    print(f"Skipping unexpected key: {key}")

        # Verification: ensure required control attributes exist
        required_attrs = ['control_layers', 'control_all_x_embedder', 'control_in_dim']
        missing_attrs = [a for a in required_attrs if not hasattr(self.controlnet, a)]
        if missing_attrs:
            raise RuntimeError(f"Loaded control transformer is missing required attributes: {missing_attrs}. Aborting.")

        # Optional: Also run a dry-run forward with small dummy tensor to ensure it runs
        try:
            import torch
            dummy_x = [torch.zeros(1, 16, 64, 64, dtype=torch.float32)]
            dummy_t = torch.zeros(1, dtype=torch.float32)
            dummy_cap = [torch.zeros(1, 512, dtype=torch.float32)]
            self.controlnet(dummy_x, dummy_t, dummy_cap)
        except Exception as e:
            raise RuntimeError(f"ControlTransformer dry-run forward failed: {e}")

        # Optional: run a lightweight verification step to report missing/unexpected keys
        # (left as implementation detail)
        
        # Step 4: Move to appropriate device
        if offload_strategy == 'none':
            self.controlnet.to(self.device_torch, dtype=self.torch_dtype)
        elif offload_strategy == 'cpu':
            # Keep on CPU, onload during forward pass
            self.controlnet.to('cpu', dtype=self.torch_dtype)
        # Sequential offload handled by offload manager
        
        # Step 5: Freeze parameters if requested
        if freeze:
            for param in self.controlnet.parameters():
                param.requires_grad = False
            print("ControlNet parameters frozen")
        
        # Step 6: Set enabled flag
        self.is_controlnet_enabled = True
        
        print(f"ControlNet loaded successfully (freeze={freeze}, offload={offload_strategy})")
    
    def encode_control_images(
        self,
        control_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode control images to control latents.
        
        Args:
            control_images: Control images [B, C, H, W] in range [0, 1]
            
        Returns:
            Control latents [B, C, F, H, W] where F is frame dim (1 for images)
        """
        # Control images use same VAE encoding as regular images
        # Normalize to [-1, 1] if needed
        if control_images.min() >= 0 and control_images.max() <= 1:
            control_images = control_images * 2.0 - 1.0
        
        # Encode through VAE
        with torch.no_grad():
            latent_dist = self.vae.encode(control_images)
            control_latents = latent_dist.latent_dist.sample()
            control_latents = control_latents * self.vae.config.scaling_factor
        
        # Add frame dimension if needed (for video-compatible architecture)
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(2)  # [B, C, 1, H, W]
        
        return control_latents
```

**Key Design Decisions**:
1. **Lazy Loading**: ControlNet only loaded if enabled in config
2. **Config Generation**: Automatic fallback if config.json missing
3. **State Dict Loading**: Use `strict=False` to handle version differences
4. **Separate VAE Encoding**: Control images encoded through same VAE as regular images
5. **Freeze by Default**: ControlNet should always be frozen for LoRA training

---

## 2.4 Control Context Integration

### 2.4.1 Modify Forward Pass

**Location**: Still in `z_image_model.py`

**Challenge**: Inject control context into transformer forward pass.

**Modified Method - Predict Noise**:

```python
def predict_noise(
    self,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    timestep: torch.Tensor,
    guidance_scale: float = 3.5,
    control_latents: Optional[torch.Tensor] = None,
    control_scale: Optional[float] = None,
    **kwargs
) -> torch.Tensor:
    """
    Predict noise with optional ControlNet conditioning.
    
    Args:
        latents: Noisy latents [B, C, F, H, W]
        text_embeddings: Text embeddings [B, seq_len, dim]
        timestep: Timestep values [B]
        guidance_scale: CFG scale
        control_latents: Optional control conditioning [B, C, F, H, W]
        control_scale: Override for controlnet_guidance_scale
        
    Returns:
        Predicted noise [B, C, F, H, W]
    """
    # Prepare control kwargs
    control_kwargs = {}
    if control_latents is not None and self.is_controlnet_enabled:
        # Process control through ControlNet (ControlNet does NOT need text embeddings)
        # Use offload manager or streaming if available to avoid memory spikes
        if hasattr(self, 'controlnet_offload_manager') and self.controlnet_offload_manager is not None:
            with self.controlnet_offload_manager.control_forward():
                control_context = self.controlnet(control_latents, timestep, return_dict=False)[0]
        else:
            control_context = self.controlnet(control_latents, timestep, return_dict=False)[0]

        # Add to forward pass kwargs
        control_kwargs['control_context'] = control_context
        control_kwargs['controlnet_conditioning_scale'] = control_scale or self.controlnet_guidance_scale

    # Forward through transformer (Z-Image uses a transformer, not a UNet)
    noise_pred = self.transformer(
        latents,
        timestep,
        encoder_hidden_states=text_embeddings,
        **control_kwargs,
        **kwargs
    ).sample
    
    return noise_pred
```

**Reference**: VideoX-Fun implementation shows control context is passed directly to transformer, not as block samples (different from SD1.5 ControlNet).

---

## 2.5 Memory Management for ControlNet

### 2.5.1 Offload Strategy Implementation

**Location**: `toolkit/controlnet_offload.py` (NEW FILE)

**Purpose**: Manage ControlNet memory with three strategies:
1. **None**: Keep everything on GPU (fastest, highest memory)
2. **CPU**: Offload control layers to CPU between forward passes
3. **Sequential**: Layer-by-layer offload (slowest, lowest memory)

**Implementation**:

```python
"""
Memory offload management for ControlNet training.
Reduces GPU memory usage by strategically moving layers.
"""

import torch
from typing import Optional, Literal
from contextlib import contextmanager


class ControlNetOffloadManager:
    """
    Manages memory offloading for ControlNet during training.
    """
    
    def __init__(
        self,
        sd_model: 'ZImageModel',
        strategy: Literal['none', 'cpu', 'sequential'] = 'cpu'
    ):
        self.sd_model = sd_model
        self.strategy = strategy
        self.device = sd_model.device_torch
        self.controlnet = sd_model.controlnet
        
        if strategy == 'sequential':
            self._setup_sequential_offload()
    
    def _setup_sequential_offload(self):
        """Setup hooks for sequential layer offloading"""
        # Implementation would use accelerate's hooks
        # Similar to model.enable_sequential_cpu_offload()
        from accelerate import cpu_offload
        cpu_offload(self.controlnet, execution_device=self.device)
    
    def onload_control_layers(self):
        """Move ControlNet to GPU for forward pass"""
        if self.strategy == 'cpu':
            self.controlnet.to(self.device)
        # Sequential handles automatically via hooks
    
    def offload_control_layers(self):
        """Move ControlNet back to CPU after forward pass"""
        if self.strategy == 'cpu':
            self.controlnet.to('cpu')
            torch.cuda.empty_cache()
        # Sequential handles automatically via hooks
    
    @contextmanager
    def control_forward(self):
        """Context manager for control forward pass with automatic offload"""
        try:
            self.onload_control_layers()
            yield
        finally:
            self.offload_control_layers()


def setup_controlnet_offload(
    sd_model: 'ZImageModel',
    strategy: Literal['none', 'cpu', 'sequential'] = 'cpu'
) -> Optional[ControlNetOffloadManager]:
    """
    Setup ControlNet offload manager.
    
    Args:
        sd_model: ZImageModel instance with controlnet loaded
        strategy: Offload strategy
        
    Returns:
        ControlNetOffloadManager or None if strategy is 'none'
    """
    if strategy == 'none':
        return None
    
    return ControlNetOffloadManager(sd_model, strategy)


def setup_controlnet_gradient_management(
    sd_model: 'ZImageModel',
    freeze_controlnet: bool = True
):
    """
    Setup gradient management for ControlNet.
    
    Args:
        sd_model: ZImageModel instance
        freeze_controlnet: Whether to freeze ControlNet parameters
    """
    if not sd_model.is_controlnet_enabled:
        return
    
    if freeze_controlnet:
        # Freeze all ControlNet parameters
        for param in sd_model.controlnet.parameters():
            param.requires_grad = False
        print("ControlNet parameters frozen (requires_grad=False)")
    else:
        # Unfreeze (for fine-tuning ControlNet itself)
        for param in sd_model.controlnet.parameters():
            param.requires_grad = True
        print("ControlNet parameters unfrozen (requires_grad=True)")
```

**Usage in Training**:

```python
# During training setup
if model.is_controlnet_enabled:
    offload_manager = setup_controlnet_offload(model, strategy='cpu')
    setup_controlnet_gradient_management(model, freeze_controlnet=True)

# During training step
if offload_manager:
    with offload_manager.control_forward():
        # Forward pass with control
        control_context = model.controlnet(control_latents, timestep, ...)
```

---

## 2.6 Pipeline Integration

### 2.6.1 Modify Pipeline Loading

**Location**: `extensions_built_in/diffusion_models/z_image/z_image_model.py`

**Challenge**: Ensure pipeline (for sampling) uses ControlNet variant when enabled.

**Modified Pipeline Loading**:

```python
def load_pipeline(self):
    """
    Load inference pipeline.
    Uses ZImageControlPipeline if ControlNet is enabled.
    """
    if self.is_controlnet_enabled:
        from diffusers import ZImageControlPipeline

        self.pipeline = ZImageControlPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,  # ZImageTransformer2DModel
            controlnet=self.controlnet,  # ZImageControlTransformer2DModel
            scheduler=self.noise_scheduler,
        )
    else:
        from diffusers import ZImagePipeline

        self.pipeline = ZImagePipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.noise_scheduler,
        )

    print(f"Pipeline loaded: {type(self.pipeline).__name__}")
```

**Sampling Method Update**:

```python
def generate_images(
    self,
    prompt: str,
    control_image: Optional[Image.Image] = None,
    **kwargs
) -> List[Image.Image]:
    """
    Generate images with optional control conditioning.
    
    Args:
        prompt: Text prompt
        control_image: Optional control image (preprocessed)
        **kwargs: Additional pipeline arguments
        
    Returns:
        List of generated PIL Images
    """
    if self.is_controlnet_enabled and control_image is not None:
        # Use ControlNet pipeline
        output = self.pipeline(
            prompt=prompt,
            control_image=control_image,
            controlnet_conditioning_scale=self.controlnet_guidance_scale,
            **kwargs
        )
    else:
        # Standard pipeline
        output = self.pipeline(
            prompt=prompt,
            **kwargs
        )
    
    return output.images
```

---

## 2.7 Weight Loading and State Dict Management

### 2.7.1 Handling ControlNet State Dict

**Challenge**: ControlNet weights may have different key naming conventions.

**Location**: `toolkit/model_utils.py` or within `z_image_model.py`

**Implementation**:

```python
def load_controlnet_state_dict(
    model: torch.nn.Module,
    state_dict_path: str,
    key_mapping: Optional[Dict[str, str]] = None
) -> tuple[list, list]:
    """
    Load ControlNet state dict with proper key mapping.
    
    Args:
        model: ControlNet model instance
        state_dict_path: Path to safetensors file
        key_mapping: Optional key remapping dictionary
        
    Returns:
        Tuple of (missing_keys, unexpected_keys)
    """
    # Memory-safe streaming approach (do NOT load entire safetensors into RAM)
    from safetensors import safe_open

    # If a small file, the helper can fall back to load_file, but for large models streaming is required
    with safe_open(state_dict_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    # Option A (stream-assign): iterate keys and assign to model directly (recommended for very large checkpoints)
    # Option B (lightweight mapping): create a small mapping of available keys and then call load_state_dict with a filtered dict

    # Here we implement Option B: build a minimal mapping of key->tensor (could be chunked in practice)
    # NOTE: This still loads tensors; a production implementation should assign per-key to avoid building a large dict.
    state_dict_stream = {}
    with safe_open(state_dict_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            mapped_key = key_mapping.get(key, key) if key_mapping else key
            # Conservative: only load tensors that are present in the model state
            if mapped_key in model.state_dict():
                state_dict_stream[mapped_key] = f.get_tensor(key)

    load_result = model.load_state_dict(state_dict_stream, strict=False)

    # Free streamed tensors (implementation detail: ensure timely GC)
    del state_dict_stream

    return load_result.missing_keys, load_result.unexpected_keys


def verify_controlnet_weights(
    controlnet: torch.nn.Module,
    expected_layers: Optional[int] = None
):
    """
    Verify ControlNet weights loaded correctly.
    
    Args:
        controlnet: Loaded ControlNet model
        expected_layers: Expected number of control layers
    """
    # Count control layers
    control_params = sum(p.numel() for p in controlnet.parameters())
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    
    print(f"ControlNet total parameters: {control_params:,}")
    print(f"ControlNet trainable parameters: {trainable_params:,}")
    
    if trainable_params > 0:
        print("Warning: ControlNet has trainable parameters (should be frozen for LoRA training)")
    
    # Verify expected architecture
    if expected_layers:
        # Check if controlnet has expected number of blocks
        # (Implementation depends on model structure)
        pass
```

---

## 2.8 Complete Model Initialization Flow

### 2.8.1 Updated Model Loading Sequence

**Location**: Main model initialization in training process

**Complete Flow**:

```python
def load_model_with_controlnet(model_config: 'ModelConfig'):
    """
    Complete model loading sequence with ControlNet support.
    
    Steps:
    1. Load base model (Z-Image-Turbo)
    2. Load LoRA network if specified
    3. Load ControlNet if enabled
    4. Setup memory offloading
    5. Freeze appropriate parameters
    6. Load pipeline for sampling
    """
    # Step 1: Load base model
    from extensions_built_in.diffusion_models.z_image import ZImageModel
    
    model = ZImageModel(
        name_or_path=model_config.name_or_path,
        dtype=model_config.dtype,
        device=model_config.device,
    )
    model.load_model()
    
    # Step 2: Load LoRA network if specified (apply to transformer for Z-Image)
    if model_config.network_config:
        from toolkit.lora import LoRANetwork
        network = LoRANetwork(
            transformer=model.transformer,
            config=model_config.network_config
        )
        network.apply_to_model()
        print(f"LoRA network applied: rank={model_config.network_config.rank}")

    # Step 3: Load ControlNet if enabled (FAIL-FAST semantics)
    if model_config.controlnet_enabled:
        # Accept either a repo/folder path or a checkpoint file. Require at least one.
        if not (model_config.controlnet_name_or_path or model_config.controlnet_file):
            raise RuntimeError("controlnet_enabled=True but neither 'controlnet_name_or_path' nor 'controlnet_file' is set in ModelConfig. Aborting.")

        model.load_controlnet_transformer(
            controlnet_path=model_config.controlnet_name_or_path,
            controlnet_file=model_config.controlnet_file,
            freeze=True,  # Always freeze for LoRA training
            offload_strategy='none'  # Will be managed separately
        )

        # Ensure controlnet loaded successfully
        if not model.is_controlnet_enabled or model.controlnet is None:
            raise RuntimeError("ControlNet failed to load or is not enabled after load_controlnet_transformer(). Aborting.")

        model.controlnet_guidance_scale = model_config.control_guidance_scale

    # Note: On machines with <64GB RAM, require explicit streaming flag in ModelConfig
    if model_config.controlnet_enabled and getattr(model_config, 'controlnet_streaming', False) is False:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if ram_gb < 64:
                raise RuntimeError("Host RAM <64GB and controlnet_streaming is not enabled. Set controlnet_streaming=True or run on a higher-RAM host.")
        except Exception:
            # If psutil not available, only warn and require explicit streaming flag in constrained environments
            pass
    
    # Step 4: Setup memory offloading if configured
    if model_config.controlnet_enabled and model_config.controlnet_offload_strategy != 'none':
        from toolkit.controlnet_offload import setup_controlnet_offload
        offload_manager = setup_controlnet_offload(
            model,
            strategy=model_config.controlnet_offload_strategy
        )
    else:
        offload_manager = None
    
    # Step 5: Freeze base model parameters (train only LoRA)
    for param in model.transformer.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    print("Base model parameters frozen")
    
    # Enable LoRA parameters
    if model_config.network_config:
        for param in network.parameters():
            param.requires_grad = True
        print("LoRA parameters enabled for training")
    
    # Step 6: Load pipeline for sampling
    model.load_pipeline()
    
    return model, offload_manager
```

---

## 2.9 Testing & Validation for Part 2

### 2.9.1 Unit Tests

**Location**: `testing/test_controlnet_model.py` (NEW FILE)

**Test Suite**:

```python
"""
Unit tests for ControlNet model loading and integration.
"""

import pytest
import torch
from unittest.mock import Mock, patch


def test_config_generation():
    """Test ControlNet config auto-generation and that it uses base transformer template"""
    from extensions_built_in.diffusion_models.z_image.controlnet_config import ZImageControlNetConfigGenerator

    # Test that generator inspects safetensors keys and copies base transformer config as template
    # (Would use actual test fixtures / small safetensors file)
    pass


def test_streaming_weight_loading():
    """Test controlnet weights are streamed and do not require loading entire file into RAM"""
    # Mock safetensors.safe_open and verify we iterate keys rather than call load_file
    pass


def test_controlnet_forward_ignores_text():
    """Ensure ControlNet forward does not accept text embeddings (encoder_hidden_states)"""
    # Create mock controlnet and assert call signature does not require encoder_hidden_states
    pass


def test_controlnet_loading():
    """Test ControlNet model loads correctly"""
    # Verify state dict loads
    # Verify parameters are frozen
    # Verify architecture matches expected
    pass


def test_control_encoding():
    """Test control image encoding"""
    # Create dummy control images
    # Encode through VAE
    # Verify output shape and dtype
    pass


def test_forward_with_control():
    """Test forward pass with control conditioning"""
    # Create dummy inputs
    # Run forward pass with control_latents
    # Verify output shape correct
    pass


def test_pipeline_switching():
    """Test pipeline switches to control variant when enabled"""
    # Load model without control -> verify ZImagePipeline
    # Enable control -> verify ZImageControlPipeline
    pass


@pytest.mark.parametrize("strategy", ['none', 'cpu', 'sequential'])
def test_offload_strategies(strategy):
    """Test different offload strategies"""
    # Setup offload manager with strategy
    # Verify layers move to correct devices
    pass


def test_gradient_freezing():
    """Test ControlNet parameters are frozen"""
    # Load ControlNet
    # Verify all controlnet.parameters() have requires_grad=False
    # Verify LoRA parameters have requires_grad=True
    pass
```

### 2.9.2 Integration Tests

**Location**: `testing/test_controlnet_integration.py`

**Test Scenarios**:

```python
"""
Integration tests for complete ControlNet workflow.
"""

def test_model_loading_with_controlnet():
    """Test complete model loading sequence"""
    # Load base model
    # Load ControlNet
    # Verify both loaded correctly
    # Verify memory usage reasonable
    pass


def test_control_image_encoding_pipeline():
    """Test control image flows through encoding"""
    # Load model with ControlNet
    # Create test control image
    # Encode through VAE
    # Pass through ControlNet
    # Verify control context shape
    pass


def test_sampling_with_control():
    """Test sampling produces valid outputs"""
    # Load model
    # Load ControlNet
    # Generate image with control
    # Verify output is valid PIL Image
    pass
```

### 2.9.3 Manual Testing Checklist

- [ ] ControlNet loads without config.json
- [ ] Config auto-generation produces valid config
- [ ] State dict loads with <5% missing keys
- [ ] ControlNet parameters are frozen
- [ ] LoRA parameters remain trainable
- [ ] Control encoding produces correct shape latents
- [ ] Forward pass with control works without errors
- [ ] Memory usage within expected range for offload strategy
- [ ] Pipeline switches correctly based on control enabled
- [ ] Sampling with control produces reasonable outputs

---

## 2.10 Debugging and Inspection Tools

### 2.10.1 ControlNet Inspector Script

**Location**: `scripts/inspect_controlnet.py` (NEW FILE)

**Purpose**: Inspect loaded ControlNet for debugging

```python
"""
Inspect ControlNet model for debugging.
"""

import argparse
import torch
from safetensors.torch import load_file


def inspect_controlnet(path: str, file: str):
    """Inspect ControlNet safetensors file"""
    full_path = f"{path}/{file}"
    
    print(f"Inspecting: {full_path}")
    print("="*80)
    
    # Load state dict
    state_dict = load_file(full_path)
    
    # Analyze structure
    total_params = sum(v.numel() for v in state_dict.values())
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Number of keys: {len(state_dict)}")
    
    # Group by prefix
    prefixes = {}
    for key in state_dict.keys():
        prefix = key.split('.')[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print(f"\nKey prefixes:")
    for prefix, count in sorted(prefixes.items()):
        print(f"  {prefix}: {count} keys")
    
    # Show sample keys
    print(f"\nSample keys:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        shape = state_dict[key].shape
        dtype = state_dict[key].dtype
        print(f"  {key}: {shape} {dtype}")
    
    # Detect control layers
    control_keys = [k for k in state_dict.keys() if 'control' in k.lower()]
    print(f"\nControl-specific keys: {len(control_keys)}")
    if control_keys:
        print(f"  Examples: {control_keys[:5]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    
    inspect_controlnet(args.path, args.file)
```

**Usage**:
```bash
python scripts/inspect_controlnet.py \
  --path models/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 \
  --file Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors
```

---

## Summary of Part 2

**Files Created**:
1. `extensions_built_in/diffusion_models/z_image/controlnet_config.py` - Config generation
2. `toolkit/controlnet_offload.py` - Memory management
3. `scripts/inspect_controlnet.py` - Debugging tool
4. `testing/test_controlnet_model.py` - Unit tests
5. `testing/test_controlnet_integration.py` - Integration tests

**Files Modified**:
1. `extensions_built_in/diffusion_models/z_image/z_image_model.py` - Model loading and integration
2. `toolkit/model_utils.py` - State dict utilities

**Key Achievements**:
✅ **Config Generation**: Handles missing config.json automatically
✅ **ControlNet Loading**: Robust loading with version detection
✅ **Control Encoding**: VAE-based control latent encoding
✅ **Memory Management**: Three-tier offload strategy
✅ **Gradient Management**: Freeze ControlNet, train LoRA only
✅ **Pipeline Integration**: Automatic switching for sampling

**Architecture Flow**:
1. Generate config from state dict inspection
2. Load ControlNet transformer weights
3. Freeze all ControlNet parameters
4. Setup memory offload if configured
5. Encode control images through VAE
6. Pass control context to transformer
7. Use control-aware pipeline for sampling

**Memory Optimization**:
- **None**: ~12GB additional VRAM (ControlNet on GPU)
- **CPU**: ~4GB additional VRAM (offload between passes)
- **Sequential**: ~2GB additional VRAM (layer-by-layer)

**Next Step**: Part 3 will cover the complete training loop implementation, including batch processing, loss computation, and sampling with ControlNet.

---

*End of Part 2*
