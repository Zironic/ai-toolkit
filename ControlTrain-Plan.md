# ControlTrain Implementation Plan

**Project Goal**: Integrate Z-Image-Turbo with ControlNet support for LoRA training in the AI Toolkit

**Reference Models**:
- Base Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Flux2-based text-to-image)
- ControlNet: [alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1)
- Reference Implementation: [VideoX-Fun GitHub](https://github.com/aigc-apps/VideoX-Fun)

**Architecture**: Z-Image-Turbo is Flux2-based, requiring control context (not block samples) for ControlNet integration

---

## PART 1: UI IMPLEMENTATION AND DATALOADER IMPLEMENTATION

### 1.1 UI Configuration Extensions

#### 1.1.1 Model Configuration UI (`ui/src/app/jobs/new/jobConfig.ts`)

**Location**: Modify `defaultJobConfig.model` section

**New Fields to Add**:
- No fields for  freeze_controlnet: true because it should always be frozen during lora training and control_guidance_scale: 1.0 because for Z-Image, it should always be 1.
```typescript
model: {
  name_or_path: 'Tongyi-MAI/Z-Image-Turbo',
  // Existing fields...
  
  // NEW: ControlNet configuration
  controlnet_config: {
    enabled: false,  // Toggle for ControlNet training
    name_or_path: 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1',
    controlnet_file: 'Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors',
  }
}
```

**Implementation Details**:
- The `controlnet_file` specifies which specific checkpoint to use from the repository
- `freeze_controlnet` should default to `true` to keep ControlNet weights frozen while training LoRA
- This follows the pattern established in `toolkit/config_modules.py` for adapter configurations

**Reference**: VideoX-Fun uses similar config structure in `config/z_image/z_image_control_2.1.yaml`

---

#### 1.1.2 Dataset Configuration UI (`ui/src/app/jobs/new/jobConfig.ts`)

**Location**: Modify `defaultDatasetConfig` to include control type selection

**New Fields to Add**:
```typescript
export const defaultDatasetConfig: DatasetConfig = {
  // Existing fields...
  
  // NEW: Control conditioning configuration
  control_type: null,  // Options: null, 'openpose', 'canny', 'depth'
  control_preprocessing: {
    openpose: {
      hand_and_face: true,
      model: 'body_with_hands',
    },
    canny: {
      low_threshold: 100,
      high_threshold: 200,
    },
    depth: {
      model: 'depth-anything-large',
    }
  },
  generate_control_on_the_fly: true,  // Generate during training vs pre-generate
  control_cache_path: null,  // Where to cache control images
}
```

**Control Type Details**:
- **Openpose**: Human pose estimation for character training
  - Uses models like `dwpose` or `openpose` from `controlnet_aux`
  - Detects body keypoints, hands, and facial landmarks
  
- **Canny**: Edge detection for structural control
  - Simple edge detection using OpenCV Canny algorithm
  - Preserves sharp structural boundaries
  
- **Depth**: Depth map estimation for spatial control
  - Uses models like `depth-anything-large` or `midas`
  - Provides 3D spatial understanding

**Phase 1 Focus**: Implement **Openpose only** for initial version, as specified in base design

**Reference Pattern**: Similar to existing `controls` array in current `DatasetConfig`, but specifically for ControlNet union model

---

#### 1.1.3 UI Form Components (`ui/src/app/jobs/new/AdvancedJob.tsx` or `SimpleJob.tsx`)

**New UI Sections to Add**:

**Section 1: Model Configuration Panel**
```typescript
// Add ControlNet toggle and configuration
<FormSection title="ControlNet Configuration">
  <Checkbox
    label="Enable ControlNet Training"
    checked={config.model.controlnet_config.enabled}
    onChange={(e) => updateConfig(['model', 'controlnet_config', 'enabled'], e.target.checked)}
  />
  
  {config.model.controlnet_config.enabled && (
    <>
      <TextInput
        label="ControlNet Name or Path"
        value={config.model.controlnet_config.name_or_path}
        onChange={(e) => updateConfig(['model', 'controlnet_config', 'name_or_path'], e.target.value)}
        helpText="HuggingFace path or local directory"
      />
      
      <Select
        label="ControlNet Checkpoint File"
        value={config.model.controlnet_config.controlnet_file}
        options={[
          { value: 'Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors', label: '2.1 - 8 steps (Recommended)' },
          { value: 'Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors', label: '2.1 - Full' },
          { value: 'Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors', label: '2.0' },
        ]}
        onChange={(e) => updateConfig(['model', 'controlnet_config', 'controlnet_file'], e.target.value)}
      />
      
      <Checkbox
        label="Freeze ControlNet (recommended)"
        checked={config.model.controlnet_config.freeze_controlnet}
        onChange={(e) => updateConfig(['model', 'controlnet_config', 'freeze_controlnet'], e.target.checked)}
        helpText="Keep ControlNet weights frozen while training LoRA adapters"
      />
    </>
  )}
</FormSection>
```

**Section 2: Dataset Configuration Panel**
```typescript
// Add control type selection per dataset
<FormSection title="Control Conditioning">
  <Select
    label="Control Type"
    value={dataset.control_type}
    options={[
      { value: null, label: 'None (standard training)' },
      { value: 'openpose', label: 'OpenPose (Human Pose)' },
      { value: 'canny', label: 'Canny (Edge Detection)' },
      { value: 'depth', label: 'Depth Map' },
    ]}
    onChange={(e) => updateDataset(index, 'control_type', e.target.value)}
  />
  
  {dataset.control_type && (
    <>
      <Checkbox
        label="Generate Control Images On-the-Fly"
        checked={dataset.generate_control_on_the_fly}
        onChange={(e) => updateDataset(index, 'generate_control_on_the_fly', e.target.checked)}
        helpText="Generate during training (slower) vs pre-cache (faster, more disk space)"
      />
      
      {!dataset.generate_control_on_the_fly && (
        <TextInput
          label="Control Cache Path"
          value={dataset.control_cache_path || `${dataset.folder_path}_control_${dataset.control_type}`}
          onChange={(e) => updateDataset(index, 'control_cache_path', e.target.value)}
          helpText="Directory to store pre-generated control images"
        />
      )}
    </>
  )}
</FormSection>
```

**TypeScript Type Additions** (`ui/src/types/index.ts` or similar):
```typescript
export interface ControlNetConfig {
  enabled: boolean;
  name_or_path: string;
  controlnet_file: string;
  freeze_controlnet: boolean;
  control_guidance_scale: number;
}

export interface ControlPreprocessingConfig {
  openpose?: {
    hand_and_face: boolean;
    model: string;
  };
  canny?: {
    low_threshold: number;
    high_threshold: number;
  };
  depth?: {
    model: string;
  };
}

export interface DatasetConfig {
  // Existing fields...
  control_type: 'openpose' | 'canny' | 'depth' | null;
  control_preprocessing: ControlPreprocessingConfig;
  generate_control_on_the_fly: boolean;
  control_cache_path: string | null;
}
```

---

### 1.2 Backend Configuration Module Extensions

#### 1.2.1 Model Configuration (`toolkit/config_modules.py`)

**Location**: `ModelConfig` class

**New Fields**:
```python
class ModelConfig:
    def __init__(self, **kwargs):
        # Existing fields...
        
        # NEW: ControlNet configuration
        self.controlnet_enabled: bool = kwargs.get('controlnet_enabled', False)
        self.controlnet_name_or_path: Optional[str] = kwargs.get('controlnet_name_or_path', None)
        self.controlnet_file: Optional[str] = kwargs.get('controlnet_file', None)
```

**Validation Logic** (add to `ModelConfig` validation):
```python
def validate_controlnet_config(self):
    """Validate ControlNet configuration"""
    if self.controlnet_enabled:
        if not self.controlnet_name_or_path:
            raise ValueError("controlnet_name_or_path is required when controlnet_enabled is True")
        
        # Verify it's a Z-Image compatible model
        if self.arch not in ['zimage', 'flux', 'flex2']:
            raise ValueError(f"ControlNet training is only supported for Flux2-based models (zimage, flux, flex2), got {self.arch}")
        
        # Verify controlnet file is specified
        if not self.controlnet_file:
            self.controlnet_file = 'Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors'
            print(f"No controlnet_file specified, using default: {self.controlnet_file}")
```

**Reference**: Follows pattern from `lora_path`, `assistant_lora_path` in existing `ModelConfig`

---

#### 1.2.2 Dataset Configuration (`toolkit/config_modules.py`)

**Location**: `DatasetConfig` class

**New Fields**:
```python
class DatasetConfig:
    def __init__(self, **kwargs):
        # Existing fields...
        
        # NEW: Control conditioning
        self.control_type: Optional[str] = kwargs.get('control_type', None)
        self.control_preprocessing: dict = kwargs.get('control_preprocessing', {})
        self.generate_control_on_the_fly: bool = kwargs.get('generate_control_on_the_fly', True)
        self.control_cache_path: Optional[str] = kwargs.get('control_cache_path', None)
        
        # Auto-generate control cache path if not provided
        if self.control_type and not self.control_cache_path and not self.generate_control_on_the_fly:
            self.control_cache_path = f"{self.folder_path}_control_{self.control_type}"
```

**Validation Logic**:
```python
def validate_control_config(self):
    """Validate control conditioning configuration"""
    valid_control_types = [None, 'openpose', 'canny', 'depth']
    
    if self.control_type not in valid_control_types:
        raise ValueError(f"control_type must be one of {valid_control_types}, got {self.control_type}")
    
    if self.control_type and not self.generate_control_on_the_fly:
        if not self.control_cache_path:
            raise ValueError("control_cache_path is required when generate_control_on_the_fly is False")
        
        # Create cache directory if it doesn't exist
        import os
        os.makedirs(self.control_cache_path, exist_ok=True)
```

---

### 1.3 Dataloader Implementation

#### 1.3.1 Control Image Processor (`toolkit/control_processor.py` - NEW FILE)

**Purpose**: Generate control images from source images using various preprocessors

**File Structure**:
```python
"""
Control image preprocessing for ControlNet training.
Supports OpenPose, Canny, Depth preprocessing.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, Literal
import cv2

try:
    from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector
    CONTROLNET_AUX_AVAILABLE = True
except ImportError:
    CONTROLNET_AUX_AVAILABLE = False
    print("Warning: controlnet_aux not installed. Install with: pip install controlnet-aux")


class ControlImageProcessor:
    """
    Processes images to generate control conditioning images for ControlNet.
    """
    
    def __init__(
        self,
        control_type: Literal['openpose', 'canny', 'depth'],
        config: Dict[str, Any],
        device: str = 'cuda'
    ):
        self.control_type = control_type
        self.config = config
        self.device = device
        self.processor = None
        
        if not CONTROLNET_AUX_AVAILABLE:
            raise ImportError("controlnet_aux is required for control image processing")
        
        self._init_processor()
    
    def _init_processor(self):
        """Initialize the appropriate processor based on control_type"""
        if self.control_type == 'openpose':
            self.processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
            # Configure for hand and face detection if requested
            hand_and_face = self.config.get('openpose', {}).get('hand_and_face', True)
            self.detect_hand_and_face = hand_and_face
            
        elif self.control_type == 'canny':
            # Canny uses OpenCV, no pretrained model needed
            self.low_threshold = self.config.get('canny', {}).get('low_threshold', 100)
            self.high_threshold = self.config.get('canny', {}).get('high_threshold', 200)
            
        elif self.control_type == 'depth':
            model_name = self.config.get('depth', {}).get('model', 'depth-anything-large')
            if 'depth-anything' in model_name:
                self.processor = MidasDetector.from_pretrained('LiheYoung/depth-anything-large-hf')
            else:
                self.processor = MidasDetector.from_pretrained('lllyasviel/ControlNet')
    
    def process_image(self, image: Image.Image) -> Image.Image:
        """
        Process a single image to generate control conditioning.
        
        Args:
            image: PIL Image to process
            
        Returns:
            PIL Image containing control conditioning
        """
        if self.control_type == 'openpose':
            return self._process_openpose(image)
        elif self.control_type == 'canny':
            return self._process_canny(image)
        elif self.control_type == 'depth':
            return self._process_depth(image)
    
    def _process_openpose(self, image: Image.Image) -> Image.Image:
        """Generate OpenPose control image"""
        control_image = self.processor(
            image,
            hand_and_face=self.detect_hand_and_face,
            output_type='pil'
        )
        return control_image
    
    def _process_canny(self, image: Image.Image) -> Image.Image:
        """Generate Canny edge detection control image"""
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        
        # Convert back to PIL
        control_image = Image.fromarray(edges)
        
        # Convert to RGB if needed
        if control_image.mode != 'RGB':
            control_image = control_image.convert('RGB')
        
        return control_image
    
    def _process_depth(self, image: Image.Image) -> Image.Image:
        """Generate depth map control image"""
        control_image = self.processor(image, output_type='pil')
        return control_image
    
    def process_batch(self, images: list[Image.Image]) -> list[Image.Image]:
        """Process a batch of images"""
        return [self.process_image(img) for img in images]


def get_control_processor(
    control_type: Optional[str],
    config: Dict[str, Any],
    device: str = 'cuda'
) -> Optional[ControlImageProcessor]:
    """
    Factory function to create control processor.
    
    Args:
        control_type: Type of control ('openpose', 'canny', 'depth', or None)
        config: Preprocessing configuration
        device: Device to run processor on
        
    Returns:
        ControlImageProcessor instance or None if control_type is None
    """
    if control_type is None:
        return None
    
    return ControlImageProcessor(control_type, config, device)
```

**Key Design Decisions**:
1. **Lazy Loading**: Processors are only initialized when needed
2. **Caching Support**: Processed images can be saved to disk and reloaded
3. **Batch Processing**: Efficient batch processing for pre-generation
4. **Error Handling**: Graceful fallback if controlnet_aux not available

**Dependencies to Add** (`requirements.txt`):
```
controlnet-aux>=0.0.7
opencv-python>=4.8.0
```

---

#### 1.3.2 Dataset Modifications (`toolkit/dataloader_mixins.py`)

**Location**: Modify existing `BucketsMixin` and dataset loading logic

**Key Changes**:

**1. Add Control Image Loading to FileItemDTO**:

The `FileItemDTO` class needs to track control image paths alongside regular images.

```python
# In toolkit/data_transfer_object/data_loader.py or similar

class FileItemDTO:
    def __init__(self, ...):
        # Existing fields...
        
        # NEW: Control image tracking
        self.control_image_path: Optional[str] = None
        self.control_image: Optional[Image.Image] = None  # Cached control image
        self.requires_control_generation: bool = False
```

**2. Modify Bucket Setup to Handle Control Images**:

```python
# In toolkit/dataloader_mixins.py

class BucketsMixin:
    def setup_buckets(self: 'AiToolkitDataset', quiet=False):
        """
        Setup buckets for training. 
        Now handles control image generation/loading if control_type is set.
        """
        # Existing bucket setup code...
        
        config: 'DatasetConfig' = self.dataset_config
        
        # NEW: Initialize control processor if needed
        control_processor = None
        if config.control_type:
            from toolkit.control_processor import get_control_processor
            control_processor = get_control_processor(
                config.control_type,
                config.control_preprocessing,
                device=self.device
            )
        
        # Iterate through files and setup control images
        for idx, file_item in enumerate(file_list):
            file_item: 'FileItemDTO' = file_item
            
            # Existing bucket assignment code...
            
            # NEW: Handle control image setup
            if config.control_type:
                file_item.requires_control_generation = True
                
                if not config.generate_control_on_the_fly:
                    # Pre-generate or load cached control image
                    control_path = self._get_control_image_path(file_item, config)
                    file_item.control_image_path = control_path
                    
                    if not os.path.exists(control_path):
                        # Generate and cache
                        self._generate_and_cache_control(
                            file_item, 
                            control_path, 
                            control_processor
                        )
```

**3. Add Control Image Path Helper**:

```python
def _get_control_image_path(
    self,
    file_item: 'FileItemDTO',
    config: 'DatasetConfig'
) -> str:
    """
    Get the path where control image should be stored/loaded from.
    
    Maintains same directory structure as source images.
    """
    relative_path = os.path.relpath(file_item.path, config.folder_path)
    base_name = os.path.splitext(relative_path)[0]
    control_filename = f"{base_name}_control.png"
    control_path = os.path.join(config.control_cache_path, control_filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(control_path), exist_ok=True)
    
    return control_path

def _generate_and_cache_control(
    self,
    file_item: 'FileItemDTO',
    control_path: str,
    processor: 'ControlImageProcessor'
):
    """Generate control image and save to cache"""
    # Load source image
    source_image = Image.open(file_item.path).convert('RGB')
    
    # Generate control image
    control_image = processor.process_image(source_image)
    
    # Save to cache
    control_image.save(control_path)
    
    print(f"Generated and cached control image: {control_path}")
```

**4. Modify `__getitem__` to Load Control Images**:

The dataset's `__getitem__` method needs to return control images alongside regular images.

```python
class AiToolkitDataset:
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Now returns control images if control_type is set.
        """
        # Existing code to load image...
        file_item = self.file_list[idx]
        image = self.load_image(file_item)  # Existing image loading

        # NEW: Load or generate control image
        control_image = None
        if self.dataset_config.control_type:
            control_image = self._get_control_image(file_item)

        # Apply transformations (resize, crop, etc.)
        # IMPORTANT: Apply SAME transformations to both image and control_image
        if control_image is not None:
            image, control_image = self._apply_synchronized_transforms(
                image,
                control_image,
                file_item
            )
        else:
            image = self._apply_transforms(image, file_item)

        # NOTE: Z-Image DOES NOT tile control images into 16 tensors.
        # Control images are encoded WHOLE through the VAE (see Part 2 `encode_control_images`).
        # If memory is constrained, enable VAE tiling during encoding with `vae.enable_tiling()` / `vae.disable_tiling()`.
        # No per-image tiling into patches is required at dataset load time; return full images and control images.

        # Return both images (non-tiling models)
        return {
            'image': image,
            'control_image': control_image,
            'file_item': file_item,
            # Other existing fields...
        }
    
    def _get_control_image(self, file_item: 'FileItemDTO') -> Image.Image:
        """Load or generate control image for a file item"""
        if file_item.control_image_path and os.path.exists(file_item.control_image_path):
            # Load from cache
            return Image.open(file_item.control_image_path).convert('RGB')
        elif self.dataset_config.generate_control_on_the_fly:
            # Generate on-the-fly
            if not hasattr(self, '_control_processor'):
                from toolkit.control_processor import get_control_processor
                self._control_processor = get_control_processor(
                    self.dataset_config.control_type,
                    self.dataset_config.control_preprocessing,
                    device=self.device
                )
            source_image = Image.open(file_item.path).convert('RGB')
            return self._control_processor.process_image(source_image)
        else:
            raise FileNotFoundError(
                f"Control image not found for {file_item.path} "
                f"and on-the-fly generation is disabled"
            )


### Fail-Fast: Dataset / Control Preflight Checks ⚠️

"""
Fail-fast policies: ensure training does not begin if required control assets or
configuration are missing. These checks should run during dataset setup and
before the training loop starts (e.g., in `hook_before_train_loop`).
"""

def validate_control_dataset(self):
    """Perform preflight checks for control conditioning.

    Raises a descriptive error if a critical precondition is missing.
    """
    # 1. If ControlNet is enabled in model config, ensure at least one dataset provides control_type
    if self.model_config.controlnet_enabled:
        has_control = any(ds.control_type is not None for ds in self.train_config.datasets)
        if not has_control:
            raise RuntimeError("ControlNet enabled but no dataset has `control_type` set. Aborting training.")

    # 2. If any dataset uses pre-generated control images, ensure the cache path exists and has files
    for ds in self.train_config.datasets:
        if ds.control_type and not ds.generate_control_on_the_fly:
            if not ds.control_cache_path or not os.path.isdir(ds.control_cache_path):
                raise RuntimeError(f"Control cache path missing for dataset {ds.folder_path}. Set `control_cache_path` or enable `generate_control_on_the_fly`.")
            # Quick sanity: require at least 1 control file
            files = list(Path(ds.control_cache_path).glob('**/*_control.*'))
            if len(files) == 0:
                raise RuntimeError(f"Control cache at {ds.control_cache_path} exists but contains no control files. Aborting.")

    # 3. If ControlNet is enabled, ensure control_processor dependency is present
    if any(ds.control_type for ds in self.train_config.datasets):
        try:
            import controlnet_aux  # noqa: F401
        except Exception:
            raise RuntimeError("`controlnet_aux` is required for control preprocessing. Install via `pip install controlnet-aux`.")

    # 4. Environment checks (optional): warn if host RAM < recommended threshold and streaming not enabled
    ram_gb = None
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass

    if self.model_config.controlnet_enabled and hasattr(self.model_config, 'controlnet_file') and self.model_config.controlnet_file:
        if ram_gb is not None and ram_gb < 64 and not getattr(self.model_config, 'controlnet_streaming', False):
            raise RuntimeError(
                "Host RAM appears to be <64GB and ControlNet streaming is not enabled. "
                "Set `controlnet_streaming=True` in the model config or use a host with more RAM. Aborting training."
            )

# Hook this validator into dataset setup and the job preflight
# Example: call self.validate_control_dataset() in BucketsMixin.setup_buckets and BaseSDTrainProcess.hook_before_train_loop

```

**5. CRITICAL: Synchronized Transforms for Image and Control**:

```python
def _apply_synchronized_transforms(
    self,
    image: Image.Image,
    control_image: Image.Image,
    file_item: 'FileItemDTO'
) -> tuple[Image.Image, Image.Image]:
    """
    Apply the SAME transformations to both image and control image.
    This ensures 1:1 pixel correspondence between them.

    CRITICAL: Must apply identical:
    - Resizing
    - Cropping (including random crops with same seed)
    - Flipping
    - Any spatial transformations

    For Z-Image tiled models: apply synchronized transforms FIRST, then TILE images
    so that tiling corresponds exactly across image and control.

    Must NOT apply to control:
    - Color jittering
    - Normalization (different for control vs image)
    """
    # Get transform parameters from file_item (set during bucket setup)
    target_width = file_item.scale_to_width or file_item.crop_width
    target_height = file_item.scale_to_height or file_item.crop_height
    crop_x = file_item.crop_x or 0
    crop_y = file_item.crop_y or 0
    crop_width = file_item.crop_width
    crop_height = file_item.crop_height

    # Apply resize
    if target_width and target_height:
        image = image.resize((target_width, target_height), Image.LANCZOS)
        control_image = control_image.resize((target_width, target_height), Image.LANCZOS)

    # Apply crop
    if crop_width and crop_height:
        image = image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))
        control_image = control_image.crop((crop_x, crop_y, crop_x + crop_width, crop_y + crop_height))

    # Apply flips if configured
    if self.dataset_config.flip_x and random.random() < 0.5:
        seed = random.randint(0, 2**32-1)
        random.seed(seed)
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        random.seed(seed)
        control_image = control_image.transpose(Image.FLIP_LEFT_RIGHT)

    return image, control_image


# --- VAE Tiling guidance for Z-Image (CRITICAL) ---

def encode_control_images(self, control_images: torch.Tensor, use_tiling: bool = False) -> torch.Tensor:
    """
    Encode control images WHOLE through the VAE.

    IMPORTANT: Control images are encoded as whole images (no per-image 16-patch tiling).
    If memory is constrained, enable the VAE's internal tiling mechanism for encoding by
    setting `use_tiling=True` (if the VAE implementation supports `enable_tiling()`/`disable_tiling()`).

    Returns:
        control_latents: Tensor with shape [B, C, F, H, W] (F=1 for images)
    """
    # Example implementation (actual VAE API may differ):
    if use_tiling and hasattr(self.vae, 'enable_tiling'):
        self.vae.enable_tiling()

    with torch.no_grad():
        latents = self.vae.encode(control_images)[0].mode()

    if use_tiling and hasattr(self.vae, 'disable_tiling'):
        self.vae.disable_tiling()

    # Ensure frame dimension for video-compatible architecture
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)  # [B, C, 1, H, W]

    return latents

# NOTE: BucketsMixin and dataset setup do NOT need to perform per-image tiling; they should
# provide full control images to the encoder and rely on the VAE's optional internal tiling.
```

**Reference**: VideoX-Fun implementation in `scripts/z_image_fun/train_control.py` shows control images are processed identically to source images for spatial transformations.

---

#### 1.3.3 Dataloader Batch Collation

**Location**: Modify batch collation to include control images

```python
def collate_fn(batch):
    """
    Collate function for dataloader.
    Now handles control images alongside regular images.
    """
    # Existing collation for images, captions, etc.
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    # NEW: Collate control images if present
    control_images = None
    if batch[0].get('control_image') is not None:
        control_images = torch.stack([item['control_image'] for item in batch])
    
    return {
        'images': images,
        'control_images': control_images,
        'captions': captions,
        # Other fields...
    }
```

---

### 1.4 Pre-Generation Utility Script

**Purpose**: Allow users to pre-generate all control images before training

**Location**: `scripts/generate_control_images.py` (NEW FILE)

```python
"""
Pre-generate control images for a dataset.
This can significantly speed up training by avoiding on-the-fly generation.

Usage:
    python scripts/generate_control_images.py \
        --input-dir datasets/my_dataset \
        --output-dir datasets/my_dataset_control_openpose \
        --control-type openpose \
        --batch-size 8
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

from toolkit.control_processor import get_control_processor


class ImageFolderDataset(Dataset):
    """Simple dataset for loading images from a folder"""
    def __init__(self, folder_path, valid_extensions=('.jpg', '.jpeg', '.png', '.webp')):
        self.folder_path = Path(folder_path)
        self.image_paths = []
        
        for ext in valid_extensions:
            self.image_paths.extend(list(self.folder_path.rglob(f'*{ext}')))
        
        print(f"Found {len(self.image_paths)} images in {folder_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        return image, img_path


def main():
    parser = argparse.ArgumentParser(description='Pre-generate control images for ControlNet training')
    parser.add_argument('--input-dir', type=str, required=True, help='Input directory containing images')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for control images')
    parser.add_argument('--control-type', type=str, required=True, choices=['openpose', 'canny', 'depth'])
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--hand-and-face', action='store_true', help='Detect hands and face (openpose only)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing control images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup control processor
    config = {}
    if args.control_type == 'openpose':
        config['openpose'] = {'hand_and_face': args.hand_and_face}
    
    processor = get_control_processor(args.control_type, config, args.device)
    
    # Create dataset and dataloader
    dataset = ImageFolderDataset(args.input_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
    
    # Process images
    for images, img_paths in tqdm(dataloader, desc='Generating control images'):
        for image, img_path in zip(images, img_paths):
            # Calculate output path (maintain directory structure)
            rel_path = Path(img_path).relative_to(args.input_dir)
            output_path = Path(args.output_dir) / rel_path.parent / f"{rel_path.stem}_control.png"
            
            # Skip if exists and not overwriting
            if output_path.exists() and not args.overwrite:
                continue
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate and save control image
            control_image = processor.process_image(image)
            control_image.save(output_path)
    
    print(f"Control image generation complete! Saved to {args.output_dir}")


if __name__ == '__main__':
    main()
```

---

### 1.5 Testing & Validation for Part 1

**Unit Tests to Write**:

1. **Control Processor Tests** (`testing/test_control_processor.py`):
   - Test each control type (openpose, canny, depth)
   - Verify output dimensions match input dimensions
   - Test batch processing
   - Test error handling for missing dependencies

2. **Dataset Tests** (`testing/test_controlnet_dataset.py`):
   - Test synchronized transforms on image and control pairs
   - Verify 1:1 correspondence after transforms
   - Test VAE tiling behavior: ensure `encode_control_images(..., use_tiling=True)` enables VAE tiling and returns latents with correct shape
   - Test both cached and on-the-fly generation modes
   - Test bucket assignment with control images

3. **UI Config Tests**:
   - Validate TypeScript types compile correctly
   - Test form validation logic

**Manual Testing Checklist**:
- [ ] UI displays ControlNet options correctly
- [ ] Control type dropdown populates and saves correctly
- [ ] Pre-generation script runs and produces valid control images
- [ ] Dataset loads control images in sync with regular images
- [ ] Bucket assignment works with control images
- [ ] Transforms are applied identically to both image types

---

## Summary of Part 1

**Files Created**:
1. `toolkit/control_processor.py` - Control image generation
2. `scripts/generate_control_images.py` - Pre-generation utility
3. `testing/test_control_processor.py` - Unit tests
4. `testing/test_controlnet_dataset.py` - Dataset tests

**Files Modified**:
1. `ui/src/app/jobs/new/jobConfig.ts` - Config defaults
2. `ui/src/app/jobs/new/AdvancedJob.tsx` - UI forms
3. `ui/src/types/index.ts` - TypeScript types
4. `toolkit/config_modules.py` - Backend config classes
5. `toolkit/dataloader_mixins.py` - Dataset loading logic
6. `toolkit/data_transfer_object/data_loader.py` - FileItemDTO class

**Key Design Principles**:
✅ **1:1 Correspondence**: Control images always match source images exactly after transforms
✅ **Flexible Caching**: Support both pre-generated and on-the-fly generation
✅ **Extensibility**: Easy to add new control types (canny, depth) later
✅ **Error Handling**: Graceful fallbacks if dependencies missing
✅ **Performance**: Batch processing and caching options

**Next Step**: Part 2 will cover the ControlNet adapter loading, transformer integration, and shim implementations.

---

*End of Part 1*
