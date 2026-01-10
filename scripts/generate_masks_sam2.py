#!/usr/bin/env python3
"""Generate binary masks for a dataset using SAM (Segment Anything Model).

This script uses facebook/sam-vit-base (or other SAM models) from HuggingFace to generate masks
for images in a dataset. Masks are saved to a specified output folder with matching
filenames.

Note: Use SAM (facebook/sam-vit-*) models for images, not SAM2 (which is optimized for video).

Usage:
    python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks
    python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --box "[[100,100,400,400]]" --output datasets/my_dataset/masks --model facebook/sam-vit-large
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    print("Error: transformers required. Install: pip install transformers")
    sys.exit(1)


def load_sam3_via_comfyui(comfyui_sam3_root: str, model_path: str = "models/sam3/sam3.pt"):
    """Load SAM3 model using ComfyUI's loader.
    
    Args:
        comfyui_sam3_root: Path to comfyui-sam3 directory
        model_path: Path to SAM3 model checkpoint (relative to AI-Toolkit root)
        
    Returns:
        SAM3UnifiedModel instance with .processor for image segmentation
    """
    sam3_root = Path(comfyui_sam3_root)
    if not sam3_root.exists():
        raise FileNotFoundError(f"comfyui-sam3 directory not found: {comfyui_sam3_root}")
    
    nodes_path = sam3_root / "nodes"
    if not nodes_path.exists():
        raise FileNotFoundError(f"comfyui-sam3/nodes not found: {nodes_path}")
    
    # Add to sys.path
    nodes_path_str = str(nodes_path)
    if nodes_path_str not in sys.path:
        sys.path.insert(0, nodes_path_str)
    
    try:
        # Inject ComfyUI shims before importing
        if 'comfy' not in sys.modules:
            # Import our shim module
            script_dir = Path(__file__).parent
            sys.path.insert(0, str(script_dir))
            from comfy_shim import ComfyShim, FolderPaths
            
            # Inject into sys.modules
            sys.modules['comfy'] = ComfyShim()
            sys.modules['comfy.model_management'] = ComfyShim.model_management
            sys.modules['folder_paths'] = FolderPaths()
        
        # Import the loader class
        from load_model import LoadSAM3Model
        
        print(f"[SAM3] Using ComfyUI loader from: {comfyui_sam3_root}")
        print(f"[SAM3] Loading model: {model_path}")
        
        loader = LoadSAM3Model()
        unified_model = loader.load_model(model_path)[0]
        
        print(f"[SAM3] Model loaded successfully")
        return unified_model
        
    except ImportError as e:
        raise ImportError(f"Failed to import ComfyUI SAM3 loader: {e}\nMake sure comfyui-sam3 is complete")
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM3 model: {e}")
    finally:
        # Clean up sys.path
        if nodes_path_str in sys.path:
            sys.path.remove(nodes_path_str)


def load_sam2_model(model_id: str = "facebook/sam-vit-base", device: str = "cuda", use_comfyui: bool = False, comfyui_sam3_path: Optional[str] = None):
    """Load SAM model from HuggingFace or SAM3 via standalone loader.
    
    Args:
        model_id: HuggingFace model ID (e.g., facebook/sam-vit-base) or SAM3 checkpoint path
        device: Device to load model on (cuda/cpu)
        use_comfyui: If True, use SAM3 standalone loader
        comfyui_sam3_path: Path to comfyui-sam3 directory
    
    Returns:
        (model, processor) tuple
    """
    if use_comfyui:
        if not comfyui_sam3_path:
            raise ValueError("comfyui_sam3_path required when use_comfyui=True")
        
        # Use standalone SAM3 loader
        from sam3_standalone import load_sam3_standalone
        model, processor = load_sam3_standalone(comfyui_sam3_path, model_id, device)
        return model, processor
    
    print(f"Loading SAM from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    return model, processor


def generate_mask_from_points(
    image_path: str,
    points: List[List[int]],
    labels: Optional[List[int]],
    model,
    processor,
    device: str
) -> np.ndarray:
    """Generate binary mask using point prompts.
    
    Args:
        image_path: Path to input image
        points: List of [x, y] points
        labels: List of labels (1 for foreground, 0 for background)
        model: SAM2 model
        processor: SAM2 processor
        device: Device string
    
    Returns:
        Binary mask [H, W] with values {0, 255}
    """
    image = Image.open(image_path).convert("RGB")
    
    if labels is None:
        labels = [1] * len(points)  # Default all points as foreground
    
    # Prepare inputs for SAM2
    # Format: [[[x, y], ...]] for batch processing
    inputs = processor(
        image,
        input_points=[[points]],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Post-process masks to original size
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        # Take the first mask from the first batch
        mask = masks[0][0, 0].numpy()
        mask_binary = (mask > 0).astype(np.uint8) * 255
    
    return mask_binary


def generate_mask_from_box(
    image_path: str,
    boxes: List[List[int]],
    model,
    processor,
    device: str
) -> np.ndarray:
    """Generate binary mask using bounding box prompts.
    
    Args:
        image_path: Path to input image
        boxes: List of [x1, y1, x2, y2] boxes
        model: SAM2 model
        processor: SAM2 processor
        device: Device string
    
    Returns:
        Binary mask [H, W] with values {0, 255}
    """
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs for SAM2
    # Format: [[[x1, y1, x2, y2], ...]]
    inputs = processor(
        image,
        input_boxes=[[boxes]],
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Post-process masks to original size
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )
        # Take the first mask from the first batch
        mask = masks[0][0, 0].numpy()
        mask_binary = (mask > 0).astype(np.uint8) * 255
    
    return mask_binary
def generate_mask_from_text(
    image_path: str,
    text: str,
    model,
    processor,
    device: str
) -> np.ndarray:
    """Generate binary mask using text prompt (SAM3 only).
    
    Args:
        image_path: Path to input image
        text: Text prompt
        model: SAM3 model wrapper
        processor: SAM3 processor
        device: Device string
    
    Returns:
        Binary mask [H, W] with values {0, 255}
    """
    image = Image.open(image_path).convert("RGB")
    
    # SAM3 uses a state-based API
    state = {}
    state = processor.set_image(image, state)
    state = processor.set_text_prompt(text, state)
    
    # Get masks from state
    if "masks" in state and len(state["masks"]) > 0:
        # Take the first (highest confidence) mask
        mask = state["masks"][0, 0].cpu().numpy()  # [H, W]
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
    else:
        # No masks found, return empty mask
        mask_binary = np.zeros((image.height, image.width), dtype=np.uint8)
    
    return mask_binary


def process_dataset(
    dataset_path: str,
    output_folder: str,
    model_id: str,
    device: str,
    points: Optional[List[List[int]]] = None,
    labels: Optional[List[int]] = None,
    boxes: Optional[List[List[int]]] = None,
    extensions: Optional[List[str]] = None,
    text: Optional[str] = None,
    use_comfyui: bool = False,
    comfyui_loader_path: Optional[str] = None,
):
    """Process all images in dataset and generate masks.
    
    Args:
        dataset_path: Path to dataset folder
        output_folder: Path to output folder for masks
        model_id: HuggingFace model ID or SAM3 checkpoint path
        device: Device to use
        points: Point prompts [[x, y], ...]
        labels: Point labels [1, 0, ...]
        boxes: Box prompts [[x1, y1, x2, y2], ...]
        extensions: List of file extensions to process
        text: Text prompt for SAM3
        use_comfyui: Use ComfyUI SAM3 loader
        comfyui_loader_path: Path to ComfyUI load_model.py
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    extensions = {ext.lower() for ext in extensions}
    
    # Find images
    images = [f for f in dataset_path.iterdir() 
              if f.is_file() and f.suffix.lower() in extensions]
    
    if not images:
        print(f"No images found in {dataset_path}")
        return
    
    print(f"Found {len(images)} images")
    
    if points:
        print(f"Using point prompts: {points} with labels: {labels}")
    elif boxes:
        print(f"Using box prompts: {boxes}")
    elif text:
        print(f"Using text prompt: {text}")

    model, processor = load_sam2_model(model_id, device, use_comfyui, comfyui_loader_path)
    
    success_count = 0
    error_count = 0
    
    for img_file in tqdm(images, desc="Generating masks"):
        try:
            if points:
                mask = generate_mask_from_points(
                    str(img_file), points, labels, model, processor, device
                )
            elif boxes:
                mask = generate_mask_from_box(
                    str(img_file), boxes, model, processor, device
                )
            elif text:
                mask = generate_mask_from_text(
                    str(img_file), text, model, processor, device
                )
            # Save mask with same name as image (as PNG)
            mask_name = img_file.stem + ".png"
            mask_path = output_dir / mask_name
            Image.fromarray(mask, 'L').save(mask_path)
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
            error_count += 1
    
    print(f"\nDone! {success_count} masks saved to {output_dir}")
    if error_count > 0:
        print(f"Errors: {error_count} images failed")


def main():
    parser = argparse.ArgumentParser(
        description="Generate masks using SAM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate masks using center point:
  python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks
  
  # Generate masks using multiple points:
  python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[100,100],[200,200]]" --labels "[1,1]" --output datasets/my_dataset/masks
  
  # Generate masks using bounding box:
  python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --box "[[100,100,400,400]]" --output datasets/my_dataset/masks
  
  # Use larger model for better quality:
  python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks --model facebook/sam-vit-large
        """
    )
    
    parser.add_argument("--dataset", "-d", required=True, 
                       help="Dataset folder path")
    parser.add_argument("--output", "-o", required=True,
                       help="Output folder for masks")
    parser.add_argument("--points", "-p", type=str,
                       help="Point prompts as JSON array: '[[x1,y1],[x2,y2]]'")
    parser.add_argument("--labels", "-l", type=str,
                       help="Point labels as JSON array: '[1,0,1]' (1=foreground, 0=background)")
    parser.add_argument("--box", "-b", type=str,
                       help="Bounding box prompts as JSON array: '[[x1,y1,x2,y2]]'")
    parser.add_argument("--model", "-m", 
                       default="facebook/sam-vit-base",
                       help="SAM model ID (default: facebook/sam-vit-base)")
    parser.add_argument("--device", 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (default: cuda if available)")
    parser.add_argument("--use-comfyui", action="store_true",
                       help="Use ComfyUI SAM3 loader")
    parser.add_argument("--comfyui-sam3", type=str,
                       default=r"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\comfyui-sam3",
                       help="Path to comfyui-sam3 directory")
    parser.add_argument("--extensions", type=str,
                       help="Comma-separated file extensions (default: jpg,jpeg,png,webp,bmp)")
    parser.add_argument("--text", type=str, default=None,
                       help="Text prompt for SAM3 (requires --use-comfyui)")
    args = parser.parse_args()
    
    # Parse points
    points = None
    if args.points:
        try:
            points = json.loads(args.points)
            if not isinstance(points, list) or not all(isinstance(p, list) and len(p) == 2 for p in points):
                print("Error: --points must be a list of [x, y] coordinates")
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing --points: {e}")
            sys.exit(1)
    
    # Parse labels
    labels = None
    if args.labels:
        try:
            labels = json.loads(args.labels)
            if not isinstance(labels, list):
                print("Error: --labels must be a list of integers")
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing --labels: {e}")
            sys.exit(1)
    
    # Parse box
    boxes = None
    if args.box:
        try:
            boxes = json.loads(args.box)
            if not isinstance(boxes, list):
                print("Error: --box must be a list of [x1, y1, x2, y2] coordinates")
                sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error parsing --box: {e}")
            sys.exit(1)
    
    # Parse extensions
    extensions = None
    if args.extensions:
        extensions = ['.' + ext.strip().lstrip('.') for ext in args.extensions.split(',')]
    
    process_dataset(
        args.dataset,
        args.output,
        args.model,
        args.device,
        points=points,
        labels=labels,
        boxes=boxes,
        text=args.text,
        extensions=extensions,
        use_comfyui=args.use_comfyui,
        comfyui_loader_path=args.comfyui_sam3 if args.use_comfyui else None,
    )



if __name__ == "__main__":
    main()
