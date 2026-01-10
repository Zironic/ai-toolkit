"""
Standalone SAM3 loader that bypasses ComfyUI dependencies.
Directly uses the SAM3 library for image segmentation.
"""
import sys
from pathlib import Path
import torch


def load_sam3_standalone(comfyui_sam3_root: str, model_path: str, device: str = "cuda"):
    """Load SAM3 model directly without ComfyUI dependencies.
    
    Args:
        comfyui_sam3_root: Path to comfyui-sam3 directory
        model_path: Path to SAM3 checkpoint file
        device: Device to load model on
        
    Returns:
        Tuple of (video_predictor, processor) for image segmentation
    """
    sam3_root = Path(comfyui_sam3_root)
    sam3_lib_path = sam3_root / "nodes" / "sam3_lib"
    nodes_path = sam3_root / "nodes"
    
    if not sam3_lib_path.exists():
        raise FileNotFoundError(f"SAM3 library not found: {sam3_lib_path}")
    
    # Add nodes directory to sys.path to enable relative imports
    sam3_lib_str = str(sam3_lib_path)
    nodes_path_str = str(nodes_path)
    sam3_root_str = str(sam3_root)
    
    # Add all necessary paths
    paths_to_add = [sam3_root_str, nodes_path_str, sam3_lib_str]
    added_paths = []
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
            added_paths.append(path)
    
    try:
        # Import SAM3 components using package-style imports
        # This requires nodes to be a package
        import sam3_lib.sam3_video_predictor as video_predictor_module
        import sam3_lib.model.sam3_image_processor as processor_module
        
        Sam3VideoPredictor = video_predictor_module.Sam3VideoPredictor
        Sam3Processor = processor_module.Sam3Processor
        
        # BPE tokenizer path
        bpe_path = sam3_lib_path / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            raise FileNotFoundError(f"BPE tokenizer not found: {bpe_path}")
        
        print(f"[SAM3] Loading model from: {model_path}")
        print(f"[SAM3] Using BPE tokenizer: {bpe_path}")
        
        # Create video predictor (supports both image and video)
        video_predictor = Sam3VideoPredictor(
            checkpoint_path=str(model_path),
            bpe_path=str(bpe_path),
            enable_inst_interactivity=True,  # Enable point/box prompts
        )
        
        print(f"[SAM3] Video predictor loaded")
        
        # Create processor for image segmentation
        detector = video_predictor.model.detector
        processor = Sam3Processor(
            model=detector,
            resolution=1008,
            device=device,
            confidence_threshold=0.2
        )
        
        print(f"[SAM3] Processor created")
        
        # Simple wrapper to match expected interface
        class SAM3Wrapper:
            def __init__(self, video_predictor, processor):
                self.video_predictor = video_predictor
                self.processor = processor
                self.model = video_predictor.model
                
            def to(self, device):
                self.model.to(device)
                return self
            
            def eval(self):
                self.model.eval()
                return self
        
        wrapper = SAM3Wrapper(video_predictor, processor)
        wrapper.to(device)
        wrapper.eval()
        
        return wrapper, processor
        
    except ImportError as e:
        raise ImportError(f"Failed to import SAM3 library: {e}\n"
                         f"Make sure comfyui-sam3/nodes/sam3_lib is complete")
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM3 model: {e}")
    finally:
        # Clean up sys.path
        for path in added_paths:
            if path in sys.path:
                sys.path.remove(path)
