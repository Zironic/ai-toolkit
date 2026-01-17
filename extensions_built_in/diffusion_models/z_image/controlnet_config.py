import os
from typing import Optional

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


class ZImageControlNetConfigGenerator:
    """Generate a minimal control transformer config when a normal base transformer
    config cannot be obtained. This is a conservative fallback that favors safety
    (small defaults) and writes a `config.json` into the controlnet_path so
    subsequent loads can re-use it.

    The generator inspects safetensors keys (if available) to infer any obvious
    attributes, but will always return a minimal config dict rather than
    attempt to perfectly re-create the original.
    """

    def __init__(self, controlnet_path: str, safetensors_file: Optional[str] = None):
        self.controlnet_path = controlnet_path
        self.safetensors_file = safetensors_file

    def _find_file(self) -> Optional[str]:
        if self.safetensors_file:
            p = os.path.join(self.controlnet_path, self.safetensors_file)
            return p if os.path.exists(p) else None
        # find any .safetensors file
        for f in os.listdir(self.controlnet_path):
            if f.endswith('.safetensors'):
                return os.path.join(self.controlnet_path, f)
        return None

    def _inspect_keys(self, path: str) -> dict:
        info = {}
        if safe_open is None or path is None:
            return info
        try:
            with safe_open(path, framework='pt', device='cpu') as f:
                keys = list(f.keys())
                # simple heuristics
                if any(k.startswith('transformer.in_channels') or 'in_channels' in k for k in keys):
                    info['inferred_has_in_channels'] = True
                if any(k.startswith('transformer') for k in keys):
                    info['has_transformer_keys'] = True
                # check for common control shapes
                for k in keys[:20]:
                    if 'embed' in k and 'x' in k:
                        info['likely_has_x_embedder'] = True

                # Count control layers by looking for control_layers.N patterns
                control_layer_indices = set()
                for k in keys:
                    if 'control_layers.' in k:
                        # Extract layer index from patterns like "control_layers.0.after_proj.weight"
                        parts = k.split('control_layers.')
                        if len(parts) > 1:
                            idx_part = parts[1].split('.')[0]
                            if idx_part.isdigit():
                                control_layer_indices.add(int(idx_part))

                if control_layer_indices:
                    info['num_control_layers'] = len(control_layer_indices)
                    info['max_control_layer_idx'] = max(control_layer_indices)
                    # Store the actual indices found (not just the count)
                    info['control_layer_indices'] = sorted(list(control_layer_indices))

                # Check for control_noise_refiner presence and count refiner layers
                if any('control_noise_refiner' in k for k in keys):
                    info['has_control_noise_refiner'] = True

                # Count control_noise_refiner layers by looking for control_noise_refiner.N patterns
                refiner_layer_indices = set()
                for k in keys:
                    if 'control_noise_refiner.' in k:
                        # Extract layer index from patterns like "control_noise_refiner.0.after_proj.weight"
                        parts = k.split('control_noise_refiner.')
                        if len(parts) > 1:
                            idx_part = parts[1].split('.')[0]
                            if idx_part.isdigit():
                                refiner_layer_indices.add(int(idx_part))

                if refiner_layer_indices:
                    info['num_refiner_layers'] = len(refiner_layer_indices)
                    info['max_refiner_layer_idx'] = max(refiner_layer_indices)

                # Check for standard 30-layer vs lite 5-layer by looking at layers count
                layer_indices = set()
                for k in keys:
                    if 'layers.' in k and 'control_layers' not in k:
                        parts = k.split('layers.')
                        if len(parts) > 1:
                            idx_part = parts[1].split('.')[0]
                            if idx_part.isdigit():
                                layer_indices.add(int(idx_part))
                if layer_indices:
                    info['num_layers'] = max(layer_indices) + 1  # 0-indexed
        except Exception:
            pass
        return info

    def generate(self) -> dict:
        """Return a conservative config dict appropriate for instantiating a
        ZImageControlTransformer2DModel or at least allowing downstream code to
        validate required attributes.

        Valid parameters for ZImageControlTransformer2DModel.__init__:
        - control_layers_places: list of layer indices where control is applied
        - control_refiner_layers_places: list of refiner layer indices
        - control_in_dim: input dimension for control (default: same as in_channels)
        - add_control_noise_refiner: whether to use control noise refiner
        - add_control_noise_refiner_correctly: variant of noise refiner
        - Plus all base ZImageTransformer2DModel parameters (in_channels, dim, n_layers, etc.)
        """
        path = self._find_file()
        hints = self._inspect_keys(path)

        # Determine number of layers from inspection or use defaults
        n_layers = hints.get('num_layers', 30)  # Default to 30 (standard) or detected
        num_control_layers = hints.get('num_control_layers', None)
        num_refiner_layers = hints.get('num_refiner_layers', 2)  # Default to 2, detect from checkpoint

        # Build control_layers_places from detected count
        # Pattern inference based on known model variants:
        # - Standard (15 control layers on 30-layer transformer): [0, 2, 4, ..., 28] (every 2nd layer)
        # - Lite (3 control layers on 30-layer transformer): [0, 10, 20] (every 10th layer)
        # - General pattern: evenly distribute N control layers across n_layers
        if num_control_layers is not None:
            if num_control_layers == 3 and n_layers == 30:
                # Lite model: 3 control layers at [0, 10, 20]
                control_layers_places = [0, 10, 20]
            elif num_control_layers == 15 and n_layers == 30:
                # Standard model: 15 control layers at [0, 2, 4, ..., 28]
                control_layers_places = [i * 2 for i in range(15)]
            elif num_control_layers > 0:
                # General case: evenly distribute control layers
                # Calculate stride to evenly space control layers across transformer layers
                stride = n_layers // num_control_layers
                control_layers_places = [i * stride for i in range(num_control_layers)]
            else:
                control_layers_places = []
        else:
            # Default: every other layer (standard pattern)
            control_layers_places = [i for i in range(0, n_layers, 2)]

        # Build control_refiner_layers_places from detected refiner layer count
        # Standard: [0, 1] for 2 refiners, [0, 1, 2] for 3 refiners
        control_refiner_layers_places = list(range(num_refiner_layers))

        # Minimal defaults that satisfy ZImageControlTransformer2DModel.__init__
        cfg = {
            # Base transformer parameters (match Z-Image defaults)
            'in_channels': 16,  # Z-Image uses 16 channels
            'dim': 3840,
            'n_layers': n_layers,
            'n_refiner_layers': num_refiner_layers,  # Detect from checkpoint (2 or 3)
            'n_heads': 30,
            'n_kv_heads': 30,
            'norm_eps': 1e-5,
            'qk_norm': True,
            'cap_feat_dim': 2560,
            'rope_theta': 256.0,
            't_scale': 1000.0,
            'axes_dims': [32, 48, 48],
            'axes_lens': [1024, 512, 512],
            'all_patch_size': [2],  # JSON-serializable list, will be converted to tuple
            'all_f_patch_size': [1],  # JSON-serializable list, will be converted to tuple
            # Control-specific parameters
            'control_layers_places': control_layers_places,
            'control_refiner_layers_places': control_refiner_layers_places,
            # control_in_dim=33 for single control: control_latent(16) + mask(1) + inpaint(16)
            # Union models technically have control_in_dim=132 (4 controls), but we use single control
            'control_in_dim': 33,
            # Diffusers expects "control_layers" or "control_noise_refiner", not boolean
            'add_control_noise_refiner': "control_noise_refiner" if hints.get('has_control_noise_refiner', False) else None,
        }

        return cfg
