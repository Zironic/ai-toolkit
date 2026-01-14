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

                # Check for control_noise_refiner presence
                if any('control_noise_refiner' in k for k in keys):
                    info['has_control_noise_refiner'] = True

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

        # Build control_layers_places from detected count
        # Standard pattern: every other layer starting from 0 (0, 2, 4, ...)
        # For lite (5-layer control), this might be different - infer from actual indices
        if num_control_layers is not None:
            # Build places list: for N control layers, use [0, 2, 4, ..., 2*(N-1)]
            # But if n_layers is smaller (lite model), adjust accordingly
            control_layers_places = [i * 2 for i in range(num_control_layers) if i * 2 < n_layers]
            if not control_layers_places:
                # Fallback: just use available indices
                control_layers_places = list(range(min(num_control_layers, n_layers)))
        else:
            # Default: every other layer
            control_layers_places = [i for i in range(0, n_layers, 2)]

        # Minimal defaults that satisfy ZImageControlTransformer2DModel.__init__
        cfg = {
            # Base transformer parameters (match Z-Image defaults)
            'in_channels': 16,  # Z-Image uses 16 channels
            'dim': 3840,
            'n_layers': n_layers,
            'n_refiner_layers': 2,
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
            # control_in_dim=33 for single control: control_latent(16) + mask(1) + inpaint(16)
            # Union models technically have control_in_dim=132 (4 controls), but we use single control
            'control_in_dim': 33,
            'add_control_noise_refiner': hints.get('has_control_noise_refiner', False),
        }

        return cfg
