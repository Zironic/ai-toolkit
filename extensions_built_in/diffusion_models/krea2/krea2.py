"""Krea 2 (K2) for ai-toolkit.

Krea 2 is a single-stream MMDiT text-to-image model:
  - text encoder: Qwen3-VL-4B-Instruct (a stack of 12 hidden-state layers is fed
    in; ``src/text_encoder.py``),
  - autoencoder: the Qwen-Image VAE (f8, 16 latent channels, the same VAE the
    ``qwen_image`` arch uses),
  - denoiser: ``SingleStreamDiT`` (``src/mmdit.py``), which fuses the text layers
    with a small ``TextFusionTransformer`` and runs the packed [text | image]
    sequence through ``SingleStreamBlock`` layers.

Flow-matching convention matches ai-toolkit exactly (t=1 noise -> t=0 clean,
target = noise - clean), so ``get_noise_prediction`` does no time flip / negation.
"""

import contextlib
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from safetensors.torch import load_file, save_file

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import EntryNotFoundError
from diffusers import AutoencoderKLQwenImage
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)
from optimum.quanto import freeze, QTensor
from tqdm import tqdm

from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.accelerator import unwrap_model
from toolkit.metadata import get_meta_for_safetensors
from toolkit.util.quantize import (
    assign_quantized_state_dict,
    assign_quantized_state_dict_subset,
    get_qtype,
    prepare_quantized_state_dict_model,
    quantize,
    quantize_model,
    tensor_subclass_leaves,
)
from toolkit.memory_management import MemoryManager
from toolkit.memory_management import vram_budget
from toolkit.memory_management.arena_offload import (
    prepare_canonical_storage,
)
from toolkit.memory_management.runtime import get_memory_runtime
from toolkit.compile_cache import load_compile_cache, save_compile_cache

from .src.mmdit import (
    DoubleSharedModulation,
    SimpleModulation,
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from .src.noise_band_scheduler import Krea2NoiseBandScheduler
from .src.skc_injection import install_skc_projector_injection
from .src.text_encoder import encode_krea_prompt, SELECT_LAYERS
from .src.pipeline import Krea2Pipeline, pad_text_features, predict_velocity

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# The reference "single_mmdit_large_wide" architecture (oss_raw / oss_turbo share it).
KREA2_MMDIT_CONFIG = dict(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

# Krea 2's mu schedule is exponential time-shifting whose mu is linearly
# interpolated in image-token count between (256-res -> 0.5) and (1280-res ->
# 1.15) -- exactly what CustomFlowMatchEulerDiscreteScheduler's dynamic shifting
# does, so we mirror those endpoints here for the training timestep distribution.
#   x1 = (256  // (8*2))**2 = 256
#   x2 = (1280 // (8*2))**2 = 6400
scheduler_config = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "min_shift": 0.33,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}

# Defaults; both overridable via model.model_kwargs.
QWEN3_VL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_IMAGE_VAE_PATH = "Qwen/Qwen-Image"

HF_TOKEN = os.getenv("HF_TOKEN", None)


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.lower() not in ("", "0", "false", "no")


def _hf_local_files_only(model_config: Optional[ModelConfig] = None) -> bool:
    if model_config is not None:
        model_kwargs = getattr(model_config, "model_kwargs", {}) or {}
        if "local_files_only" in model_kwargs:
            return bool(model_kwargs["local_files_only"])
    return any(
        _truthy_env(name)
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    )


_SAFETENSORS_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}


def _read_safetensors_header(path: str) -> tuple[dict, int]:
    """Read the JSON header without mapping the weight payload."""
    with open(path, "rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise RuntimeError(f"[krea2] invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(handle.read(header_length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + header_length


def _read_safetensors_tensor(handle, data_start: int, spec: dict) -> torch.Tensor:
    """Read one tensor byte range through normal I/O, avoiding a whole-file mmap."""
    dtype_name = spec["dtype"]
    if dtype_name not in _SAFETENSORS_DTYPES:
        raise RuntimeError(f"[krea2] unsupported safetensors dtype {dtype_name!r}")
    begin, end = spec["data_offsets"]
    payload = bytearray(end - begin)
    handle.seek(data_start + begin)
    view = memoryview(payload)
    position = 0
    while position < len(payload):
        read = handle.readinto(view[position:])
        if not read:
            raise EOFError("[krea2] truncated safetensors tensor payload")
        position += read
    tensor = torch.frombuffer(payload, dtype=_SAFETENSORS_DTYPES[dtype_name])
    return tensor.reshape(spec["shape"])


def patch_qwen_vl_patch_embed(model):
    """Replace strided Qwen-VL Conv3d patch projection with equivalent GEMM."""
    patched = 0
    for module in model.modules():
        proj = getattr(module, "proj", None)
        if isinstance(proj, torch.nn.Conv3d) and tuple(proj.kernel_size) == tuple(
            proj.stride
        ):
            def fast_forward(hidden_states, _proj=proj):
                weight = _proj.weight.reshape(_proj.weight.shape[0], -1)
                inputs = hidden_states.view(-1, weight.shape[1]).to(weight.dtype)
                return F.linear(inputs, weight, _proj.bias)

            module.forward = fast_forward
            patched += 1
    return patched

def _resolve_mmdit_checkpoint_path(name_or_path: str, filename: Optional[str]) -> str:
    """Resolve the MMDiT weights to a local ``.safetensors`` file path (downloading from the hub
    if needed) without loading it. Returning the path lets the caller stream tensors in lazily
    instead of materializing the whole checkpoint (plus a cast copy) in RAM at once.

    ``name_or_path`` may be: a ``.safetensors`` file, a directory containing one
    (``filename`` or the lone ``.safetensors`` in it), or a hub repo id (the
    file ``filename`` is downloaded, defaulting to ``model.safetensors``).
    """
    if name_or_path.endswith(".safetensors") and os.path.isfile(name_or_path):
        return name_or_path

    if os.path.isdir(name_or_path):
        if filename is not None:
            return os.path.join(name_or_path, filename)
        candidates = [f for f in os.listdir(name_or_path) if f.endswith(".safetensors")]
        if len(candidates) == 1:
            return os.path.join(name_or_path, candidates[0])
        raise FileNotFoundError(
            f"Could not pick an MMDiT checkpoint in {name_or_path}: found "
            f"{candidates}. Set model.model_kwargs.checkpoint_filename."
        )

    # Treat as a hub repo id. When no filename is given, derive it from the repo
    # name's trailing segment (e.g. "krea/Krea-2-Raw" -> "raw.safetensors",
    # "krea/Krea-2-Turbo" -> "turbo.safetensors").
    fname = filename or (
        name_or_path.split("/")[-1].split("-")[-1].lower() + ".safetensors"
    )
    try:
        path = huggingface_hub.hf_hub_download(
            repo_id=name_or_path,
            filename=fname,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(),
        )
    except EntryNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find {fname!r} in hub repo {name_or_path!r}. Set "
            "model.model_kwargs.checkpoint_filename to the weight file name."
        ) from e
    return path


def _assign_tensor_by_name(module: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    """Place ``tensor`` at the dotted ``key`` in ``module``, replacing the (meta) param/buffer in
    place — the single-entry equivalent of ``load_state_dict(assign=True)``. Preserves the original
    parameter's ``requires_grad`` so streaming a checkpoint in matches the eager load exactly.
    """
    *parents, leaf = key.split(".")
    target = module
    for p in parents:
        target = getattr(target, p)
    if leaf in target._parameters:
        old = target._parameters[leaf]
        requires_grad = bool(old.requires_grad) if old is not None else False
        target._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
    elif leaf in target._buffers:
        target._buffers[leaf] = tensor
    else:
        raise KeyError(f"[krea2] no parameter/buffer named {key!r} in transformer")


def _module_by_name(module: torch.nn.Module, name: str) -> torch.nn.Module:
    target = module
    for part in name.split("."):
        target = getattr(target, part)
    return target


def _quantization_unit_for_key(key: str) -> str:
    """Return a bounded submodule that can be fully materialized and quantized alone."""
    parts = key.split(".")
    if parts[0] == "blocks":
        return ".".join(parts[:2])
    if parts[:2] in (["txtfusion", "layerwise_blocks"], ["txtfusion", "refiner_blocks"]):
        return ".".join(parts[:3])
    if parts[:2] == ["txtfusion", "projector"]:
        return "txtfusion.projector"
    return parts[0]


def _validate_checkpoint_keys(transformer, header: dict, checkpoint_path: str) -> list[str]:
    target_keys = list(transformer.state_dict().keys())
    file_keys = set(header)
    missing = [key for key in target_keys if key not in file_keys]
    unexpected = sorted(file_keys.difference(target_keys))
    if missing or unexpected:
        raise RuntimeError(
            f"[krea2] checkpoint key mismatch for {checkpoint_path}: "
            f"{len(missing)} missing ({missing[:5]}), "
            f"{len(unexpected)} unexpected ({unexpected[:5]})"
        )
    return target_keys


def _load_checkpoint_tensor(handle, data_start, header, key, expected_shape, dtype):
    tensor = _read_safetensors_tensor(handle, data_start, header[key])
    if tuple(tensor.shape) != tuple(expected_shape):
        raise RuntimeError(
            f"[krea2] shape mismatch for {key}: checkpoint {tuple(tensor.shape)}, "
            f"model {tuple(expected_shape)}"
        )
    if tensor.is_floating_point() and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def _arena_destination_key(state_key: str):
    """Map a Krea checkpoint leaf to the generic arena destination key."""
    parts = state_key.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
        return None
    tail = parts[-1]
    if tail in ("_data", "_scale") and len(parts) >= 5 and parts[-2] == "weight":
        role = "qdata" if tail == "_data" else "scale"
        linear_parts = parts[2:-2]
    elif tail in ("weight", "bias"):
        role = tail
        linear_parts = parts[2:-1]
    else:
        return None
    if not linear_parts:
        return None
    return (f"blocks.{parts[1]}", ".".join(linear_parts), role)


def _stream_checkpoint(transformer, checkpoint_path: str, dtype, *, canonical_build=None) -> None:
    header, data_start = _read_safetensors_header(checkpoint_path)
    target_state = transformer.state_dict()
    target_keys = _validate_checkpoint_keys(transformer, header, checkpoint_path)

    def populate(destinations):
        with open(checkpoint_path, "rb", buffering=0) as handle:
            for key in tqdm(target_keys, desc="Loading Krea 2 tensors"):
                tensor = _load_checkpoint_tensor(
                    handle, data_start, header, key, target_state[key].shape, dtype
                )
                destination_key = _arena_destination_key(key)
                if destination_key is not None and destination_key in destinations:
                    destinations[destination_key].copy_(tensor)
                else:
                    _assign_tensor_by_name(transformer, key, tensor)
                del tensor

    if canonical_build is None:
        populate({})
    else:
        canonical_build.populate(populate)


def _populate_canonical_build_from_model(canonical_build) -> None:
    """Feed a loaded quantized cache into final typed arena destinations."""
    def populate(destinations):
        for destination_key, source in canonical_build.model_source_leaves():
            destinations[destination_key].copy_(source)

    canonical_build.populate(populate)


def _stream_and_quantize_checkpoint(
    base_model,
    transformer,
    checkpoint_path,
    dtype,
    *,
    canonical_build=None,
    adapter=None,
) -> None:
    """Materialize, quantize, and release one bounded submodule at a time."""
    header, data_start = _read_safetensors_header(checkpoint_path)
    target_state = transformer.state_dict()
    target_keys = _validate_checkpoint_keys(transformer, header, checkpoint_path)
    units = {}
    for key in target_keys:
        units.setdefault(_quantization_unit_for_key(key), []).append(key)

    quantization_type = get_qtype(base_model.model_config.qtype)
    unit_items = list(units.items())
    base_model.print_and_status_update(f"  - streaming and quantizing {len(unit_items)} transformer units")
    with open(checkpoint_path, "rb", buffering=0) as handle:
        for index, (unit_name, keys) in enumerate(unit_items, start=1):
            if index == 1 or index == len(unit_items) or index % 5 == 0:
                base_model.print_and_status_update(f"    quantizing unit {index}/{len(unit_items)}: {unit_name}")
                flush(garbage_collect=False)
            for key in keys:
                tensor = _load_checkpoint_tensor(
                    handle, data_start, header, key, target_state[key].shape, dtype
                )
                _assign_tensor_by_name(transformer, key, tensor)
                del tensor
            unit = _module_by_name(transformer, unit_name)
            unit.to(base_model.device_torch, dtype=dtype)
            quantize(unit, weights=quantization_type)
            freeze(unit)
            unit.requires_grad_(False)
            unit.to("cpu")
            if canonical_build is not None and unit_name.startswith("blocks."):
                block_index = int(unit_name.split(".")[1])
                block_key = adapter.block_key(transformer, block_index)
                canonical_build.add_block(block_key, adapter.leaf_entries(unit))
                canonical_build.populate_block_from_model(block_key)
                canonical_build.release_block_sources_to_meta(block_key)
    if canonical_build is not None:
        canonical_build.finish_population()
    base_model.print_and_status_update("  - finished streaming and quantizing transformer units")
    flush(garbage_collect=False)



def _sampling_shape_key(gen_config) -> tuple:
    """Bucket key for a sampling working-set peak: what actually moves the peak."""
    reference_count = sum(
        value is not None
        for value in (
            getattr(gen_config, "ctrl_img", None),
            getattr(gen_config, "ctrl_img_1", None),
            getattr(gen_config, "ctrl_img_2", None),
            getattr(gen_config, "ctrl_img_3", None),
        )
    )
    return (
        int(gen_config.width),
        int(gen_config.height),
        bool(getattr(gen_config, "batch_cfg", False)),
        int(reference_count),
    )


def _compile_cache_key(base_model) -> str:
    """Identity for the torch.compile mega-cache: resolved checkpoint + quant,
    not the raw `name_or_path` (which may be an unresolved HF repo id or a
    local path -- either way, not itself a stable model identity).

    No shape/resolution tag needed: sampling compiles are static-shape
    (`dynamic=False`) under the `eager_then_compile` stance, so torch's own
    guard system (not us) decides whether a given call reuses or misses the
    cached graph -- that's exactly the "safe miss" property the mega-cache
    already relies on.
    """
    checkpoint_path = getattr(base_model, "_resolved_checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_path = base_model.model_config.name_or_path
    identity = {
        "checkpoint_path": os.path.abspath(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path,
        "qtype": str(base_model.model_config.qtype),
        "torch_version": torch.__version__,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def _train_compile_cache_key(base_model) -> str:
    """Identity for the TRAIN-side block-kernel mega-cache.

    Unlike sampling, the immutable runtime's train compile is NOT always
    static-shape: compile_dynamic/compile_dynamic_hints choose whether (and
    how) the block kernel treats input shapes as dynamic, so two runs with
    different settings can compile structurally different graphs for the
    same checkpoint. Folding those settings into the key keeps them from
    silently reusing each other's cached artifacts (found via a settings
    sweep in scripts/smoke_krea2_train_cuda.py loading a stale cache after
    a --compile-dynamic change).
    """
    config = base_model.model_config
    compile_identity = {
        "compile_dynamic": config.compile_dynamic,
        "compile_dynamic_hints": tuple(config.compile_dynamic_hints or ()),
    }
    compile_tag = hashlib.sha256(
        json.dumps(compile_identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    return f"{_compile_cache_key(base_model)}_immutable_train_{compile_tag}"


def _quantized_transformer_cache_info(base_model, checkpoint_path: str, dtype, config: SingleMMDiTConfig):
    model_kwargs = base_model.model_config.model_kwargs
    cache_enabled = bool(model_kwargs.get("quantized_transformer_cache", True))
    if not cache_enabled:
        return None, None
    cache_root = Path(
        model_kwargs.get("quantized_transformer_cache_dir")
        or os.getenv("AI_TOOLKIT_KREA2_QUANT_CACHE")
        or (Path(HF_HUB_CACHE) / "ai-toolkit" / "krea2_quantized_transformers")
    )
    checkpoint_stat = os.stat(checkpoint_path)
    metadata = {
        "schema": "krea2_quantized_transformer_cache.v1",
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_size": int(checkpoint_stat.st_size),
        "checkpoint_mtime_ns": int(checkpoint_stat.st_mtime_ns),
        "dtype": str(dtype).replace("torch.", ""),
        "qtype": str(base_model.model_config.qtype),
        "mmdit_config": config.__dict__,
        "torch_version": torch.__version__,
    }
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
    return cache_root / f"krea2_transformer_{digest}.pt", metadata


def _try_load_quantized_transformer_cache(
    base_model,
    transformer,
    cache_path: Path,
    metadata: dict,
    *,
    canonical_build=None,
    canonical_adapter=None,
    canonical_device=None,
) -> tuple[bool, object | None]:
    created_build = False
    if cache_path is None or not cache_path.exists():
        return False, canonical_build
    try:
        base_model.print_and_status_update(f"  - loading cached quantized transformer state from {cache_path}")
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            base_model.print_and_status_update("  - cached quantized transformer metadata mismatch; ignoring")
            return False, canonical_build
        state_dict = payload["state_dict"]
        if canonical_build is None and canonical_adapter is not None:
            prepare_quantized_state_dict_model(
                transformer,
                state_dict,
                base_model.model_config.qtype,
            )
            transformer.requires_grad_(False)
            canonical_build = prepare_canonical_storage(
                transformer,
                canonical_adapter,
                device=canonical_device,
            )
            created_build = True
        if canonical_build is None:
            assign_quantized_state_dict(
                transformer,
                state_dict,
                base_model.model_config.qtype,
            )
        else:
            destinations = canonical_build.destinations
            canonical_values = {}
            canonical_keys = set()
            for key, value in state_dict.items():
                destination_key = _arena_destination_key(key)
                if (
                    destination_key not in destinations
                    and destination_key is not None
                    and destination_key[2] == "weight"
                ):
                    qdata_key = (*destination_key[:2], "qdata")
                    if qdata_key in destinations:
                        destination_key = qdata_key
                if destination_key not in destinations:
                    continue
                leaves = tensor_subclass_leaves(value)
                if (
                    destination_key[2] == "qdata"
                    and len(leaves) == 2
                    and (*destination_key[:2], "scale") in destinations
                ):
                    canonical_values[destination_key] = leaves[0]
                    canonical_values[(*destination_key[:2], "scale")] = leaves[1]
                else:
                    canonical_values[destination_key] = value
                canonical_keys.add(key)
            required = set(destinations)
            provided = set(canonical_values)
            if provided != required:
                raise RuntimeError(
                    "cached arena payload mismatch: "
                    f"missing={sorted(required - provided)[:5]} "
                    f"unexpected={sorted(provided - required)[:5]}"
                )
            assign_quantized_state_dict_subset(
                transformer,
                state_dict,
                base_model.model_config.qtype,
                excluded_keys=canonical_keys,
            )

            def populate(final_destinations):
                for destination_key, value in canonical_values.items():
                    final_destinations[destination_key].copy_(value)

            canonical_build.populate(populate)
            transformer.requires_grad_(False)
        from toolkit.dequantize import patch_dequantization_on_save
        patch_dequantization_on_save(transformer)
        return True, canonical_build
    except Exception as error:
        if created_build and canonical_build is not None:
            canonical_build.rollback()
        base_model.print_and_status_update(f"  - failed to load cached quantized transformer; rebuilding ({error})")
        return False, None


def _save_quantized_transformer_cache(base_model, transformer, cache_path: Path, metadata: dict) -> None:
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        base_model.print_and_status_update(f"  - saving cached quantized transformer state to {cache_path}")
        flush(garbage_collect=False)
        torch.save({"metadata": metadata, "state_dict": transformer.state_dict()}, str(tmp_path))
        os.replace(tmp_path, cache_path)
    except Exception as error:
        base_model.print_and_status_update(f"  - failed to save cached quantized transformer (ignored): {error}")
    finally:
        from toolkit.dequantize import patch_dequantization_on_save
        patch_dequantization_on_save(transformer)


class Krea2Model(BaseModel):
    arch = "krea2"

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["SingleStreamDiT"]

        self.patch_size = KREA2_MMDIT_CONFIG["patch"]
        self.vae_scale_factor = 8  # Qwen-Image VAE is f8
        # Prompts are unlimited by default. Strict mode rejects prompts over the
        # configured threshold during text-embedding setup instead of truncating.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 512)
        )
        self.prompt_overflow_policy = str(
            self.model_config.model_kwargs.get(
                "prompt_overflow_policy", "unlimited"
            )
        ).lower()
        if self.prompt_overflow_policy not in ("unlimited", "error"):
            raise ValueError(
                "model.model_kwargs.prompt_overflow_policy must be "
                "'unlimited' or 'error'"
            )
        if self.max_text_length < 1:
            raise ValueError("model.model_kwargs.max_text_length must be at least 1")
        # Qwen2TokenizerFast used to tokenize the assistant suffix (matches the
        # reference's separate processor pass).
        self.processor = None
        # Qwen3-VL AutoProcessor for encoding reference images into the prompt.
        self.vl_processor = None
        self.use_old_lokr_format = False
        self._transformer_quantized_during_load = False

        # Optional reference-image (edit) conditioning, enabled with
        # model_kwargs.edit = true. Control images feed the model in two places:
        # through the Qwen3-VL encoder alongside the prompt (edit-plus style, so
        # the text embeddings see them) and as clean VAE latents appended to the
        # image sequence at t=0 (ComfyUI Kontext "index_timestep_zero"). Runs in
        # ComfyUI with the ComfyUI-Krea2-Ostris-Edit custom nodes. With edit off
        # (the default) all of it is skipped and this is the plain T2I model.
        self.is_edit = bool(self.model_config.model_kwargs.get("edit", False))
        self.encode_control_in_text_embeddings = self.is_edit
        self.has_multiple_control_images = self.is_edit
        # Reference images keep their own aspect/size (not resized to the target).
        self.use_raw_control_images = self.is_edit
        # model_kwargs.kv_cache = true: train with an asymmetric attention mask
        # where the clean reference tokens attend only to each other (never to
        # text / noisy tokens). Their hidden states then depend only on the
        # refs + t=0 modulation, so at inference their per-layer K/V can be
        # computed once and reused across all denoising steps
        # (OminiControl2-style conditioning feature reuse). Off by default:
        # the base model was trained fully bidirectional, so a LoRA must be
        # trained with kv_cache enabled for kv-cached inference (the ComfyUI
        # node / hub pipeline kv_cache toggles) to work properly.
        self.kv_cache = bool(self.model_config.model_kwargs.get("kv_cache", False))

    @property
    def text_embedding_space_version(self):
        # v2 invalidates embeddings created by the old silent 512-token truncation.
        if self.prompt_overflow_policy == "error":
            return f"krea2-v2-error-{self.max_text_length}"
        return "krea2-v2-unlimited"

    @staticmethod
    def get_train_scheduler(model_config: Optional[ModelConfig] = None):
        model_kwargs = model_config.model_kwargs if model_config is not None else {}
        # Compresses the training-timestep ladder into a bounded noise-fraction
        # band, e.g. model.model_kwargs.noise_band_max: 0.9 excludes the top 10%
        # (near-total-noise) tail from training instead of merely biasing away
        # from it. Defaults to [0, 1] (disabled / behaviour-preserving). Requires
        # train.timestep_type: shift when a band is set (see noise_band_scheduler.py).
        noise_band_min = float(model_kwargs.get("noise_band_min", 0.0))
        noise_band_max = float(model_kwargs.get("noise_band_max", 1.0))
        return Krea2NoiseBandScheduler(
            noise_band_min=noise_band_min,
            noise_band_max=noise_band_max,
            **scheduler_config,
        )

    def _install_skc_injection(self, transformer):
        """Attach a fixed SKC projector perturbation if configured.

        Reads model.model_kwargs.skc_lora_path / skc_strength. The vector is a
        frozen teacher-side perturbation on txtfusion.projector, present in every
        forward (training and sampling) -- distinct from assistant_lora_path,
        which is merged for training and INVERTED at inference. strength is the
        raw multiplier on the extracted vector: for skc3vo (scale-1, no alpha)
        this equals its ComfyUI strength 1:1 (no 2.5x -- that factor is z0-only).
        """
        self._skc_injection_handle = None
        if transformer is None:
            return
        mk = self.model_config.model_kwargs
        skc_lora_path = mk.get("skc_lora_path")
        if skc_lora_path is None:
            return
        skc_strength = float(mk.get("skc_strength", 0.0))
        if skc_strength == 0.0:
            self.print_and_status_update(
                "skc_lora_path set but skc_strength=0; SKC injection disabled"
            )
            return
        source, handle = install_skc_projector_injection(
            transformer, skc_lora_path, skc_strength
        )
        self._skc_injection_handle = handle
        self.print_and_status_update(
            f"  - SKC projector injection active (strength={skc_strength}, "
            f"from {source}); present in training and sampling forwards"
        )

    def get_bucket_divisibility(self):
        # 8 for the VAE downsample, 2 for the patch size.
        return self.vae_scale_factor * self.patch_size

    def estimate_sampling_working_reserve_bytes(self, gen_configs):
        """Shape-aware cold-start hint for MemoryManager.inference_resident.

        Sized from the LARGEST pending sample so the residency plan streams
        enough blocks up front -- a mid-denoise OOM demote republishes residency
        under an active execution, so planning for the worst sample beats
        reacting per image. Returns None
        when there is nothing to estimate; the manager then keeps its flat
        cold-start default. A learned measured reserve replaces the estimate
        after the first sample either way.
        """
        gen_configs = [cfg for cfg in (gen_configs or []) if cfg is not None]
        if not gen_configs:
            return None
        token_div = self.vae_scale_factor * self.patch_size
        image_tokens = max(
            (max(1, int(cfg.width)) // token_div)
            * (max(1, int(cfg.height)) // token_div)
            for cfg in gen_configs
        )
        batch_cfg = any(getattr(cfg, "batch_cfg", False) for cfg in gen_configs)
        return vram_budget.estimate_sampling_working_reserve_bytes(
            image_tokens,
            self.max_text_length,
            batch_cfg=batch_cfg,
            fp8_native=bool(
                getattr(self.model_config, "layer_offloading_fp8_sampling", False)
            ),
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_transformer(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading transformer (SingleStreamDiT)")

        from toolkit.sdpa_patch import set_gqa_backend_mode

        gqa_backend = self.model_config.model_kwargs.get("sdpa_gqa_backend", "auto")
        set_gqa_backend_mode(gqa_backend)
        self.print_and_status_update(f"  - SDPA GQA backend: {gqa_backend}")

        mmdit_kwargs = dict(KREA2_MMDIT_CONFIG)
        mmdit_kwargs.update(self.model_config.model_kwargs.get("mmdit_config", {}))
        config = SingleMMDiTConfig(**mmdit_kwargs)

        self.print_and_status_update("  - fetching transformer weights")
        checkpoint_path = _resolve_mmdit_checkpoint_path(
            self.model_config.name_or_path,
            self.model_config.model_kwargs.get("checkpoint_filename", None),
        )
        self._resolved_checkpoint_path = checkpoint_path

        stream_quantized = (
            self.model_config.quantize
            and self.model_config.accuracy_recovery_adapter is None
            and self.model_config.assistant_lora_path is None
        )
        arena_requested = bool(
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_smart
        )
        direct_ranged_population = bool(
            arena_requested
            and not self.model_config.quantize
            and self.model_config.assistant_lora_path is None
        )
        cache_path, cache_metadata = _quantized_transformer_cache_info(self, checkpoint_path, dtype, config)
        # Build on meta, then materialize either from the quantized cache or the checkpoint.
        with torch.device("meta"):
            transformer = SingleStreamDiT(config)

        canonical_build = None
        try:
            cache_adapter = None
            if arena_requested:
                from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter

                cache_adapter = SingleStreamMMDiTAdapter()
            cache_loaded, canonical_build = (
                _try_load_quantized_transformer_cache(
                    self,
                    transformer,
                    cache_path,
                    cache_metadata,
                    canonical_adapter=cache_adapter,
                    canonical_device=self.device_torch,
                )
                if stream_quantized
                else (False, None)
            )
            if cache_loaded:
                self._transformer_quantized_during_load = True
            else:
                if stream_quantized and cache_path is not None and cache_path.exists():
                    # Cache validation can reconstruct wrapper metadata before a
                    # later failure. Start the ranged fallback from a pristine
                    # meta model so no partial cache state survives.
                    with torch.device("meta"):
                        transformer = SingleStreamDiT(config)
                    canonical_build = None
                self.print_and_status_update("  - loading transformer through ranged disk reads")
                if stream_quantized:
                    adapter = None
                    if arena_requested:
                        from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter

                        adapter = SingleStreamMMDiTAdapter()
                        canonical_build = prepare_canonical_storage(
                            transformer,
                            adapter,
                            device=self.device_torch,
                            defer_blocks=True,
                        )
                    _stream_and_quantize_checkpoint(
                        self,
                        transformer,
                        checkpoint_path,
                        dtype,
                        canonical_build=canonical_build,
                        adapter=adapter,
                    )
                    if canonical_build is None:
                        _save_quantized_transformer_cache(
                            self, transformer, cache_path, cache_metadata
                        )
                    else:
                        self._pending_quantized_transformer_cache = (
                            cache_path,
                            cache_metadata,
                        )
                    self._transformer_quantized_during_load = True
                else:
                    if direct_ranged_population:
                        from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter

                        transformer.requires_grad_(False)
                        canonical_build = prepare_canonical_storage(
                            transformer,
                            SingleStreamMMDiTAdapter(),
                            device=self.device_torch,
                        )
                    _stream_checkpoint(
                        transformer,
                        checkpoint_path,
                        dtype,
                        canonical_build=canonical_build,
                    )
        except Exception:
            if canonical_build is not None:
                canonical_build.rollback()
            raise

        self._prepared_canonical_build = canonical_build
        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        te_path = self.model_config.model_kwargs.get("text_encoder_path", QWEN3_VL_PATH)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            te_path,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        processor = Qwen2TokenizerFast.from_pretrained(
            te_path,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        vl_processor = None
        if self.is_edit:
            # Edit mode: reference images are encoded into the text embeddings,
            # so the vision tower stays. Swap its Conv3d patch_embed for an
            # equivalent GEMM (bf16 Conv3d has no fast cuDNN kernel).
            vl_processor = AutoProcessor.from_pretrained(te_path, token=HF_TOKEN)
            patch_qwen_vl_patch_embed(text_encoder)
        else:
            # We only ever encode text, so the vision tower is dead weight -- drop it to
            # free VRAM and skip loading its (bf16-slow) Conv3d patch_embed onto the GPU.
            if getattr(text_encoder.model, "visual", None) is not None:
                text_encoder.model.visual = None
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, processor, vl_processor, text_encoder

    def _load_vae(self):
        vae_path = self.model_config.model_kwargs.get("vae_path", QWEN_IMAGE_VAE_PATH)
        self.print_and_status_update(f"Loading Qwen-Image VAE from {vae_path}")
        vae = AutoencoderKLQwenImage.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=self.vae_torch_dtype,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        vae.eval()
        vae.requires_grad_(False)
        return vae

    def load_training_adapter(self, transformer: SingleStreamDiT):
        self.print_and_status_update("Loading assistant LoRA")
        lora_path = self.model_config.assistant_lora_path
        if not os.path.exists(lora_path):
            # assume it is a hub path
            lora_splits = lora_path.split("/")
            if len(lora_splits) != 3:
                raise ValueError(
                    f"Assistant LoRA path {lora_path} is not a valid local path or hub path."
                )
            repo_id = "/".join(lora_splits[:2])
            filename = lora_splits[2]
            try:
                lora_path = huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    token=HF_TOKEN,
                )
                # upgrade path to the local download
                self.model_config.assistant_lora_path = lora_path
            except Exception as e:
                raise ValueError(
                    f"Failed to download assistant LoRA from {lora_path}: {e}"
                )
        # load the adapter and merge it in. We will inference with a -1.0 multiplier so the adapter effects only work during training.
        lora_state_dict = load_file(lora_path)
        # detect the LoRA rank from the first down-projection weight.
        dim_key = next(k for k in lora_state_dict if k.endswith("lora_A.weight"))
        dim = int(lora_state_dict[dim_key].shape[0])

        new_sd = {}
        for key, value in lora_state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        lora_state_dict = new_sd

        network_config = {
            "type": "lora",
            "linear": dim,
            "linear_alpha": dim,
            "transformer_only": True,
        }

        network_config = NetworkConfig(**network_config)
        LoRASpecialNetwork.LORA_PREFIX_UNET = "lora_transformer"
        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=transformer,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=True,
            target_lin_modules=self.target_lora_modules,
            is_assistant_adapter=True,
            base_model=self,
            is_ara=True,
        )
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        self.print_and_status_update("Merging in assistant LoRA")
        network.force_to(self.device_torch, dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)

        network.merge_in(merge_weight=1.0)

        # mark it as not merged so inference ignores it.
        network.is_merged_in = False

        # add the assistant so sampler will activate it while sampling
        self.assistant_lora: LoRASpecialNetwork = network

        # deactivate lora during training
        self.assistant_lora.multiplier = -1.0
        self.assistant_lora.is_active = False

        # tell the model to invert assistant on inference since we want remove lora effects
        self.invert_assistant_lora = True

    def get_quantization_exclude_modules(self):
        # sensitive modules kept in full precision (fnmatch patterns on module
        # names within SingleStreamDiT):
        #   first             - patchified latent input projection
        #   tmlp* / tproj*    - timestep embedder + modulation projection; feed
        #                       every block's DoubleSharedModulation and LastLayer
        #   txtmlp*           - text feature -> model width projection
        #   txtfusion.projector - tiny (num_txt_layers -> 1) encoder-layer mixer
        #   last*             - final norm/modulated output projection
        return [
            "first",
            "tmlp*",
            "tproj*",
            "txtmlp*",
            "txtfusion.projector",
            "last*",
        ]

    def _attach_immutable_training_memory(self, transformer, ignore_modules):
        """Select the arena offload backend for the transformer.

        Sequencing is strict and owned by the caller: load/quantize -> merge
        assistant LoRA -> freeze -> prepare arena. Singleton (non-canonical)
        modules stay resident; only the canonical blocks stream through the
        arena. LoRA/optimizer construction and runtime finalization happen later
        in the trainer, after all frozen base Parameter identities are final.
        """
        from toolkit.memory_management.adapters import SingleStreamMMDiTAdapter
        from toolkit.memory_management.arena_offload import (
            ArenaOffloadConfig,
            prepare_arena_offload,
        )

        canonical_build = getattr(self, "_prepared_canonical_build", None)
        try:
            return prepare_arena_offload(
                transformer,
                device=self.device_torch,
                adapter=SingleStreamMMDiTAdapter(),
                config=ArenaOffloadConfig.from_model_config(self.model_config),
                ignore_modules=ignore_modules,
                canonical_build=canonical_build,
            )
        finally:
            self._prepared_canonical_build = None

    def cleanup_memory_runtime_preparation(self):
        build = getattr(self, "_prepared_canonical_build", None)
        self._prepared_canonical_build = None
        if build is not None:
            build.rollback()

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Krea 2 model")

        transformer = None
        if self.te_only:
            # TE cache worker: only the text encoder is needed, skip the transformer.
            self.print_and_status_update("Skipping transformer (te_only load)")
        else:
            transformer = self._load_transformer()

            # Load and merge assistant LoRA before quantization; the adapter expects
            # ordinary module weights.
            if self.model_config.assistant_lora_path is not None:
                self.load_training_adapter(transformer)

            if self.model_config.quantize and not self._transformer_quantized_during_load:
                self.print_and_status_update("Quantizing transformer")
                quantize_model(self, transformer)
                flush()

            arena_runtime = None
            if (
                self.model_config.layer_offloading
                and (
                    self.model_config.layer_offloading_smart
                    or self.model_config.layer_offloading_transformer_percent > 0
                )
            ):
                ignore_modules = [
                    module
                    for module in transformer.modules()
                    if isinstance(module, (SimpleModulation, DoubleSharedModulation))
                ]
                if self.model_config.layer_offloading_smart:
                    # The compile-neutral immutable runtime is the SOLE smart
                    # memory/compile backend for Krea2. It streams the repeated
                    # blocks through the canonical arena and keeps every
                    # singleton module resident. It also checkpoints each block
                    # in its own train trunk, so the model's gradient
                    # checkpointing stays off here.
                    arena_runtime = self._attach_immutable_training_memory(
                        transformer, ignore_modules
                    )
                else:
                    MemoryManager.attach(
                        transformer,
                        self.device_torch,
                        offload_percent=self.model_config.layer_offloading_transformer_percent,
                        ignore_modules=ignore_modules,
                    )

            if arena_runtime is not None:
                pending_cache = getattr(
                    self, "_pending_quantized_transformer_cache", None
                )
                if pending_cache is not None:
                    _save_quantized_transformer_cache(
                        self, transformer, pending_cache[0], pending_cache[1]
                    )
                    self._pending_quantized_transformer_cache = None
                arena_runtime.place_permanent_modules(self.device_torch, dtype)
            elif self.model_config.low_vram:
                self.print_and_status_update("Moving transformer to CPU")
                transformer.to("cpu")
            elif self.model_config.quantize:
                # Move device-only. Passing dtype to .to() on a quantized model dequantizes the
                # ENTIRE model to fp32 in a single allocation (~52 GB for Krea2) and OOMs. The
                # weights are already at the correct (quantized) precision from quantize_model.
                transformer.to(self.device_torch)
            else:
                transformer.to(self.device_torch, dtype=dtype)
            flush()

        if self.skip_te:
            # Trainer running off a pre-built embedding cache: never load the heavy Qwen3-VL
            # text encoder. Keep the cheap tokenizers so the model stays well-formed.
            from toolkit.unloader import FakeTextEncoder
            self.print_and_status_update("Skipping text encoder (skip_te load)")
            te_path = self.model_config.model_kwargs.get("text_encoder_path", QWEN3_VL_PATH)
            tokenizer = AutoTokenizer.from_pretrained(
                te_path,
                token=HF_TOKEN,
                local_files_only=_hf_local_files_only(self.model_config),
            )
            processor = Qwen2TokenizerFast.from_pretrained(
                te_path,
                token=HF_TOKEN,
                local_files_only=_hf_local_files_only(self.model_config),
            )
            vl_processor = None
            text_encoder = FakeTextEncoder(device=self.device_torch, dtype=dtype)
        else:
            tokenizer, processor, vl_processor, text_encoder = self._load_text_encoder()
            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing text encoder")
                text_encoder.to(self.device_torch)
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                flush()
            arena_runtime = None
            if (
                self.model_config.layer_offloading
                and self.model_config.layer_offloading_text_encoder_percent > 0
            ):
                MemoryManager.attach(
                    text_encoder,
                    self.device_torch,
                    offload_percent=self.model_config.layer_offloading_text_encoder_percent,
                )

            if self.model_config.low_vram:
                self.print_and_status_update("Moving text encoder to CPU")
                text_encoder.to("cpu")
            else:
                text_encoder.to(self.device_torch)
            flush()

        vae = None
        if self.te_only:
            self.print_and_status_update("Skipping VAE (te_only load)")
        else:
            vae = self._load_vae()
            vae.to(self.vae_device_torch, dtype=self.vae_torch_dtype)

        self.noise_scheduler = Krea2Model.get_train_scheduler(self.model_config)

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.vl_processor = vl_processor
        self.model = transformer
        self._install_skc_injection(transformer)
        self.pipeline = Krea2Pipeline(self)
        self.print_and_status_update("Model Loaded")

    # ------------------------------------------------------------------
    # Generation (training previews)
    # ------------------------------------------------------------------
    def get_generation_pipeline(self):
        return Krea2Pipeline(self)

    def generate_single_image(
        self,
        pipeline: Krea2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: AdvancedPromptEmbeds,
        unconditional_embeds: AdvancedPromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        extra = extra or {}
        skip_sampling_guard = bool(extra.get("skip_sampling_guard", False))
        arena_runtime = get_memory_runtime(self.model)
        if arena_runtime is not None:
            arena_runtime.place_permanent_modules(self.device_torch, self.torch_dtype)
        elif self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        # Reference images become clean t=0 VAE tokens. The Qwen3-VL side has
        # already incorporated the same images into the prompt embeddings.
        ctrl_paths = []
        if self.is_edit:
            if gen_config.ctrl_img is not None:
                ctrl_paths.append(gen_config.ctrl_img)
            elif gen_config.ctrl_img_1 is not None:
                ctrl_paths.append(gen_config.ctrl_img_1)
            if gen_config.ctrl_img_2 is not None:
                ctrl_paths.append(gen_config.ctrl_img_2)
            if gen_config.ctrl_img_3 is not None:
                ctrl_paths.append(gen_config.ctrl_img_3)

        ref_latents = None
        if ctrl_paths:
            ctrl_tensors = [
                to_tensor(Image.open(path).convert("RGB")) for path in ctrl_paths
            ]
            target_pixels = gen_config.width * gen_config.height
            ref_latents = [
                self._encode_ref_latents(ctrl_tensors, target_pixels=target_pixels)
            ]

        # CFG is zero-normalized for Krea 2.
        guidance = max(0.0, gen_config.guidance_scale - 1.0)
        compile_cache_dir = getattr(self.model_config, 'compile_cache_dir', None)
        compile_cache_key = _compile_cache_key(self)
        if (
            self.model_config.compile_sample
            and compile_cache_dir
            and not getattr(self, '_compile_cache_load_attempted', False)
        ):
            self._compile_cache_load_attempted = True
            if load_compile_cache(compile_cache_dir, compile_cache_key):
                self.print_and_status_update(
                    f"Loaded torch.compile cache from {compile_cache_dir}"
                )

        # The immutable runtime owns the whole sampling trunk via its permanent
        # SAMPLE program (_blocks_trunk routes to it first) and compiles the
        # block kernels itself. Its residency plan was activated at the sampling
        # boundary by the trainer.

        # Sampling compiles are static-shape (dynamic=False); running the
        # call under eager_then_compile defers each compile to the second
        # call with a given shape instead of wasting one on the very first
        # -- so "did a new compile happen" must be re-checked on every
        # call, not just the first one this process.
        frames_before = None
        if compile_cache_dir and self.model_config.compile_sample:
            frames_before = torch._dynamo.utils.counters["frames"].get("total", 0)

        compile_stance = (
            torch.compiler.set_stance("eager_then_compile")
            if self.model_config.compile_sample
            else contextlib.nullcontext()
        )
        immutable_context = (
            arena_runtime.sampling_image(
                shape_key=_sampling_shape_key(gen_config),
                cold_working_bytes=int(
                    self.estimate_sampling_working_reserve_bytes([gen_config])
                    or 3 * (1024 ** 3)
                ),
            )
            if arena_runtime is not None
            else (
                MemoryManager.sampling_image(self.model)
                if not skip_sampling_guard
                else contextlib.nullcontext()
            )
        )
        with immutable_context:
            with compile_stance:
                img = pipeline(
                    conditional_embeds=conditional_embeds,
                    unconditional_embeds=unconditional_embeds,
                    height=gen_config.height,
                    width=gen_config.width,
                    num_inference_steps=gen_config.num_inference_steps,
                    guidance_scale=guidance,
                    latents=gen_config.latents,
                    generator=generator,
                    batch_cfg=getattr(gen_config, "batch_cfg", False),
                    ref_latents=ref_latents,
                )[0]
        if frames_before is not None:
            frames_after = torch._dynamo.utils.counters["frames"].get("total", 0)
            if frames_after > frames_before and save_compile_cache(
                compile_cache_dir, compile_cache_key
            ):
                self.print_and_status_update(
                    f"Saved torch.compile cache to {compile_cache_dir}"
                )
        return img

    # ------------------------------------------------------------------
    # Reference-image helpers
    # ------------------------------------------------------------------
    def _ref_target_pixels(self, target_pixels: Optional[int]) -> int:
        """Pixel budget each reference image is resized to fit within.

        - default: ``control_image_max_pixels`` model_kwarg (1 MP) -- a hard cap
          so raw, full-size control images don't blow up the token count / VRAM.
        - ``match_target_res`` model_kwarg: use the target generation area instead.
        """
        max_pixels = int(
            self.model_config.model_kwargs.get("control_image_max_pixels", 1024 * 1024)
        )
        if (
            self.model_config.model_kwargs.get("match_target_res", False)
            and target_pixels
        ):
            return int(target_pixels)
        return max_pixels

    def _encode_ref_latents(
        self, control_tensors, target_pixels: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Encode ``[0, 1]`` reference image tensors to VAE latents.

        Returns a list of ``(16, h, w)`` latents (one per reference image). Each
        control image is resized so its area fits within the pixel budget (see
        ``_ref_target_pixels``) -- preserving aspect ratio -- then snapped so the
        latent grid is divisible by the patch size. ``control_tensors`` is a list
        of ``(C, H, W)`` or ``(1, C, H, W)`` tensors in ``[0, 1]``.
        """
        sc = self.get_bucket_divisibility()  # 16: VAE(8) * patch(2)
        budget = self._ref_target_pixels(target_pixels)
        match = self.model_config.model_kwargs.get("match_target_res", False)

        latents = []
        for img in control_tensors:
            if img.dim() == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device_torch, dtype=self.torch_dtype)

            h, w = img.shape[2], img.shape[3]
            # match_target_res: scale area *to* the budget; otherwise only scale
            # *down* when the image is larger than the budget.
            area = h * w
            if match or area > budget:
                ratio = h / w
                new_h = math.sqrt(budget * ratio)
                new_w = new_h / ratio
            else:
                new_h, new_w = float(h), float(w)

            # snap to a multiple of the bucket divisibility so the VAE latent grid
            # is patchifiable (the transformer rearranges 2x2 latent patches).
            new_h = max(sc, int(round(new_h / sc)) * sc)
            new_w = max(sc, int(round(new_w / sc)) * sc)
            if (new_h, new_w) != (h, w):
                img = F.interpolate(img, size=(new_h, new_w), mode="bilinear")

            # encode_images expects [-1, 1]; control tensors arrive in [0, 1].
            latent = self.encode_images(
                img * 2 - 1, device=self.device_torch, dtype=self.torch_dtype
            )
            latents.append(latent[0])  # drop batch dim -> (16, h, w)
        return latents

    def _batch_ref_latents_from_batch(
        self,
        batch: "DataLoaderBatchDTO",
        batch_size: int,
        target_pixels: Optional[int] = None,
    ) -> Optional[List[List[torch.Tensor]]]:
        """Build predict_velocity's ``ref_latents`` from a train batch."""
        control_list = batch.control_tensor_list
        if control_list is None and batch.control_tensor is not None:
            control_list = [batch.control_tensor[b : b + 1] for b in range(batch_size)]
        if control_list is None:
            return None
        if len(control_list) != batch_size:
            raise ValueError("Control tensor list length does not match batch size")
        return [
            self._encode_ref_latents(controls, target_pixels=target_pixels)
            for controls in control_list
        ]

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 16, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        arena_runtime = get_memory_runtime(self.model)
        if arena_runtime is not None:
            arena_runtime.place_permanent_modules(self.device_torch, self.torch_dtype)
        elif self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # Clean reference latents from the batch's control images (if any); they
        # ride along in the sequence at t=0 and are never noised.
        ref_latents = None
        if batch is not None and self.is_edit:
            with torch.no_grad():
                _, _, lh, lw = latent_model_input.shape
                target_pixels = (lh * self.vae_scale_factor) * (
                    lw * self.vae_scale_factor
                )
                ref_latents = self._batch_ref_latents_from_batch(
                    batch, latent_model_input.shape[0], target_pixels=target_pixels
                )

        # toolkit timestep (0..1000, 1000 = pure noise) -> Krea flow time t in
        # [0, 1] with t=1 = pure noise. Same convention -> straight divide.
        t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != latent_model_input.shape[0]:
            t = t.expand(latent_model_input.shape[0])

        context, text_mask = pad_text_features(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )

        pred = predict_velocity(
            self.transformer,
            latent_model_input.to(self.device_torch, self.torch_dtype),
            t,
            context,
            text_mask,
            ref_latents=ref_latents,
            isolate_refs=self.kv_cache,
        )
        return pred

    def _prep_vlm_images(self, ctrl: List[torch.Tensor]) -> List[torch.Tensor]:
        """Resize reference images for the Qwen3-VL pass.

        Downscaled (aspect-preserved, never upscaled) to fit ``vlm_max_pixels``
        total area (384^2 by default, the boogu_image_edit / ComfyUI
        TextEncodeQwenImageEditPlus budget) -- the MLLM only needs a coarse
        understanding of the reference; high-res detail flows through the VAE
        ref latents.
        """
        target = int(self.model_config.model_kwargs.get("vlm_max_pixels", 384 * 384))
        images = []
        for img in ctrl:
            if img.dim() == 4:
                img = img[0]
            img = img.to(self.device_torch)
            h, w = img.shape[1], img.shape[2]
            scale = min(1.0, math.sqrt(target / (h * w)))
            nh, nw = max(round(h * scale), 28), max(round(w * scale), 28)
            if (nh, nw) != (h, w):
                img = (
                    F.interpolate(
                        img.unsqueeze(0).float(),
                        size=(nh, nw),
                        mode="bicubic",
                        antialias=True,
                    )
                    .squeeze(0)
                    .clamp(0, 1)
                )
            images.append(img.float())
        return images

    def get_prompt_embeds(self, prompt, control_images=None) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        # Normalize control images to a per-prompt list (List[List[Tensor]]).
        # They arrive as a (B, C, H, W) batch tensor (control_tensor), a list of
        # per-sample lists (control_tensor_list), or a flat list of (1, C, H, W)
        # tensors for a single prompt (sampling / blank-embed caching).
        if control_images is not None:
            if isinstance(control_images, torch.Tensor):
                control_images = [
                    [control_images[i]] for i in range(control_images.shape[0])
                ]
            elif len(control_images) > 0 and not isinstance(control_images[0], list):
                control_images = [control_images]
            if len(control_images) == 1 and len(prompt) > 1:
                control_images = control_images * len(prompt)
            if len(control_images) != len(prompt):
                raise ValueError(
                    "Number of prompts must match number of control image sets"
                )
        else:
            control_images = [None] * len(prompt)

        # Encode each prompt at its natural length and store one (L, 12*2560)
        # tensor per batch item. The (L, 12, 2560) stack is flattened to 2D so the
        # toolkit's batching reads the list length (not the seq length) as the
        # batch size; predict_velocity restores the layer axis. Padding to the
        # batch max is deferred to the model call so caches stay small and any
        # prompts can share a batch.
        features_list = []
        for p, ctrl in zip(prompt, control_images):
            images = self._prep_vlm_images(ctrl) if ctrl is not None else None
            features = encode_krea_prompt(
                self.text_encoder,
                self.tokenizer,
                self.processor,
                p,
                max_length=self.max_text_length,
                overflow_policy=self.prompt_overflow_policy,
                select_layers=SELECT_LAYERS,
                images=images,
                vl_processor=self.vl_processor,
                dtype=self.torch_dtype,
            )
            # (L, n, d) -> (L, n*d)
            features = features.reshape(features.shape[0], -1)
            features_list.append(features.to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=features_list)

    def get_compile_sequence_layout(self):
        # The trunk runs cat([text, image]) and pads that to a 256-token bucket
        # before the blocks (see MMDiT._forward_impl), so the compiled block
        # kernels only ever see multiples of 256.
        from toolkit.compile_shape_bounds import SequenceLayout

        return SequenceLayout(
            sequence_alignment=256,
            includes_text=True,
            extra_tokens=0,
        )

    def get_text_length_bounds(self) -> Optional[tuple]:
        # Unlimited prompts cannot provide a safe finite compile bound. Strict
        # mode uses its configured setup-time limit.
        if self.prompt_overflow_policy == "unlimited":
            return None
        return (0, int(self.max_text_length))

    def get_loss_target(self, *args, **kwargs):
        # Flow-matching velocity target: noise - clean.
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    # ------------------------------------------------------------------
    # VAE (Qwen-Image AutoencoderKLQwenImage -- same handling as qwen_image arch)
    # ------------------------------------------------------------------
    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)

        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        # AutoencoderKLQwenImage is a video VAE: add a frame dim.
        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.squeeze(2)  # drop frame dim
        return latents.to(device, dtype=dtype)

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)
        latents = latents.unsqueeze(2)  # add frame dim

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # Full-resolution decode spikes VRAM; tile it when low on VRAM (decode
        # only -- encode stays untiled).
        tiled = self.model_config.low_vram
        if tiled:
            self.vae.enable_tiling()
        try:
            images = self.vae.decode(latents).sample
        finally:
            if tiled:
                self.vae.disable_tiling()
        images = images.squeeze(2)  # drop frame dim
        return images.to(device, dtype=dtype)

    # ------------------------------------------------------------------
    # Saving / bookkeeping
    # ------------------------------------------------------------------
    def save_model(self, output_path, meta, save_dtype):
        from toolkit.util.quantize import dequantize_if_quantized

        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: SingleStreamDiT = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            # dequantize any quantized (e.g. quanto/torchao) weights so we save plain full precision tensors
            save_dict[k] = (
                dequantize_if_quantized(v).clone().to("cpu", dtype=save_dtype)
            )
        meta = get_meta_for_safetensors(meta, name="krea2")
        save_file(save_dict, output_path, metadata=meta)

    def get_base_model_version(self):
        return "krea2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        return {
            k.replace("transformer.", "diffusion_model."): v
            for k, v in state_dict.items()
        }

    def convert_lora_weights_before_load(self, state_dict):
        return {
            k.replace("diffusion_model.", "transformer."): v
            for k, v in state_dict.items()
        }
