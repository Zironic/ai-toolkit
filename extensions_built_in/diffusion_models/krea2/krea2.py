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
import os
import struct
from pathlib import Path
from typing import List, Optional

import torch
from safetensors.torch import load_file, save_file

import huggingface_hub
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import EntryNotFoundError
from diffusers import AutoencoderKLQwenImage
from transformers import (
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
from toolkit.util.quantize import quantize, get_qtype, quantize_model
from toolkit.memory_management import MemoryManager
from toolkit.memory_management import vram_budget
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


def _stream_checkpoint(transformer, checkpoint_path: str, dtype) -> None:
    header, data_start = _read_safetensors_header(checkpoint_path)
    target_state = transformer.state_dict()
    target_keys = _validate_checkpoint_keys(transformer, header, checkpoint_path)
    with open(checkpoint_path, "rb", buffering=0) as handle:
        for key in tqdm(target_keys, desc="Loading Krea 2 tensors"):
            tensor = _load_checkpoint_tensor(
                handle, data_start, header, key, target_state[key].shape, dtype
            )
            _assign_tensor_by_name(transformer, key, tensor)
            del tensor


def _stream_and_quantize_checkpoint(base_model, transformer, checkpoint_path, dtype) -> None:
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
            unit.to("cpu")
    base_model.print_and_status_update("  - finished streaming and quantizing transformer units")
    flush(garbage_collect=False)



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


def _try_load_quantized_transformer_cache(base_model, transformer, cache_path: Path, metadata: dict) -> bool:
    if cache_path is None or not cache_path.exists():
        return False
    try:
        base_model.print_and_status_update(f"  - loading cached quantized transformer state from {cache_path}")
        payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        if payload.get("metadata") != metadata:
            base_model.print_and_status_update("  - cached quantized transformer metadata mismatch; ignoring")
            return False
        missing, unexpected = transformer.load_state_dict(payload["state_dict"], strict=True, assign=True)
        if missing or unexpected:
            raise RuntimeError(f"missing={missing[:5]} unexpected={unexpected[:5]}")
        from toolkit.dequantize import patch_dequantization_on_save
        patch_dequantization_on_save(transformer)
        return True
    except Exception as error:
        base_model.print_and_status_update(f"  - failed to load cached quantized transformer; rebuilding ({error})")
        return False


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
        # Safety cap on prompt token length (truncation only); embeds are stored
        # per-sample at natural length and padded to the batch max at the model call.
        self.max_text_length = int(
            self.model_config.model_kwargs.get("max_text_length", 512)
        )
        # Qwen2TokenizerFast used to tokenize the assistant suffix (matches the
        # reference's separate processor pass).
        self.processor = None
        self.use_old_lokr_format = False
        self._transformer_quantized_during_load = False

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
        enough blocks up front -- a mid-denoise OOM demote invalidates
        compiled state and (under strict ingraph) changes the pack set, so
        planning for the worst sample beats reacting per image. Returns None
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
        cache_path, cache_metadata = _quantized_transformer_cache_info(self, checkpoint_path, dtype, config)
        # Build on meta, then materialize either from the quantized cache or the checkpoint.
        with torch.device("meta"):
            transformer = SingleStreamDiT(config)

        if stream_quantized and _try_load_quantized_transformer_cache(self, transformer, cache_path, cache_metadata):
            self._transformer_quantized_during_load = True
        else:
            self.print_and_status_update("  - loading transformer through ranged disk reads")
            if stream_quantized:
                _stream_and_quantize_checkpoint(self, transformer, checkpoint_path, dtype)
                _save_quantized_transformer_cache(self, transformer, cache_path, cache_metadata)
                self._transformer_quantized_during_load = True
            else:
                _stream_checkpoint(transformer, checkpoint_path, dtype)

        flush()
        return transformer

    def _load_text_encoder(self):
        dtype = self.torch_dtype
        te_path = self.model_config.model_kwargs.get("text_encoder_path", QWEN3_VL_PATH)
        self.print_and_status_update(f"Loading Qwen3-VL text encoder from {te_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            te_path,
            max_length=self.max_text_length,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        processor = Qwen2TokenizerFast.from_pretrained(
            te_path,
            max_length=self.max_text_length,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            te_path,
            torch_dtype=dtype,
            token=HF_TOKEN,
            local_files_only=_hf_local_files_only(self.model_config),
        )
        # We only ever encode text, so the vision tower is dead weight -- drop it to
        # free VRAM and skip loading its (bf16-slow) Conv3d patch_embed onto the GPU.
        if getattr(text_encoder.model, "visual", None) is not None:
            text_encoder.model.visual = None
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        flush()
        return tokenizer, processor, text_encoder

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
                    keep_last = self.model_config.layer_offloading_checkpoint_keep_last
                    pinned_resident_keys = MemoryManager.training_pinned_keys_for_keep_last(
                        transformer, max(0, keep_last)
                    )
                    if self.model_config.layer_offloading_pinned_arena:
                        # Phase 3 Slice B: the pinned arena covers FROZEN base
                        # weights only and is built inside attach. The trainer
                        # freezes the base (BaseSDTrainProcess:2524) only AFTER
                        # this load_model runs, so freeze here to satisfy the
                        # "frozen before attach" invariant. Safe: the base is
                        # fp8-quantized (never genuinely trainable) and LoRA
                        # trains separate adapters, so an early freeze is
                        # behavior-neutral for adapter training.
                        transformer.requires_grad_(False)
                    MemoryManager.attach_smart_training(
                        transformer,
                        self.device_torch,
                        working_reserve_gib=self.model_config.layer_offloading_smart_working_reserve_gb,
                        wddm_margin_gib=self.model_config.layer_offloading_smart_wddm_margin_gb,
                        wddm_hard_gib=self.model_config.layer_offloading_smart_wddm_hard_gb,
                        ignore_modules=ignore_modules,
                        pinned_resident_keys=pinned_resident_keys,
                        block_stream_only=self.model_config.layer_offloading_block_stream_only,
                        # Ingraph training WITHOUT the arena pins its own block
                        # packs (repoint=False duplicates); per-tensor attach
                        # pins for the same weights would double-commit the
                        # shared WDDM pinned budget, so zero the attach budget.
                        # WITH the arena, the arena IS the pin authority and
                        # must be sized (auto/config value) to cover the whole
                        # streamed set -- zeroing it would make every block
                        # pageable and fail the borrow.
                        pinned_weight_gib=(
                            0.0
                            if (
                                self.model_config.layer_offloading_ingraph_training
                                and not self.model_config.layer_offloading_pinned_arena
                            )
                            else self.model_config.layer_offloading_pinned_weight_gb
                        ),
                        wddm_spill_reserve_pct=self.model_config.layer_offloading_wddm_spill_reserve_pct,
                        fp8_training_forward=(
                            self.model_config.quantize
                            and self.model_config.qtype in ('qfloat8', 'float8')
                            and self.model_config.layer_offloading_fp8_forward
                        ),
                        # Phase 3 (Slice B): the arena and ingraph training now
                        # coexist -- enable_ingraph_training BORROWS the arena
                        # flats for the frozen base (see mmdit.enable_ingraph_-
                        # training) instead of pinning a second, independent copy.
                        # The arena is the single pin authority for the base
                        # weights across both train and sample.
                        use_pinned_arena=self.model_config.layer_offloading_pinned_arena,
                    )
                    # Smart offload budgets weights, but an uncheckpointed Krea
                    # graph retains every block's activations (~16 GiB at the
                    # observed training shape). On low-VRAM cards that defeats
                    # offloading before backward can begin. Checkpointing is a
                    # requirement for this mode; it is gated by grad-enabled in
                    # SingleStreamDiT.forward, so inference/sampling is unchanged.
                    # -1 = auto: start fully checkpointed; the trainer hill-climbs
                    # elapsed time per resolution under a hard spill guard.
                    transformer.enable_gradient_checkpointing(
                        keep_last=max(0, keep_last)
                    )
                    if keep_last == -1:
                        self.print_and_status_update(
                            "  - gradient checkpointing enabled; auto-tuning "
                            "uncheckpointed trailing blocks"
                        )
                    elif keep_last:
                        self.print_and_status_update(
                            "  - selective gradient checkpointing enabled "
                            f"(last {keep_last} blocks kept resident)"
                        )
                    else:
                        self.print_and_status_update(
                            "  - gradient checkpointing enabled for smart training offload"
                        )
                else:
                    MemoryManager.attach(
                        transformer,
                        self.device_torch,
                        offload_percent=self.model_config.layer_offloading_transformer_percent,
                        ignore_modules=ignore_modules,
                    )

            if self.model_config.low_vram:
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
                max_length=self.max_text_length,
                token=HF_TOKEN,
                local_files_only=_hf_local_files_only(self.model_config),
            )
            processor = Qwen2TokenizerFast.from_pretrained(
                te_path,
                max_length=self.max_text_length,
                token=HF_TOKEN,
                local_files_only=_hf_local_files_only(self.model_config),
            )
            text_encoder = FakeTextEncoder(device=self.device_torch, dtype=dtype)
        else:
            tokenizer, processor, text_encoder = self._load_text_encoder()
            if self.model_config.quantize_te:
                self.print_and_status_update("Quantizing text encoder")
                text_encoder.to(self.device_torch)
                quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
                freeze(text_encoder)
                flush()
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
        keep_ingraph_sampling = bool(extra.get("keep_ingraph_sampling", False))
        skip_sampling_guard = bool(extra.get("skip_sampling_guard", False))
        sample_ok = False
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        # Reactive cohabitation guard: if external VRAM growth (Windows desktop,
        # another app) since the last image would push this forward's peak within
        # the WDDM spill margin, stream one resident block back to CPU first.
        # Runs before compile so enable_compiled_sampling() rebuilds for the new
        # resident set. Paging is silent (not an OOM), so this must be proactive.
        guard = getattr(self.model, "_mm_sampling_guard", None)
        if guard is not None and not skip_sampling_guard:
            try:
                guard()
            except Exception as error:
                print(f"[MemoryManager] sampling cohabitation guard failed: {error}")

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

        ingraph_requested = (
            self.model_config.compile_sample
            and (
                getattr(self.model_config, 'layer_offloading_compile_streamed', False)
                or getattr(self.model_config, 'layer_offloading_ingraph_sampling', False)
            )
        )
        strict_ingraph = bool(getattr(self.model_config, 'layer_offloading_ingraph_sampling', False))
        if ingraph_requested:
            self.model._last_ingraph_sampling_compile_state = None
            try:
                if getattr(self.model_config, 'layer_offloading_ingraph_stream_all', False):
                    streamed_blocks = tuple(range(len(self.model.blocks)))
                else:
                    # enable_ingraph_sampling() strips the legacy streaming
                    # markers that ingraph_streamed_block_indices() detects, so
                    # on a layout retained across calls (keep_ingraph_sampling)
                    # the live packs are the source of truth for the streamed
                    # set; markers only reappear if the layout changed since.
                    retained_packs = getattr(self.model, "_ingraph_sampling_packs", {}) or {}
                    streamed_blocks = tuple(sorted(
                        {int(i) for i in self.model.ingraph_streamed_block_indices()}
                        | {int(i) for i in retained_packs}
                    ))
                if strict_ingraph and not streamed_blocks:
                    raise RuntimeError("in-graph sampling unavailable: dynamic_streamed_block_set")
                requested_streamed = tuple(sorted(int(index) for index in streamed_blocks))
                current_packs = getattr(self.model, "_ingraph_sampling_packs", {}) or {}
                current_streamed = tuple(sorted(int(index) for index in current_packs))
                compiled_ingraph_blocks = getattr(
                    self.model, "_compiled_ingraph_sampling_blocks", {}
                ) or {}
                reuse_ingraph = bool(
                    current_streamed
                    and current_streamed == requested_streamed
                    and (
                        self.model._compiled_ingraph_sampling is not None
                        or len(compiled_ingraph_blocks) == len(current_streamed)
                    )
                )
                if reuse_ingraph:
                    packed = len(current_streamed)
                else:
                    packed = self.model.enable_ingraph_sampling(
                        streamed_blocks=streamed_blocks if streamed_blocks else None,
                        depth=getattr(self.model_config, 'layer_offloading_ingraph_depth', 2),
                        compile=True,
                    )
                if not getattr(self, '_ingraph_compile_sample_reported', False):
                    self.print_and_status_update(
                        f"Compiling transformer trunk for in-graph sampling: "
                        f"{packed} streamed block pack(s). First preview will be slow."
                    )
                    self._ingraph_compile_sample_reported = True
            except Exception as error:
                if not (keep_ingraph_sampling and sample_ok):
                    self.model.disable_ingraph_sampling()
                if strict_ingraph:
                    raise RuntimeError(f"strict in-graph sampling failed: {error}") from error
                if not getattr(self, '_ingraph_compile_sample_reported', False):
                    self.print_and_status_update(
                        "In-graph sampling compile unavailable for this layout "
                        f"({error}); falling back to regional compile."
                    )
                    self._ingraph_compile_sample_reported = True

        if self.model_config.compile_sample and self.model._compiled_ingraph_sampling is None:
            # Regional (per-block) compilation. We are inside the sampling
            # context (inference_resident) here, so residency is already
            # decided: blocks the manager made GPU-resident have NO offload
            # hooks and are compile-clean; any block still streaming weights
            # carries _BouncingLinearFn (a device-mutating custom autograd fn)
            # and must stay eager. enable_compiled_sampling() inspects each
            # block and selects accordingly, so this is safe even with
            # layer_offloading=True — fully-resident sampling gets the full
            # win, partial offload compiles whatever is resident.
            compiled_count, eager_count = self.model.enable_compiled_sampling()
            if not getattr(self, '_compile_sample_reported', False):
                if compiled_count == 0:
                    self.print_and_status_update(
                        "compile_sample=True but every transformer block still "
                        "streams weights (offload hooks present); nothing to "
                        "compile. Reduce offload (or sampling working reserve) so whole "
                        "blocks fit resident."
                    )
                else:
                    self.print_and_status_update(
                        f"Compiling transformer blocks for sampling (regional): "
                        f"{compiled_count} compiled, {eager_count} left eager "
                        f"(still streaming). First preview will be slow."
                    )
                self._compile_sample_reported = True

        # Sampling compiles are static-shape (dynamic=False); running the
        # call under eager_then_compile defers each compile to the second
        # call with a given shape instead of wasting one on the very first
        # -- so "did a new compile happen" must be re-checked on every
        # call, not just the first one this process.
        frames_before = None
        if compile_cache_dir and self.model_config.compile_sample:
            frames_before = torch._dynamo.utils.counters["frames"].get("total", 0)

        try:
            compile_stance = (
                torch.compiler.set_stance("eager_then_compile")
                if self.model_config.compile_sample
                else contextlib.nullcontext()
            )
            with compile_stance:
                img = pipeline(
                    conditional_embeds=conditional_embeds,
                    unconditional_embeds=unconditional_embeds,
                    height=gen_config.height,
                    width=gen_config.width,
                    num_inference_steps=gen_config.num_inference_steps,
                    guidance_scale=gen_config.guidance_scale,
                    latents=gen_config.latents,
                    generator=generator,
                    batch_cfg=getattr(gen_config, "batch_cfg", False),
                )[0]
            if frames_before is not None:
                frames_after = torch._dynamo.utils.counters["frames"].get("total", 0)
                if frames_after > frames_before and save_compile_cache(compile_cache_dir, compile_cache_key):
                    self.print_and_status_update(
                        f"Saved torch.compile cache to {compile_cache_dir}"
                    )
            sample_ok = True
            return img
        finally:
            if ingraph_requested:
                self.model._last_ingraph_sampling_compile_state = {
                    "ingraph_compiled": self.model._compiled_ingraph_sampling is not None,
                    "ingraph_compiled_blocks": len(
                        getattr(self.model, "_compiled_ingraph_sampling_blocks", {}) or {}
                    ),
                    "ingraph_packs": len(getattr(self.model, "_ingraph_sampling_packs", {}) or {}),
                    "regional_compiled_blocks": sum(
                        1 for block in (getattr(self.model, "_compiled_blocks", None) or [])
                        if block is not None
                    ),
                    "unavailable_reasons": tuple(
                        getattr(self.model, "_ingraph_unavailable_reasons", ()) or ()
                    ),
                }
                if not (keep_ingraph_sampling and sample_ok):
                    self.model.disable_ingraph_sampling()

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------
    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,  # (B, 16, h, w)
        timestep: torch.Tensor,  # 0..1000 scale
        text_embeddings: AdvancedPromptEmbeds,
        **kwargs,
    ):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

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
        )
        return pred

    def get_prompt_embeds(self, prompt) -> AdvancedPromptEmbeds:
        if isinstance(prompt, str):
            prompt = [prompt]

        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)

        # Encode each prompt at its natural length and store one (L, 12*2560)
        # tensor per batch item. The (L, 12, 2560) stack is flattened to 2D so the
        # toolkit's batching reads the list length (not the seq length) as the
        # batch size; predict_velocity restores the layer axis. Padding to the
        # batch max is deferred to the model call so caches stay small and any
        # prompts can share a batch.
        features_list = []
        for p in prompt:
            features = encode_krea_prompt(
                self.text_encoder,
                self.tokenizer,
                self.processor,
                p,
                max_length=self.max_text_length,
                select_layers=SELECT_LAYERS,
            )
            # (L, n, d) -> (L, n*d)
            features = features.reshape(features.shape[0], -1)
            features_list.append(features.to(self.torch_dtype))

        return AdvancedPromptEmbeds(text_embeds=features_list)

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
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        transformer: SingleStreamDiT = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)
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
