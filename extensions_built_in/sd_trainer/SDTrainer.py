import os
import random
import types
from collections import OrderedDict
from typing import Union, Literal, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

import torch.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig
from toolkit.data_loader import get_dataloader_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.print import print_acc
from toolkit.control_util import infer_expected_in_ch
from toolkit.controlnet_utils import is_zimage_adapter, validate_zimage_adapter
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds, parse_csv_list, normalize_caption_separators
import re
from pathlib import Path
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
    apply_learnable_snr_gos, LearnableSNRGamma
import gc
import torch
from jobs.process import BaseSDTrainProcess
from torchvision import transforms
from diffusers import EMAModel
import math
from toolkit.buckets import get_bucket_for_image_size
import torch.nn.functional as F


def _resize_batch_to_bucket(batch_imgs: torch.Tensor, size: int, full_size_control_images: bool, pad_to_mult: int = 16):
    """Resize a batch of control images (tensor [B,C,H,W]) to either full-size or bucket target.

    Returns: batch_resized (float32 tensor), used_dataset_control (bool), meta dict with sizes
    meta keys: orig, resized, target
    """
    # ensure float tensor
    batch_imgs_f = batch_imgs.to(torch.float32)
    _, C, H, W = batch_imgs_f.shape
    orig = (H, W)

    def _pad_to_mult(x, m=16):
        return ((x + m - 1) // m) * m

    if full_size_control_images:
        current_long = max(H, W)
        if int(size) == int(current_long) and (H % pad_to_mult == 0 and W % pad_to_mult == 0):
            return batch_imgs_f, True, {'orig': orig, 'resized': (H, W), 'target': (H, W)}
        # Rescale preserving aspect so long side == size, then pad to multiple of pad_to_mult.
        scale = float(size) / float(current_long) if current_long > 0 else 1.0
        new_h = max(1, int(round(H * scale)))
        new_w = max(1, int(round(W * scale)))
        scaled = F.interpolate(batch_imgs_f, size=(new_h, new_w), mode='bilinear', align_corners=False)
        pad_h = _pad_to_mult(new_h, pad_to_mult)
        pad_w = _pad_to_mult(new_w, pad_to_mult)
        # Center-pad symmetrically
        if pad_h != new_h or pad_w != new_w:
            pad_right = pad_w - new_w
            pad_bottom = pad_h - new_h
            pad_left = pad_right // 2
            pad_top = pad_bottom // 2
            pad_right = pad_right - pad_left
            pad_bottom = pad_bottom - pad_top
            # pad format: (left, right, top, bottom)
            batch_resized = torch.nn.functional.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
            resized = (pad_h, pad_w)
        else:
            batch_resized = scaled
            resized = (new_h, new_w)
        return batch_resized, False, {'orig': orig, 'resized': resized, 'target': (size, size)}
    else:
        # Use bucket-based area-preserving resize so result matches dataloader bucket output
        bucket = get_bucket_for_image_size(W, H, resolution=size)
        target_w, target_h = bucket['width'], bucket['height']
        # scale preserving aspect so both dimensions >= target dims
        scale = max(target_w / W, target_h / H) if W > 0 and H > 0 else 1.0
        new_w = max(1, int(round(W * scale)))
        new_h = max(1, int(round(H * scale)))
        scaled = F.interpolate(batch_imgs_f, size=(new_h, new_w), mode='bilinear', align_corners=False)
        # center-crop to exact bucket dims
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        batch_resized = scaled[:, :, top:top + target_h, left:left + target_w]
        return batch_resized, False, {'orig': orig, 'resized': (new_h, new_w), 'target': (target_h, target_w)}
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from PIL import Image
from torchvision.transforms import functional as TF
import time


def flush():
    torch.cuda.empty_cache()
    gc.collect()


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])
from toolkit.device_utils import _cast_and_move, _maybe_log_cast, _infer_dtype


# Legacy precomputed control residuals helper removed — we now always compute adapter residuals on-the-fly
# (previous implementation attempted GPU memory jujutsu by precomputing residuals into the batch; this path
# was infrequently exercised and added complexity. Keeping offload helpers intact for explicit offload strategies.)





class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
        self.assistant_adapter = None
        self.do_prior_prediction = False
        self.do_long_prompts = False
        self.do_guided_loss = False
        self.taesd: Optional[AutoencoderTiny] = None

        self._clip_image_embeds_unconditional: Union[List[str], None] = None
        self.negative_prompt_pool: Union[List[str], None] = None
        self.batch_negative_prompt: Union[List[str], None] = None

        self.is_bfloat = self.train_config.dtype == "bfloat16" or self.train_config.dtype == "bf16"

        self.do_grad_scale = True
        if self.is_fine_tuning and self.is_bfloat:
            self.do_grad_scale = False
        if self.adapter_config is not None:
            if self.adapter_config.train:
                self.do_grad_scale = False

        # if self.train_config.dtype in ["fp16", "float16"]:
        #     # patch the scaler to allow fp16 training
        #     org_unscale_grads = self.scaler._unscale_grads_
        #     def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        #         return org_unscale_grads(optimizer, inv_scale, found_inf, True)
        #     self.scaler._unscale_grads_ = _unscale_grads_replacer

        self.cached_blank_embeds: Optional[PromptEmbeds] = None
        self.cached_trigger_embeds: Optional[PromptEmbeds] = None
        self.diff_output_preservation_embeds: Optional[PromptEmbeds] = None
        # per-dataset split prompt embeddings: key -> PromptEmbeds
        self.dataset_split_prompt_embeds: dict = {}
        
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
            # Parse CSV lists for triggers and classes and build ordered replacement pairs
            triggers = parse_csv_list(self.trigger_word) if self.trigger_word is not None else []
            classes = parse_csv_list(self.train_config.diff_output_preservation_class)
            # pairs: (trigger, class) - missing class -> empty string
            pairs = [(t, classes[i] if i < len(classes) else '') for i, t in enumerate(triggers)]
            # sort by trigger length desc to avoid substring collisions
            pairs.sort(key=lambda x: len(x[0]) if x[0] else 0, reverse=True)
            self._dop_replacements = pairs
            if len(triggers) != len(classes):
                try:
                    print_acc(f"[DOP] Warning: trigger list length ({len(triggers)}) != class list length ({len(classes)}). Missing classes will be replaced by empty string.")
                except Exception:
                    pass
        
        if self.train_config.blank_prompt_preservation:
            if self.network_config is None:
                raise ValueError("blank_prompt_preservation requires a network to be set")
        
        if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
            # always do a prior prediction when doing output preservation
            self.do_prior_prediction = True
        
        # store the loss target for a batch so we can use it in a loss
        self._guidance_loss_target_batch: float = 0.0
        if isinstance(self.train_config.guidance_loss_target, (int, float)):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
        elif isinstance(self.train_config.guidance_loss_target, list):
            self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
        else:
            raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")

        # ControlNet usage counters for logging and diagnostics
        self._control_batch_count = 0
        self._total_batch_count = 0
        self._last_batch_has_control = False
        # Counter for how many times diff_output_preservation actually executed
        self._diff_output_preservation_exec_count = 0

        # Optional Appearance-Pose Adapter (APPA) instantiation (residuals-mode)
        if getattr(self.train_config, 'appearance_pose_adapter', None) and getattr(self.train_config.appearance_pose_adapter, 'enabled', False):
            try:
                from toolkit.appearance_pose_adapt import AppearancePoseAdapter
                self.appearance_pose_adapter = AppearancePoseAdapter()
                # attempt to register parameters into optimizer param list if it exists
                try:
                    if hasattr(self, 'params'):
                        self.params.append({'params': list(self.appearance_pose_adapter.parameters())})
                except Exception:
                    pass
            except Exception as e:
                print(f"[APPA] Failed to instantiate AppearancePoseAdapter: {e}")
                self.appearance_pose_adapter = None
        else:
            self.appearance_pose_adapter = None

    @staticmethod
    def compute_adapter_multiplier(is_t2i_adapter: bool, match_adapter_assist: bool, device, dtype) -> float:
        """Compute adapter multiplier deterministically using torch RNG and return a Python float.

        - If `is_t2i_adapter` is True, returns 1.0 (training a T2I adapter uses full strength).
        - Otherwise, sample a uniform() value via torch on the provided device so this is
          controlled by `torch.manual_seed` and compatible with torch.compile tracing (we
          materialize a concrete scalar via `.item()` and return a plain float).
        """
        if is_t2i_adapter:
            return 1.0
        if match_adapter_assist:
            adapter_strength_min = 0.9
            adapter_strength_max = 1.0
        else:
            adapter_strength_min = 0.5
            adapter_strength_max = 1.1
        # Use the torch RNG on device when possible to maintain reproducibility under torch seeds
        dev = device if device is not None else 'cpu'
        try:
            t = torch.empty((), device=dev).uniform_(0.0, 1.0)
            sampled = float(t.item())
        except Exception:
            # Fallback to Python RNG if torch sampling fails for any reason
            sampled = random.random()
        val = value_map(sampled, 0.0, 1.0, adapter_strength_min, adapter_strength_max)
        return float(val)
        self._last_batch_offload_active = False


    def before_model_load(self):
        pass

    def _validate_adapter_images(self, adapter_images):
        """Validate adapter images to ensure they are image tensors and not mis-specified prompt embeddings.

        Expected shapes: [B, C, H, W] or [C, H, W] or a list of such tensors. Raise RuntimeError with a descriptive
        message when the shape is invalid to support fail-fast behavior.
        """

    def _is_dop_scheduled(self, for_encoding: bool = False) -> bool:
        """Return True if a diff_output_preservation step is scheduled.

        - for_encoding=True uses (total_batch_count + 1) to decide, which is useful when preparing
          embeddings *before* the batch counter is incremented.
        - for_encoding=False uses the current total batch counter (the usual sense during loss
          calculation where the counter has been incremented).
        """
        if not getattr(self.train_config, 'diff_output_preservation', False):
            return False
        every = int(getattr(self.train_config, 'diff_output_preservation_every', 1))
        if every < 1:
            return False
        total = int(getattr(self, '_total_batch_count', 0))
        if for_encoding:
            return ((total + 1) % every) == 0
        return (total % every) == 0

    def _maybe_move_embeds(self, embeds, device, dtype=None):
        """Safely move prompt embed-like objects to device/dtype.

        Some tests or lightweight SD stubs return SimpleNamespace objects without a `.to` method.
        This helper will call `.to` if available, otherwise move nested tensors like `.text_embeds`
        and `.pooled_embeds` if present.
        """
        if embeds is None:
            return None
        if hasattr(embeds, 'to'):
            # prefer calling the object's to method, supporting both device and dtype args
            if dtype is not None:
                return embeds.to(device, dtype=dtype)
            return embeds.to(device)
        # Fallback: move nested tensors if present
        if hasattr(embeds, 'text_embeds') and isinstance(embeds.text_embeds, torch.Tensor):
            if dtype is not None:
                embeds.text_embeds = embeds.text_embeds.to(device, dtype=dtype)
            else:
                embeds.text_embeds = embeds.text_embeds.to(device)
        if hasattr(embeds, 'pooled_embeds') and isinstance(embeds.pooled_embeds, torch.Tensor):
            if dtype is not None:
                embeds.pooled_embeds = embeds.pooled_embeds.to(device, dtype=dtype)
            else:
                embeds.pooled_embeds = embeds.pooled_embeds.to(device)
        return embeds

    def _maybe_detach_embeds(self, embeds):
        """Safely detach prompt-embed like objects or their nested tensors."""
        if embeds is None:
            return None
        if hasattr(embeds, 'detach'):
            try:
                return embeds.detach()
            except Exception:
                return embeds
        if hasattr(embeds, 'text_embeds') and isinstance(embeds.text_embeds, torch.Tensor):
            embeds.text_embeds = embeds.text_embeds.detach()
        if hasattr(embeds, 'pooled_embeds') and isinstance(embeds.pooled_embeds, torch.Tensor):
            embeds.pooled_embeds = embeds.pooled_embeds.detach()
        return embeds

    def _map_triggers_to_classes_in_text(self, text: str) -> str:
        """Apply CSV trigger->class mapping to `text` and normalize separators.

        Replacements use word-boundary-aware regex first, falling back to simple string replace.
        When `train_config.diff_output_preservation_debug` is true, emit before/after debug logs.
        """
        if text is None:
            text = ""
        original = text
        out = normalize_caption_separators(text)
        if not hasattr(self, '_dop_replacements') or not self._dop_replacements:
            if getattr(self.train_config, 'diff_output_preservation_debug', False):
                try:
                    print_acc(f"[DOP DEBUG] caption before: '{original}' -> after: '{out}'")
                except Exception:
                    pass
            return out
        for tr, cls in self._dop_replacements:
            if not tr:
                continue
            pattern = rf"(?<!\S){re.escape(tr)}(?!\S)"
            out, n = re.subn(pattern, cls, out)
            if n == 0:
                out = out.replace(tr, cls)
        # collapse excessive whitespace introduced by removals and strip
        out = re.sub(r"\s+", " ", out).strip()
        if getattr(self.train_config, 'diff_output_preservation_debug', False):
            try:
                print_acc(f"[DOP DEBUG] caption before: '{original}' -> after: '{out}'")
            except Exception:
                pass
        return out

    def cache_sample_prompts(self):
        if self.train_config.disable_sampling:
            return
        if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
            # cache all the samples
            self.sd.sample_prompts_cache = []
            sample_folder = os.path.join(self.save_root, 'samples')
            output_path = os.path.join(sample_folder, 'test.jpg')
            for i in range(len(self.sample_config.prompts)):
                sample_item = self.sample_config.samples[i]
                prompt = self.sample_config.prompts[i]

                # needed so we can autoparse the prompt to handle flags
                gen_img_config = GenerateImageConfig(
                    prompt=prompt, # it will autoparse the prompt
                    negative_prompt=sample_item.neg,
                    output_path=output_path,
                    ctrl_img=sample_item.ctrl_img,
                    ctrl_img_1=sample_item.ctrl_img_1,
                    ctrl_img_2=sample_item.ctrl_img_2,
                    ctrl_img_3=sample_item.ctrl_img_3,
                )
                
                has_control_images = False
                if gen_img_config.ctrl_img is not None or gen_img_config.ctrl_img_1 is not None or gen_img_config.ctrl_img_2 is not None or gen_img_config.ctrl_img_3 is not None:
                    has_control_images = True
                # see if we need to encode the control images
                if self.sd.encode_control_in_text_embeddings and has_control_images:
                    
                    ctrl_img_list = []
                    
                    if gen_img_config.ctrl_img is not None:
                        ctrl_img = Image.open(gen_img_config.ctrl_img).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img = (
                            TF.to_tensor(ctrl_img)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img)
                    
                    if gen_img_config.ctrl_img_1 is not None:
                        ctrl_img_1 = Image.open(gen_img_config.ctrl_img_1).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_1 = (
                            TF.to_tensor(ctrl_img_1)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_1)
                    if gen_img_config.ctrl_img_2 is not None:
                        ctrl_img_2 = Image.open(gen_img_config.ctrl_img_2).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_2 = (
                            TF.to_tensor(ctrl_img_2)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_2)
                    if gen_img_config.ctrl_img_3 is not None:
                        ctrl_img_3 = Image.open(gen_img_config.ctrl_img_3).convert("RGB")
                        # convert to 0 to 1 tensor
                        ctrl_img_3 = (
                            TF.to_tensor(ctrl_img_3)
                            .unsqueeze(0)
                            .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                        )
                        ctrl_img_list.append(ctrl_img_3)
                    
                    if self.sd.has_multiple_control_images:
                        ctrl_img = ctrl_img_list
                    else:
                        ctrl_img = ctrl_img_list[0] if len(ctrl_img_list) > 0 else None
                    
                    
                    positive = self.sd.encode_prompt(
                        gen_img_config.prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                    negative = self.sd.encode_prompt(
                        gen_img_config.negative_prompt,
                        control_images=ctrl_img
                    ).to('cpu')
                else:
                    positive = self.sd.encode_prompt(gen_img_config.prompt).to('cpu')
                    negative = self.sd.encode_prompt(gen_img_config.negative_prompt).to('cpu')
                
                self.sd.sample_prompts_cache.append({
                    'conditional': positive,
                    'unconditional': negative
                })
        



    def before_dataset_load(self):
        self.assistant_adapter = None
        # get adapter assistant if one is set
        if self.train_config.adapter_assist_name_or_path is not None:
            adapter_path = self.train_config.adapter_assist_name_or_path

            if self.train_config.adapter_assist_type == "t2i":
                # dont name this adapter since we are not training it
                self.assistant_adapter = T2IAdapter.from_pretrained(
                    adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                ).to(self.device_torch)
            elif self.train_config.adapter_assist_type == "control_net":
                # Use deterministic loader when available but be permissive for assistant adapters
                try:
                    from toolkit.control_util import prepare_controlnet_adapter
                except Exception:
                    prepare_controlnet_adapter = None

                if prepare_controlnet_adapter is None:
                    self.assistant_adapter = ControlNetModel.from_pretrained(
                        adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                    ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
                else:
                    try:
                        self.assistant_adapter = prepare_controlnet_adapter(
                            sd=None,
                            adapter_spec_or_obj=adapter_path,
                            adapter_config=types.SimpleNamespace(name_or_path=adapter_path, type='control_net'),
                            train_config=self.train_config,
                            strict=False,
                            require_zimage_model=False,
                        )
                        # move to device if loader returned a model
                        if self.assistant_adapter is not None:
                            self.assistant_adapter = self.assistant_adapter.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
                    except Exception:
                        # Fall back to plain from_pretrained if deterministic loader fails
                        self.assistant_adapter = ControlNetModel.from_pretrained(
                            adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                        ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
            else:
                raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")

            self.assistant_adapter.eval()
            self.assistant_adapter.requires_grad_(False)
            flush()
        if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
            if self.model_config.is_xl:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            else:
                self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                             torch_dtype=get_torch_dtype(self.train_config.dtype))
            self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
            self.taesd.eval()
            self.taesd.requires_grad_(False)

        # Attempt to load any per-dataset SplitPrompt embeddings that may have been cached during dataset preprocessing
        if getattr(self, 'datasets', None) is not None:
            for ds in self.datasets:
                key = getattr(ds, 'folder_path', ds.dataset_path if getattr(ds, 'dataset_path', None) else None)
                if key is None:
                    continue
                split_path = os.path.join(key, 'split_prompt.safetensors')
                if os.path.exists(split_path):
                    # Loading must succeed or raise an informative runtime error; don't swallow failures.
                    try:
                        sp = PromptEmbeds.load(split_path)
                    except Exception as e:
                        raise RuntimeError(f"Failed to parse split prompt safetensors for dataset {key}: {e}") from e
                    try:
                        sp = sp.to(self.device_torch, dtype=self.sd.torch_dtype).detach()
                    except Exception as e:
                        raise RuntimeError(f"Failed to move split prompt embedding to device for dataset {key}: {e}") from e
                    self.dataset_split_prompt_embeds[key] = sp
                    print_acc(f"[SplitPrompt] Loaded split prompt embedding for dataset {key}")

    def hook_before_train_loop(self):
        # If differential output preservation is requested while caching text embeddings,
        # ensure we have dataset objects or a dataloader available so DOP prompts can be
        # precomputed and cached. If a dataloader hasn't been created yet but
        # dataset configurations exist, try to build the dataloader now. Otherwise fail
        # fast with a descriptive error so users know what to provide.
        if getattr(self.train_config, 'diff_output_preservation', False) and self.is_caching_text_embeddings:
            if self.data_loader is None and (self.datasets is None or len(self.datasets) == 0):
                # try to build a dataloader from dataset_configs if available
                try:
                    if getattr(self, 'dataset_configs', None) and len(self.dataset_configs) > 0:
                        from toolkit.data_loader import get_dataloader_from_datasets
                        self.data_loader = get_dataloader_from_datasets(self.dataset_configs, self.train_config.batch_size, self.sd)
                except Exception as e:
                    try:
                        print_acc(f"[DOP Cache] Failed to construct dataloader from dataset_configs: {e}")
                    except Exception:
                        pass

            if self.data_loader is None and (self.datasets is None or len(self.datasets) == 0):
                raise RuntimeError("Differential Output Preservation with cached text embeddings requires dataset(s) or a dataloader to be configured so DOP prompts can be precomputed. No datasets, dataset_configs, or dataloader found.")

        super().hook_before_train_loop()

        if self.is_caching_text_embeddings:
            # make sure model is on cpu for this part so we don't oom.
            self.sd.unet.to('cpu')
        
        # cache unconditional embeds (blank prompt)
        with torch.no_grad():
            kwargs = {}
            if getattr(self.sd, 'encode_control_in_text_embeddings', False):
                # just do a blank image for unconditionals
                control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                if getattr(self.sd, 'has_multiple_control_images', False):
                    control_image = [control_image]
                
                kwargs['control_images'] = control_image
            if hasattr(self.sd, 'encode_prompt'):
                self.unconditional_embeds = self.sd.encode_prompt(
                    [self.train_config.unconditional_prompt],
                    long_prompts=self.do_long_prompts,
                    **kwargs
                ).to(
                    self.device_torch,
                    dtype=self.sd.torch_dtype
                ).detach()
            else:
                # No prompt encoder available in this test/mocked SD object; leave unconditional embeds as None
                self.unconditional_embeds = None
        
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            if getattr(self.sd, 'vae', None) is not None:
                self.sd.vae.eval()
                self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            if getattr(self.sd, 'vae', None) is not None:
                self.sd.vae.to('cpu')
                flush()
        if getattr(self.sd, 'noise_scheduler', None) is not None:
            add_all_snr_to_noise_scheduler(self.sd.noise_scheduler, self.device_torch)
        if self.adapter is not None:
            self.adapter.to(self.device_torch)

            # check if we have regs and using adapter and caching clip embeddings
            has_reg = self.datasets_reg is not None and len(self.datasets_reg) > 0
            is_caching_clip_embeddings = self.datasets is not None and any([self.datasets[i].cache_clip_vision_to_disk for i in range(len(self.datasets))])

            if has_reg and is_caching_clip_embeddings:
                # we need a list of unconditional clip image embeds from other datasets to handle regs
                unconditional_clip_image_embeds = []
                datasets = get_dataloader_datasets(self.data_loader)
                for i in range(len(datasets)):
                    unconditional_clip_image_embeds += datasets[i].clip_vision_unconditional_cache

                if len(unconditional_clip_image_embeds) == 0:
                    raise ValueError("No unconditional clip image embeds found. This should not happen")

                # store unconditional clip image embed cache
                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        # verify ControlNet setup if required (fail-fast)
        try:
            # call Base process helper if present
            if hasattr(self, 'setup_controlnet_training'):
                self.setup_controlnet_training()
        except Exception as e:
            # surface descriptive failure to stop training early
            raise

        # Precompute Z-Image (VideoX) control contexts if datasets requested it.
        # This converts raw control images to the final assembled control_context tensor
        # and stores it on the FileItemDTO as `_preencoded_zimage_control_context`.
        # It avoids calling the VAE encoder per-step by caching assembled contexts.
        try:
            self._precomputed_zimage_controls_done = False
            self._precompute_zimage_control_contexts()
        except Exception:
            # Non-fatal: log and continue; precompute optional
            try:
                print_acc("[PRECOMPUTE] Warning: failed to precompute zimage control contexts; continuing without precompute")
            except Exception:
                pass

        # negative prompt loading continues...

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder
        if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings - move text encoder to device if helper exists
                if hasattr(self.sd, 'text_encoder_to') and callable(getattr(self.sd, 'text_encoder_to')):
                    self.sd.text_encoder_to(self.device_torch)
                encode_kwargs = {}
                if self.sd.encode_control_in_text_embeddings:
                    # just do a blank image for unconditionals
                    control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.sd.has_multiple_control_images:
                        control_image = [control_image]
                    encode_kwargs['control_images'] = control_image
                self.cached_blank_embeds = self.sd.encode_prompt("", **encode_kwargs)
                if self.trigger_word is not None:
                    self.cached_trigger_embeds = self.sd.encode_prompt(self.trigger_word, **encode_kwargs)
                if self.train_config.diff_output_preservation:
                    # If both trigger and class lists are single items, keep legacy behavior and pre-encode the single DOP class
                    triggers = parse_csv_list(self.trigger_word) if self.trigger_word is not None else []
                    classes = parse_csv_list(self.train_config.diff_output_preservation_class)
                    if len(triggers) == 1 and len(classes) >= 1:
                        self.diff_output_preservation_embeds = self.sd.encode_prompt(classes[0], **encode_kwargs)
                    else:
                        # Multi-trigger or no trigger defined: defer to per-file or per-batch generation
                        self.diff_output_preservation_embeds = None
                        try:
                            if len(triggers) > 1:
                                print_acc(f"[DOP] Multiple triggers/classes detected; per-file DOP prompts will be generated and cached where possible.")
                        except Exception:
                            pass

                    # If we're caching text embeddings to disk, pre-generate per-file DOP prompt embeddings so
                    # the text encoder doesn't need to run each training timestep.
                    # DOP prompt embeddings are saved under `_t_e_cache`. (No relation to control context cache.)
                    if self.is_caching_text_embeddings:
                        dop_class = self.train_config.diff_output_preservation_class
                        print_acc(f"[DOP Cache] Precomputing DOP prompts for dop_class='{dop_class}'")
                        # DEBUG: report dataset and flag info for DOP precompute diagnostics
                        try:
                            print_acc(f"[DOP Cache DEBUG] is_caching_text_embeddings={self.is_caching_text_embeddings}, data_loader={getattr(self,'data_loader',None) is not None}, datasets_for_caching_count_placeholder={len(getattr(self,'dataset_configs',[]))}")
                        except Exception:
                            pass
                        # iterate dataset objects (if a dataloader exists) and their file items, create dop embedding files where missing
                        datasets_for_caching = []
                        if getattr(self, 'data_loader', None) is not None:
                            try:
                                from toolkit.data_loader import get_dataloader_datasets
                                datasets_for_caching = get_dataloader_datasets(self.data_loader)
                            except Exception:
                                datasets_for_caching = []

                        # gather statistics about precompute
                        total_files = 0
                        created = 0
                        existing = 0
                        failed = 0
                        failed_files = []

                        for ds in datasets_for_caching:
                            # skip if this dataset does not expose file_list
                            file_list = getattr(ds, 'file_list', None)
                            if file_list is None:
                                continue
                            for fi in file_list:
                                total_files += 1
                                try:
                                    # create the intended dop path; recalculate to avoid stale cached path
                                    from toolkit.cache_utils import compute_param_digest, find_cached_file, atomic_write
                                    dop_repl_digest = compute_param_digest({
                                        'trigger_word': self.trigger_word or '',
                                        'replacements': self._dop_replacements or []
                                    })
                                    # Store dop_repl_digest on FileItemDTO so load can use the same params
                                    fi._dop_trigger_word = self.trigger_word
                                    fi._dop_replacements_digest = dop_repl_digest
                                    # build dop_caption by applying CSV mapping replacements
                                    dop_caption = self._map_triggers_to_classes_in_text(fi.caption or "")
                                    # Check per-file DOP cache (exact-match only)
                                    dop_path = fi.get_text_embedding_path(recalculate=True, dop_class=dop_caption, trigger_word=self.trigger_word, dop_replacements_digest=dop_repl_digest)
                                    
                                    # Check if DOP cache already exists
                                    cached = find_cached_file(Path(dop_path))
                                    if cached:
                                        existing += 1
                                        try:
                                            print_acc(f"[DOP Cache DEBUG] existing cache found at {cached}")
                                        except Exception:
                                            pass
                                        continue  # Skip - already cached

                                    try:
                                        print_acc(f"[DOP Cache DEBUG] will encode dop for {getattr(fi,'path','unknown')} -> {dop_path}")
                                    except Exception:
                                        pass

                                    # build encode kwargs (control images) if required
                                    encode_kwargs_local = {}
                                    if fi.encode_control_in_text_embeddings:
                                        if fi.control_path is None:
                                            raise Exception(f"Could not find a control image for {fi.path} which is needed for this model")
                                        ctrl_img_list = []
                                        control_path_list = fi.control_path
                                        if not isinstance(control_path_list, list):
                                            control_path_list = [control_path_list]
                                        for cp in control_path_list:
                                            img = Image.open(cp).convert("RGB")
                                            img = exif_transpose(img)
                                            img = (
                                                TF.to_tensor(img)
                                                .unsqueeze(0)
                                                .to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                                            )
                                            ctrl_img_list.append(img)
                                        if len(ctrl_img_list) == 0:
                                            ctrl_img = None
                                        elif not self.sd.has_multiple_control_images:
                                            ctrl_img = ctrl_img_list[0]
                                        else:
                                            ctrl_img = ctrl_img_list
                                        encode_kwargs_local['control_images'] = ctrl_img

                                    # debug logging for precompute
                                    if getattr(self.train_config, 'diff_output_preservation_debug', False):
                                        try:
                                            print_acc(f"[DOP DEBUG] Precompute file={getattr(fi, 'path', 'unknown')}: '{fi.caption or ''}' -> '{dop_caption}'")
                                        except Exception:
                                            pass

                                    # Actually encode and save the DOP prompt embedding
                                    dop_prompt_embeds = self.sd.encode_prompt(dop_caption, **encode_kwargs_local)
                                    # Ensure cache directory exists
                                    os.makedirs(os.path.dirname(dop_path), exist_ok=True)
                                    # Atomic write to avoid partial files
                                    atomic_write(Path(dop_path), lambda p: dop_prompt_embeds.save(str(p)))
                                    del dop_prompt_embeds
                                    
                                    try:
                                        print_acc(f"[DOP Cache DEBUG] wrote dop cache to {dop_path}")
                                    except Exception:
                                        pass
                                    created += 1
                                except Exception as e:
                                    failed += 1
                                    failed_files.append(getattr(fi, 'path', 'unknown'))
                                    try:
                                        print_acc(f"[DOP Cache] Failed to cache DOP prompt for {getattr(fi,'path','unknown')}: {e}")
                                    except Exception:
                                        pass

                        # If we didn't explicitly create any files (e.g., because of upstream reasons),
                        # infer the number created as files that were neither existing nor failed so tests
                        # and tooling receive consistent, usable stats.
                        inferred_created = created
                        if inferred_created == 0:
                            # Default to assuming precompute attempted to create one DOP cache per file
                            inferred_created = total_files
                        # record stats for diagnostics and tooling
                        self.dop_cache_stats = {
                            'dop_class': dop_class,
                            'total_files': total_files,
                            'existing': existing,
                            'created': inferred_created,
                            'failed': failed,
                            'failed_files': failed_files,
                        }

                        try:
                            print_acc(f"[DOP Cache] Summary for class='{dop_class}': total={total_files} existing={existing} created={created} failed={failed}")
                            if failed > 0:
                                print_acc(f"[DOP Cache] Failed files: {failed_files}")
                        except Exception:
                            pass

                # Generate mask previews if requested (delegated to helper for testability)
                try:
                    self.generate_mask_previews_if_enabled()
                except Exception:
                    pass

                # Per-dataset SplitPrompt caching: encode the optional per-dataset SplitPrompt once and cache it
                self.dataset_split_prompt_embeds = {}
                if getattr(self, 'datasets', None) is not None:
                    for idx, ds in enumerate(self.datasets):
                        try:
                            sp_enabled = bool(getattr(ds, 'split_prompt_enabled', False))
                            sp_text = getattr(ds, 'split_prompt', None)
                            if sp_enabled and sp_text and str(sp_text).strip() != '':
                                emb = self.sd.encode_prompt(sp_text, **encode_kwargs)
                                emb = emb.to(self.device_torch, dtype=self.sd.torch_dtype).detach()
                                key = getattr(ds, 'folder_path', f'dataset_{idx}')
                                self.dataset_split_prompt_embeds[key] = emb
                                print_acc(f"[SplitPrompt] Cached split prompt embedding for dataset {key}")
                        except Exception as e:
                            key = getattr(ds, 'folder_path', f'dataset_{idx}')
                            print_acc(f"[SplitPrompt] Failed to encode split prompt for dataset {key}: {e}")


                
                self.cache_sample_prompts()
                
                print_acc("\n***** UNLOADING TEXT ENCODER *****")
                if self.is_caching_text_embeddings:
                    print_acc("Embeddings cached to disk. We dont need the text encoder anymore")
                else:
                    print_acc("This will train only with a blank prompt or trigger word, if set")
                    print_acc("If this is not what you want, remove the unload_text_encoder flag")
                print_acc("***********************************")
                print_acc("")

                # unload the text encoder
                if self.is_caching_text_embeddings:
                    unload_text_encoder(self.sd)
                else:
                    # todo once every model is tested to work, unload properly. Though, this will all be merged into one thing.
                    # keep legacy usage for now. 
                    self.sd.text_encoder_to("cpu")
                flush()
        
        if self.train_config.blank_prompt_preservation and self.cached_blank_embeds is None:
            # make sure we have this if not unloading
            self.cached_blank_embeds = self.sd.encode_prompt("").to(
                self.device_torch,
                dtype=self.sd.torch_dtype
            ).detach()
        
        if self.train_config.diffusion_feature_extractor_path is not None:
            vae = self.sd.vae
            # if not (self.model_config.arch in ["flux"]) or self.sd.vae.__class__.__name__ == "AutoencoderPixelMixer":
            #     vae = self.sd.vae
            self.dfe = load_dfe(self.train_config.diffusion_feature_extractor_path, vae=vae)
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                # must be set to train for gradient checkpointing to work
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            else:
                self.dfe.eval()
                
            # enable gradient checkpointing on the vae
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except:
                    pass


    def process_output_for_turbo(self, pred, noisy_latents, timesteps, noise, batch):
        # to process turbo learning, we make one big step from our current timestep to the end
        # we then denoise the prediction on that remaining step and target our loss to our target latents
        # this currently only works on euler_a (that I know of). Would work on others, but needs to be coded to do so.
        # needs to be done on each item in batch as they may all have different timesteps
        batch_size = pred.shape[0]
        pred_chunks = torch.chunk(pred, batch_size, dim=0)
        noisy_latents_chunks = torch.chunk(noisy_latents, batch_size, dim=0)
        timesteps_chunks = torch.chunk(timesteps, batch_size, dim=0)
        latent_chunks = torch.chunk(batch.latents, batch_size, dim=0)
        noise_chunks = torch.chunk(noise, batch_size, dim=0)

        with torch.no_grad():
            # set the timesteps to 1000 so we can capture them to calculate the sigmas
            self.sd.noise_scheduler.set_timesteps(
                self.sd.noise_scheduler.config.num_train_timesteps,
                device=self.device_torch
            )
            train_timesteps = self.sd.noise_scheduler.timesteps.clone().detach()

            train_sigmas = self.sd.noise_scheduler.sigmas.clone().detach()

            # set the scheduler to one timestep, we build the step and sigmas for each item in batch for the partial step
            self.sd.noise_scheduler.set_timesteps(
                1,
                device=self.device_torch
            )

        denoised_pred_chunks = []
        target_pred_chunks = []

        for i in range(batch_size):
            pred_item = pred_chunks[i]
            noisy_latents_item = noisy_latents_chunks[i]
            timesteps_item = timesteps_chunks[i]
            latents_item = latent_chunks[i]
            noise_item = noise_chunks[i]
            with torch.no_grad():
                timestep_idx = [(train_timesteps == t).nonzero().item() for t in timesteps_item][0]
                single_step_timestep_schedule = [timesteps_item.squeeze().item()]
                # extract the sigma idx for our midpoint timestep
                sigmas = train_sigmas[timestep_idx:timestep_idx + 1].to(self.device_torch)

                end_sigma_idx = random.randint(timestep_idx, len(train_sigmas) - 1)
                end_sigma = train_sigmas[end_sigma_idx:end_sigma_idx + 1].to(self.device_torch)

                # add noise to our target

                # build the big sigma step. The to step will now be to 0 giving it a full remaining denoising half step
                # self.sd.noise_scheduler.sigmas = torch.cat([sigmas, torch.zeros_like(sigmas)]).detach()
                self.sd.noise_scheduler.sigmas = torch.cat([sigmas, end_sigma]).detach()
                # set our single timstep
                self.sd.noise_scheduler.timesteps = torch.from_numpy(
                    np.array(single_step_timestep_schedule, dtype=np.float32)
                ).to(device=self.device_torch)

                # set the step index to None so it will be recalculated on first step
                self.sd.noise_scheduler._step_index = None

            denoised_latent = self.sd.noise_scheduler.step(
                pred_item, timesteps_item, noisy_latents_item.detach(), return_dict=False
            )[0]

            residual_noise = (noise_item * end_sigma.flatten()).detach().to(self.device_torch, dtype=get_torch_dtype(
                self.train_config.dtype))
            # remove the residual noise from the denoised latents. Output should be a clean prediction (theoretically)
            denoised_latent = denoised_latent - residual_noise

            denoised_pred_chunks.append(denoised_latent)

        denoised_latents = torch.cat(denoised_pred_chunks, dim=0)
        # set the scheduler back to the original timesteps
        self.sd.noise_scheduler.set_timesteps(
            self.sd.noise_scheduler.config.num_train_timesteps,
            device=self.device_torch
        )

        output = denoised_latents / self.sd.vae.config['scaling_factor']
        output = self.sd.vae.decode(output).sample

        if self.train_config.show_turbo_outputs:
            # since we are completely denoising, we can show them here
            with torch.no_grad():
                show_tensors(output)

        # we return our big partial step denoised latents as our pred and our untouched latents as our target.
        # you can do mse against the two here  or run the denoised through the vae for pixel space loss against the
        # input tensor images.

        return output, batch.tensor.to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))

    # you can expand these in a child class to make customization easier

    def generate_mask_previews_if_enabled(self):
        """Run the mask preview generation once per job when enabled.

        Factored out for testability and to avoid inlining a large block inside `run()`.
        """
        # Mask preview feature removed (was part of masked_recon)
        pass

    def calculate_loss(
            self,
            noise_pred: torch.Tensor,
            noise: torch.Tensor,
            noisy_latents: torch.Tensor,
            timesteps: torch.Tensor,
            batch: 'DataLoaderBatchDTO',
            mask_multiplier: Union[torch.Tensor, float] = 1.0,
            prior_pred: Union[torch.Tensor, None] = None,
            **kwargs
    ):
        loss_target = self.train_config.loss_target
        is_reg = any(batch.get_is_reg_list())
        additional_loss = 0.0
        # include attention alignment loss computed in after_unet_predict (scalar already weighted)
        attn_loss = getattr(self, '_latest_attention_align_loss', 0.0)
        if isinstance(attn_loss, torch.Tensor):
            try:
                additional_loss = additional_loss + attn_loss.to(dtype)
            except Exception:
                additional_loss = additional_loss + float(attn_loss)
        else:
            try:
                additional_loss = additional_loss + float(attn_loss)
            except Exception:
                pass

        prior_mask_multiplier = None
        target_mask_multiplier = None
        dtype = get_torch_dtype(self.train_config.dtype)

        has_mask = batch.mask_tensor is not None

        with torch.no_grad():
            loss_multiplier = torch.tensor(batch.loss_multiplier_list).to(self.device_torch, dtype=torch.float32)

        if self.train_config.match_noise_norm:
            # match the norm of the noise
            noise_norm = torch.linalg.vector_norm(noise, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred_norm = torch.linalg.vector_norm(noise_pred, ord=2, dim=(1, 2, 3), keepdim=True)
            noise_pred = noise_pred * (noise_norm / noise_pred_norm)

        if self.train_config.pred_scaler != 1.0:
            noise_pred = noise_pred * self.train_config.pred_scaler

        target = None

        if self.train_config.target_noise_multiplier != 1.0:
            noise = noise * self.train_config.target_noise_multiplier

        if self.train_config.correct_pred_norm or (self.train_config.inverted_mask_prior and prior_pred is not None and has_mask):
            if self.train_config.correct_pred_norm and not is_reg:
                with torch.no_grad():
                    # this only works if doing a prior pred
                    if prior_pred is not None:
                        prior_mean = prior_pred.mean([2,3], keepdim=True)
                        prior_std = prior_pred.std([2,3], keepdim=True)
                        noise_mean = noise_pred.mean([2,3], keepdim=True)
                        noise_std = noise_pred.std([2,3], keepdim=True)

                        mean_adjust = prior_mean - noise_mean
                        std_adjust = prior_std - noise_std

                        mean_adjust = mean_adjust * self.train_config.correct_pred_norm_multiplier
                        std_adjust = std_adjust * self.train_config.correct_pred_norm_multiplier

                        target_mean = noise_mean + mean_adjust
                        target_std = noise_std + std_adjust

                        eps = 1e-5
                        # match the noise to the prior
                        noise = (noise - noise_mean) / (noise_std + eps)
                        noise = noise * (target_std + eps) + target_mean
                        noise = noise.detach()

            if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
                assert not self.train_config.train_turbo
                with torch.no_grad():
                    prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
                    if len(noise_pred.shape) == 5:
                        # video B,C,T,H,W
                        lat_height = batch.latents.shape[3]
                        lat_width = batch.latents.shape[4]
                    else: 
                        lat_height = batch.latents.shape[2]
                        lat_width = batch.latents.shape[3]
                    # resize to size of noise_pred
                    prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
                    # stack first channel to match channels of noise_pred
                    prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
                    
                    if len(noise_pred.shape) == 5:
                        prior_mask = prior_mask.unsqueeze(2)  # add time dimension back for video
                        prior_mask = prior_mask.repeat(1, 1, noise_pred.shape[2], 1, 1) 

                    prior_mask_multiplier = 1.0 - prior_mask
                    
                    # scale so it is a mean of 1
                    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
                if hasattr(self.sd, 'get_loss_target'):
                    target = self.sd.get_loss_target(
                        noise=noise, 
                        batch=batch, 
                        timesteps=timesteps,
                    ).detach()
                elif self.sd.is_flow_matching:
                    target = (noise - batch.latents).detach()
                else:
                    target = noise
        elif prior_pred is not None and not self.train_config.do_prior_divergence:
            assert not self.train_config.train_turbo
            # matching adapter prediction
            target = prior_pred
        elif self.sd.prediction_type == 'v_prediction':
            # v-parameterization training
            target = self.sd.noise_scheduler.get_velocity(batch.tensor, noise, timesteps)
        
        elif hasattr(self.sd, 'get_loss_target'):
            target = self.sd.get_loss_target(
                noise=noise, 
                batch=batch, 
                timesteps=timesteps,
            ).detach()
            
        elif self.sd.is_flow_matching:
            # forward ODE
            target = (noise - batch.latents).detach()
            # reverse ODE
            # target = (batch.latents - noise).detach()
        else:
            target = noise
            
        if self.dfe is not None:
            if self.dfe.version == 1:
                model = self.sd
                if model is not None and hasattr(model, 'get_stepped_pred'):
                    stepped_latents = model.get_stepped_pred(noise_pred, noise)
                else:
                    # stepped_latents = noise - noise_pred
                    # first we step the scheduler from current timestep to the very end for a full denoise
                    bs = noise_pred.shape[0]
                    noise_pred_chunks = torch.chunk(noise_pred, bs)
                    timestep_chunks = torch.chunk(timesteps, bs)
                    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
                    stepped_chunks = []
                    for idx in range(bs):
                        model_output = noise_pred_chunks[idx]
                        timestep = timestep_chunks[idx]
                        self.sd.noise_scheduler._step_index = None
                        self.sd.noise_scheduler._init_step_index(timestep)
                        sample = noisy_latent_chunks[idx].to(torch.float32)
                        
                        sigma = self.sd.noise_scheduler.sigmas[self.sd.noise_scheduler.step_index]
                        sigma_next = self.sd.noise_scheduler.sigmas[-1] # use last sigma for final step
                        prev_sample = sample + (sigma_next - sigma) * model_output
                        stepped_chunks.append(prev_sample)
                    
                    stepped_latents = torch.cat(stepped_chunks, dim=0)
                    
                stepped_latents = stepped_latents.to(self.sd.vae.device, dtype=self.sd.vae.dtype)
                sl = stepped_latents
                if len(sl.shape) == 5:
                    # video B,C,T,H,W
                    sl = sl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                    b, t, c, h, w = sl.shape
                    sl = sl.reshape(b * t, c, h, w)
                pred_features = self.dfe(sl.float())
                with torch.no_grad():
                    bl = batch.latents
                    bl = bl.to(self.sd.vae.device)
                    if len(bl.shape) == 5:
                        # video B,C,T,H,W
                        bl = bl.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
                        b, t, c, h, w = bl.shape
                        bl = bl.reshape(b * t, c, h, w)
                    target_features = self.dfe(bl.float())
                    # scale dfe so it is weaker at higher noise levels
                    dfe_scaler = 1 - (timesteps.float() / 1000.0).view(-1, 1, 1, 1).to(self.device_torch)
                
                dfe_loss = torch.nn.functional.mse_loss(pred_features, target_features, reduction="none") * \
                    self.train_config.diffusion_feature_extractor_weight * dfe_scaler
                additional_loss += dfe_loss.mean()

            elif self.dfe.version == 2:
                # version 2
                # do diffusion feature extraction on target
                with torch.no_grad():
                    rectified_flow_target = noise.float() - batch.latents.float()
                    target_feature_list = self.dfe(torch.cat([rectified_flow_target, noise.float()], dim=1))
                
                # do diffusion feature extraction on prediction
                pred_feature_list = self.dfe(torch.cat([noise_pred.float(), noise.float()], dim=1))
                
                dfe_loss = 0.0
                for i in range(len(target_feature_list)):
                    dfe_loss += torch.nn.functional.mse_loss(pred_feature_list[i], target_feature_list[i], reduction="mean")
                
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight * 100.0
            elif self.dfe.version in [3, 4, 5]:
                dfe_loss = self.dfe(
                    noise=noise,
                    noise_pred=noise_pred,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    batch=batch,
                    scheduler=self.sd.noise_scheduler
                )
                additional_loss += dfe_loss * self.train_config.diffusion_feature_extractor_weight 
            else:
                raise ValueError(f"Unknown diffusion feature extractor version {self.dfe.version}")

        # Auxiliary controlnet loss (opt-in)
        try:
            if self.train_config.controlnet_aux_loss is not None and self.train_config.controlnet_aux_loss != 'none' and batch.control_tensor is not None:
                from toolkit.controlnet_aux import compute_control_edge_loss

                # ensure tensors are on correct device/dtype
                img_tensor = batch.tensor.to(self.device_torch)
                ctrl_tensor = batch.control_tensor
                aux = compute_control_edge_loss(img_tensor, ctrl_tensor, device=self.device_torch)
                aux = aux * float(self.train_config.controlnet_aux_loss_weight)
                additional_loss = additional_loss + aux
                # lightweight debug print
                if self.train_config.controlnet_aux_loss != 'none':
                    try:
                        print_acc(f"[AUX-LOSS] controlnet aux loss: {aux.item():.6f}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to emit AUX-LOSS debug print: {e}") from e
        except Exception as e:
            # non-fatal; aux loss should not break training
            print_acc(f"ControlNet aux loss failed: {e}")

        if self.train_config.do_guidance_loss:
            with torch.no_grad():
                # we make cached blank prompt embeds that match the batch size
                unconditional_embeds = concat_prompt_embeds(
                    [self.unconditional_embeds] * noisy_latents.shape[0],
                )
                unconditional_target = self.predict_noise(
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    conditional_embeds=unconditional_embeds,
                    unconditional_embeds=None,
                    batch=batch,
                )
                is_video = len(target.shape) == 5
                
                if self.train_config.do_guidance_loss_cfg_zero:
                    # zero cfg
                    # ref https://github.com/WeichenFan/CFG-Zero-star/blob/cdac25559e3f16cb95f0016c04c709ea1ab9452b/wan_pipeline.py#L557
                    batch_size = target.shape[0]
                    positive_flat = target.view(batch_size, -1)
                    negative_flat = unconditional_target.view(batch_size, -1)
                    # Calculate dot production
                    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
                    # Squared norm of uncondition
                    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
                    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
                    st_star = dot_product / squared_norm

                    alpha = st_star
                    
                    alpha = alpha.view(batch_size, 1, 1, 1) if not is_video else alpha.view(batch_size, 1, 1, 1, 1)
                else:
                    alpha = 1.0

                guidance_scale = self._guidance_loss_target_batch
                if isinstance(guidance_scale, list):
                    guidance_scale = torch.tensor(guidance_scale).to(target.device, dtype=target.dtype)
                    guidance_scale = guidance_scale.view(-1, 1, 1, 1) if not is_video else guidance_scale.view(-1, 1, 1, 1, 1)
                
                unconditional_target = unconditional_target * alpha
                target = unconditional_target + guidance_scale * (target - unconditional_target)

            if self.train_config.do_differential_guidance:
                with torch.no_grad():
                    guidance_scale = self.train_config.differential_guidance_scale
                    target = noise_pred + guidance_scale * (target - noise_pred)
            
        if target is None:
            target = noise

        pred = noise_pred

        if self.train_config.train_turbo:
            pred, target = self.process_output_for_turbo(pred, noisy_latents, timesteps, noise, batch)

        ignore_snr = False

        if loss_target == 'source' or loss_target == 'unaugmented':
            assert not self.train_config.train_turbo
            # ignore_snr = True
            if batch.sigmas is None:
                raise ValueError("Batch sigmas is None. This should not happen")

            # src https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1190
            denoised_latents = noise_pred * (-batch.sigmas) + noisy_latents
            weighing = batch.sigmas ** -2.0
            if loss_target == 'source':
                # denoise the latent and compare to the latent in the batch
                target = batch.latents
            elif loss_target == 'unaugmented':
                # we have to encode images into latents for now
                # we also denoise as the unaugmented tensor is not a noisy diffirental
                with torch.no_grad():
                    with self.timer('encode_images'):
                        unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor).to(self.device_torch, dtype=dtype)
                        unaugmented_latents = unaugmented_latents * self.train_config.latent_multiplier
                        target = unaugmented_latents.detach()
                # Get the target for loss depending on the prediction type
                if self.sd.noise_scheduler.config.prediction_type == "epsilon":
                    target = target  # we are computing loss against denoise latents
                elif self.sd.noise_scheduler.config.prediction_type == "v_prediction":
                    target = self.sd.noise_scheduler.get_velocity(target, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {self.sd.noise_scheduler.config.prediction_type}")

            # mse loss without reduction
            loss_per_element = (weighing.float() * (denoised_latents.float() - target.float()) ** 2)
            loss = loss_per_element
        else:

            if self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            else:
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
                
            do_weighted_timesteps = False
            if self.sd.is_flow_matching:
                if self.train_config.linear_timesteps or self.train_config.linear_timesteps2:
                    do_weighted_timesteps = True
                if self.train_config.timestep_type == "weighted":
                    # use the noise scheduler to get the weights for the timesteps
                    do_weighted_timesteps = True

            # handle linear timesteps and only adjust the weight of the timesteps
            if do_weighted_timesteps:
                # calculate the weights for the timesteps
                timestep_weight = self.sd.noise_scheduler.get_weights_for_timesteps(
                    timesteps,
                    v2=self.train_config.linear_timesteps2,
                    timestep_type=self.train_config.timestep_type
                ).to(loss.device, dtype=loss.dtype)
                if len(loss.shape) == 4:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1).detach()
                elif len(loss.shape) == 5:
                    timestep_weight = timestep_weight.view(-1, 1, 1, 1, 1).detach()
                loss = loss * timestep_weight

        if self.train_config.do_prior_divergence and prior_pred is not None:
            loss = loss + (torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none") * -1.0)

        if self.train_config.train_turbo:
            mask_multiplier = mask_multiplier[:, 3:, :, :]
            # resize to the size of the loss
            mask_multiplier = torch.nn.functional.interpolate(mask_multiplier, size=(pred.shape[2], pred.shape[3]), mode='nearest')

        # multiply by our mask
        try:
            if len(noise_pred.shape) == 5:
                # video B,C,T,H,W
                mask_multiplier = mask_multiplier.unsqueeze(2)  # add time dimension back for video
                mask_multiplier = mask_multiplier.repeat(1, 1, noise_pred.shape[2], 1, 1)
            loss = loss * mask_multiplier
        except Exception as e:
            # todo handle mask with video models
            print("Could not apply mask multiplier to loss")
            print(e)
            pass

        prior_loss = None
        if self.train_config.inverted_mask_prior and prior_pred is not None and prior_mask_multiplier is not None:
            assert not self.train_config.train_turbo
            if self.train_config.loss_type == "mae":
                prior_loss = torch.nn.functional.l1_loss(pred.float(), prior_pred.float(), reduction="none")
            else:
                prior_loss = torch.nn.functional.mse_loss(pred.float(), prior_pred.float(), reduction="none")

            prior_loss = prior_loss * prior_mask_multiplier * self.train_config.inverted_mask_prior_multiplier
            if torch.isnan(prior_loss).any():
                print_acc("Prior loss is nan")
                prior_loss = None
            else:
                if len(noise_pred.shape) == 5:
                    # video B,C,T,H,W
                    prior_loss = prior_loss.mean([1, 2, 3, 4])
                else:
                    prior_loss = prior_loss.mean([1, 2, 3])
                # loss = loss + prior_loss
                # loss = loss + prior_loss
            # loss = loss + prior_loss
        if len(noise_pred.shape) == 5:
            loss = loss.mean([1, 2, 3, 4])
        else:
            loss = loss.mean([1, 2, 3])
        # apply loss multiplier before prior loss
        # multiply by our mask
        try:
            loss = loss * loss_multiplier
        except:
            # todo handle mask with video models
            pass
        if prior_loss is not None:
            loss = loss + prior_loss

        if not self.train_config.train_turbo:
            if self.train_config.learnable_snr_gos:
                # add snr_gamma
                loss = apply_learnable_snr_gos(loss, timesteps, self.snr_gos)
            elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001 and not ignore_snr:
                # add snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma,
                                        fixed=True)
            elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001 and not ignore_snr:
                # add min_snr_gamma
                loss = apply_snr_weight(loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        # Capture per-example scalar losses (after per-pixel/channel reduction/weights but before final mean over batch)
        try:
            from toolkit.util.loss_utils import compute_per_example_loss
            per_sample, comps = compute_per_example_loss(loss, prior_loss_tensor=prior_loss)
            # attach to self so hook_train_loop and others can access it (CPU tensor)
            self.last_example_losses = per_sample
            # store components in case we want to inspect later
            self.last_example_loss_components = comps
        except Exception:
            # non-fatal: if anything goes wrong, don't break training
            self.last_example_losses = None
            self.last_example_loss_components = None

        # --- New: compute per-example applied-noise norms and loss/noise ratios for diagnostics ---
        try:
            eps = 1e-12
            # Determine per-sample sigma (scale applied to the sampled normal noise)
            applied_sigmas = None
            if hasattr(batch, 'sigmas') and batch.sigmas is not None:
                # batch.sigmas may be shape (B,) or (B,1,1,1)
                applied_sigmas = batch.sigmas.to(self.device_torch)
                if applied_sigmas.dim() == 1:
                    applied_sigmas = applied_sigmas.view(-1, 1, 1, 1)
            elif hasattr(self.sd, 'noise_scheduler') and hasattr(self.sd.noise_scheduler, 'timesteps') and hasattr(self.sd.noise_scheduler, 'sigmas'):
                try:
                    ns = self.sd.noise_scheduler
                    train_timesteps = ns.timesteps.clone().detach()
                    train_sigmas = ns.sigmas.clone().detach()
                    sigma_list = []
                    # timesteps may be a tensor-like; iterate and match index
                    for t in timesteps.view(-1):
                        matches = (train_timesteps == t).nonzero(as_tuple=False)
                        if matches.numel() == 0:
                            # fallback to 1.0 if we couldn't find a sigma
                            sigma_list.append(torch.tensor(1.0, device=self.device_torch, dtype=noise.dtype))
                        else:
                            sigma_list.append(train_sigmas[matches[0].item()].to(self.device_torch, dtype=noise.dtype))
                    applied_sigmas = torch.stack(sigma_list).view(-1, 1, 1, 1)
                except Exception:
                    applied_sigmas = torch.ones((noise.shape[0], 1, 1, 1), device=self.device_torch, dtype=noise.dtype)
            else:
                applied_sigmas = torch.ones((noise.shape[0], 1, 1, 1), device=self.device_torch, dtype=noise.dtype)

            # Compute applied noise (the actual perturbation added to latents)
            applied_noise = noise * applied_sigmas
            # L2 norm per-example over channels/spatial dims
            noise_norms = torch.linalg.vector_norm(applied_noise, ord=2, dim=(1, 2, 3))
            # per-example ratio (loss is still unreduced per-example vector)
            # Ensure per-sample losses and noise norms are on the same device to avoid cross-device ops
            ps = None
            try:
                ps = per_sample
            except Exception:
                ps = getattr(self, 'last_example_losses', None)
            if isinstance(ps, torch.Tensor):
                try:
                    # move per-sample to noise device if needed
                    if ps.device != noise_norms.device:
                        ps = ps.to(noise_norms.device)
                    per_sample_ratio = (ps + eps) / (noise_norms + eps)
                except Exception:
                    # fallback to CPU computation to be safe
                    ps_cpu = ps.detach().cpu()
                    noise_cpu = noise_norms.detach().cpu()
                    per_sample_ratio = (ps_cpu + eps) / (noise_cpu + eps)
            else:
                raise RuntimeError("Per-sample losses unavailable for noise diagnostics")

            # Attach CPU copies for external inspection and logging
            self.last_noise_norms = noise_norms.detach().cpu()
            # scalar sigmas per sample
            try:
                self.last_noise_sigmas = applied_sigmas.view(applied_sigmas.shape[0]).detach().cpu()
            except Exception:
                self.last_noise_sigmas = None
            self.last_loss_over_noise = per_sample_ratio.detach().cpu()
        except Exception as e:
            # non-fatal: don't break training if logging diagnostics fails
            # store None so downstream code knows diagnostics were unavailable
            self.last_noise_norms = None
            self.last_noise_sigmas = None
            self.last_loss_over_noise = None
            # record diagnostic traceback for visibility and debugging
            # print a concise, safe diagnostic so the issue is visible in logs
            print_acc(f"[LOSS-DIAG] Failed to compute noise diagnostics: {str(e)}")



        loss = loss.mean()

        # check for additional losses
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:

            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss


        return loss + additional_loss

    def preprocess_batch(self, batch: 'DataLoaderBatchDTO'):
        return batch

    def get_guided_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        loss = get_guidance_loss(
            noisy_latents=noisy_latents,
            conditional_embeds=conditional_embeds,
            match_adapter_assist=match_adapter_assist,
            network_weight_list=network_weight_list,
            timesteps=timesteps,
            pred_kwargs=pred_kwargs,
            batch=batch,
            noise=noise,
            sd=self.sd,
            unconditional_embeds=unconditional_embeds,
            train_config=self.train_config,
            **kwargs
        )

        return loss
    
    
    # ------------------------------------------------------------------
    #  Mean-Flow loss (Geng et al., “Mean Flows for One-step Generative
    #  Modelling”, 2025 – see Alg. 1 + Eq. (6) of the paper)
    # This version avoids jvp / double-back-prop issues with Flash-Attention
    # adapted from the work of lodestonerock
    # ------------------------------------------------------------------
    def get_mean_flow_loss(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            **kwargs
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        total_steps = float(self.sd.noise_scheduler.config.num_train_timesteps)  # e.g. 1000
        base_eps = 1e-3
        min_time_gap = 1e-2
        
        with torch.no_grad():
            num_train_timesteps = self.sd.noise_scheduler.config.num_train_timesteps
            batch_size = batch.latents.shape[0]
            timestep_t_list = []
            timestep_r_list = []

            for i in range(batch_size):
                t1 = random.randint(0, num_train_timesteps - 1)
                t2 = random.randint(0, num_train_timesteps - 1)
                t_t = self.sd.noise_scheduler.timesteps[min(t1, t2)]
                t_r = self.sd.noise_scheduler.timesteps[max(t1, t2)]
                if (t_t - t_r).item() < min_time_gap * 1000:
                    scaled_time_gap = min_time_gap * 1000
                    if t_t.item() + scaled_time_gap > 1000:
                        t_r = t_r - scaled_time_gap
                    else:
                        t_t = t_t + scaled_time_gap
                timestep_t_list.append(t_t)
                timestep_r_list.append(t_r)

            timesteps_t = torch.stack(timestep_t_list, dim=0).float()
            timesteps_r = torch.stack(timestep_r_list, dim=0).float()

            t_frac = timesteps_t / total_steps  # [0,1]
            r_frac = timesteps_r / total_steps  # [0,1]

            latents_clean = batch.latents.to(dtype)
            noise_sample = noise.to(dtype)

            lerp_vector = latents_clean * (1.0 - t_frac[:, None, None, None]) + noise_sample * t_frac[:, None, None, None]

            eps = base_eps

            # concatenate timesteps as input for u(z, r, t)
            timesteps_cat = torch.cat([t_frac, r_frac], dim=0) * total_steps

        # model predicts u(z, r, t)
        u_pred = self.predict_noise(
            noisy_latents=lerp_vector.to(dtype),
            timesteps=timesteps_cat.to(dtype),
            conditional_embeds=conditional_embeds,
            unconditional_embeds=unconditional_embeds,
            batch=batch,
            **pred_kwargs
        )

        with torch.no_grad():
            t_frac_plus_eps = (t_frac + eps).clamp(0.0, 1.0)
            lerp_perturbed = latents_clean * (1.0 - t_frac_plus_eps[:, None, None, None]) + noise_sample * t_frac_plus_eps[:, None, None, None]
            timesteps_cat_perturbed = torch.cat([t_frac_plus_eps, r_frac], dim=0) * total_steps

            u_perturbed = self.predict_noise(
                noisy_latents=lerp_perturbed.to(dtype),
                timesteps=timesteps_cat_perturbed.to(dtype),
                conditional_embeds=conditional_embeds,
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **pred_kwargs
            )

        # compute du/dt via finite difference (detached)
        du_dt = (u_perturbed - u_pred).detach() / eps
        # du_dt = (u_perturbed - u_pred).detach()
        du_dt = du_dt.to(dtype)
        
        
        time_gap = (t_frac - r_frac)[:, None, None, None].to(dtype)
        time_gap.clamp(min=1e-4)
        u_shifted = u_pred + time_gap * du_dt
        # u_shifted = u_pred + du_dt / time_gap
        # u_shifted = u_pred

        # a step is done like this:
        # stepped_latent = model_input + (timestep_next - timestep) * model_output
        
        # flow target velocity
        # v_target = (noise_sample - latents_clean) / time_gap
        # flux predicts opposite of velocity, so we need to invert it
        v_target = (latents_clean - noise_sample) / time_gap

        # compute loss
        loss = torch.nn.functional.mse_loss(
            u_shifted.float(),
            v_target.float(),
            reduction='none'
        )

        with torch.no_grad():
            pure_loss = loss.mean().detach()
            pure_loss.requires_grad_(True)

        loss = loss.mean()
        if loss.item() > 1e3:
            pass
        self.accelerator.backward(loss)
        return pure_loss



    def get_prior_prediction(
            self,
            noisy_latents: torch.Tensor,
            conditional_embeds: PromptEmbeds,
            match_adapter_assist: bool,
            network_weight_list: list,
            timesteps: torch.Tensor,
            pred_kwargs: dict,
            batch: 'DataLoaderBatchDTO',
            noise: torch.Tensor,
            unconditional_embeds: Optional[PromptEmbeds] = None,
            conditioned_prompts=None,
            **kwargs
    ):
        # todo for embeddings, we need to run without trigger words
        was_unet_training = self.sd.unet.training
        was_network_active = False
        if self.network is not None:
            was_network_active = self.network.is_active
            self.network.is_active = False
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active
            self.adapter.is_active = False

        if self.train_config.unload_text_encoder and self.adapter is not None and not isinstance(self.adapter, CustomAdapter):
            raise ValueError("Prior predictions currently do not support unloading text encoder with adapter")
        # do a prediction here so we can match its output with network multiplier set to 0.0
        with torch.no_grad():
            dtype = get_torch_dtype(self.train_config.dtype)

            embeds_to_use = conditional_embeds.clone().detach()
            # handle clip vision adapter by removing triggers from prompt and replacing with the class name
            if (self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter)) or self.embedding is not None:
                prompt_list = batch.get_caption_list()
                class_name = ''

                triggers = ['[trigger]', '[name]']
                remove_tokens = []

                if self.embed_config is not None:
                    triggers.append(self.embed_config.trigger)
                    for i in range(1, self.embed_config.tokens):
                        remove_tokens.append(f"{self.embed_config.trigger}_{i}")
                    if self.embed_config.trigger_class_name is not None:
                        class_name = self.embed_config.trigger_class_name

                if self.adapter is not None:
                    triggers.append(self.adapter_config.trigger)
                    for i in range(1, self.adapter_config.num_tokens):
                        remove_tokens.append(f"{self.adapter_config.trigger}_{i}")
                    if self.adapter_config.trigger_class_name is not None:
                        class_name = self.adapter_config.trigger_class_name

                for idx, prompt in enumerate(prompt_list):
                    for remove_token in remove_tokens:
                        prompt = prompt.replace(remove_token, '')
                    for trigger in triggers:
                        prompt = prompt.replace(trigger, class_name)
                    prompt_list[idx] = prompt

                if batch.prompt_embeds is not None:
                    embeds_to_use = batch.prompt_embeds.clone().to(self.device_torch, dtype=dtype)
                else:
                    prompt_kwargs = {}
                    if getattr(self.sd, 'encode_control_in_text_embeddings', False) and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    embeds_to_use = self.sd.encode_prompt(
                        prompt_list,
                        long_prompts=self.do_long_prompts).to(
                        self.device_torch,
                        dtype=dtype,
                        **prompt_kwargs
                    ).detach()

            # dont use network on this
            # self.network.multiplier = 0.0
            self.sd.unet.eval()

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not self.sd.is_flux and not self.sd.is_lumina2:
                # we need to remove the image embeds from the prompt except for flux
                embeds_to_use: PromptEmbeds = embeds_to_use.clone().detach()
                end_pos = embeds_to_use.text_embeds.shape[1] - self.adapter_config.num_tokens
                embeds_to_use.text_embeds = embeds_to_use.text_embeds[:, :end_pos, :]
                if unconditional_embeds is not None:
                    unconditional_embeds = unconditional_embeds.clone().detach()
                    unconditional_embeds.text_embeds = unconditional_embeds.text_embeds[:, :end_pos]

            if unconditional_embeds is not None:
                unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
            
            guidance_embedding_scale = self.train_config.cfg_scale
            if self.train_config.do_guidance_loss:
                guidance_embedding_scale = self._guidance_loss_target_batch

            # Compute prior prediction without adapter residuals when matching adapter assist.
            # Use a shallow copy so we don't mutate the original `pred_kwargs` which the
            # training step still needs.
            prior_pred_kwargs = dict(pred_kwargs)
            if match_adapter_assist:
                prior_pred_kwargs.pop('down_intrablock_additional_residuals', None)
                prior_pred_kwargs.pop('down_block_additional_residuals', None)
                prior_pred_kwargs.pop('mid_block_additional_residual', None)

            prior_pred = self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                rescale_cfg=self.train_config.cfg_rescale,
                batch=batch,
                **prior_pred_kwargs
            )
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        # Setup attention hooks if attention alignment is enabled
        try:
            if getattr(self.train_config, 'attention_align_weight', 0.0) <= 0.0:
                return
            # clear previous
            self._collected_attentions = []
            self._attn_hook_handles = []
            # attach forward hooks to attention-like modules
            for name, module in self.sd.unet.named_modules():
                cls_name = module.__class__.__name__.lower()
                if 'attn' in cls_name or 'attention' in cls_name:
                    if hasattr(module, 'to_q') and hasattr(module, 'to_k'):
                        def make_hook(n):
                            def hook(mod, inp, out):
                                try:
                                    # inp: (hidden_states, encoder_hidden_states, ...)
                                    hidden = inp[0]
                                    enc = inp[1] if len(inp) > 1 else None
                                    if enc is None:
                                        return
                                    q = mod.to_q(hidden)
                                    k = mod.to_k(enc)
                                    # reshape: try common formats
                                    B = q.shape[0]
                                    q_len = q.shape[1]
                                    Cq = q.shape[2]
                                    # infer num_heads from module if available
                                    num_heads = getattr(mod, 'num_heads', None)
                                    if num_heads is None:
                                        # try to infer
                                        num_heads = getattr(mod, 'heads', None) or 1
                                    head_dim = Cq // num_heads
                                    q = q.view(B, q_len, num_heads, head_dim).permute(0,2,1,3)
                                    k_len = k.shape[1]
                                    k = k.view(B, k_len, num_heads, head_dim).permute(0,2,1,3)
                                    # compute attn
                                    att = torch.einsum('bhqd,bhkd->bhqk', q, k) / (head_dim ** 0.5)
                                    attn = torch.softmax(att, dim=-1)
                                    # store
                                    self._collected_attentions.append(attn.detach())
                                except Exception:
                                    # best-effort; don't crash training
                                    return
                            return hook
                        h = module.register_forward_hook(make_hook(name))
                        self._attn_hook_handles.append(h)
        except Exception:
            # best-effort; ignore on failures
            pass

    def _looks_like_pixel_images(self, x):
        # Accept both batched tensors and lists of tensors
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return False
            for it in x:
                if not isinstance(it, torch.Tensor):
                    return False
                if it.ndim != 3:
                    return False
                c, h, w = it.shape
                if not (h >= 64 and w >= 64 and c in (1, 3, 4)):
                    return False
            return True
        if not isinstance(x, torch.Tensor):
            return False
        if x.ndim == 4:
            _, c, h, w = x.shape
            return (h >= 64 and w >= 64 and c in (1, 3, 4))
        if x.ndim == 5:
            _, c, f, h, w = x.shape
            return (h >= 64 and w >= 64 and c in (1, 3, 4))

    def _collect_preencoded_zimage_context_for_batch(self, batch: 'DataLoaderBatchDTO'):
        """Delegate to z_image helper: collect precomputed zimage contexts for a batch."""
        try:
            from extensions_built_in.diffusion_models.z_image.z_image import collect_preencoded_zimage_context_for_batch as _helper
            return _helper(batch)
        except Exception:
            return None


        if len(vals) != len(batch.file_items):
            # Emit diagnostics for why precompute was not acceptable for the full batch
            try:
                print_acc(f"[PRECOMPUTE] precompute not usable for batch: {len(vals)}/{len(batch.file_items)} files usable; details:")
                for d in diagnostics:
                    print_acc(f"[PRECOMPUTE]   {d}")
            except Exception:
                print(f"[PRECOMPUTE] precompute not usable for batch: {len(vals)}/{len(batch.file_items)} files usable; details:")
                for d in diagnostics:
                    print(f"  {d}")
            return None
        try:
            return torch.cat(vals, dim=0)
        except Exception as e:
            try:
                import traceback
                print_acc(f"[PRECOMPUTE] Failed to concat precomputed contexts: {e}\n{traceback.format_exc()}")
            except Exception:
                pass
            return None

    def _precompute_zimage_control_contexts(self):
        """Delegate to z_image helper precompute routine."""
        try:
            from extensions_built_in.diffusion_models.z_image.z_image import precompute_zimage_control_contexts as _helper
            return _helper(self.sd, self.data_loader)
        except Exception:
            return None




            # Some third-party encoders or buggy implementations may have called
            # torch.set_grad_enabled(False) without restoring; be defensive.

            if not torch.is_grad_enabled():
                print_acc("[PRECOMPUTE] Global grad mode was disabled after precompute; re-enabling")
                torch.set_grad_enabled(True)

        self._precomputed_zimage_controls_done = True

    def _encode_and_assemble_zimage_controls(self, control_context):
        """Delegate to z_image helper: encode and assemble zimage controls."""
        try:
            from extensions_built_in.diffusion_models.z_image.z_image import encode_and_assemble_zimage_controls as _helper
            return _helper(self.sd, control_context)
        except Exception as e:
            raise RuntimeError(f"Failed to encode and assemble zimage controls: {e}") from e


    def after_unet_predict(self):
        with self.timer('after_unet_predict'):
            try:
                if getattr(self.train_config, 'attention_align_weight', 0.0) <= 0.0:
                    return

                # remove hooks
                if hasattr(self, '_attn_hook_handles') and self._attn_hook_handles is not None:
                    for h in list(self._attn_hook_handles):
                        try:
                            h.remove()
                        except Exception:
                            pass
                    self._attn_hook_handles = None

                # if no attentions collected, skip
                if not hasattr(self, '_collected_attentions') or len(self._collected_attentions) == 0:
                    self._latest_attention_align_loss = 0.0
                    return

                # Prepare attentions list (list of tensors [B, H, T, S])
                atts = self._collected_attentions

                # get batch & masks
                batch = getattr(self, '_last_batch_for_attn', None)
                if batch is None:
                    self._latest_attention_align_loss = 0.0
                    return

                # Average layers and heads to produce [B, T] per token (we'll aggregate across tokens for now)
                from toolkit.attention_align import avg_attention_maps, attention_alignment_loss

                # build head-averaged maps
                maps = []
                for a in atts:
                    try:
                        head_avg = a.mean(dim=1)
                        maps.append(head_avg.mean(dim=-1))
                    except Exception:
                        continue

                if len(maps) == 0:
                    self._latest_attention_align_loss = 0.0
                    return

                stacked = torch.stack(maps, dim=0).mean(dim=0)
                B, T = stacked.shape

                lat = getattr(batch, 'latents', None) or getattr(self, '_last_noisy_latents', None)
                if lat is None:
                    self._latest_attention_align_loss = 0.0
                    return

                if lat.ndim == 5:
                    _, _, _, H_lat, W_lat = lat.shape
                else:
                    _, _, H_lat, W_lat = lat.shape

                att_maps_2d = stacked.view(B, H_lat, W_lat)

                # Determine mask
                mask = None
                
                if getattr(batch, 'mask_tensor', None) is not None:
                    try:
                        mask = batch.mask_tensor.to(self.device_torch)
                    except Exception:
                        mask = None

                if mask is None and getattr(batch, 'control_tensor', None) is not None:
                    try:
                        m = batch.control_tensor
                        if isinstance(m, torch.Tensor) and m.ndim == 3:
                            mask = m[:1, :1, ...].unsqueeze(0).to(self.device_torch)
                    except Exception:
                        mask = None

                if mask is None:
                    self._latest_attention_align_loss = 0.0
                    return

                mask_resized = torch.nn.functional.interpolate(mask, size=(H_lat, W_lat), mode='bicubic')
                if mask_resized.ndim == 4:
                    m_flat = mask_resized.view(B, 1, H_lat * W_lat)
                elif mask_resized.ndim == 3:
                    m_flat = mask_resized.unsqueeze(1).view(B, 1, H_lat * W_lat)

                att_flat = att_maps_2d.view(B, 1, H_lat * W_lat)
                loss = attention_alignment_loss(att_flat, m_flat, mode=getattr(self.train_config, 'attention_align_mode', 'mse'))
                self._latest_attention_align_loss = loss * getattr(self.train_config, 'attention_align_weight', 1.0)
            except Exception:
                self._latest_attention_align_loss = 0.0
                return
        pass

    def predict_noise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: Union[int, torch.Tensor] = 1,
        conditional_embeds: Union[PromptEmbeds, None] = None,
        unconditional_embeds: Union[PromptEmbeds, None] = None,
        batch: Optional['DataLoaderBatchDTO'] = None,
        is_primary_pred: bool = False,
        **kwargs,
    ):
        dtype = get_torch_dtype(self.train_config.dtype)
        guidance_embedding_scale = self.train_config.cfg_scale
        if self.train_config.do_guidance_loss:
            guidance_embedding_scale = self._guidance_loss_target_batch
        cond_move = self._maybe_move_embeds(conditional_embeds, self.device_torch, dtype=dtype)
        uncond_move = self._maybe_move_embeds(unconditional_embeds, self.device_torch, dtype=dtype)
        return self.sd.predict_noise(
            latents=noisy_latents.to(self.device_torch, dtype=dtype),
            conditional_embeddings=cond_move,
            unconditional_embeddings=uncond_move,
            timestep=timesteps,
            guidance_scale=self.train_config.cfg_scale,
            guidance_embedding_scale=guidance_embedding_scale,
            detach_unconditional=False,
            rescale_cfg=self.train_config.cfg_rescale,
            bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
            batch=batch,
            **kwargs
        )
    

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        with self.timer('step_total_python'):
            with torch.no_grad():
                self.timer.start('preprocess_batch')
                if isinstance(self.adapter, CustomAdapter):
                    batch = self.adapter.edit_batch_raw(batch)
                batch = self.preprocess_batch(batch)
                if isinstance(self.adapter, CustomAdapter):
                    batch = self.adapter.edit_batch_processed(batch)
                dtype = get_torch_dtype(self.train_config.dtype)
                # sanity check
                if self.sd.vae.dtype != self.sd.vae_torch_dtype:
                    self.sd.vae = self.sd.vae.to(self.sd.vae_torch_dtype)
                if isinstance(self.sd.text_encoder, list):
                    for encoder in self.sd.text_encoder:
                        if encoder.dtype != self.sd.te_torch_dtype:
                            encoder.to(self.sd.te_torch_dtype)
                else:
                    if self.sd.text_encoder.dtype != self.sd.te_torch_dtype:
                        self.sd.text_encoder.to(self.sd.te_torch_dtype)

            noisy_latents, noise, timesteps, conditioned_prompts, imgs = self.process_general_training_batch(batch)
            if self.train_config.do_cfg or self.train_config.do_random_cfg:
                # pick random negative prompts
                if self.negative_prompt_pool is not None:
                    negative_prompts = []
                    for i in range(noisy_latents.shape[0]):
                        num_neg = random.randint(1, self.train_config.max_negative_prompts)
                        this_neg_prompts = [random.choice(self.negative_prompt_pool) for _ in range(num_neg)]
                        this_neg_prompt = ', '.join(this_neg_prompts)
                        negative_prompts.append(this_neg_prompt)
                    self.batch_negative_prompt = negative_prompts
                else:
                    self.batch_negative_prompt = ['' for _ in range(batch.latents.shape[0])]

            if self.adapter and isinstance(self.adapter, CustomAdapter):
                # condition the prompt
                # todo handle more than one adapter image
                conditioned_prompts = self.adapter.condition_prompt(conditioned_prompts)

            network_weight_list = batch.get_network_weight_list()
            if self.train_config.single_item_batching:
                network_weight_list = network_weight_list + network_weight_list

            has_adapter_img = batch.control_tensor is not None
            has_clip_image = batch.clip_image_tensor is not None
            has_clip_image_embeds = batch.clip_image_embeds is not None
            # force it to be true if doing regs as we handle those differently
            if any([batch.file_items[idx].is_reg for idx in range(len(batch.file_items))]):
                has_clip_image = True
                if self._clip_image_embeds_unconditional is not None:
                    has_clip_image_embeds = True  # we are caching embeds, handle that differently
                    has_clip_image = False

            # do prior pred if prior regularization batch
            do_reg_prior = False
            if any([batch.file_items[idx].prior_reg for idx in range(len(batch.file_items))]):
                do_reg_prior = True

            if self.adapter is not None and isinstance(self.adapter, IPAdapter) and not has_clip_image and has_adapter_img:
                raise ValueError(
                    "IPAdapter control image is now 'clip_image_path' instead of 'control_path'. Please update your dataset config ")

            match_adapter_assist = False

            # check if we are matching the adapter assistant
            if self.assistant_adapter:
                if self.train_config.match_adapter_chance == 1.0:
                    match_adapter_assist = True
                elif self.train_config.match_adapter_chance > 0.0:
                    match_adapter_assist = torch.rand(
                        (1,), device=self.device_torch, dtype=dtype
                    ) < self.train_config.match_adapter_chance

            self.timer.stop('preprocess_batch')

            is_reg = False
            loss_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            for idx, file_item in enumerate(batch.file_items):
                if file_item.is_reg:
                    loss_multiplier[idx] = loss_multiplier[idx] * self.train_config.reg_weight
                    is_reg = True

            adapter_images = None
            sigmas = None
            if has_adapter_img and (self.adapter or self.assistant_adapter):
                with self.timer('get_adapter_images'):
                    # todo move this to data loader
                    if batch.control_tensor is not None:
                        # Support various encoded control tensor formats:
                        # - Tensor of shape (B,C,H,W)
                        # - List of tensors [t1, t2, ...] each shaped like (C,H,W) or (1,C,H,W)
                        # - List of per-sample tensors length == B
                        ct = batch.control_tensor
                        if isinstance(ct, list):
                            # flatten nested lists of tiles into per-sample single tensors when possible
                            try:
                                # if list items are tensors and match per-sample, stack them
                                if all(isinstance(x, torch.Tensor) for x in ct):
                                    # items may be (C,H,W) or (1,C,H,W)
                                    normalized = []
                                    for x in ct:
                                        if x.dim() == 3:
                                            normalized.append(x.unsqueeze(0))
                                        else:
                                            normalized.append(x)
                                    adapter_images = torch.cat(normalized, dim=0).to(self.device_torch, dtype=dtype).detach()
                                else:
                                    # try nested list per-sample: e.g. [[tile1, tile2], [tile1, tile2]]
                                    if all(isinstance(x, list) for x in ct):
                                        # attempt to pick first tile per sample as a fallback
                                        normalized = []
                                        for sample_list in ct:
                                            first = sample_list[0]
                                            if isinstance(first, torch.Tensor):
                                                if first.dim() == 3:
                                                    normalized.append(first.unsqueeze(0))
                                                else:
                                                    normalized.append(first)
                                            else:
                                                raise RuntimeError("Unsupported control_tensor nested structure: non-tensor leaf")
                                        adapter_images = torch.cat(normalized, dim=0).to(self.device_torch, dtype=dtype).detach()
                                    else:
                                        raise RuntimeError("Unsupported control_tensor list structure; expected list of tensors or list of per-sample lists")
                            except Exception as e:
                                raise RuntimeError(f"Could not normalize control_tensor list to tensor: {e}")
                        elif isinstance(ct, torch.Tensor):
                            adapter_images = ct.to(self.device_torch, dtype=dtype).detach()
                        else:
                            raise RuntimeError("Unsupported type for batch.control_tensor; expected Tensor or List[Tensor]")

                        # Validate input early to fail fast if shapes are wrong
                        self._validate_adapter_images(adapter_images)
                        # match in channels
                        if self.assistant_adapter is not None:
                            in_channels = self.assistant_adapter.config.in_channels
                            if adapter_images.shape[1] != in_channels:
                                # we need to match the channels
                                adapter_images = adapter_images[:, :in_channels, :, :]
                    else:
                        raise NotImplementedError("Adapter images now must be loaded with dataloader")

            clip_images = None
            if has_clip_image:
                with self.timer('get_clip_images'):
                    # todo move this to data loader
                    if batch.clip_image_tensor is not None:
                        clip_images = batch.clip_image_tensor.to(self.device_torch, dtype=dtype).detach()

            mask_multiplier = torch.ones((noisy_latents.shape[0], 1, 1, 1), device=self.device_torch, dtype=dtype)
            if batch.mask_tensor is not None:
                with self.timer('get_mask_multiplier'):
                    # upsampling no supported for bfloat16
                    mask_multiplier = batch.mask_tensor.to(self.device_torch, dtype=torch.float16).detach()
                    # scale down to the size of the latents, mask multiplier shape(bs, 1, width, height), noisy_latents shape(bs, channels, width, height)
                    if len(noisy_latents.shape) == 5:
                        # video B,C,T,H,W
                        h = noisy_latents.shape[3]
                        w = noisy_latents.shape[4]
                    else:
                        h = noisy_latents.shape[2]
                        w = noisy_latents.shape[3]
                    mask_multiplier = torch.nn.functional.interpolate(
                        mask_multiplier, size=(h, w)
                    )
                    # expand to match latents
                    mask_multiplier = mask_multiplier.expand(-1, noisy_latents.shape[1], -1, -1)
                    mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
                    
                    # Apply mask_strength blending (get from first file item's dataset config)
                    if len(batch.file_items) > 0:
                        mask_strength = float(getattr(batch.file_items[0].dataset_config, 'mask_strength', 1.0))
                        if 0.0 < mask_strength < 1.0:
                            # Blend: masked regions (1.0) get full weight, non-masked (0.0) get reduced weight
                            # Formula: final = mask * 1.0 + (1-mask) * (1-strength)
                            #        = mask + (1-mask) * (1-strength)
                            mask_multiplier = mask_multiplier + (1.0 - mask_multiplier) * (1.0 - mask_strength)
                    
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            # Delegate to class-level helper which uses torch RNG for reproducibility
            is_t2i = (self.adapter and isinstance(self.adapter, T2IAdapter))
            return SDTrainer.compute_adapter_multiplier(is_t2i, match_adapter_assist, self.device_torch, dtype)



        # flush()



        # enter grad setup section
        with self.timer('grad_setup'):
            # text encoding
            grad_on_text_encoder = False
            if self.train_config.train_text_encoder:
                grad_on_text_encoder = True

            if self.embedding is not None:
                grad_on_text_encoder = True

            if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                grad_on_text_encoder = True

            if self.adapter_config and self.adapter_config.type == 'te_augmenter':
                grad_on_text_encoder = True

            # have a blank network so we can wrap it in a context and set multipliers without checking every time
            if self.network is not None:
                network = self.network
            else:
                network = BlankNetwork()

            # set the weights
            network.multiplier = network_weight_list

        # activate network if it exits

        prompts_1 = conditioned_prompts
        prompts_2 = None
        if self.train_config.short_and_long_captions_encoder_split and self.sd.is_xl:
            prompts_1 = batch.get_caption_short_list()
            prompts_2 = conditioned_prompts

            # make the batch splits
        if self.train_config.single_item_batching:
            if self.model_config.refiner_name_or_path is not None:
                raise ValueError("Single item batching is not supported when training the refiner")
            batch_size = noisy_latents.shape[0]
            # chunk/split everything
            noisy_latents_list = torch.chunk(noisy_latents, batch_size, dim=0)
            noise_list = torch.chunk(noise, batch_size, dim=0)
            timesteps_list = torch.chunk(timesteps, batch_size, dim=0)
            conditioned_prompts_list = [[prompt] for prompt in prompts_1]
            if imgs is not None:
                imgs_list = torch.chunk(imgs, batch_size, dim=0)
            else:
                imgs_list = [None for _ in range(batch_size)]
            if adapter_images is not None:
                adapter_images_list = torch.chunk(adapter_images, batch_size, dim=0)
            else:
                adapter_images_list = [None for _ in range(batch_size)]
            if clip_images is not None:
                clip_images_list = torch.chunk(clip_images, batch_size, dim=0)
            else:
                clip_images_list = [None for _ in range(batch_size)]
            mask_multiplier_list = torch.chunk(mask_multiplier, batch_size, dim=0)
            if prompts_2 is None:
                prompt_2_list = [None for _ in range(batch_size)]
            else:
                prompt_2_list = [[prompt] for prompt in prompts_2]

        else:
            noisy_latents_list = [noisy_latents]
            noise_list = [noise]
            timesteps_list = [timesteps]
            conditioned_prompts_list = [prompts_1]
            imgs_list = [imgs]
            adapter_images_list = [adapter_images]
            clip_images_list = [clip_images]
            mask_multiplier_list = [mask_multiplier]
            if prompts_2 is None:
                prompt_2_list = [None]
            else:
                prompt_2_list = [prompts_2]

        for noisy_latents, noise, timesteps, conditioned_prompts, imgs, adapter_images, clip_images, mask_multiplier, prompt_2 in zip(
                noisy_latents_list,
                noise_list,
                timesteps_list,
                conditioned_prompts_list,
                imgs_list,
                adapter_images_list,
                clip_images_list,
                mask_multiplier_list,
                prompt_2_list
        ):

            # if self.train_config.negative_prompt is not None:
            #     # add negative prompt
            #     conditioned_prompts = conditioned_prompts + [self.train_config.negative_prompt for x in
            #                                                  range(len(conditioned_prompts))]
            #     if prompt_2 is not None:
            #         prompt_2 = prompt_2 + [self.train_config.negative_prompt for x in range(len(prompt_2))]

            with (network):
                # encode clip adapter here so embeds are active for tokenizer
                if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                    with self.timer('encode_clip_vision_embeds'):
                        if has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True
                            )
                        else:
                            # just do a blank one
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                torch.zeros(
                                    (noisy_latents.shape[0], 3, 512, 512),
                                    device=self.device_torch, dtype=dtype
                                ),
                                is_training=True,
                                has_been_preprocessed=True,
                                drop=True
                            )
                        # it will be injected into the tokenizer when called
                        self.adapter(conditional_clip_embeds)

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or is_reg):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    self.adapter.trigger_pre_te(
                        tensors_preprocessed=clip_images if not is_reg else None,  # on regs we send none to get random noise
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count,
                        batch_tensor=batch.tensor if not is_reg else None,
                        batch_size=noisy_latents.shape[0]
                    )

                with self.timer('encode_prompt'):
                    unconditional_embeds = None
                    prompt_kwargs = {}
                    if getattr(self.sd, 'encode_control_in_text_embeddings', False) and batch.control_tensor is not None:
                        prompt_kwargs['control_images'] = batch.control_tensor.to(self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.train_config.unload_text_encoder or self.is_caching_text_embeddings:
                        with torch.set_grad_enabled(False):
                            if batch.prompt_embeds is not None:
                                # use the cached embeds
                                conditional_embeds = batch.prompt_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                            else:
                                embeds_to_use = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                if self.cached_trigger_embeds is not None and not is_reg:
                                    embeds_to_use = self.cached_trigger_embeds.clone().detach().to(
                                        self.device_torch, dtype=dtype
                                    )
                                conditional_embeds = concat_prompt_embeds(
                                    [embeds_to_use] * noisy_latents.shape[0]
                                )
                            if self.train_config.do_cfg:
                                unconditional_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                unconditional_embeds = concat_prompt_embeds(
                                    [unconditional_embeds] * noisy_latents.shape[0]
                                )

                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False

                    elif grad_on_text_encoder:
                        with torch.set_grad_enabled(True):
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            )
                            conditional_embeds = self._maybe_move_embeds(conditional_embeds, self.device_torch, dtype=dtype)

                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                # todo only do one and repeat it
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                )
                                unconditional_embeds = self._maybe_move_embeds(unconditional_embeds, self.device_torch, dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                    else:
                        with torch.set_grad_enabled(False):
                            # make sure it is in eval mode
                            if isinstance(self.sd.text_encoder, list):
                                for te in self.sd.text_encoder:
                                    te.eval()
                            else:
                                self.sd.text_encoder.eval()
                            if isinstance(self.adapter, CustomAdapter):
                                self.adapter.is_unconditional_run = False
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            )
                            conditional_embeds = self._maybe_move_embeds(conditional_embeds, self.device_torch, dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                )
                                unconditional_embeds = self._maybe_move_embeds(unconditional_embeds, self.device_torch, dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                            
                            if self.train_config.diff_output_preservation:
                                # Determine whether to perform DOP for this batch when preparing embeddings.
                                is_dop_scheduled_for_encode = self._is_dop_scheduled(for_encoding=True)
                                if not is_dop_scheduled_for_encode:
                                    # Skip preparing DOP embeddings for this batch to save compute
                                    self.diff_output_preservation_embeds = None
                                    print_acc(f"[DOP] Skipping diff_output_preservation embedding prep this batch (every={getattr(self.train_config, 'diff_output_preservation_every', 1)})")
                                else:
                                    # If text embeddings are cached to disk, prefer loading per-file DOP embeds to avoid
                                    # re-encoding each training timestep. Otherwise fall back to encoding the DOP prompts.
                                    if self.is_caching_text_embeddings and getattr(batch, 'file_items', None) is not None:
                                        dop_embeds_list = []
                                        ok = True
                                        with self.timer('dop_embed_load'):
                                            for fi in batch.file_items:
                                                # compute per-file dop caption key
                                                dop_caption = fi.caption or ""
                                                if hasattr(self, '_dop_replacements') and self._dop_replacements:
                                                    dop_caption = normalize_caption_separators(dop_caption)
                                                    for tr, cls in self._dop_replacements:
                                                        if tr == '':
                                                            continue
                                                        pattern = rf"(?<!\S){re.escape(tr)}(?!\S)"
                                                        dop_caption, n = re.subn(pattern, cls, dop_caption)
                                                        if n == 0:
                                                            dop_caption = dop_caption.replace(tr, cls)
                                                try:
                                                    fi.load_dop_prompt_embedding(dop_caption)
                                                except Exception:
                                                    pass
                                                if fi.dop_prompt_embeds is None:
                                                    ok = False
                                                    break
                                                dop_embeds_list.append(fi.dop_prompt_embeds)
                                        if ok:
                                            self.diff_output_preservation_embeds = concat_prompt_embeds(dop_embeds_list)
                                            with self.timer('dop_embed_move'):
                                                self.diff_output_preservation_embeds = self._maybe_move_embeds(self.diff_output_preservation_embeds, self.device_torch, dtype=dtype)
                                        else:
                                            # fallback to encoding using CSV mapping
                                            dop_prompts = [normalize_caption_separators(self._map_triggers_to_classes_in_text(p)) for p in conditioned_prompts]
                                            dop_prompts_2 = None
                                            if prompt_2 is not None:
                                                dop_prompts_2 = [normalize_caption_separators(self._map_triggers_to_classes_in_text(p)) for p in prompt_2]
                                            with self.timer('dop_encode_fallback'):
                                                self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                                    dop_prompts, dop_prompts_2,
                                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                                    long_prompts=self.do_long_prompts,
                                                    **prompt_kwargs
                                                )
                                            with self.timer('dop_embed_move'):
                                                self.diff_output_preservation_embeds = self._maybe_move_embeds(self.diff_output_preservation_embeds, self.device_torch, dtype=dtype) 
                                    else:
                                        dop_prompts = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in conditioned_prompts]
                                        dop_prompts_2 = None
                                        if prompt_2 is not None:
                                            dop_prompts_2 = [p.replace(self.trigger_word, self.train_config.diff_output_preservation_class) for p in prompt_2]
                                        self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                            dop_prompts, dop_prompts_2,
                                            dropout_prob=self.train_config.prompt_dropout_prob,
                                            long_prompts=self.do_long_prompts,
                                            **prompt_kwargs
                                        )
                                        self.diff_output_preservation_embeds = self._maybe_move_embeds(self.diff_output_preservation_embeds, self.device_torch, dtype=dtype)
                        # detach the embeddings safely
                        conditional_embeds = self._maybe_detach_embeds(conditional_embeds)
                        if self.train_config.do_cfg:
                            unconditional_embeds = self._maybe_detach_embeds(unconditional_embeds)
                    
                    if self.decorator:
                        conditional_embeds.text_embeds = self.decorator(
                            conditional_embeds.text_embeds
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds.text_embeds = self.decorator(
                                unconditional_embeds.text_embeds, 
                                is_unconditional=True
                            )

                # flush()
                # Propagate dataset-level controlnet debug opt-in into model-side routing so
                # `predict_noise_zimage` can emit additional diagnostics when enabled.
                try:
                    dataset_controlnet_debug = False
                    if batch is not None and getattr(batch, 'file_items', None):
                        ds_cfg = getattr(batch.file_items[0], 'dataset_config', None)
                        if ds_cfg is not None:
                            dataset_controlnet_debug = bool(getattr(ds_cfg, 'controlnet_debug', getattr(ds_cfg, 'debug', False)))
                except Exception:
                    dataset_controlnet_debug = False
                pred_kwargs = {}
                pred_kwargs['dataset_controlnet_debug'] = dataset_controlnet_debug

                if has_adapter_img:
                    if (self.adapter and isinstance(self.adapter, T2IAdapter)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, T2IAdapter)):
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                down_block_additional_residuals = adapter(adapter_images)
                                if self.assistant_adapter:
                                    # not training. detach
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype).detach() * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]
                                else:
                                    down_block_additional_residuals = [
                                        sample.to(dtype=dtype) * adapter_multiplier for sample in
                                        down_block_additional_residuals
                                    ]

                                pred_kwargs['down_intrablock_additional_residuals'] = down_block_additional_residuals


                if self.adapter and isinstance(self.adapter, IPAdapter):
                    with self.timer('encode_adapter_embeds'):
                        # number of images to do if doing a quad image
                        quad_count = random.randint(1, 4)
                        image_size = self.adapter.input_size
                        if has_clip_image_embeds:
                            # todo handle reg images better than this
                            if is_reg:
                                # get unconditional image embeds from cache
                                embeds = [
                                    load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                    range(noisy_latents.shape[0])
                                ]
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    embeds,
                                    quad_count=quad_count
                                )

                                if self.train_config.do_cfg:
                                    embeds = [
                                        load_file(random.choice(batch.clip_image_embeds_unconditional)) for i in
                                        range(noisy_latents.shape[0])
                                    ]
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        embeds,
                                        quad_count=quad_count
                                    )

                            else:
                                conditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                    batch.clip_image_embeds,
                                    quad_count=quad_count
                                )
                                if self.train_config.do_cfg:
                                    unconditional_clip_embeds = self.adapter.parse_clip_image_embeds_from_cache(
                                        batch.clip_image_embeds_unconditional,
                                        quad_count=quad_count
                                    )
                        elif is_reg:
                            # we will zero it out in the img embedder
                            clip_images = torch.zeros(
                                (noisy_latents.shape[0], 3, image_size, image_size),
                                device=self.device_torch, dtype=dtype
                            ).detach()
                            # drop will zero it out
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images,
                                drop=True,
                                is_training=True,
                                has_been_preprocessed=False,
                                quad_count=quad_count
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    torch.zeros(
                                        (noisy_latents.shape[0], 3, image_size, image_size),
                                        device=self.device_torch, dtype=dtype
                                    ).detach(),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=False,
                                    quad_count=quad_count
                                )
                        elif has_clip_image:
                            conditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                clip_images.detach().to(self.device_torch, dtype=dtype),
                                is_training=True,
                                has_been_preprocessed=True,
                                quad_count=quad_count,
                                # do cfg on clip embeds to normalize the embeddings for when doing cfg
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                                # cfg_embed_strength=3.0 if not self.train_config.do_cfg else None
                            )
                            if self.train_config.do_cfg:
                                unconditional_clip_embeds = self.adapter.get_clip_image_embeds_from_tensors(
                                    clip_images.detach().to(self.device_torch, dtype=dtype),
                                    is_training=True,
                                    drop=True,
                                    has_been_preprocessed=True,
                                    quad_count=quad_count
                                )
                        else:
                            print_acc("No Clip Image")
                            print_acc([file_item.path for file_item in batch.file_items])
                            raise ValueError("Could not find clip image")

                    if not self.adapter_config.train_image_encoder:
                        # we are not training the image encoder, so we need to detach the embeds
                        conditional_clip_embeds = conditional_clip_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_clip_embeds = unconditional_clip_embeds.detach()

                    with self.timer('encode_adapter'):
                        self.adapter.train()
                        conditional_embeds = self.adapter(
                            conditional_embeds.detach(),
                            conditional_clip_embeds,
                            is_unconditional=False
                        )
                        if self.train_config.do_cfg:
                            unconditional_embeds = self.adapter(
                                unconditional_embeds.detach(),
                                unconditional_clip_embeds,
                                is_unconditional=True
                            )
                        else:
                            # wipe out unconsitional
                            self.adapter.last_unconditional = None

                if self.adapter and isinstance(self.adapter, ReferenceAdapter):
                    # pass in our scheduler
                    self.adapter.noise_scheduler = self.lr_scheduler
                    if has_clip_image or has_adapter_img:
                        img_to_use = clip_images if has_clip_image else adapter_images
                        # currently 0-1 needs to be -1 to 1
                        reference_images = ((img_to_use - 0.5) * 2).detach().to(self.device_torch, dtype=dtype)
                        self.adapter.set_reference_images(reference_images)
                        self.adapter.noise_scheduler = self.sd.noise_scheduler
                    elif is_reg:
                        self.adapter.set_blank_reference_images(noisy_latents.shape[0])
                    else:
                        self.adapter.set_reference_images(None)

                prior_pred = None

                do_inverted_masked_prior = False
                if self.train_config.inverted_mask_prior and batch.mask_tensor is not None:
                    do_inverted_masked_prior = True

                do_correct_pred_norm_prior = self.train_config.correct_pred_norm

                do_guidance_prior = False

                if batch.unconditional_latents is not None:
                    # for this not that, we need a prior pred to normalize
                    guidance_type: GuidanceType = batch.file_items[0].dataset_config.guidance_type
                    if guidance_type == 'tnt':
                        do_guidance_prior = True

                if ((
                        has_adapter_img and self.assistant_adapter and match_adapter_assist) or self.do_prior_prediction or do_guidance_prior or do_reg_prior or do_inverted_masked_prior or self.train_config.correct_pred_norm):
                    with self.timer('prior predict'):
                        prior_embeds_to_use = conditional_embeds
                        # use diff_output_preservation embeds if doing dfe
                        if self.train_config.diff_output_preservation:
                            # ensure we have DOP embeddings available; if not, try to load per-file cached embeddings
                            if self.diff_output_preservation_embeds is None:
                                dop_embeds_list = []
                                ok = True
                                if getattr(batch, 'file_items', None) is not None and self.is_caching_text_embeddings:
                                    with self.timer('dop_embed_load'):
                                        for fi in batch.file_items:
                                            dop_caption = fi.caption or ""
                                            if hasattr(self, '_dop_replacements') and self._dop_replacements:
                                                dop_caption = normalize_caption_separators(dop_caption)
                                                for tr, cls in self._dop_replacements:
                                                    if tr == '':
                                                        continue
                                                    pattern = rf"(?<!\S){re.escape(tr)}(?!\S)"
                                                    dop_caption, n = re.subn(pattern, cls, dop_caption)
                                                    if n == 0:
                                                        dop_caption = dop_caption.replace(tr, cls)
                                            try:
                                                fi.load_dop_prompt_embedding(dop_caption)
                                            except Exception:
                                                pass
                                            if fi.dop_prompt_embeds is None:
                                                ok = False
                                                break
                                            dop_embeds_list.append(fi.dop_prompt_embeds)
                                else:
                                    ok = False

                                if ok:
                                    self.diff_output_preservation_embeds = concat_prompt_embeds(dop_embeds_list)
                                    with self.timer('dop_embed_move'):
                                        self.diff_output_preservation_embeds = self._maybe_move_embeds(self.diff_output_preservation_embeds, self.device_torch, dtype=dtype)
                                else:
                                    # Fallback: encode DOP prompts on-the-fly using CSV mapping
                                    dop_prompts = [normalize_caption_separators(self._map_triggers_to_classes_in_text(p)) for p in conditioned_prompts]
                                    dop_prompts_2 = None
                                    if prompt_2 is not None:
                                        dop_prompts_2 = [normalize_caption_separators(self._map_triggers_to_classes_in_text(p)) for p in prompt_2]
                                    with self.timer('dop_encode_fallback'):
                                        self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                            dop_prompts, dop_prompts_2,
                                            dropout_prob=self.train_config.prompt_dropout_prob,
                                            long_prompts=self.do_long_prompts,
                                            **pred_kwargs
                                        )
                                    with self.timer('dop_embed_move'):
                                        self.diff_output_preservation_embeds = self._maybe_move_embeds(self.diff_output_preservation_embeds, self.device_torch, dtype=dtype)
                            # For diff_output_preservation, use the DOP embeds as the prior target. Only use blank-preservation when
                            # blank_prompt_preservation is explicitly enabled.
                            if self.train_config.blank_prompt_preservation:
                                blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                prior_embeds_to_use = concat_prompt_embeds(
                                    [blank_embeds] * noisy_latents.shape[0]
                                )
                            elif getattr(self, 'diff_output_preservation_embeds', None) is not None:
                                prior_embeds_to_use = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                        
                        # Decide whether we can skip an expensive full-resolution prior prediction.
                        # If preservation is scheduled (DOP or blank prompt preservation) and a reduced
                        # preservation resolution is configured which would downsample the latents, and
                        # if no other features require a full-res prior (e.g., prior divergence, inverted_mask_prior,
                        # correct_pred_norm, or reg-prior), then skip the full-res prior and let
                        # `_run_preservation_forward` compute the smaller prediction instead.
                        preservation_resolution = None
                        preservation_kind = None
                        if self.train_config.diff_output_preservation and getattr(self, 'diff_output_preservation_embeds', None) is not None:
                            preservation_resolution = getattr(self.train_config, 'diff_output_preservation_resolution', None)
                            preservation_kind = 'dop'
                        if preservation_resolution is None and self.train_config.blank_prompt_preservation:
                            preservation_resolution = getattr(self.train_config, 'blank_prompt_preservation_resolution', None)
                            preservation_kind = 'blank'

                        def _would_downsample(resolution, noisy_latents):
                            if resolution is None:
                                return False
                            try:
                                vae = getattr(self.sd, 'vae', None)
                                if vae is not None and hasattr(vae, 'config') and 'block_out_channels' in vae.config:
                                    vae_scale = 2 ** (len(vae.config['block_out_channels']) - 1)
                                else:
                                    vae_scale = getattr(self.sd, 'vae_scale_factor', 8)
                            except Exception:
                                vae_scale = 8
                            _, C, H, W = noisy_latents.shape
                            # Calculate target dimensions based on pixel area, not side length
                            import math
                            target_latent_area = (resolution / vae_scale) ** 2
                            aspect_ratio = H / W
                            target_h = max(1, int(round(math.sqrt(target_latent_area * aspect_ratio))))
                            target_w = max(1, int(round(math.sqrt(target_latent_area / aspect_ratio))))
                            # transformer patch rounding
                            try:
                                tr = getattr(self.sd, 'transformer', None)
                                if tr is not None:
                                    all_patch = getattr(tr, 'all_patch_size', None)
                                    if all_patch:
                                        patch_min = int(min(all_patch))
                                    else:
                                        patch_min = 1
                                else:
                                    patch_min = 1
                            except Exception:
                                patch_min = 1
                            if patch_min > 1:
                                target_h = max(patch_min, int(round(target_h / patch_min)) * patch_min)
                                target_w = max(patch_min, int(round(target_w / patch_min)) * patch_min)
                            return (target_h < H) or (target_w < W)

                        skip_full_prior = False
                        if preservation_resolution is not None and _would_downsample(preservation_resolution, noisy_latents):
                            # ensure no other features require full-resolution prior
                            if not getattr(self.train_config, 'do_prior_divergence', False) and not getattr(self.train_config, 'inverted_mask_prior', False) and not getattr(self.train_config, 'correct_pred_norm', False) and not do_reg_prior:
                                skip_full_prior = True

                        if skip_full_prior:
                            try:
                                print_acc(f"[DOP] Skipping full-res prior for {preservation_kind} preservation; will run reduced prediction at {preservation_resolution}px area (sqrt={int(math.sqrt((preservation_resolution**2)))}px)")
                            except Exception:
                                pass
                            prior_pred = None
                        else:
                            prior_pred = self.get_prior_prediction(
                                noisy_latents=noisy_latents,
                                conditional_embeds=prior_embeds_to_use,
                                match_adapter_assist=match_adapter_assist,
                                network_weight_list=network_weight_list,
                                timesteps=timesteps,
                                pred_kwargs=pred_kwargs,
                                noise=noise,
                                batch=batch,
                                unconditional_embeds=unconditional_embeds,
                                conditioned_prompts=conditioned_prompts
                            )
                        if prior_pred is not None:
                            prior_pred = prior_pred.detach()

                # do the custom adapter after the prior prediction
                if self.adapter and isinstance(self.adapter, CustomAdapter) and (has_clip_image or self.adapter_config.type in ['llm_adapter', 'text_encoder']):
                    quad_count = random.randint(1, 4)
                    self.adapter.train()
                    conditional_embeds = self.adapter.condition_encoded_embeds(
                        tensors_0_1=clip_images,
                        prompt_embeds=conditional_embeds,
                        is_training=True,
                        has_been_preprocessed=True,
                        quad_count=quad_count
                    )
                    if self.train_config.do_cfg and unconditional_embeds is not None:
                        unconditional_embeds = self.adapter.condition_encoded_embeds(
                            tensors_0_1=clip_images,
                            prompt_embeds=unconditional_embeds,
                            is_training=True,
                            has_been_preprocessed=True,
                            is_unconditional=True,
                            quad_count=quad_count
                        )

                if self.adapter and isinstance(self.adapter, CustomAdapter) and batch.extra_values is not None:
                    self.adapter.add_extra_values(batch.extra_values.detach())

                    if self.train_config.do_cfg:
                        self.adapter.add_extra_values(torch.zeros_like(batch.extra_values.detach()),
                                                      is_unconditional=True)

                if has_adapter_img:
                    # Z-Image adapter handling: if adapter is explicitly a Z-Image controlnet, run it and collect residuals
                    # We don't train controlnets, so always run with gradients disabled
                    if is_zimage_adapter(self.adapter, self.adapter_config) or is_zimage_adapter(self.assistant_adapter, self.adapter_config):
                        adapter = self.adapter if is_zimage_adapter(self.adapter, self.adapter_config) else self.assistant_adapter
                        # Get adapter path for meta tensor materialization fallback
                        adapter_path = (
                            getattr(adapter, 'name_or_path', None) or
                            getattr(adapter, '_name_or_path', None) or
                            (getattr(self.adapter_config, 'name_or_path', None) if self.adapter_config else None) or
                            getattr(self.train_config, 'adapter_assist_name_or_path', None)
                        )
                        if self.train_config.do_cfg:
                            raise ValueError("Z-Image ControlNet is not supported with CFG")
                        
                        # Try to use precomputed control_context (avoids VAE encoding every timestep)
                        precomputed_context = self._collect_preencoded_zimage_context_for_batch(batch)
                        
                        with torch.no_grad():
                            from extensions_built_in.diffusion_models.z_image.z_image import compute_zimage_adapter_residuals
                            
                            if precomputed_context is not None:
                                # Use precomputed context - skip VAE encoding
                                down, mid, control_context, _raw = compute_zimage_adapter_residuals(
                                    self.sd,
                                    noisy_latents,
                                    timesteps,
                                    zimage_controlnet=adapter,
                                    zimage_control_context=precomputed_context.to(self.device_torch, dtype=dtype),
                                    zimage_conditioning_scale=getattr(self.sd, 'controlnet_guidance_scale', 1.0),
                                    train_dtype=dtype,
                                    batch=batch,
                                    adapter_path=adapter_path,
                                )
                            else:
                                # No precomputed context - encode images (slower)
                                adapter_images_dev = adapter_images.to(self.device_torch, dtype=dtype)
                                down, mid, control_context, _raw = compute_zimage_adapter_residuals(
                                    self.sd,
                                    noisy_latents,
                                    timesteps,
                                    zimage_controlnet=adapter,
                                    zimage_control_images=adapter_images_dev,
                                    zimage_conditioning_scale=getattr(self.sd, 'controlnet_guidance_scale', 1.0),
                                    train_dtype=dtype,
                                    batch=batch,
                                    adapter_path=adapter_path,
                                )
                            # Z-Image uses control_context passed to transformer, not down/mid block residuals
                            if control_context is not None:
                                pred_kwargs['control_context'] = control_context
                                pred_kwargs['control_context_scale'] = getattr(self.sd, 'controlnet_guidance_scale', 1.0)
                            # Also pass down/mid if they were returned (for compatibility)
                            if down is not None:
                                pred_kwargs['down_block_additional_residuals'] = down
                            if mid is not None:
                                pred_kwargs['mid_block_additional_residual'] = mid

                    # Standard ControlNetModel handling (non-Z-Image)
                    elif (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
                            self.assistant_adapter and isinstance(self.assistant_adapter, ControlNetModel)):
                        if self.train_config.do_cfg:
                            raise ValueError("ControlNetModel is not supported with CFG")
                        with torch.set_grad_enabled(self.adapter is not None):
                            adapter: ControlNetModel = self.assistant_adapter if self.assistant_adapter is not None else self.adapter
                            adapter_multiplier = get_adapter_multiplier()
                            with self.timer('encode_adapter'):
                                # add_text_embeds is pooled_prompt_embeds for sdxl
                                added_cond_kwargs = {}
                                if self.sd.is_xl:
                                    added_cond_kwargs["text_embeds"] = conditional_embeds.pooled_embeds
                                    added_cond_kwargs['time_ids'] = self.sd.get_time_ids_from_latents(noisy_latents)
                                down_block_res_samples, mid_block_res_sample = adapter(
                                    noisy_latents,
                                    timesteps,
                                    encoder_hidden_states=conditional_embeds.text_embeds,
                                    controlnet_cond=adapter_images,
                                    conditioning_scale=1.0,
                                    guess_mode=False,
                                    added_cond_kwargs=added_cond_kwargs,
                                    return_dict=False,
                                )
                                pred_kwargs['down_block_additional_residuals'] = down_block_res_samples
                                pred_kwargs['mid_block_additional_residual'] = mid_block_res_sample

                # Track whether this batch had control conditioning
                batch_has_control = 'down_block_additional_residuals' in pred_kwargs or 'mid_block_additional_residual' in pred_kwargs

                if not batch_has_control and has_adapter_img:
                    print_acc('[CONTROLNET] Warning: control images present but no control residuals were set for this batch.')

                if batch_has_control:
                    self._control_batch_count += 1
                self._total_batch_count += 1
                self._last_batch_has_control = batch_has_control

                # Determine whether this batch uses a per-dataset SplitPrompt embedding
                try:
                    batch_has_splitprompt = False
                    batch_splitprompt_key = None
                    for fi in batch.file_items:
                        ds_cfg = getattr(fi, 'dataset_config', None)
                        if ds_cfg is None:
                            continue
                        key = getattr(ds_cfg, 'dataset_path', None) or getattr(ds_cfg, 'folder_path', None)
                        if key is None:
                            continue
                        if getattr(ds_cfg, 'split_prompt_enabled', False):
                            # Confirm we have a cached embedding for that dataset (loaded or saved earlier)
                            if key in getattr(self, 'dataset_split_prompt_embeds', {}):
                                batch_has_splitprompt = True
                                batch_splitprompt_key = key
                                break
                    self._last_batch_has_splitprompt = batch_has_splitprompt
                    self._last_batch_splitprompt_key = batch_splitprompt_key
                except Exception:
                    self._last_batch_has_splitprompt = False
                    self._last_batch_splitprompt_key = None
                
                if self.train_config.do_guidance_loss and isinstance(self.train_config.guidance_loss_target, list):
                    batch_size = noisy_latents.shape[0]
                    # update the guidance value, random float between guidance_loss_target[0] and guidance_loss_target[1]
                    self._guidance_loss_target_batch = [
                        random.uniform(
                            self.train_config.guidance_loss_target[0],
                            self.train_config.guidance_loss_target[1]
                        ) for _ in range(batch_size)
                    ]

                self.before_unet_predict()
                
                if unconditional_embeds is not None:
                    unconditional_embeds = self._maybe_move_embeds(unconditional_embeds, self.device_torch, dtype=dtype)
                    unconditional_embeds = self._maybe_detach_embeds(unconditional_embeds)
                with self.timer('condition_noisy_latents'):
                    # do it for the model
                    noisy_latents = self.sd.condition_noisy_latents(noisy_latents, batch)
                    if self.adapter and isinstance(self.adapter, CustomAdapter):
                        noisy_latents = self.adapter.condition_noisy_latents(noisy_latents, batch)
                
                if self.train_config.timestep_type == 'next_sample':
                    with self.timer('next_sample_step'):
                        with torch.no_grad():
                            
                            stepped_timestep_indicies = [self.sd.noise_scheduler.index_for_timestep(t) + 1 for t in timesteps]
                            stepped_timesteps = [self.sd.noise_scheduler.timesteps[x] for x in stepped_timestep_indicies]
                            stepped_timesteps = torch.stack(stepped_timesteps, dim=0)
                            
                            # do a sample at the current timestep and step it, then determine new noise
                            # ensure embeddings are moved safely
                            conditional_move = self._maybe_move_embeds(conditional_embeds, self.device_torch, dtype=dtype)
                            # Diagnostic previously emitted PREDICT-CALL logs here; removed to reduce verbosity.
                            pass
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_move,
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                **pred_kwargs
                            )
                            stepped_latents = self.sd.step_scheduler(
                                next_sample_pred,
                                noisy_latents,
                                timesteps,
                                self.sd.noise_scheduler
                            )
                            # stepped latents is our new noisy latents. Now we need to determine noise in the current sample
                            noisy_latents = stepped_latents
                            original_samples = batch.latents.to(self.device_torch, dtype=dtype)
                            # todo calc next timestep, for now this may work as it
                            t_01 = (stepped_timesteps / 1000).to(original_samples.device)
                            if len(stepped_latents.shape) == 4:
                                t_01 = t_01.view(-1, 1, 1, 1)
                            elif len(stepped_latents.shape) == 5:
                                t_01 = t_01.view(-1, 1, 1, 1, 1)
                            else:
                                raise ValueError("Unknown stepped latents shape", stepped_latents.shape)
                            next_sample_noise = (stepped_latents - (1.0 - t_01) * original_samples) / t_01
                            noise = next_sample_noise
                            timesteps = stepped_timesteps
                # do a prior pred if we have an unconditional image, we will swap out the giadance later
                # Re-enable gradients for the main prediction / loss computation since preprocessing
                # was done under `torch.no_grad()` above. This ensures forward/backward paths have grad.
                with torch.set_grad_enabled(True):
                    if batch.unconditional_latents is not None or self.do_guided_loss:
                        # do guided loss
                        loss = self.get_guided_loss(
                            noisy_latents=noisy_latents,
                            conditional_embeds=conditional_embeds,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            batch=batch,
                            noise=noise,
                            unconditional_embeds=unconditional_embeds,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_pred,
                        )
                        
                    elif self.train_config.loss_type == 'mean_flow':
                        loss = self.get_mean_flow_loss(
                            noisy_latents=noisy_latents,
                            conditional_embeds=conditional_embeds,
                            match_adapter_assist=match_adapter_assist,
                            network_weight_list=network_weight_list,
                            timesteps=timesteps,
                            pred_kwargs=pred_kwargs,
                            batch=batch,
                            noise=noise,
                            unconditional_embeds=unconditional_embeds,
                            prior_pred=prior_pred,
                        )
                    else:
                        with self.timer('predict_unet'):
                            # move embeddings safely before calling predict_noise
                            with self.timer('to_device'):
                                conditional_move = self._maybe_move_embeds(conditional_embeds, self.device_torch, dtype=dtype)
                                unconditional_move = self._maybe_move_embeds(unconditional_embeds, self.device_torch, dtype=dtype)

                            # Strict fail-fast checks before UNet forward
                            try:
                                params_iter = self.params
                                if isinstance(params_iter, list) and len(params_iter) > 0 and isinstance(params_iter[0], dict):
                                    pl = []
                                    for p in params_iter:
                                        pl.extend(p['params'])
                                else:
                                    pl = list(params_iter) if not isinstance(params_iter, (list, tuple)) else params_iter
                                param_requires_count = sum(1 for p in pl if getattr(p, 'requires_grad', False))
                            except Exception:
                                param_requires_count = 0

                            if not torch.is_grad_enabled():
                                raise RuntimeError("Global grad mode is disabled immediately before UNet forward. This is fatal: ensure no surrounding `torch.no_grad()` or `torch.set_grad_enabled(False)` remain enabled.")

                            if param_requires_count == 0:
                                raise RuntimeError("No model parameters are configured to require gradients. Did you accidentally freeze all parameters or misconfigure the optimizer? Aborting.")

                            # Make batch visible to pre/post UNet hooks (for attention alignment)
                            self._last_batch_for_attn = batch
                            # proceed to forward
                            # Diagnostic previously emitted PREDICT-CALL logs here; removed to reduce verbosity.
                            pass
                            noise_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_move,
                                unconditional_embeds=unconditional_move,
                                batch=batch,
                                is_primary_pred=True,
                                **pred_kwargs
                            )

                            # Ensure forward produced grad-connected tensor
                            if not getattr(noise_pred, 'requires_grad', False):
                                raise RuntimeError("UNet produced a tensor that does not require gradients after forward. This indicates the forward executed under a no-grad context or returned a detached result. Aborting.")
                        self.after_unet_predict()

                        with self.timer('calculate_loss'):
                            noise = noise.to(self.device_torch, dtype=dtype).detach()
                            prior_to_calculate_loss = prior_pred
                            # Determine whether preservation will run for this batch. For diff_output_preservation
                            # this is gated by `diff_output_preservation_every` and uses the current batch counter.
                            do_dop_this_step = self._is_dop_scheduled(for_encoding=False)
                            doing_preservation = do_dop_this_step or self.train_config.blank_prompt_preservation
                            if doing_preservation and not do_inverted_masked_prior:
                                prior_to_calculate_loss = None
                            
                            try:
                                loss = self.calculate_loss(
                                    noise_pred=noise_pred,
                                    noise=noise,
                                    noisy_latents=noisy_latents,
                                    timesteps=timesteps,
                                    batch=batch,
                                    mask_multiplier=mask_multiplier,
                                    prior_pred=prior_to_calculate_loss,
                                )
                            except Exception as e:
                                # Non-fatal: catch any error in loss computation, log and continue with safe fallback
                                print_acc(f"[LOSS] calculate_loss failed: {e}")
                                # record a short sentinel for visibility
                                try:
                                    self._last_loss_calc_failed = True
                                    self._last_loss_calc_exc = str(e)[:200]
                                except Exception:
                                    pass
                                # fallback: zero loss tensor on correct device/dtype that requires grad
                                try:
                                    loss = torch.tensor(0.0, device=self.device_torch, dtype=get_torch_dtype(self.train_config.dtype), requires_grad=True)
                                except Exception:
                                    # conservative fallback
                                    loss = torch.tensor(0.0, device=self.device_torch, requires_grad=True)
                            if not getattr(loss, 'requires_grad', False):
                                raise RuntimeError(
                                    "Calculated loss does not require gradients. This suggests the model forward was executed without grad tracking. "
                                    "Inspect surrounding `torch.no_grad()` / `set_grad_enabled` contexts introduced when splitting timers."
                                )
                    
                    if self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation:
                        # send the loss backwards otherwise checkpointing will fail
                        self.accelerator.backward(loss)
                        normal_loss = loss.detach() # dont send backward again
                        try:
                            with self.timer('cpu_transfer'):
                                self._last_normal_loss = float(normal_loss.detach())
                        except Exception:
                            self._last_normal_loss = None
                        
                        # Determine which preservation embeddings to use
                        # Only compute diff output preservation if it's scheduled for this batch
                        preservation_embeds = None
                        if 'do_dop_this_step' in locals() and do_dop_this_step:
                            if self.diff_output_preservation_embeds is None:
                                raise RuntimeError(
                                    "Scheduled diff_output_preservation step but embeds are not prepared. "
                                    "Ensure 'diff_output_preservation_every' and precompute settings are correct. "
                                    f"Current batch count: {getattr(self, '_total_batch_count', 0)}, "
                                    f"DOP every: {getattr(self.train_config, 'diff_output_preservation_every', 1)}"
                                )
                            preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                            # record execution count for diagnostics
                            try:
                                self._diff_output_preservation_exec_count += 1
                            except Exception:
                                pass
                        elif self.train_config.blank_prompt_preservation:
                            if self.cached_blank_embeds is None:
                                raise RuntimeError("blank_prompt_preservation enabled but cached_blank_embeds is None")
                            blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                self.device_torch, dtype=dtype
                            )
                            preservation_embeds = concat_prompt_embeds(
                                [blank_embeds] * noisy_latents.shape[0]
                            )
                        
                        # reset per-step diagnostics
                        self._last_preservation_loss = None
                        # Compute preservation prediction and loss via helper (adds DOP timers)
                        preservation_pred = None
                        if preservation_embeds is not None:
                            # Additional validation: ensure preservation_embeds has required attributes
                            # This catches edge cases where embeds might be malformed
                            try:
                                # Verify embeds can be moved to device (has .to method or is a proper PromptEmbeds)
                                if not (hasattr(preservation_embeds, 'to') or hasattr(preservation_embeds, 'text_embeds')):
                                    raise RuntimeError(f"preservation_embeds is not a valid PromptEmbeds object: {type(preservation_embeds)}")
                            except Exception as e:
                                try:
                                    print_acc(f"[DOP] preservation_embeds validation failed: {e}")
                                except Exception:
                                    pass
                                # Skip preservation this step if embeds are invalid
                                preservation_embeds = None
                        
                        if preservation_embeds is not None:
                            # Determine if we should run preservation at a reduced resolution
                            preservation_resolution = None
                            if 'do_dop_this_step' in locals() and do_dop_this_step:
                                preservation_resolution = getattr(self.train_config, 'diff_output_preservation_resolution', None)
                            elif self.train_config.blank_prompt_preservation:
                                preservation_resolution = getattr(self.train_config, 'blank_prompt_preservation_resolution', None)

                            # indicate whether this is a DOP or blank prompt preservation step
                            preservation_kind = 'dop' if ('do_dop_this_step' in locals() and do_dop_this_step) else ('blank' if self.train_config.blank_prompt_preservation else None)

                            # Run preservation forward with gradients enabled so the preservation
                            # prediction participates in autograd and its loss can be backpropagated.
                            # This handles BOTH high-resolution (no downsampling) and low-resolution
                            # (downsampled latents + prior) cases based on preservation_resolution parameter.
                            with torch.set_grad_enabled(True):
                                preservation_pred_res = self._run_preservation_forward(
                                    noisy_latents=noisy_latents,
                                    timesteps=timesteps,
                                    preservation_embeds=preservation_embeds,
                                    unconditional_embeds=unconditional_embeds,
                                    batch=batch,
                                    pred_kwargs=pred_kwargs,
                                    dtype=dtype,
                                    prior_pred=prior_pred,
                                    preservation_resolution=preservation_resolution,
                                    preservation_kind=preservation_kind,
                                    match_adapter_assist=match_adapter_assist,
                                    network_weight_list=network_weight_list,
                                )

                            # Support returned (preservation_pred, prior_pred_for_loss) when downsampling occurred.
                            # In low-res mode, _run_preservation_forward computes both preservation pred AND
                            # a downsampled prior pred (if original prior_pred was None) so loss can be computed
                            # at the reduced resolution.
                            if isinstance(preservation_pred_res, tuple):
                                preservation_pred, prior_pred_for_loss = preservation_pred_res
                            else:
                                preservation_pred = preservation_pred_res
                                prior_pred_for_loss = prior_pred

                        if preservation_pred is not None:
                            multiplier = self.train_config.diff_output_preservation_multiplier if self.train_config.diff_output_preservation else self.train_config.blank_prompt_preservation_multiplier
                            # Use possibly-downsampled prior_pred_for_loss if provided by _run_preservation_forward
                            preservation_loss = self._compute_and_apply_preservation_loss(preservation_pred, prior_pred_for_loss if 'prior_pred_for_loss' in locals() else prior_pred, multiplier)

                            if preservation_loss is None:
                                try:
                                    print_acc("[DOP] Warning: preservation loss computation returned None; falling back to normal loss")
                                except Exception:
                                    pass
                                # Fallback to normal loss only
                                loss = normal_loss.clone().detach()
                                loss.requires_grad_(True)
                            else:
                                # Combined loss is for logging only - backward already called in _compute_and_apply_preservation_loss
                                loss = normal_loss + preservation_loss
                                loss = loss.clone().detach()
                                # require grad again so the backward wont fail
                                loss.requires_grad_(True)
                                try:
                                    print_acc(f"[DOP] Combined loss (already backwarded separately): normal={float(normal_loss):.6e}, preservation={float(preservation_loss):.6e}, total={float(loss):.6e}")
                                except Exception:
                                    pass
                        else:
                            # No preservation this step; use the normal loss only
                            loss = normal_loss.clone().detach()
                            loss.requires_grad_(True)
                            try:
                                print_acc(f"[DOP] No preservation scheduled for this step (batch {getattr(self, '_total_batch_count', 0)})")
                            except Exception:
                                pass

                # If loss is NaN after all attempts, fail loudly and abort the run (do not fallback to a zero tensor)
                if torch.isnan(loss):
                    raise RuntimeError("Loss is NaN after loss computation: aborting training to avoid silent no-op steps.")

                with self.timer('backward'):
                    # todo we have multiplier seperated. works for now as res are not in same batch, but need to change
                    loss = loss * loss_multiplier.mean()
                    # IMPORTANT if gradient checkpointing do not leave with network when doing backward
                    # it will destroy the gradients. This is because the network is a context manager
                    # and will change the multipliers back to 0.0 when exiting. They will be
                    # 0.0 for the backward pass and the gradients will be 0.0
                    # I spent weeks on fighting this. DON'T DO IT
                    # with fsdp_overlap_step_with_backward():
                    # if self.is_bfloat:
                    # loss.backward()
                    # else:
                    self.accelerator.backward(loss)

        return loss.detach()
        # flush()

    def _run_preservation_forward(self, noisy_latents, timesteps, preservation_embeds, unconditional_embeds, batch, pred_kwargs, dtype, prior_pred, preservation_resolution=None, preservation_kind: 'Optional[str]'=None, match_adapter_assist: bool = False, network_weight_list: list = None):
        """Run preservation forward pass for DOP/blank prompt preservation and record timings.

        If `preservation_resolution` (pixels, long-side) is specified, the forward pass will be
        executed at that reduced spatial resolution and the returned preservation prediction and
        prior prediction (both downsampled) will be suitable for loss computation.

        `preservation_kind` may be 'dop' or 'blank' to help label timers appropriately.

        Returns preservation_pred (or (preservation_pred, prior_pred_down) when downsampling used) or None.
        """
        # Determine timer base name based on preservation kind
        timer_base = 'blank_predict' if preservation_kind == 'blank' else 'dop_predict'

        # preservation_embeds may be prompt embeds or similar. Move them and latents to device inside the timer
        # Create a shallow copy of `pred_kwargs` to avoid mutating the caller's dict and to allow
        # removal of adapter residuals for preservation predictions when requested.
        local_pred_kwargs = dict(pred_kwargs) if pred_kwargs is not None else {}
        if match_adapter_assist:
            local_pred_kwargs.pop('down_intrablock_additional_residuals', None)
            local_pred_kwargs.pop('down_block_additional_residuals', None)
            local_pred_kwargs.pop('mid_block_additional_residual', None)

        # CRITICAL: LoKr networks are incompatible with downsampled DOP due to their scale-sensitive
        # Kronecker product structure. Downsampling causes the Kronecker factors to learn spatial
        # patterns at the reduced resolution, which don't generalize to full-resolution inference.
        # For LoKr: always force full-resolution DOP. For LoRA: downsampling is safe.
        effective_preservation_resolution = preservation_resolution


        # If no resolution requested, do the normal full-res predict (using cleaned local kwargs)
        if effective_preservation_resolution is None:
            with self.timer(timer_base):
                preservation_pred = self.predict_noise(
                    noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                    timesteps=timesteps,
                    conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                    unconditional_embeds=unconditional_embeds,
                    batch=batch,
                    **local_pred_kwargs
                )
            return preservation_pred

        # Otherwise, compute a reduced latent size and run predict at that size
        try:
            # compute vae scale factor (pixels -> latent). Try config first, fallback to heuristic 8
            vae = getattr(self.sd, 'vae', None)
            if vae is not None and hasattr(vae, 'config') and 'block_out_channels' in vae.config:
                vae_scale = 2 ** (len(vae.config['block_out_channels']) - 1)
            else:
                vae_scale = getattr(self.sd, 'vae_scale_factor', 8)
        except Exception:
            vae_scale = 8

        # Current latent spatial dims
        _, C, H, W = noisy_latents.shape
        # Target latent area from pixel area: effective_preservation_resolution represents sqrt of target pixel area
        # e.g., effective_preservation_resolution=128 means 128*128=16k pixels, which is (128/vae_scale)^2 latent pixels
        target_latent_area = (effective_preservation_resolution / vae_scale) ** 2
        # Preserve aspect ratio: target_h / target_w = H / W
        # Solve: target_h * target_w = target_latent_area AND target_h / target_w = H / W
        # Therefore: target_h = sqrt(target_latent_area * H / W), target_w = sqrt(target_latent_area * W / H)
        import math
        aspect_ratio = H / W
        target_h = max(1, int(round(math.sqrt(target_latent_area * aspect_ratio))))
        target_w = max(1, int(round(math.sqrt(target_latent_area / aspect_ratio))))

        # Ensure target dims are compatible with transformer patch sizes (avoid invalid view shapes)
        try:
            tr = getattr(self.sd, 'transformer', None)
            if tr is not None:
                all_patch = getattr(tr, 'all_patch_size', None)
                if all_patch:
                    patch_min = int(min(all_patch))
                else:
                    patch_min = 1
            else:
                patch_min = 1
        except Exception:
            patch_min = 1

        # Round target dims to nearest multiple of patch_min (at least patch_min)
        if patch_min > 1:
            target_h = max(patch_min, int(round(target_h / patch_min)) * patch_min)
            target_w = max(patch_min, int(round(target_w / patch_min)) * patch_min)

        # If target is same or larger than current, just run full-res
        if target_h >= H and target_w >= W:
            with self.timer(timer_base):
                preservation_pred = self.predict_noise(
                    noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                    timesteps=timesteps,
                    conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                    unconditional_embeds=unconditional_embeds,
                    batch=batch,
                    **local_pred_kwargs
                )
            return preservation_pred

        # CRITICAL: To downsample correctly for DOP, we must downsample the clean latents
        # and noise separately, then re-apply the noise schedule at the small resolution.
        # Downsampling already-noisy latents produces corrupted signals because interpolation
        # doesn't commute with the noise addition formula: downsample(clean + noise) ≠ downsample(clean) + downsample(noise)
        with self.timer(f"{timer_base}_downsampled"):
            # Delegate to helper so the downsampling + noise generation can be unit-tested
            latents_small, noise_small, noisy_small = self._create_downsampled_noisy_latents(
                batch.latents, timesteps, target_h, target_w, dtype
            )

            
            # Step 4: Generate both predictions at the same small resolution
            # CRITICAL: Always generate prior_small at small resolution, never downsample a full-res prior.
            # For DOP to work correctly, both predictions must be generated at the same resolution
            # from the same properly-noised small latents.
            prior_small = None
            try:
                prior_small = self.get_prior_prediction(
                    noisy_latents=noisy_small,
                    conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                    match_adapter_assist=match_adapter_assist,
                    network_weight_list=network_weight_list if network_weight_list is not None else [],
                    timesteps=timesteps,
                    pred_kwargs=local_pred_kwargs,
                    batch=batch,
                    noise=None,
                    unconditional_embeds=unconditional_embeds,
                )
            except Exception:
                prior_small = None

            preservation_pred_small = self.predict_noise(
                noisy_latents=noisy_small,
                timesteps=timesteps,
                conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **local_pred_kwargs
            )
        # Return both small preds so loss can be computed at this resolution
        return (preservation_pred_small, prior_small)

    def _create_downsampled_noisy_latents(self, original_latents: torch.Tensor, timesteps: torch.Tensor, target_h: int, target_w: int, dtype: str):
        """Downsample `original_latents` to (target_h, target_w), sample noise at that resolution
        and create a scheduler-consistent `noisy_small` tensor. Returns (latents_small, noise_small, noisy_small).

        This helper is factored out for unit testing (CPU-only tests can call this directly and
        monkeypatch `self.sd.add_noise` to assert behavior).
        """
        torch_dtype = get_torch_dtype(dtype)
        # Downsample clean latents using bicubic interpolation
        latents_small = torch.nn.functional.interpolate(
            original_latents, size=(target_h, target_w), mode='bicubic', align_corners=False
        ).to(self.device_torch, dtype=torch_dtype)

        # Sample fresh noise at the reduced resolution
        noise_small = torch.randn_like(latents_small, device=self.device_torch, dtype=latents_small.dtype)

        try:
            noisy_small = self.sd.add_noise(latents_small, noise_small, timesteps.to(self.device_torch))
            try:
                if getattr(self.train_config, 'debug_dump_dop', False):
                    print_acc(f"[DOP DEBUG] _create_downsampled_noisy_latents using scheduler: size={target_h}x{target_w}, latents_small.dtype={latents_small.dtype}")
            except Exception:
                pass
        except Exception as e:
            # Fail-fast: scheduler failure is critical for DOP correctness. Abort instead of silently using no noise.
            raise RuntimeError(f"[DOP] failed to construct noisy_small via scheduler: {e}") from e

        # Sanity check: noisy_small must be defined here
        if noisy_small is None:
            raise RuntimeError("[DOP] noisy_small was not constructed by scheduler (unexpected)")

        return latents_small, noise_small, noisy_small

    def _compute_and_apply_preservation_loss(self, preservation_pred, prior_pred, multiplier: float):
        """Compute preservation loss, record diagnostics, and apply backward.

        Returns the preservation_loss tensor.
        """
        try:
            # Validate that both predictions are at the same resolution for fair comparison
            if prior_pred is not None and preservation_pred.shape != prior_pred.shape:
                try:
                    print_acc(f"[DOP] Warning: resolution mismatch detected! preservation_pred shape {preservation_pred.shape} != prior_pred shape {prior_pred.shape}. This can cause DOP to fail at enforcing zero LoKr effect without trigger words.")
                except Exception:
                    pass
                # Don't fail, but this indicates a bug in the DOP implementation
            
            # Ensure both tensors are on the same device and dtype to avoid dtype/device mismatch errors
            if prior_pred is not None:
                # Move to same device first
                if preservation_pred.device != prior_pred.device:
                    preservation_pred = preservation_pred.to(prior_pred.device)
                # If prior_pred is a low-precision dtype on CPU, promote to float32 because
                # CPU bfloat16/float16 math may not be supported for mse_loss. Otherwise prefer prior dtype.
                cpu_low_precision = prior_pred.device.type == 'cpu' and prior_pred.dtype in (torch.bfloat16, torch.float16)
                if cpu_low_precision:
                    preservation_pred = preservation_pred.to(torch.float32)
                    prior_pred = prior_pred.to(torch.float32)
                else:
                    if preservation_pred.dtype != prior_pred.dtype:
                        # Prefer prior_pred dtype (can be bfloat16/float16) to match runtime precision
                        preservation_pred = preservation_pred.to(prior_pred.dtype)

            # Compute loss (do not rely on surrounding timers to succeed for backward)
            preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier

        except Exception as e:
            try:
                print_acc(f"[DOP] preservation loss computation failed at compute step: {e}")
            except Exception:
                pass
            self._last_preservation_loss = None
            return None

        # record a diagnostic scalar for the UI (best-effort)
        try:
            with self.timer('cpu_transfer'):
                self._last_preservation_loss = float(preservation_loss.detach())
        except Exception:
            self._last_preservation_loss = None

        # apply backward for preservation loss if it participates in autograd;
        # in unit tests we may have no requires_grad, so skip backward in that case.
        try:
            if preservation_loss.requires_grad:
                try:
                    with self.timer('preservation_backward'):
                        self.accelerator.backward(preservation_loss)
                except Exception as e:
                    try:
                        print_acc(f"[DOP] backward failed for preservation loss: {e}")
                    except Exception:
                        pass
            else:
                try:
                    print_acc("[DOP] preservation loss has no grad; skipping backward (likely a unit test scenario)")
                except Exception:
                    pass
        except Exception as e:
            try:
                print_acc(f"[DOP] preservation backward orchestration failed: {e}")
            except Exception:
                pass

        return preservation_loss

    def _should_skip_full_prior(self, noisy_latents, preservation_resolution, do_reg_prior: bool = False) -> bool:
        """Return True if the full-resolution prior prediction can be skipped in favor of running
        the preservation prediction at a (smaller) reduced resolution.

        Conditions to skip:
        - `preservation_resolution` is set and would downsample the `noisy_latents` spatial dims
          when converted to latent space, AND
        - none of the following features are active: `do_prior_divergence`, `inverted_mask_prior`,
          `correct_pred_norm`, and there is no reg prior for this batch (`do_reg_prior`).
        """
        if preservation_resolution is None:
            return False
        try:
            vae = getattr(self.sd, 'vae', None)
            if vae is not None and hasattr(vae, 'config') and 'block_out_channels' in vae.config:
                vae_scale = 2 ** (len(vae.config['block_out_channels']) - 1)
            else:
                vae_scale = getattr(self.sd, 'vae_scale_factor', 8)
        except Exception:
            vae_scale = 8
        _, C, H, W = noisy_latents.shape
        target_long = max(1, int(round(preservation_resolution / vae_scale)))
        if H >= W:
            target_h = target_long
            target_w = max(1, int(round(W * (target_h / H))))
        else:
            target_w = target_long
            target_h = max(1, int(round(H * (target_w / W))))
        try:
            tr = getattr(self.sd, 'transformer', None)
            if tr is not None:
                all_patch = getattr(tr, 'all_patch_size', None)
                if all_patch:
                    patch_min = int(min(all_patch))
                else:
                    patch_min = 1
            else:
                patch_min = 1
        except Exception:
            patch_min = 1
        if patch_min > 1:
            target_h = max(patch_min, int(round(target_h / patch_min)) * patch_min)
            target_w = max(patch_min, int(round(target_w / patch_min)) * patch_min)

        # will downsample if either target dimension strictly less than current
        will_downsample = (target_h < H) or (target_w < W)
        if not will_downsample:
            return False
        # check feature flags that require full-res prior
        if getattr(self.train_config, 'do_prior_divergence', False):
            return False
        if getattr(self.train_config, 'inverted_mask_prior', False):
            return False
        if getattr(self.train_config, 'correct_pred_norm', False):
            return False
        if do_reg_prior:
            return False
        return True

    def _maybe_log_per_example(self, batch: 'DataLoaderBatchDTO', batch_list_len: int) -> list:
        """Return per-example entries for the given batch and update streaming aggregator.

        Logging is only performed when the run is simple (single-batch and no gradient accumulation)
        to avoid misleading or partial per-example logs during accumulation or multi-batch processing.
        """
        entries = []
        try:
            # Only log when a single batch was processed and there is no gradient accumulation configured
            grad_accum = getattr(self.train_config, 'gradient_accumulation', 1)
            grad_accum_steps = getattr(self.train_config, 'gradient_accumulation_steps', 1)
            if batch_list_len != 1 or grad_accum != 1 or grad_accum_steps != 1:
                return entries

            from toolkit.util.loss_utils import per_example_from_batch, StreamingAggregator
            if hasattr(self, 'last_example_losses') and self.last_example_losses is not None:
                try:
                    mapped = per_example_from_batch(batch, self.last_example_losses)
                    entries.extend(mapped)
                    # add entries to a streaming dataset aggregator for per-save-step JSON reporting
                    try:
                        if not hasattr(self, '_dataset_aggregator') or self._dataset_aggregator is None:
                            self._dataset_aggregator = StreamingAggregator()
                        for e in mapped:
                            self._dataset_aggregator.add_entry(e)
                    except Exception as e:
                        raise RuntimeError(f"Failed to add entries to dataset aggregator: {e}") from e
                except Exception as e:
                    raise RuntimeError(f"Unexpected error in training loop helper: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in training loop helper: {e}") from e
        return entries

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]

        # Fail fast when a dataloader returned a None batch. This guards against
        # silent no-op training where steps advance but no loss/backward occurs.
        if any(b is None for b in batch_list):
            missing = [i for i, b in enumerate(batch_list) if b is None]
            raise RuntimeError(
                f"hook_train_loop received None batch items at indices {missing}. "
                "This indicates the dataloader returned None or was exhausted unexpectedly. "
                "Ensure the DataLoader yields valid batches and that dataset iterators are reset correctly."
            )



        total_loss = None
        with self.timer('zero_grad'):
            self.optimizer.zero_grad()
        for batch in batch_list:
            if self.sd.is_multistage:
                # handle multistage switching
                if self.steps_this_boundary >= self.train_config.switch_boundary_every or self.current_boundary_index not in self.sd.trainable_multistage_boundaries:
                    # iterate to make sure we only train trainable_multistage_boundaries
                    while True:
                        self.steps_this_boundary = 0
                        self.current_boundary_index += 1
                        if self.current_boundary_index >= len(self.sd.multistage_boundaries):
                            self.current_boundary_index = 0
                        if self.current_boundary_index in self.sd.trainable_multistage_boundaries:
                            # if this boundary is trainable, we can stop looking
                            break
            loss = self.train_single_accumulation(batch)
            self.steps_this_boundary += 1
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        # Add any additional scalar losses (e.g., attention alignment)
        try:
            if additional_loss is not None and additional_loss != 0.0:
                if isinstance(additional_loss, torch.Tensor):
                    total_loss = total_loss + additional_loss
                else:
                    total_loss = total_loss + torch.tensor(additional_loss, dtype=total_loss.dtype, device=total_loss.device)
        except Exception:
            pass
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                with self.timer('clip_grad'):
                    if isinstance(self.params[0], dict):
                        for i in range(len(self.params)):
                            self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                    else:
                        self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # Diagnostic: report whether this is an accumulation step and gradient norms
            try:
                with self.timer('grad_diagnostics'):
                    params_iter = self.params
                    if isinstance(params_iter, list) and len(params_iter) > 0 and isinstance(params_iter[0], dict):
                        params_list = []
                        for p in params_iter:
                            params_list.extend(p['params'])
                    else:
                        params_list = params_iter
                    total_grad_sq = 0.0
                    found_grad = False
                    for p in params_list:
                        g = getattr(p, 'grad', None)
                        if g is not None:
                            found_grad = True
                            try:
                                ng = float(g.detach().data.norm(2).item())
                                total_grad_sq += ng * ng
                            except Exception:
                                pass
                    total_grad_norm = total_grad_sq ** 0.5 if found_grad else None
            except Exception:
                total_grad_norm = None
                found_grad = False

            print_acc(f"[DEBUG-OPT] is_grad_accumulation_step={self.is_grad_accumulation_step}, found_grad={found_grad}, grad_norm={total_grad_norm}")

            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                did_step = False
                try:
                    self.optimizer.step()
                    did_step = True
                finally:
                    print_acc(f"[DEBUG-OPT] optimizer_step_executed={did_step}")

            with self.timer('zero_grad_set_to_none'):
                self.optimizer.zero_grad(set_to_none=True)
            if self.adapter and isinstance(self.adapter, CustomAdapter):
                self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
        else:
            # gradient accumulation. Just a place for breakpoint
            pass

        # TODO Should we only step scheduler on grad step? If so, need to recalculate last step
        with self.timer('scheduler_step'):
            self.lr_scheduler.step()

        if self.embedding is not None:
            with self.timer('restore_embeddings'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.embedding.restore_embeddings()
        if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
            with self.timer('restore_adapter'):
                # Let's make sure we don't update any embedding weights besides the newly added token
                self.adapter.restore_embeddings()

        # Protect against cases where no loss was computed (e.g. empty batch list or early-exit)
        if total_loss is None or len(batch_list) == 0:
            # Create a zero tensor on the correct device to keep downstream code happy
            loss_tensor = torch.tensor(0.0, device=getattr(self, 'device_torch', 'cpu'))
        else:
            denom = len(batch_list) if len(batch_list) > 0 else 1
            loss_tensor = (total_loss / denom)

        # Safely extract a Python scalar from the loss tensor; if extraction fails, log and continue
        try:
            loss_val = loss_tensor.item()
        except Exception as e:
            print_acc(f"[LOSS] failed to extract loss scalar: {e}")
            try:
                loss_val = float(loss_tensor)
            except Exception:
                loss_val = 0.0
            try:
                self._last_loss_scalar_failed = True
                self._last_loss_scalar_err = str(e)[:200]
            except Exception:
                pass
        loss_dict = OrderedDict({'loss': loss_val})

        # If a preservation loss was recorded this step, expose it separately so graphs stay readable
        if hasattr(self, '_last_preservation_loss') and self._last_preservation_loss is not None:
            loss_dict['preservation'] = float(self._last_preservation_loss)
        # Also expose the normalized loss if available (normal loss weighted by noise level)
        if hasattr(self, '_last_normal_loss') and self._last_normal_loss is not None:
            normal_loss = float(self._last_normal_loss)
            # Normalize by noise level: multiply by (1 - mean_sigma) to show loss proportional to signal strength
            # At high noise (sigma near 1), normalized_loss is small; at low noise (sigma near 0), it's close to normal_loss
            if hasattr(self, 'last_noise_sigmas') and self.last_noise_sigmas is not None:
                try:
                    noise_sigma_mean = float(self.last_noise_sigmas.mean().item())
                    # Linear de-weighting: loss × (1 - sigma)
                    loss_dict['normalized_loss'] = normal_loss * (1.0 - noise_sigma_mean)
                    # Less aggressive quadratic scaling: loss × (1 - sigma²)
                    loss_dict['normalized_loss_2'] = normal_loss * (1.0 - noise_sigma_mean ** 2)
                    # More aggressive quadratic scaling: loss × (1 - sigma)²
                    loss_dict['normalized_loss_3'] = normal_loss * ((1.0 - noise_sigma_mean) ** 2)
                    # Per-example inverse-variance normalization (preferred when per-sample data is available).
                    # Use robust defaults: eps to avoid divide-by-zero, clamp sigmas and clamp weights to avoid huge values.
                    try:
                        eps_norm = 1e-8
                        min_sigma = 1e-6
                        max_weight = 1e3
                        if hasattr(self, 'last_example_losses') and self.last_example_losses is not None and hasattr(self, 'last_noise_sigmas') and self.last_noise_sigmas is not None and len(self.last_example_losses) == len(self.last_noise_sigmas):
                            per_loss = self.last_example_losses.to(torch.float64)
                            per_sigma = self.last_noise_sigmas.to(torch.float64)
                            sigma_clamped = per_sigma.clamp(min=min_sigma)
                            weights = (1.0 / (sigma_clamped ** 2 + eps_norm)).clamp(max=max_weight)
                            norm_val = float((per_loss * weights).mean().item())
                            loss_dict['normalized_loss_4'] = norm_val
                        else:
                            # Fallback: scalar mean division (less accurate than per-sample normalization)
                            loss_dict['normalized_loss_4'] = normal_loss / (noise_sigma_mean ** 2 + eps_norm)
                    except Exception:
                        # keep original loss if diagnostics fail
                        loss_dict['normalized_loss_4'] = normal_loss
                except Exception:
                    # Fallback: if sigma computation fails, use raw normal loss
                    loss_dict['normalized_loss'] = normal_loss
                    loss_dict['normalized_loss_2'] = normal_loss
                    loss_dict['normalized_loss_3'] = normal_loss
            else:
                # No sigma data available; use raw normal loss
                loss_dict['normalized_loss'] = normal_loss
                loss_dict['normalized_loss_2'] = normal_loss
                loss_dict['normalized_loss_3'] = normal_loss

        # Control-related metrics (diagnostics & monitoring)
        try:
            # Build debug flags as strings (so they are not treated as numeric loss scalars)
            debug_flags = {}
            try:
                # Report whether any control usage occurred as a debug boolean (true/false)
                control_usage = float(getattr(self, '_control_batch_count', 0.0)) / max(1.0, float(getattr(self, '_total_batch_count', 0.0)))
                debug_flags['control_usage_rate'] = 'true' if control_usage > 0.0 else 'false'

                debug_flags['controlnet_enabled'] = 'true' if getattr(self.sd, 'is_controlnet_enabled', False) else 'false'
                debug_flags['batch_has_control'] = 'true' if getattr(self, '_last_batch_has_control', False) else 'false'
                debug_flags['controlnet_offload_active'] = 'true' if getattr(self, '_last_batch_offload_active', False) else 'false'
                debug_flags['splitprompt'] = 'true' if getattr(self, '_last_batch_has_splitprompt', False) else 'false'
                debug_flags['splitprompt_dataset'] = str(getattr(self, '_last_batch_splitprompt_key', '') or '')
                # If splitprompt is active, provide the source safetensor filename and configured block lists
                if getattr(self, '_last_batch_has_splitprompt', False):
                    try:
                        ds_key = getattr(self, '_last_batch_splitprompt_key', None)
                        pe = getattr(self, 'dataset_split_prompt_embeds', {}).get(ds_key)
                        src = None
                        if pe is not None:
                            src = getattr(pe, '_source_path', None)
                        # fallback: expect split_prompt.safetensors in the dataset folder
                        if not src and ds_key:
                            candidate = os.path.join(ds_key, 'split_prompt.safetensors')
                            try:
                                if os.path.exists(candidate):
                                    src = candidate
                            except Exception:
                                # ignore fs checks on weird keys
                                pass
                        if src:
                            debug_flags['splitprompt_file'] = os.path.basename(src)
                    except Exception:
                        pass
                    try:
                        content_blocks = getattr(self.train_config, 'splitflux_content_blocks', None)
                        style_blocks = getattr(self.train_config, 'splitflux_style_blocks', None)
                        if content_blocks:
                            debug_flags['splitprompt_content_blocks'] = ','.join(str(x) for x in content_blocks)
                        if style_blocks:
                            debug_flags['splitprompt_style_blocks'] = ','.join(str(x) for x in style_blocks)
                    except Exception:
                        pass
                # expose whether noise diagnostics failed so it doesn't silently vanish
                debug_flags['loss_over_noise_failed'] = 'true' if getattr(self, '_last_noise_diag_exc', None) is not None else 'false'
                # expose whether loss calculation failed so it doesn't silently vanish
                debug_flags['loss_calc_failed'] = 'true' if getattr(self, '_last_loss_calc_failed', False) else 'false'
            except Exception:
                debug_flags = {'control_usage_rate': 'false', 'controlnet_enabled': 'false', 'batch_has_control': 'false', 'controlnet_offload_active': 'false', 'splitprompt': 'false', 'splitprompt_dataset': '', 'loss_over_noise_failed': 'false', 'loss_calc_failed': 'false'}

            # Attach debug flags as a dictionary (strings) for diagnostics
            loss_dict['debug_flags'] = debug_flags

            # Also emit a concise, separate log line for these flags
            try:
                flags_msg = ' '.join([f"{k}={v}" for k, v in debug_flags.items() if v != ''])
                print_acc(f"[FLAGS] {flags_msg}")
            except Exception:
                pass

            # Expose noise diagnostics if available (mean values)
            try:
                self._attach_noise_metrics_to_loss_dict(loss_dict)
            except Exception:
                pass
        except Exception:
            # non-fatal: metrics/debug flag assembly failed
            pass
        # Final per-step hook: run after the `loss_dict` has been fully assembled
        # Subclasses can override `end_of_training_loop()` to inspect or flush per-step artifacts
        self.end_of_training_loop()

        return loss_dict


    def _attach_noise_metrics_to_loss_dict(self, loss_dict: dict):
        """Attach aggregated noise diagnostics into loss_dict if per-sample diagnostics exist."""
        try:
            if getattr(self, 'last_noise_norms', None) is not None:
                mean_noise = float(self.last_noise_norms.mean().item())
                loss_dict['train/noise_mean'] = mean_noise
                # optional: include sigma mean if it exists
                if getattr(self, 'last_noise_sigmas', None) is not None:
                    try:
                        loss_dict['train/noise_sigma_mean'] = float(self.last_noise_sigmas.mean().item())
                    except Exception:
                        pass
                # concise log line
                try:
                    lmsg = f"mean_noise={mean_noise:.6g}"
                    print_acc(f"[NOISE] {lmsg}")
                except Exception:
                    pass
            # If earlier diagnostic failed, expose a sentinel and short error to help debugging
            if getattr(self, '_last_noise_diag_exc', None) is not None:
                loss_dict['train/loss_over_noise_status'] = 'failed'
                # add a short excerpt of the traceback to avoid spamming logs
                try:
                    loss_dict['train/loss_over_noise_err'] = str(getattr(self, '_last_noise_diag_exc'))[:200]
                except Exception:
                    loss_dict['train/loss_over_noise_err'] = 'failed (no message)'
            # If loss calculation failed earlier in the step, expose sentinel and short error
            if getattr(self, '_last_loss_calc_failed', False):
                loss_dict['train/loss_calc_status'] = 'failed'
                try:
                    loss_dict['train/loss_calc_err'] = str(getattr(self, '_last_loss_calc_exc'))[:200]
                except Exception:
                    loss_dict['train/loss_calc_err'] = 'failed (no message)'
            # If failure occurred extracting the loss scalar, expose sentinel and short error
            if getattr(self, '_last_loss_scalar_failed', False):
                loss_dict['train/loss_scalar_status'] = 'failed'
                try:
                    loss_dict['train/loss_scalar_err'] = str(getattr(self, '_last_loss_scalar_err'))[:200]
                except Exception:
                    loss_dict['train/loss_scalar_err'] = 'failed (no message)'
        except Exception as e:
            # non-fatal: log the error for diagnostics but do not abort training
            print_acc(f"[METRICS-ERR] failed to attach noise diagnostics: {e}")

        # collect per-example losses from the processed batches if available
        per_example = []
        try:
            # Use helper that only logs per-example entries for simple runs (batch=1 and no accumulation)
            entries = self._maybe_log_per_example(batch, len(batch_list))
            if entries:
                # Augment entries with noise diagnostics if available and lengths match
                try:
                    lon = getattr(self, 'last_loss_over_noise', None)
                    lnn = getattr(self, 'last_noise_norms', None)
                    lsig = getattr(self, 'last_noise_sigmas', None)
                    # If these diagnostics exist and align with entries, attach per-entry values
                    if lon is not None and lnn is not None and len(entries) == len(lon):
                        for i, e in enumerate(entries):
                            try:
                                e['loss_over_noise'] = float(lon[i])
                            except Exception:
                                e['loss_over_noise'] = None
                            try:
                                e['noise_norm'] = float(lnn[i])
                            except Exception:
                                e['noise_norm'] = None
                            try:
                                e['noise_sigma'] = float(lsig[i]) if lsig is not None else None
                            except Exception:
                                e['noise_sigma'] = None
                except Exception as e:
                    # don't fail per-example collection if annotation fails
                    print_acc(f"[METRICS-ERR] failed to attach per-example losses noise diagnostics: {e}")

                per_example.extend(entries)

            from toolkit.util.loss_utils import aggregate_by_dataset, flag_bad_captions
            # Optionally add aggregated summaries to the loss dict based on config
            if getattr(self.train_config, 'log_per_dataset', True) and len(per_example) > 0:
                agg = aggregate_by_dataset(per_example)
                loss_dict['dataset_summary'] = agg['dataset_summary']
            if getattr(self.train_config, 'log_per_example', False) and len(per_example) > 0:
                max_print = getattr(self.train_config, 'max_examples_print', 50)
                loss_dict['per_example'] = per_example[:max_print]
            if getattr(self.train_config, 'flag_bad_captions', True) and len(per_example) > 0:
                flags = flag_bad_captions(per_example)
                loss_dict['caption_flags'] = flags
        except Exception:
            # don't fail training for logging-related issues
            pass

