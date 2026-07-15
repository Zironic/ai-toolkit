import os
import random
import hashlib
import json
import re
import concurrent.futures
from collections import OrderedDict
from contextlib import contextmanager
from typing import Union, Literal, List, Optional

import numpy as np
from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel

from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader, ConcatDataset

from toolkit import train_tools
from toolkit.basic import value_map, adain, get_mean_std
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.config_modules import GenerateImageConfig, WeightNoiseConfig
from toolkit.data_loader import get_dataloader_datasets, get_dataloader_from_datasets
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
from toolkit.image_utils import show_tensors, show_latents
from toolkit.ip_adapter import IPAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.print import print_acc
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds, normalize_caption_separators
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
from toolkit.train_tools import precondition_model_outputs_flow_match
from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
from toolkit.util.losses import wavelet_loss, stepped_loss
import torch.nn.functional as F
from toolkit.unloader import unload_text_encoder
from toolkit import aux_embed_cache
from PIL import Image
from torchvision.transforms import functional as TF
from toolkit.basic import flush
from toolkit.memory_management.runtime import get_memory_runtime


adapter_transforms = transforms.Compose([
    transforms.ToTensor(),
])


class SDTrainer(BaseSDTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super().__init__(process_id, job, config, **kwargs)
        self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
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
        self._gpu_phase_events = {}
        self._resolution_bucket_counts = {}
        self._resolution_memory_stats = {}
        self._resolution_memory_baseline = None
        # Step-level peak high-water, aggregated across accumulations. The inner
        # per-accumulation sampler resets the global CUDA peak counter, so the
        # live counter only reflects the LAST accumulation. The smart-offload
        # controller governs on the within-step peak, so it must see the max
        # across all accumulations, not just the last one.
        self._step_peak_allocated_bytes = 0
        self._step_peak_reserved_bytes = 0
        self._current_resolution_bucket = 256
        self._checkpoint_autotuner = None
        self._checkpoint_tunable = None
        self._checkpoint_autotuner_off = False
        self._checkpoint_timing_start = None
        # Saved-tensor (autograd) memory probe. Debug-only: a handful of measures
        # per UNet to see what the within-step activation footprint actually is
        # (normal vs DOP forward, attention vs MLP via top shapes), so decisions
        # like FP8-autograd or swapping attention backends are grounded in data.
        # Set AITK_SAVED_TENSOR_PROBE=<n_steps> to capture; 0/unset = off.
        self._saved_tensor_probe_steps = self._read_saved_tensor_probe_steps()
        self._saved_tensor_probe_active = False
        self._saved_tensor_probe_stats = {}
        self._saved_tensor_probe_param_ptrs = None
        self.dfe: Optional[DiffusionFeatureExtractor] = None
        self.unconditional_embeds = None
        self._dop_cache_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        
        if self.train_config.diff_output_preservation:
            if self.trigger_word is None:
                raise ValueError("diff_output_preservation requires a trigger_word to be set")
            if self.network_config is None:
                raise ValueError("diff_output_preservation requires a network to be set")
            if self.train_config.train_text_encoder:
                raise ValueError("diff_output_preservation is not supported with train_text_encoder")
        
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

        self._last_weight_noise_norm: Optional[float] = None

    def before_model_load(self):
        pass
    
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

    # ------------------------------------------------------------------
    # Text-encoder worker support
    # ------------------------------------------------------------------
    def _aux_config_params(self) -> dict:
        """Inputs that determine the aux (blank/trigger/uncond/sample) embeds. Changing
        any of these invalidates the on-disk aux cache. Used identically by the worker
        (writer) and the trainer (reader) so they agree on the cache key."""
        sample_prompts: List[str] = []
        sample_negs: List[Optional[str]] = []
        if (
            self.sample_config is not None
            and getattr(self.sample_config, 'prompts', None) is not None
            and not self.train_config.disable_sampling
        ):
            sample_prompts = list(self.sample_config.prompts)
            if getattr(self.sample_config, 'samples', None) is not None:
                sample_negs = [getattr(s, 'neg', None) for s in self.sample_config.samples]
        return {
            'model': str(self.model_config.name_or_path),
            'arch': str(self.model_config.arch),
            'trigger_word': self.trigger_word,
            'unconditional_prompt': self.train_config.unconditional_prompt,
            'diff_output_preservation': bool(self.train_config.diff_output_preservation),
            'diff_output_preservation_class': self.train_config.diff_output_preservation_class,
            'sample_prompts': sample_prompts,
            'sample_negatives': sample_negs,
        }

    def _expected_aux_sample_count(self) -> int:
        if self.train_config.disable_sampling:
            return 0
        if self.sample_config is None or getattr(self.sample_config, 'prompts', None) is None:
            return 0
        return len(self.sample_config.prompts)


    def _iter_text_cache_file_items(self):
        def walk(ds):
            if ds is None:
                return
            if isinstance(ds, ConcatDataset):
                for child in ds.datasets:
                    yield from walk(child)
                return
            if hasattr(ds, "datasets"):
                for child in ds.datasets:
                    yield from walk(child)
                return
            for item in getattr(ds, "file_list", []) or []:
                yield ds, item

        for root in (getattr(self, "datasets", None), getattr(self, "datasets_reg", None)):
            yield from walk(root)


    def _prepare_file_item_text_cache_signature(self, dataset, file_item):
        from toolkit.util.get_model import get_model_class

        model_class = get_model_class(self.model_config)
        embedding_space = getattr(model_class, "text_embedding_space_version", None)
        if not isinstance(embedding_space, str):
            embedding_space = str(self.model_config.arch)
        file_item.text_embedding_space_version = embedding_space

        # Match the default used by FileItemDTO/TextEmbeddingFileItemDTOMixin.
        if not hasattr(file_item, "text_embedding_version"):
            file_item.text_embedding_version = 1

        # Pre-worker check cannot inspect sd.encode_control_in_text_embeddings.
        # Use a config-derived value if you have one; otherwise preserve whatever
        # the file item already has.
        if not hasattr(file_item, "encode_control_in_text_embeddings"):
            file_item.encode_control_in_text_embeddings = bool(
                getattr(self.model_config, "encode_control_in_text_embeddings", False)
            )

        # Always reload the source. A FileItem may already hold a caption from
        # before its sidecar/captions.json entry was edited; hashing that memoized
        # value would incorrectly let the parent skip the TE worker.
        file_item.refresh_caption_for_text_embedding_cache()
    def dataset_text_embedding_cache_is_ready(self) -> bool:
        if not self.is_caching_text_embeddings:
            return True

        from pathlib import Path
        from toolkit.cache_utils import find_cached_file

        checked = 0

        for dataset, file_item in self._iter_text_cache_file_items():
            self._prepare_file_item_text_cache_signature(dataset, file_item)

            expected = Path(file_item.get_text_embedding_path(recalculate=True))
            if find_cached_file(expected) is None:
                return False
            file_item.is_text_embedding_cached = True
            checked += 1

            if self.train_config.diff_output_preservation:
                from toolkit.prompt_utils import (
                    build_dop_replacement_pairs,
                    apply_dop_replacements,
                )

                pairs = build_dop_replacement_pairs(
                    triggers_csv=self.trigger_word,
                    classes_csv=self.train_config.diff_output_preservation_class,
                    case_insensitive=False,
                )

                dop_caption = apply_dop_replacements(
                    caption=file_item.caption,
                    replacement_pairs=pairs,
                    case_insensitive=False,
                    debug=False,
                )

                dop_expected = Path(
                    file_item.get_text_embedding_path(
                        recalculate=True,
                        dop_caption=dop_caption,
                    )
                )

                # DOP loader waits on the exact DOP path, so use strict lookup here.
                if not dop_expected.exists():
                    return False

        # If there were no dataset file items, do not block aux-only caching.
        return True
    def aux_cache_is_ready(self) -> bool:
        """True only if every embedding required by skip_te is already on disk."""
        aux_ready = aux_embed_cache.aux_cache_is_complete(
            self.save_root,
            aux_embed_cache.compute_aux_config_hash(self._aux_config_params()),
            self._expected_aux_sample_count(),
        )
        if not aux_ready:
            return False

        return self.dataset_text_embedding_cache_is_ready()

    def cache_text_encoder_outputs_to_disk(self):
        """Run in the throwaway TE worker process (model loaded te_only). Encodes and
        persists every text embedding the trainer needs — dataset captions (via the
        dataloader cache), DOP embeds, and the aux blank/trigger/uncond/sample embeds —
        then returns so the process can exit and free the text encoder."""
        self.sd.text_encoder_to(self.device_torch)

        # these are not defaulted on the process; ensure they exist for the DOP path below
        self.data_loader = None
        self.data_loader_reg = None

        # The worker loads te_only (no VAE / clip / control models), so building the
        # dataloader must ONLY trigger text-embedding caching. Disable every other
        # per-dataset caching path here; the trainer process does those with the VAE.
        for ds_cfg in (self.dataset_configs or []):
            ds_cfg.cache_latents = False
            ds_cfg.cache_latents_to_disk = False
            ds_cfg.cache_clip_vision_to_disk = False
            ds_cfg.cache_text_embeddings_to_memory = False
            ds_cfg.controls = []

        # 1) dataset caption embeddings (+reg) -> disk, triggered by building the loaders
        if self.datasets is not None:
            self.data_loader = get_dataloader_from_datasets(
                self.datasets, self.train_config.batch_size, self.sd
            )
        if self.datasets_reg is not None:
            self.data_loader_reg = get_dataloader_from_datasets(
                self.datasets_reg, self.train_config.batch_size, self.sd
            )

        # The worker's only contract is the on-disk cache. Drop normal dataloader
        # references before aux encoding so large text-encoder outputs cannot pile up
        # across datasets in this short-lived process.
        dop_datasets = []
        if self.train_config.diff_output_preservation and self.data_loader is not None:
            dop_datasets = list(get_dataloader_datasets(self.data_loader))
        self.data_loader = None
        self.data_loader_reg = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2) aux embeds (blank / unconditional / trigger / DOP / samples)
        with torch.no_grad():
            encode_kwargs = {}
            if self.sd.encode_control_in_text_embeddings:
                control_image = torch.zeros(
                    (1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype
                )
                if self.sd.has_multiple_control_images:
                    control_image = [control_image]
                encode_kwargs['control_images'] = control_image

            blank = self.sd.encode_prompt("", **encode_kwargs).to('cpu')
            unconditional = blank
            uncond_prompt = self.train_config.unconditional_prompt
            if uncond_prompt is not None and uncond_prompt != "":
                unconditional = self.sd.encode_prompt(uncond_prompt, **encode_kwargs).to('cpu')

            trigger = None
            if self.trigger_word is not None:
                trigger = self.sd.encode_prompt(self.trigger_word, **encode_kwargs).to('cpu')

            # DOP embeds write their own per-file disk cache
            if self.train_config.diff_output_preservation and dop_datasets:
                from toolkit.prompt_utils import build_dop_replacement_pairs
                triggers_csv = self.trigger_word
                classes_csv = self.train_config.diff_output_preservation_class
                self._dop_replacement_pairs = build_dop_replacement_pairs(
                    triggers_csv=triggers_csv, classes_csv=classes_csv, case_insensitive=False
                )
                for dataset in dop_datasets:
                    dataset.precompute_dop_embeddings(
                        triggers_csv=triggers_csv,
                        classes_csv=classes_csv,
                        encode_fn=lambda caption: self.sd.encode_prompt(caption, **encode_kwargs),
                        case_insensitive=False,
                        debug=getattr(self.train_config, 'diff_output_preservation_debug', False),
                    )

            # sample-prompt embeds -> self.sd.sample_prompts_cache (in memory), persisted below
            self.cache_sample_prompts()
            samples = self.sd.sample_prompts_cache or []

        config_hash = aux_embed_cache.compute_aux_config_hash(self._aux_config_params())
        aux_embed_cache.save_aux_embeds(
            self.save_root,
            config_hash,
            blank=blank,
            trigger=trigger,
            unconditional=unconditional,
            samples=samples,
        )
        print_acc(
            f"[te-worker] cached aux embeds (samples={len(samples)}, trigger={'yes' if trigger is not None else 'no'}) to {aux_embed_cache.aux_cache_dir(self.save_root)}"
        )

    def load_cached_text_encoder_outputs_from_disk(self) -> bool:
        """Trainer side: populate the in-memory aux embeds from the worker's on-disk cache.
        Returns True on success, False if the cache is not present/valid for this config."""
        loaded = aux_embed_cache.load_aux_embeds(
            self.save_root,
            aux_embed_cache.compute_aux_config_hash(self._aux_config_params()),
            self._expected_aux_sample_count(),
        )
        if loaded is None:
            return False

        def _to_device(pe):
            if pe is None:
                return None
            return pe.to(self.device_torch, dtype=self.sd.torch_dtype)

        self.cached_blank_embeds = _to_device(loaded.get('blank'))
        self.cached_trigger_embeds = _to_device(loaded.get('trigger'))
        uncond = _to_device(loaded.get('unconditional'))
        if uncond is None:
            uncond = self.cached_blank_embeds
        self.unconditional_embeds = uncond.detach() if uncond is not None else None
        self.sd.sample_prompts_cache = loaded.get('samples') or []
        return True

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

    def hook_before_train_loop(self):
        super().hook_before_train_loop()
        if self.is_caching_text_embeddings and not self._use_cached_te:
            # make sure model is on cpu for this part so we don't oom.
            from toolkit.memory_management.runtime import get_memory_runtime

            arena_runtime = get_memory_runtime(self.sd.unet)
            if arena_runtime is not None:
                arena_runtime.park_residency_for_external_phase()
                arena_runtime.place_permanent_modules("cpu")
            else:
                self.sd.unet.to('cpu')

        # cache unconditional embeds (blank prompt). When a TE worker has already cached
        # everything, self.unconditional_embeds was populated from disk and the text
        # encoder is not loaded — skip the in-process encode.
        if not self._use_cached_te:
            with torch.no_grad():
                kwargs = {}
                if self.sd.encode_control_in_text_embeddings:
                    # just do a blank image for unconditionals
                    control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                    if self.sd.has_multiple_control_images:
                        control_image = [control_image]

                    kwargs['control_images'] = control_image
                self.unconditional_embeds = self.sd.encode_prompt(
                    [self.train_config.unconditional_prompt],
                    long_prompts=self.do_long_prompts,
                    **kwargs
                ).to(
                    self.device_torch,
                    dtype=self.sd.torch_dtype
                ).detach()
        if self.train_config.do_prior_divergence:
            self.do_prior_prediction = True
        # move vae to device if we did not cache latents
        if not self.is_latents_cached:
            self.sd.vae.eval()
            self.sd.vae.to(self.device_torch)
        else:
            # offload it. Already cached
            self.sd.vae.to('cpu')
            flush()
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

                self._clip_image_embeds_unconditional = unconditional_clip_image_embeds

        if self.train_config.negative_prompt is not None:
            if os.path.exists(self.train_config.negative_prompt):
                with open(self.train_config.negative_prompt, 'r') as f:
                    self.negative_prompt_pool = f.readlines()
                    # remove empty
                    self.negative_prompt_pool = [x.strip() for x in self.negative_prompt_pool if x.strip() != ""]
            else:
                # single prompt
                self.negative_prompt_pool = [self.train_config.negative_prompt]

        # handle unload text encoder. When a TE worker already cached everything to disk,
        # the embeds are loaded and the text encoder is not present — skip this entirely.
        if (self.train_config.unload_text_encoder or self.is_caching_text_embeddings) and not self._use_cached_te:
            print_acc("Caching embeddings and unloading text encoder")
            with torch.no_grad():
                if self.train_config.train_text_encoder:
                    raise ValueError("Cannot unload text encoder if training text encoder")
                # cache embeddings
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

                # DOP: Precompute embeddings via dataloader (new pattern)
                if self.train_config.diff_output_preservation:
                    from toolkit.prompt_utils import build_dop_replacement_pairs

                    triggers_csv = self.trigger_word
                    classes_csv = self.train_config.diff_output_preservation_class

                    # Build replacement pairs for fallback encoding
                    self._dop_replacement_pairs = build_dop_replacement_pairs(
                        triggers_csv=triggers_csv,
                        classes_csv=classes_csv,
                        case_insensitive=False
                    )

                    # Delegate precompute to dataloader
                    datasets = get_dataloader_datasets(self.data_loader)
                    for dataset in datasets:
                        dataset.precompute_dop_embeddings(
                            triggers_csv=triggers_csv,
                            classes_csv=classes_csv,
                            encode_fn=lambda caption: self.sd.encode_prompt(caption, **encode_kwargs),
                            case_insensitive=False,
                            debug=getattr(self.train_config, 'diff_output_preservation_debug', False)
                        )

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

        # When a TE worker pre-cached everything, the in-process caching block above was
        # skipped. DOP still needs its per-dataset state (_dop_enabled + replacement pairs)
        # so the dataloader loads the worker-cached DOP embeds for each item. Calling
        # precompute_dop_embeddings here sets that state and returns early on a full cache
        # hit, so no encoding happens and the (absent) text encoder is never touched.
        if self._use_cached_te and self.train_config.diff_output_preservation and self.data_loader is not None:
            from toolkit.prompt_utils import build_dop_replacement_pairs
            triggers_csv = self.trigger_word
            classes_csv = self.train_config.diff_output_preservation_class
            self._dop_replacement_pairs = build_dop_replacement_pairs(
                triggers_csv=triggers_csv,
                classes_csv=classes_csv,
                case_insensitive=False,
            )

            def _dop_no_encode(caption):
                raise RuntimeError(
                    "DOP embedding cache miss in a skip_te trainer: the TE worker should "
                    f"have cached all DOP embeddings. Missing for caption: {caption!r}"
                )

            for dataset in get_dataloader_datasets(self.data_loader):
                dataset.precompute_dop_embeddings(
                    triggers_csv=triggers_csv,
                    classes_csv=classes_csv,
                    encode_fn=_dop_no_encode,
                    case_insensitive=False,
                    debug=getattr(self.train_config, 'diff_output_preservation_debug', False),
                )

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
            self.dfe = load_dfe(
                self.train_config.diffusion_feature_extractor_path, 
                vae=vae,
                sd=self.sd
            )
            self.dfe.to(self.device_torch)
            if hasattr(self.dfe, 'vision_encoder') and self.train_config.gradient_checkpointing:
                # must be set to train for gradient checkpointing to work
                self.dfe.vision_encoder.train()
                self.dfe.vision_encoder.gradient_checkpointing = True
            elif hasattr(self.dfe, 'model') and self.train_config.gradient_checkpointing:
                if hasattr(self.dfe.model, 'enable_gradient_checkpointing'): 
                    self.dfe.model.train()
                    self.dfe.model.enable_gradient_checkpointing()
                if hasattr(self.dfe.model, 'gradient_checkpointing_enable'): 
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing_enable()
                elif hasattr(self.dfe.model, 'gradient_checkpointing'):
                    self.dfe.model.train()
                    self.dfe.model.gradient_checkpointing = True
                else:
                    print_acc("Warning: Could not enable gradient checkpointing on diffusion feature extractor model.")
            else:
                self.dfe.eval()
                
            # enable gradient checkpointing on the vae
            if vae is not None and self.train_config.gradient_checkpointing:
                try:
                    vae.enable_gradient_checkpointing()
                    vae.train()
                except Exception:
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
        elif self.train_config.do_signal_amplification:
            if not self.sd.is_flow_matching:
                raise ValueError("Signal amplification is only supported for flow matching models")
            with torch.no_grad():
                nas = 1.0 - (timesteps / 1000).to(noise.device, dtype=noise.dtype)
                nas = nas * self.train_config.signal_amplification_strength
                while len(nas.shape) < len(noise.shape):
                    nas = nas.unsqueeze(-1)
                aug = batch.latents * nas
                target = noise - (batch.latents + aug)
                target = target.detach()
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
            elif self.dfe.version in [3, 4, 5, 6, 7, 8, 9, 10]:
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
            local_loss_scale = 1.0
            if self.train_config.t0_loss_target or self.train_config.do_fft_loss:
                # do the loss on a stepped timestep 0 prediction
                # doto handle doing priors, preservations, masking, etc
                with torch.no_grad():
                    tv = timesteps.to(noise_pred.device).to(noise_pred.dtype) / 1000.0
                    # expand shape to match noise_pred
                    while len(tv.shape) < len(noise_pred.shape):
                        tv = tv.unsqueeze(-1)
                        # min 0.001
                        tv = torch.clamp(tv, min=0.001)
                
                # step latent, use here or with do_fft_loss
                t0 = noisy_latents - tv * noise_pred
                
                if self.train_config.t0_loss_target:
                    # replace the loss targets and pred
                    target = batch.latents.detach()
                    pred = t0
                    # handle velocity equiv loss if set. This scales t0 loss to match velocity of flowmatchhing loss
                    if self.train_config.t0_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        local_loss_scale = velocity_equiv_weight
                        
                if self.train_config.do_fft_loss:
                    with torch.no_grad():
                        target_mag = torch.fft.rfft2(batch.latents.to(t0.device).float(), norm="ortho").abs()
                    pred_mag = torch.fft.rfft2(t0.float(), norm="ortho").abs()
                    fft_loss = F.mse_loss(pred_mag, target_mag, reduction="none")
                    if self.train_config.do_fft_velocity_equiv_weight:
                        velocity_equiv_weight = (1.0 / torch.clamp(tv, min=0.1) ** 2)
                        fft_loss = fft_loss * velocity_equiv_weight
                    additional_loss += fft_loss.mean()
            if self.train_config.loss_type == "pseudo_huber":
                diff = pred.float() - target.float()
                c=0.01
                loss =(torch.sqrt(diff.pow(2) + c ** 2) - c)
            elif self.train_config.loss_type == "mae":
                loss = torch.nn.functional.l1_loss(pred.float(), target.float(), reduction="none")
            elif self.train_config.loss_type == "wavelet":
                loss = wavelet_loss(pred, batch.latents, noise)
            elif self.train_config.loss_type == "stepped":
                loss = stepped_loss(pred, batch.latents, noise, noisy_latents, timesteps, self.sd.noise_scheduler)
                # the way this loss works, it is low, increase it to match predictable LR effects
                loss = loss * 10.0
            else:
                loss = torch.nn.functional.mse_loss(pred.float(), target.float(), reduction="none")
            
            loss = loss * local_loss_scale
            
            # apply model specific loss scaling
            loss = self.sd.scale_loss(loss)
                
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
        except Exception:
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

        if self.train_config.log_per_file_loss:
            self._log_per_file_loss(batch, loss, timesteps)

        loss = loss.mean()

        # check for audio loss
        if batch.audio_pred is not None and batch.audio_target is not None:
            audio_loss = torch.nn.functional.mse_loss(batch.audio_pred.float(), batch.audio_target.float(), reduction="mean")
            audio_loss = audio_loss * self.train_config.audio_loss_multiplier
            loss = loss + audio_loss

        # check for additional losses
        if self.adapter is not None and hasattr(self.adapter, "additional_loss") and self.adapter.additional_loss is not None:

            loss = loss + self.adapter.additional_loss.mean()
            self.adapter.additional_loss = None

        if self.train_config.target_norm_std:
            # seperate out the batch and channels
            pred_std = noise_pred.std([2, 3], keepdim=True)
            norm_std_loss = torch.abs(self.train_config.target_norm_std_value - pred_std).mean()
            loss = loss + norm_std_loss


        loss = loss + additional_loss
        
        if hasattr(self.sd, "get_additional_loss"):
            additional_model_loss = self.sd.get_additional_loss(pred, target)
            if additional_model_loss is not None:
                loss = loss + additional_model_loss
                self.additional_logs["additional_model_loss"] = additional_model_loss.item()

        if self.train_config.max_loss_debug and self.train_config.max_loss is not None:
            if loss.item() > self.train_config.max_loss:
                print_acc(f"Loss {loss.item()} is greater than max loss {self.train_config.max_loss}. Clipping to max loss.")
                print_acc(f"timesteps: {timesteps}")

        if self.train_config.max_loss is not None:
            loss = torch.clamp(loss, max=self.train_config.max_loss)
        
        return loss

    def _log_per_file_loss(self, batch: 'DataLoaderBatchDTO', per_sample_loss: torch.Tensor, timesteps: torch.Tensor):
        # per_sample_loss is shape [B], aligned with batch.file_items. One JSONL row per sample.
        try:
            if getattr(self, "_per_file_loss_path", None) is None:
                cfg = self.train_config.log_per_file_loss
                if isinstance(cfg, str) and cfg.strip():
                    path = cfg
                else:
                    path = os.path.join(self.save_root, "per_file_loss.jsonl")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self._per_file_loss_path = path

            losses = per_sample_loss.detach().float().flatten().cpu().tolist()
            ts = timesteps.detach().float().flatten().cpu().tolist() if timesteps is not None else None
            rows = []
            for i, fi in enumerate(batch.file_items):
                if i >= len(losses):
                    break
                rows.append(json.dumps({
                    "step": self.step_num,
                    "path": fi.path,
                    "name": os.path.basename(fi.path),
                    "orig_w": fi.width,
                    "orig_h": fi.height,
                    "train_w": fi.crop_width,
                    "train_h": fi.crop_height,
                    "timestep": (ts[i] if ts is not None and i < len(ts) else None),
                    "is_reg": bool(fi.is_reg),
                    "loss": losses[i],
                }))
            with open(self._per_file_loss_path, "a", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
        except Exception as e:
            print_acc(f"[per_file_loss] failed to log: {e}")

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
        with self._gpu_phase(f'normal_backward/{self._current_resolution_bucket}'):
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
        can_disable_adapter = False
        was_adapter_active = False
        if self.adapter is not None and (isinstance(self.adapter, IPAdapter) or
                                         isinstance(self.adapter, ReferenceAdapter) or
                                         (isinstance(self.adapter, CustomAdapter))
        ):
            can_disable_adapter = True
            was_adapter_active = self.adapter.is_active

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
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
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

            if self.network is not None:
                self.network.is_active = False
            if can_disable_adapter:
                self.adapter.is_active = False
            self.sd.unet.eval()
            try:
                prior_pred = self.sd.predict_noise(
                    latents=noisy_latents.to(self.device_torch, dtype=dtype).detach(),
                    conditional_embeddings=embeds_to_use.to(self.device_torch, dtype=dtype).detach(),
                    unconditional_embeddings=unconditional_embeds,
                    timestep=timesteps,
                    guidance_scale=self.train_config.cfg_scale,
                    guidance_embedding_scale=guidance_embedding_scale,
                    rescale_cfg=self.train_config.cfg_rescale,
                    batch=batch,
                    **pred_kwargs  # adapter residuals in here
                )
                if prior_pred is None:
                    raise RuntimeError('base model returned no prior prediction')
            finally:
                if was_unet_training:
                    self.sd.unet.train()
                if can_disable_adapter:
                    self.adapter.is_active = was_adapter_active
                if self.network is not None:
                    self.network.is_active = was_network_active
            if was_unet_training:
                self.sd.unet.train()
            prior_pred = prior_pred.detach()
            # remove the residuals as we wont use them on prediction when matching control
            if match_adapter_assist and 'down_intrablock_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_intrablock_additional_residuals']
            if match_adapter_assist and 'down_block_additional_residuals' in pred_kwargs:
                del pred_kwargs['down_block_additional_residuals']
            if match_adapter_assist and 'mid_block_additional_residual' in pred_kwargs:
                del pred_kwargs['mid_block_additional_residual']

            if can_disable_adapter:
                self.adapter.is_active = was_adapter_active
            # restore network
            # self.network.multiplier = network_weight_list
            if self.network is not None:
                self.network.is_active = was_network_active
        return prior_pred

    def before_unet_predict(self):
        pass

    def after_unet_predict(self):
        pass

    def end_of_training_loop(self):
        pass

    def done_hook(self):
        if self._dop_cache_executor is not None:
            self._dop_cache_executor.shutdown(wait=True)
            self._dop_cache_executor = None

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
        # Primary training prediction is the "normal" forward; everything else
        # building a graph here (DOP/blank preservation, guidance/perturbation)
        # is bucketed separately for the saved-tensor probe.
        probe_label = 'normal_forward' if is_primary_pred else 'dop_forward'
        with self._capture_saved_tensors(probe_label):
            return self.sd.predict_noise(
                latents=noisy_latents.to(self.device_torch, dtype=dtype),
                conditional_embeddings=conditional_embeds.to(self.device_torch, dtype=dtype),
                unconditional_embeddings=unconditional_embeds,
                timestep=timesteps,
                guidance_scale=self.train_config.cfg_scale,
                guidance_embedding_scale=guidance_embedding_scale,
                detach_unconditional=False,
                rescale_cfg=self.train_config.cfg_rescale,
                bypass_guidance_embedding=self.train_config.bypass_guidance_embedding,
                batch=batch,
                **kwargs
            )
    

    @contextmanager
    def _gpu_phase(self, name):
        enabled = (
            self.performance_log_every > 0
            and torch.cuda.is_available()
            and self.device_torch.type == 'cuda'
        )
        if not enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.device_torch)
        start.record(stream)
        try:
            yield
        except BaseException:
            # Recording another CUDA event while unwinding an OOM can itself
            # fail and replace the useful original exception. An incomplete
            # phase is not a valid timing sample, so discard it unchanged.
            raise
        else:
            end.record(stream)
            self._gpu_phase_events.setdefault(name, []).append((start, end))

    def _performance_resolution_bucket(self, batch):
        items = getattr(batch, 'file_items', None)
        if items:
            height = getattr(items[0], 'crop_height', None)
            width = getattr(items[0], 'crop_width', None)
            if height and width:
                pixel_count = int(height) * int(width)
                return min(
                    (256, 512, 768, 1024),
                    key=lambda size: abs(pixel_count - size * size),
                )
        latents = getattr(batch, 'latents', None)
        tensor = latents if latents is not None else getattr(batch, 'tensor', None)
        if tensor is None or tensor.ndim < 4:
            return 256
        height, width = tensor.shape[-2:]
        if latents is not None:
            scale = getattr(self.sd, 'vae_scale_factor', 8) or 8
            height, width = height * scale, width * scale
        pixel_count = int(height) * int(width)
        return min((256, 512, 768, 1024), key=lambda size: abs(pixel_count - size * size))

    def _find_checkpoint_tunable(self):
        """Find the module exposing selective-checkpointing (_checkpoint_keep_last
        + a .blocks list), e.g. Krea's SingleStreamDiT, under the transformer."""
        root = getattr(self.sd, 'unet', None)
        if root is None:
            return None
        if hasattr(root, '_checkpoint_keep_last') and hasattr(root, 'blocks'):
            return root
        for module in root.modules():
            if hasattr(module, '_checkpoint_keep_last') and hasattr(module, 'blocks'):
                return module
        return None

    def _ensure_checkpoint_autotuner(self):
        """Lazily build the autotuner when checkpoint_keep_last is set to -1 (auto)."""
        if self._checkpoint_autotuner is not None:
            return self._checkpoint_autotuner
        if self._checkpoint_autotuner_off:
            return None
        keep_last_cfg = getattr(
            self.model_config, 'layer_offloading_checkpoint_keep_last', 0
        )
        if keep_last_cfg != -1 or not (
            torch.cuda.is_available() and self.device_torch.type == 'cuda'
        ):
            self._checkpoint_autotuner_off = True
            return None
        module = self._find_checkpoint_tunable()
        if module is None or not getattr(module, 'gradient_checkpointing', False):
            self._checkpoint_autotuner_off = True
            return None
        from toolkit.memory_management import vram_budget
        from toolkit.memory_management.checkpoint_autotuner import (
            CheckpointKeepLastAutotuner,
        )
        total = vram_budget.device_total_bytes(self.device_torch)
        max_keep = max(0, len(module.blocks) - 1)

        def _set(n, _m=module):
            n = int(n)
            if _m._checkpoint_keep_last != n:
                _m._checkpoint_keep_last = n
                # Selective checkpointing changes access order. Re-record now,
                # before this step's first model access; the tuner treats this
                # candidate's first step as an untimed prefetch warmup.
                from toolkit.memory_management import MemoryManager
                MemoryManager.set_training_pinned_resident_blocks(_m, n)
                MemoryManager.reset_offload_trace_for_tuning()

        self._checkpoint_tunable = module
        self._checkpoint_autotuner = CheckpointKeepLastAutotuner(
            total, max_keep, set_keep_last=_set
        )
        print(
            f"[CheckpointAutotune] time-based keep_last enabled: blocks={len(module.blocks)} "
            f"device_total={total / 1024 ** 3:.2f} GiB"
        )
        return self._checkpoint_autotuner

    def _memory_sample_active(self):
        return (
            torch.cuda.is_available()
            and self.device_torch.type == 'cuda'
            and (
                self.performance_log_every > 0
                or self._checkpoint_autotuner is not None
            )
        )

    def _start_resolution_memory_sample(self):
        if not self._memory_sample_active():
            self._resolution_memory_baseline = None
            self._checkpoint_timing_start = None
            return
        torch.cuda.reset_peak_memory_stats(self.device_torch)
        self._resolution_memory_baseline = torch.cuda.memory_allocated(self.device_torch)
        if (
            self._checkpoint_autotuner is not None
            and not self._checkpoint_autotuner.is_settled(
                self._current_resolution_bucket
            )
        ):
            self._checkpoint_timing_start = torch.cuda.Event(enable_timing=True)
            self._checkpoint_timing_start.record(
                torch.cuda.current_stream(self.device_torch)
            )

    def _finish_resolution_memory_sample(self, bucket):
        baseline = self._resolution_memory_baseline
        if baseline is None:
            return
        step_time_s = None
        if self._checkpoint_timing_start is not None:
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_end.record(torch.cuda.current_stream(self.device_torch))
            timing_end.synchronize()
            step_time_s = self._checkpoint_timing_start.elapsed_time(timing_end) / 1000.0
            self._checkpoint_timing_start = None
        allocated = torch.cuda.max_memory_allocated(self.device_torch)
        reserved = torch.cuda.max_memory_reserved(self.device_torch)
        # Carry this accumulation's peak into the step-level high-water before the
        # next accumulation's _start resets the global counter. The controller
        # reads max(this accumulator, live counter), so earlier accumulations'
        # peaks are not lost on multi-accumulation steps.
        self._step_peak_allocated_bytes = max(self._step_peak_allocated_bytes, allocated)
        self._step_peak_reserved_bytes = max(self._step_peak_reserved_bytes, reserved)
        incremental = max(0, allocated - baseline)
        # Time chooses the candidate; reserved memory is only the hard WDDM
        # spill guard. The CUDA event covers this accumulation's forward and
        # backward without data-loading noise.
        if self._checkpoint_autotuner is not None:
            crossed_spill_guard = self._checkpoint_autotuner.observe(
                bucket, reserved, step_time_s or 0.0
            )
            if crossed_spill_guard:
                # The rejected candidate's cached segments would otherwise keep
                # memory_reserved above the ceiling after we back off.
                torch.cuda.empty_cache()
        if self.performance_log_every <= 0:
            self._resolution_memory_baseline = None
            return
        stats = self._resolution_memory_stats.setdefault(bucket, {
            'count': 0, 'allocated_sum': 0, 'allocated_max': 0,
            'reserved_sum': 0, 'reserved_max': 0,
            'incremental_sum': 0, 'incremental_max': 0,
        })
        stats['count'] += 1
        stats['allocated_sum'] += allocated
        stats['allocated_max'] = max(stats['allocated_max'], allocated)
        stats['reserved_sum'] += reserved
        stats['reserved_max'] = max(stats['reserved_max'], reserved)
        stats['incremental_sum'] += incremental
        stats['incremental_max'] = max(stats['incremental_max'], incremental)
        self._resolution_memory_baseline = None

    @staticmethod
    def _read_saved_tensor_probe_steps():
        raw = os.environ.get('AITK_SAVED_TENSOR_PROBE', '0')
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            # Truthy-but-not-an-int (e.g. "true") → a few measures.
            return 3 if str(raw).strip().lower() in ('true', 'yes', 'on') else 0

    @staticmethod
    def _storage_ptr(t):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return t.storage().data_ptr()

    def _saved_tensor_probe_param_ptr_set(self):
        """Storage data_ptrs of the transformer's parameters, so saved weight
        tensors can be told apart from saved activations (the streamed/FP8
        weights we already account for in resident+ring vs. true activations)."""
        if self._saved_tensor_probe_param_ptrs is None:
            ptrs = set()
            unet = getattr(self.sd, 'unet', None)
            if unet is not None:
                for p in unet.parameters():
                    try:
                        ptrs.add(self._storage_ptr(p.data))
                    except Exception:
                        pass
            self._saved_tensor_probe_param_ptrs = ptrs
        return self._saved_tensor_probe_param_ptrs

    @contextmanager
    def _capture_saved_tensors(self, label):
        """Tally autograd saved-tensor footprint for the wrapped forward.

        Dedups by storage (saved tensors routinely alias one allocation, so a
        naive numel*element_size sum over-counts wildly) and splits weight saves
        from activation saves. Read-only: the unpack hook returns the tensor
        unchanged, so this does not alter what is kept alive or move any memory.
        """
        if not self._saved_tensor_probe_active:
            yield
            return
        bucket = self._saved_tensor_probe_stats.setdefault(label, {
            'param_bytes': 0, 'param_count': 0,
            'act_bytes': 0, 'act_count': 0, 'by_shape': {},
        })
        seen = set()
        param_ptrs = self._saved_tensor_probe_param_ptr_set()

        def pack_hook(t):
            try:
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    ptr = self._storage_ptr(t)
                    if ptr not in seen:
                        seen.add(ptr)
                        nbytes = t.numel() * t.element_size()
                        if ptr in param_ptrs:
                            bucket['param_bytes'] += nbytes
                            bucket['param_count'] += 1
                        else:
                            bucket['act_bytes'] += nbytes
                            bucket['act_count'] += 1
                            key = (str(t.dtype).replace('torch.', ''), tuple(t.shape))
                            bucket['by_shape'][key] = bucket['by_shape'].get(key, 0) + nbytes
            except Exception:
                pass
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
            yield

    def _emit_saved_tensor_report(self):
        stats = self._saved_tensor_probe_stats
        if not stats:
            return
        gib = 1024 ** 3
        peak_alloc = (
            torch.cuda.max_memory_allocated(self.device_torch) / gib
            if torch.cuda.is_available() else 0.0
        )
        lines = [
            "",
            "============== saved-tensor probe (autograd activation footprint) ==============",
        ]
        if getattr(self.model_config, 'compile', False):
            lines.append(
                "WARNING: model_config.compile is on — compiled blocks bypass these hooks, "
                "so activation totals UNDERCOUNT. Probe with compile off for true numbers."
            )
        total_act = 0
        for label, b in stats.items():
            total_act += b['act_bytes']
            lines.append(
                f"[{label}] activation_saved={b['act_bytes'] / gib:.3f} GiB "
                f"({b['act_count']} tensors)   weight_saved={b['param_bytes'] / gib:.3f} GiB "
                f"({b['param_count']} tensors)"
            )
            top = sorted(b['by_shape'].items(), key=lambda kv: kv[1], reverse=True)[:8]
            for (dt, shape), nb in top:
                lines.append(f"      {nb / gib:.3f} GiB  {dt} {list(shape)}")
        lines.append(
            f"activation_saved_total={total_act / gib:.3f} GiB   "
            f"peak_allocated={peak_alloc:.3f} GiB   "
            "(saved ⊂ peak; remainder = attention/GEMM workspace + scratch)"
        )
        lines.append(
            "================================================================================"
        )
        print_acc("\n".join(lines))

    def train_single_accumulation(self, batch: DataLoaderBatchDTO):
        resolution_bucket = self._performance_resolution_bucket(batch)
        self._current_resolution_bucket = resolution_bucket
        # Arm the saved-tensor probe for this accumulation (a few measures total).
        self._saved_tensor_probe_active = (
            self._saved_tensor_probe_steps > 0
            and torch.cuda.is_available()
            and self.device_torch.type == 'cuda'
        )
        if self._saved_tensor_probe_active:
            self._saved_tensor_probe_stats = {}
        if self.performance_log_every > 0:
            self._resolution_bucket_counts[resolution_bucket] = (
                self._resolution_bucket_counts.get(resolution_bucket, 0) + 1
            )
        # Auto selective-checkpointing: pick keep_last for this resolution before
        # the forward reads it. Hill-climb elapsed time; reserved memory is only
        # the hard spill ceiling.
        tuner = self._ensure_checkpoint_autotuner()
        if tuner is not None:
            tuner.recommend(resolution_bucket)
        self._start_resolution_memory_sample()
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
                        adapter_images = batch.control_tensor.to(self.device_torch, dtype=dtype).detach()
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
            if batch.mask_tensor is not None and self.sd.do_masked_loss:
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
                    # make avg 1.0
                    mask_multiplier = mask_multiplier / mask_multiplier.mean()

        def get_adapter_multiplier():
            if self.adapter and isinstance(self.adapter, T2IAdapter):
                # training a t2i adapter, not using as assistant.
                return 1.0
            elif match_adapter_assist:
                # training a texture. We want it high
                adapter_strength_min = 0.9
                adapter_strength_max = 1.0
            else:
                # training with assistance, we want it low
                # adapter_strength_min = 0.4
                # adapter_strength_max = 0.7
                adapter_strength_min = 0.5
                adapter_strength_max = 1.1

            adapter_conditioning_scale = torch.rand(
                (1,), device=self.device_torch, dtype=dtype
            )

            adapter_conditioning_scale = value_map(
                adapter_conditioning_scale,
                0.0,
                1.0,
                adapter_strength_min,
                adapter_strength_max
            )
            return adapter_conditioning_scale

        # flush()
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
                    if self.sd.encode_control_in_text_embeddings and batch.control_tensor is not None:
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
                            if self.train_config.diff_output_preservation:

                                if batch.dop_prompt_embeds is not None:
                                    # use the cached embeds
                                    self.diff_output_preservation_embeds = batch.dop_prompt_embeds.clone().detach().to(
                                        self.device_torch, dtype=dtype
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
                            ).to(
                                self.device_torch,
                                dtype=dtype)

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
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
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
                            if self.sd.encode_control_in_text_embeddings and batch.control_tensor_list is not None:
                                prompt_kwargs['control_images'] = batch.control_tensor_list
                            conditional_embeds = self.sd.encode_prompt(
                                conditioned_prompts, prompt_2,
                                dropout_prob=self.train_config.prompt_dropout_prob,
                                long_prompts=self.do_long_prompts,
                                **prompt_kwargs
                            ).to(
                                self.device_torch,
                                dtype=dtype)
                            if self.train_config.do_cfg:
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = True
                                unconditional_embeds = self.sd.encode_prompt(
                                    self.batch_negative_prompt,
                                    dropout_prob=self.train_config.prompt_dropout_prob,
                                    long_prompts=self.do_long_prompts,
                                    **prompt_kwargs
                                ).to(
                                    self.device_torch,
                                    dtype=dtype)
                                if isinstance(self.adapter, CustomAdapter):
                                    self.adapter.is_unconditional_run = False
                            
                            if self.train_config.diff_output_preservation:
                                # Use dataloader-provided DOP embeddings (new pattern)
                                self.diff_output_preservation_embeds = batch.dop_prompt_embeds

                                # Fallback: encode on-the-fly if cache missing (text encoder still available in this branch)
                                if self.diff_output_preservation_embeds is None:
                                    from toolkit.prompt_utils import apply_dop_replacements
                                    print_acc("[DOP] Cache missing for batch - encoding on-the-fly")

                                    # Apply trigger→class replacements using utility
                                    dop_prompts = [
                                        apply_dop_replacements(p, self._dop_replacement_pairs, debug=False)
                                        for p in conditioned_prompts
                                    ]
                                    dop_prompts_2 = None
                                    if prompt_2 is not None:
                                        dop_prompts_2 = [
                                            apply_dop_replacements(p, self._dop_replacement_pairs, debug=False)
                                            for p in prompt_2
                                        ]
                                    self.diff_output_preservation_embeds = self.sd.encode_prompt(
                                        dop_prompts, dop_prompts_2,
                                        dropout_prob=self.train_config.prompt_dropout_prob,
                                        long_prompts=self.do_long_prompts,
                                        **prompt_kwargs
                                    ).to(
                                        self.device_torch,
                                        dtype=dtype)
                                else:
                                    # Move cached embeddings to device
                                    self.diff_output_preservation_embeds = self.diff_output_preservation_embeds.to(
                                        self.device_torch,
                                        dtype=dtype
                                    )
                        # detach the embeddings
                        conditional_embeds = conditional_embeds.detach()
                        if self.train_config.do_cfg:
                            unconditional_embeds = unconditional_embeds.detach()
                    
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
                pred_kwargs = {}

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
                            patch_min = self._dop_patch_min()
                            if patch_min > 1:
                                target_h = max(patch_min, int(round(target_h / patch_min)) * patch_min)
                                target_w = max(patch_min, int(round(target_w / patch_min)) * patch_min)
                            return (target_h < H) or (target_w < W)

                        # Decide via helper whether we can skip the full-resolution prior.
                        skip_full_prior = self._should_skip_full_prior(noisy_latents, preservation_resolution, do_reg_prior=do_reg_prior)

                        if skip_full_prior:
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
                    if (self.adapter and isinstance(self.adapter, ControlNetModel)) or (
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
                    unconditional_embeds = unconditional_embeds.to(self.device_torch, dtype=dtype).detach()
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
                            next_sample_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
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
                        with self._gpu_phase(
                            f'normal_training_forward/{self._current_resolution_bucket}'
                        ):
                            noise_pred = self.predict_noise(
                                noisy_latents=noisy_latents.to(self.device_torch, dtype=dtype),
                                timesteps=timesteps,
                                conditional_embeds=conditional_embeds.to(self.device_torch, dtype=dtype),
                                unconditional_embeds=unconditional_embeds,
                                batch=batch,
                                is_primary_pred=True,
                                **pred_kwargs
                            )
                    self.after_unet_predict()

                    with self.timer('calculate_loss'):
                        noise = noise.to(self.device_torch, dtype=dtype).detach()
                        prior_to_calculate_loss = prior_pred
                        # if we are doing diff_output_preservation and not noing inverted masked prior
                        # then we need to send none here so it will not target the prior
                        doing_preservation = self.train_config.diff_output_preservation or self.train_config.blank_prompt_preservation
                        if doing_preservation and not do_inverted_masked_prior:
                            prior_to_calculate_loss = None
                        
                        loss = self.calculate_loss(
                            noise_pred=noise_pred,
                            noise=noise,
                            noisy_latents=noisy_latents,
                            timesteps=timesteps,
                            batch=batch,
                            mask_multiplier=mask_multiplier,
                            prior_pred=prior_to_calculate_loss,
                        )
                    
                    # Check if DOP is actually available this step (embeds might be missing if cache failed)
                    do_dop_this_step = self.train_config.diff_output_preservation and self.diff_output_preservation_embeds is not None
                    if self.train_config.diff_output_preservation and self.diff_output_preservation_embeds is None:
                        print_acc("[DOP WARNING] diff_output_preservation enabled but embeddings missing - skipping preservation loss this step")

                    if do_dop_this_step or self.train_config.blank_prompt_preservation:
                        single_backward = self.train_config.dop_single_backward
                        if single_backward:
                            # Keep the main loss graph alive so it can be combined with the
                            # preservation loss into one backward at the end of the step.
                            normal_loss = loss
                        else:
                            # Two-pass: send the loss backwards now (frees its graph before the
                            # preservation forward, keeping peak VRAM low) otherwise checkpointing will fail
                            with self._gpu_phase(
                                f'normal_backward/{self._current_resolution_bucket}'
                            ):
                                self.accelerator.backward(loss)
                            normal_loss = loss.detach() # dont send backward again

                        # Determine preservation embeddings and resolution
                        preservation_embeds = None
                        preservation_resolution = None
                        preservation_kind = None

                        with torch.no_grad():
                            if do_dop_this_step:
                                preservation_embeds = self.diff_output_preservation_embeds.expand_to_batch(noisy_latents.shape[0])
                                preservation_resolution = getattr(self.train_config, 'diff_output_preservation_resolution', None)
                                preservation_kind = 'dop'
                            elif self.train_config.blank_prompt_preservation:
                                blank_embeds = self.cached_blank_embeds.clone().detach().to(
                                    self.device_torch, dtype=dtype
                                )
                                preservation_embeds = concat_prompt_embeds(
                                    [blank_embeds] * noisy_latents.shape[0]
                                )
                                preservation_resolution = getattr(self.train_config, 'blank_prompt_preservation_resolution', None)
                                preservation_kind = 'blank'

                        # Run preservation forward with optional downsampling
                        with torch.set_grad_enabled(True):
                            with self._gpu_phase('dop_section'):
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

                        # Handle tuple return (preservation_pred, prior_pred_for_loss) when downsampling
                        if isinstance(preservation_pred_res, tuple):
                            preservation_pred, prior_pred_for_loss = preservation_pred_res
                        else:
                            preservation_pred = preservation_pred_res
                            prior_pred_for_loss = prior_pred

                        # Compute preservation loss; in two-pass mode this also backprops it.
                        multiplier = self.train_config.diff_output_preservation_multiplier if do_dop_this_step else self.train_config.blank_prompt_preservation_multiplier
                        preservation_loss = self._compute_and_apply_preservation_loss(
                            preservation_pred, prior_pred_for_loss, multiplier,
                            apply_backward=not single_backward,
                        )

                        if single_backward:
                            # Both losses are still graph-connected; sum them and let the
                            # trailing self.accelerator.backward(loss) run a single combined pass.
                            if preservation_loss is not None:
                                loss = normal_loss + preservation_loss
                            else:
                                loss = normal_loss
                        else:
                            # Two-pass: gradients are already applied; rebuild a detached loss
                            # purely for logging / the nan-check below (trailing backward is a no-op).
                            if preservation_loss is not None:
                                loss = normal_loss + preservation_loss.detach()
                            else:
                                loss = normal_loss
                            loss = loss.clone().detach()
                            # require grad again so the backward wont fail
                            loss.requires_grad_(True)

                # check if nan
                if torch.isnan(loss):
                    print_acc("loss is nan")
                    loss = torch.zeros_like(loss).requires_grad_(True)

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
                    backward_phase = (
                        'combined_backward'
                        if self.train_config.dop_single_backward
                        else f'normal_backward/{self._current_resolution_bucket}'
                    )
                    with self._gpu_phase(backward_phase):
                        self.accelerator.backward(loss)

        self._finish_resolution_memory_sample(resolution_bucket)
        if self._saved_tensor_probe_active:
            self._emit_saved_tensor_report()
            self._saved_tensor_probe_steps -= 1
            self._saved_tensor_probe_active = False
        return loss.detach()
        # flush()

    def _dop_patch_size(self) -> int:
        """Patchify factor used when computing image_seq_len for the flow-match shift.

        Mirrors the logic in BaseSDTrainProcess.process_general_training_batch: flux/flex/zimage
        latents are divided by 2 (patch_size 2), then a model-declared self.patch_size (e.g. krea2),
        then unet.config.patch_size, default 1.
        """
        sd = self.sd
        try:
            if getattr(sd, 'is_flux', False) or 'flex' in getattr(sd, 'arch', '') or getattr(sd, 'arch', '') == 'zimage':
                return 2
            ps = getattr(sd, 'patch_size', None)
            if ps is not None:
                return int(ps)
            unet = getattr(sd, 'unet', None)
            if unet is not None and hasattr(unet, 'config') and hasattr(unet.config, 'patch_size'):
                return int(unet.config.patch_size)
        except Exception:
            pass
        return 1

    def _dop_reproject_timesteps(self, timesteps, main_h, main_w, dop_h, dop_w):
        """Re-project the main-resolution timestep(s) onto the reduced DOP resolution's dynamic-shift
        curve so the preservation forward runs at the noise level appropriate to that resolution.

        The flow-match shift is resolution-aware: image_seq_len scales with latent area, so a smaller
        DOP resolution yields a smaller shift (mu) and a lower average noise level. The live path
        otherwise reuses the main-resolution timestep, which applies main-resolution noise to the
        downsampled latents. This is a quantile-preserving remap: invert the main-res shift to recover
        the base quantile, then re-apply the shift at the DOP resolution. It preserves the configured
        content/style timestep weighting and adds no new RNG.

        No-op (returns timesteps unchanged) unless the scheduler uses plain dynamic shifting; exotic
        sigma transforms (karras/exponential/beta/terminal/invert) break the closed-form remap.
        """
        sched = self.sd.noise_scheduler
        try:
            cfg = sched.config
            if not bool(cfg.get('use_dynamic_shifting', False)):
                return timesteps
            if self.train_config.timestep_type not in ('shift', 'flux_shift', 'lumina2_shift'):
                return timesteps
            if (cfg.get('use_karras_sigmas') or cfg.get('use_exponential_sigmas')
                    or cfg.get('use_beta_sigmas') or cfg.get('shift_terminal') or cfg.get('invert_sigmas')):
                return timesteps
            if not hasattr(sched, 'time_shift'):
                return timesteps
        except Exception:
            return timesteps

        from toolkit.samplers.custom_flowmatch_sampler import calculate_shift

        patch_size = self._dop_patch_size()

        def _mu(h, w):
            seq_len = (h * w) // (patch_size ** 2)
            mu = calculate_shift(
                seq_len,
                cfg.get('base_image_seq_len', 256),
                cfg.get('max_image_seq_len', 4096),
                cfg.get('base_shift', 0.5),
                cfg.get('max_shift', 1.16),
            )
            min_shift = getattr(sched, '_min_shift', None)
            if min_shift is not None:
                mu = max(mu, min_shift)
            return float(mu)

        try:
            mu_main = _mu(main_h, main_w)
            mu_dop = _mu(dop_h, dop_w)
            if abs(mu_main - mu_dop) < 1e-6:
                return timesteps
            s_main = math.exp(mu_main)
            sigma = (timesteps.float() / 1000.0).clamp(1e-6, 1.0 - 1e-6)
            # invert the main-resolution shift to the base quantile
            base = (sigma / (s_main + sigma * (1.0 - s_main))).clamp(1e-6, 1.0 - 1e-6)
            # re-apply the shift at the DOP resolution using the scheduler's own transform
            shifted = sched.time_shift(mu_dop, 1.0, base)
            return (shifted * 1000.0).to(dtype=timesteps.dtype)
        except Exception:
            return timesteps

    def _dop_patch_min(self) -> int:
        """Latent-space patch divisibility for DOP downsample targets.

        DOP downsamples the latents and re-patchifies them through the transformer, so
        the reduced latent dims must stay divisible by the transformer patch size or the
        patchify rearrange fails (e.g. an odd latent dim with patch=2). Prefer the
        transformer's advertised ``all_patch_size``; then the model's ``patch_size``
        (single-stream DiTs like krea2 only expose this, not ``all_patch_size``); else 1.
        """
        try:
            tr = getattr(self.sd, 'transformer', None) or getattr(self.sd, 'unet', None)
            all_patch = getattr(tr, 'all_patch_size', None) if tr is not None else None
            if all_patch:
                return max(1, int(min(all_patch)))
            ps = getattr(self.sd, 'patch_size', None)
            if isinstance(ps, (list, tuple)) and ps:
                return max(1, int(min(ps)))
            if ps:
                return max(1, int(ps))
        except Exception:
            pass
        return 1

    def _dop_prior_cache_hash(self, file_item, target_h, target_w, samples):
        """Return the stable namespace digest for one image and DOP configuration."""
        caption = getattr(file_item, '_dop_transformed_caption', None)
        if caption is None:
            caption = normalize_caption_separators(
                self._map_triggers_to_classes_in_text(file_item.caption or '')
            )
        params = {'version': 2, 'prompt': caption, 'size': [target_h, target_w], 'samples': samples}
        params['model'] = str(self.model_config.name_or_path_original)
        params['arch'] = self.model_config.arch
        params['qtype'] = getattr(self.model_config, 'qtype', None)
        params['dtype'] = getattr(self.model_config, 'dtype', None)
        params['patch_size'] = self._dop_patch_size()
        params['timestep_type'] = getattr(self.train_config, 'timestep_type', None)
        params['content_or_style'] = getattr(self.train_config, 'content_or_style', None)
        params['scheduler'] = dict(getattr(self.sd.noise_scheduler, 'config', {}))
        return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _dop_prior_cache_valid(state, target_h, target_w):
        try:
            noisy, ts, prior = state['noisy_latents'], state['timesteps'], state['prior_predictions']
            return (noisy.ndim == 4 and prior.shape == noisy.shape and ts.ndim == 1
                    and noisy.shape[0] == ts.shape[0]
                    and tuple(noisy.shape[-2:]) == (target_h, target_w))
        except (KeyError, AttributeError):
            return False

    def _dop_prior_cache_load_batch(self, batch, target_h, target_w, samples, dtype):
        """Load one cached sample for every image, or return None on any miss."""
        file_items = getattr(batch, 'file_items', None)
        if not file_items:
            return
        chosen = []
        for item in file_items:
            path = item.get_dop_prior_path(self._dop_prior_cache_hash(item, target_h, target_w, samples))
            if not os.path.exists(path):
                return None
            try:
                state = load_file(path, device='cpu')
            except Exception:
                return None
            if not self._dop_prior_cache_valid(state, target_h, target_w):
                return None
            if state['timesteps'].shape[0] < samples:
                return None
            idx = random.randrange(state['timesteps'].shape[0])
            chosen.append((state['noisy_latents'][idx], state['timesteps'][idx], state['prior_predictions'][idx]))
        torch_dtype = get_torch_dtype(dtype)
        noisy = torch.stack([x[0] for x in chosen]).to(self.device_torch, dtype=torch_dtype)
        ts = torch.stack([x[1] for x in chosen]).to(self.device_torch)
        prior = torch.stack([x[2] for x in chosen]).to(self.device_torch, dtype=torch_dtype)
        return noisy, ts, prior

    def _dop_prior_cache_add_batch(self, batch, target_h, target_w, samples,
                                   noisy_small, timesteps, prior_small):
        """Append each live image result to its persistent cache until full.

        GPU→CPU transfers are non-blocking; the actual disk write runs in a
        background thread so the caller can submit loss kernels to the GPU
        immediately rather than stalling during file I/O.
        """
        items = getattr(batch, 'file_items', None)
        if not items or len(items) != prior_small.shape[0]:
            return

        # Shape checks use tensor metadata only — no data access, no sync.
        ts_gpu = timesteps.detach().float().reshape(-1)
        if ts_gpu.shape[0] == 1 and len(items) > 1:
            ts_gpu = ts_gpu.expand(len(items)).contiguous()
        if ts_gpu.shape[0] != len(items):
            return

        # Kick off async GPU→CPU copies. Data is not valid until event fires.
        ts_cpu = ts_gpu.to('cpu', non_blocking=True)
        noisy_cpu = noisy_small.detach().to('cpu', non_blocking=True)
        prior_cpu = prior_small.detach().to('cpu', non_blocking=True)

        # Record a CUDA event so the worker knows when the transfers are done.
        event = torch.cuda.Event()
        event.record()

        paths = [
            item.get_dop_prior_path(self._dop_prior_cache_hash(item, target_h, target_w, samples))
            for item in items
        ]

        if self._dop_cache_executor is None:
            self._dop_cache_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _write():
            try:
                event.synchronize()
                self._dop_prior_cache_write_items(
                    paths, target_h, target_w, samples, ts_cpu, noisy_cpu, prior_cpu
                )
            except Exception as exc:
                print_acc(f'[DOP cache] background write failed: {exc}')

        self._dop_cache_executor.submit(_write)

    def _dop_prior_cache_write_items(self, paths, target_h, target_w, samples,
                                     ts, noisy_cpu, prior_cpu):
        """Write per-image cache entries to disk (called from background thread)."""
        for idx, path in enumerate(paths):
            state = None
            if os.path.exists(path):
                try:
                    state = load_file(path, device='cpu')
                except Exception:
                    pass
            if state is not None and not self._dop_prior_cache_valid(state, target_h, target_w):
                state = None
            if state is None:
                state = {'noisy_latents': noisy_cpu[idx:idx + 1],
                         'timesteps': ts[idx:idx + 1],
                         'prior_predictions': prior_cpu[idx:idx + 1]}
            elif state['timesteps'].shape[0] >= samples:
                continue
            else:
                state = {key: torch.cat((state[key], value[idx:idx + 1]), dim=0).contiguous()
                         for key, value in (('noisy_latents', noisy_cpu), ('timesteps', ts),
                                            ('prior_predictions', prior_cpu))}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temp_path = f'{path}.{os.getpid()}.{random.randrange(1 << 30)}.tmp.safetensors'
            try:
                save_file(state, temp_path, metadata={'format': 'aitk_dop_prior_v2'})
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def _run_preservation_forward(self, noisy_latents, timesteps, preservation_embeds, unconditional_embeds, batch, pred_kwargs, dtype, prior_pred, preservation_resolution=None, preservation_kind: 'Optional[str]'=None, match_adapter_assist: bool = False, network_weight_list: list = None):
        """Run preservation forward pass for DOP/blank prompt preservation and record timings.

        If `preservation_resolution` (pixels, long-side) is specified, the forward pass will be
        executed at that reduced spatial resolution and the returned preservation prediction and
        prior prediction (both downsampled) will be suitable for loss computation.

        `preservation_kind` may be 'dop' or 'blank' to help label timers appropriately.

        Returns preservation_pred (or (preservation_pred, prior_pred_down) when downsampling used) or None.
        """
        # Preservation/DOP loss compares the LoRA-on prediction against the frozen-base
        # prediction. Classifier-free guidance serves no purpose for this comparison: it would
        # double every forward in this path (cond+uncond batch) and amplify the prior target
        # with guidance the base model wouldn't apply. Always run the preservation forwards and
        # the matching prior without CFG, regardless of the training CFG setting.
        unconditional_embeds = None

        # Determine timer base name based on preservation kind
        timer_base = 'blank_predict' if preservation_kind == 'blank' else 'dop_predict'

        # preservation_embeds may be prompt embeds or similar. Move them and latents to device inside the timer
        # Create a shallow copy of `pred_kwargs` to avoid mutating the caller's dict and to allow
        # removal of adapter residuals for preservation predictions when requested.
        local_pred_kwargs = dict(pred_kwargs) if pred_kwargs is not None else {}
        if match_adapter_assist:
            # Remove controlnet influence for preservation loss
            local_pred_kwargs.pop('down_intrablock_additional_residuals', None)
            local_pred_kwargs.pop('down_block_additional_residuals', None)
            local_pred_kwargs.pop('mid_block_additional_residual', None)
            # For Z-Image controlnet (unified model), remove control context entirely.
            local_pred_kwargs.pop('control_context', None)
            local_pred_kwargs['control_context_scale'] = 0.0

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
        target_latent_area = (effective_preservation_resolution / vae_scale) ** 2
        # Preserve aspect ratio: target_h / target_w = H / W
        import math
        aspect_ratio = H / W
        target_h = max(1, int(round(math.sqrt(target_latent_area * aspect_ratio))))
        target_w = max(1, int(round(math.sqrt(target_latent_area / aspect_ratio))))

        # Ensure target dims are compatible with transformer patch sizes (avoid invalid view shapes)
        patch_min = self._dop_patch_min()

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
        # Re-project the main-resolution timestep onto this reduced resolution's dynamic-shift curve
        # so the preservation forward runs at the noise level appropriate to the DOP resolution
        # (lower res -> lower shift -> lower average noise) rather than reusing the main-resolution
        # noise level. No-op unless the scheduler is using plain dynamic shifting. The same remapped
        # timestep feeds the noisy-latent construction, the prior, and the preservation forward so
        # all three stay consistent.
        dop_timesteps = self._dop_reproject_timesteps(
            timesteps, noisy_latents.shape[2], noisy_latents.shape[3], target_h, target_w
        )

        # Frozen-base prior cache (DOP only). The prior is a LoRA-disabled forward, so it is
        # deterministic and reusable. Disabled when extra per-batch conditioning is present
        # (local_pred_kwargs non-empty, e.g. controlnet) since the cached noisy latent wouldn't
        # match the current batch's conditioning. Cache files are keyed by latent identity plus
        # prompt/model/resolution/scheduler configuration.
        cache_on = (
            getattr(self.train_config, 'dop_prior_cache', False)
            and preservation_kind == 'dop'
            and not local_pred_kwargs
        )
        samples = max(1, int(getattr(self.train_config, 'dop_prior_cache_samples', 12)))

        # Steady state: load one matching sample per image and skip the frozen-base forward.
        cached_batch = None
        if cache_on:
            with self.timer('dop_prior_cache_lookup'):
                cached_batch = self._dop_prior_cache_load_batch(
                    batch, target_h, target_w, samples, dtype
                )
            self.timer.record(
                'dop/cache_hit' if cached_batch is not None else 'dop/cache_miss',
                1.0,
            )
        if cached_batch is not None:
            noisy_small, cached_timesteps, prior_small = cached_batch
            with self.timer(f"{timer_base}_downsampled"):
                preservation_pred_small = self.predict_noise(
                    noisy_latents=noisy_small,
                    timesteps=cached_timesteps,
                    conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                    unconditional_embeds=unconditional_embeds,
                    batch=batch,
                    **local_pred_kwargs
                )
            return (preservation_pred_small, prior_small)

        with self.timer(f"{timer_base}_downsampled"):
            latents_small, noise_small, noisy_small = self._create_downsampled_noisy_latents(
                batch.latents, dop_timesteps, target_h, target_w, dtype
            )

            # Generate both predictions at the same small resolution
            prior_small = None
            try:
                with self._gpu_phase('dop_prior_generation'):
                    prior_small = self.get_prior_prediction(
                        noisy_latents=noisy_small,
                        conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                        match_adapter_assist=match_adapter_assist,
                        network_weight_list=network_weight_list if network_weight_list is not None else [],
                        timesteps=dop_timesteps,
                        pred_kwargs=local_pred_kwargs,
                        batch=batch,
                        noise=None,
                        unconditional_embeds=unconditional_embeds,
                    )
            except Exception as e:
                print_acc(
                    f'[DOP] frozen prior prediction failed; skipping DOP for this step: '
                    f'{type(e).__name__}: {e}'
                )
                prior_small = None

            if prior_small is None:
                return (None, None)

            preservation_pred_small = self.predict_noise(
                noisy_latents=noisy_small,
                timesteps=dop_timesteps,
                conditional_embeds=preservation_embeds.to(self.device_torch, dtype=dtype),
                unconditional_embeds=unconditional_embeds,
                batch=batch,
                **local_pred_kwargs
            )

        # Warmup: append this live result to each image's persistent cache.
        if cache_on and prior_small is not None:
            with self.timer('dop_prior_cache_write'):
                self._dop_prior_cache_add_batch(
                    batch, target_h, target_w, samples, noisy_small, dop_timesteps, prior_small
                )

        # Return both small preds so loss can be computed at this resolution
        return (preservation_pred_small, prior_small)

    def _create_downsampled_noisy_latents(self, original_latents: torch.Tensor, timesteps: torch.Tensor, target_h: int, target_w: int, dtype: str):
        """Downsample `original_latents` to (target_h, target_w), sample noise at that resolution
        and create a scheduler-consistent `noisy_small` tensor. Returns (latents_small, noise_small, noisy_small).
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
        except Exception as e:
            raise RuntimeError(f"[DOP] failed to construct noisy_small via scheduler: {e}") from e

        if noisy_small is None:
            raise RuntimeError("[DOP] noisy_small was not constructed by scheduler (unexpected)")

        return latents_small, noise_small, noisy_small

    def _compute_and_apply_preservation_loss(self, preservation_pred, prior_pred, multiplier: float, apply_backward: bool = True):
        """Compute preservation loss, record diagnostics, and optionally apply backward.

        When ``apply_backward`` is True (default, two-pass mode) the preservation loss is
        backpropagated here so its forward graph can be freed immediately. When False
        (single-backward mode) the still-connected loss tensor is returned for the caller to
        sum into the main loss and backprop once.

        Returns the preservation_loss tensor.
        """
        if preservation_pred is None or prior_pred is None:
            self._last_preservation_loss = None
            return None
        try:
            # Validate that both predictions are at the same resolution for fair comparison
            if prior_pred is not None and preservation_pred.shape != prior_pred.shape:
                try:
                    print_acc(f"[DOP] Warning: resolution mismatch detected! preservation_pred shape {preservation_pred.shape} != prior_pred shape {prior_pred.shape}")
                except Exception:
                    pass

            # Ensure both tensors are on the same device and dtype to avoid dtype/device mismatch errors
            if prior_pred is not None:
                if preservation_pred.device != prior_pred.device:
                    preservation_pred = preservation_pred.to(prior_pred.device)
                cpu_low_precision = prior_pred.device.type == 'cpu' and prior_pred.dtype in (torch.bfloat16, torch.float16)
                if cpu_low_precision:
                    preservation_pred = preservation_pred.to(torch.float32)
                    prior_pred = prior_pred.to(torch.float32)
                else:
                    if preservation_pred.dtype != prior_pred.dtype:
                        preservation_pred = preservation_pred.to(prior_pred.dtype)

            preservation_loss = torch.nn.functional.mse_loss(preservation_pred, prior_pred) * multiplier

        except Exception as e:
            try:
                print_acc(f"[DOP] preservation loss computation failed: {e}")
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

        # apply backward for preservation loss (two-pass mode only; single-backward mode
        # defers this so the caller can combine it with the main loss for one backward)
        if apply_backward:
            try:
                if preservation_loss.requires_grad:
                    with self.timer('preservation_backward'):
                        with self._gpu_phase('dop_backward'):
                            self.accelerator.backward(preservation_loss)
            except Exception as e:
                try:
                    print_acc(f"[DOP] backward failed for preservation loss: {e}")
                except Exception:
                    pass

        return preservation_loss

    def _inject_weight_noise(self) -> None:
        """Add Gaussian noise directly to LoRA parameter values after the optimizer step.

        No-op when weight_noise.enabled is False. Runs after ema.update() so the EMA
        shadow tracks the clean optimizer trajectory while live weights are the noisy
        ones used for the next forward.

        Modes:
          'absolute': σ fixed at cfg.sigma.
          'relative': σ = cfg.sigma × per-param weight RMS. Zero-init LoRA-up params
                      get zero noise until they learn — avoids destabilizing early training.
        """
        cfg = self.train_config.weight_noise
        if not getattr(cfg, 'enabled', False):
            return

        mode = cfg.mode
        step = max(0, int(getattr(self, 'step_num', 0)))
        do_log = cfg.log_every > 0 and step % cfg.log_every == 0
        noise_sq = 0.0

        groups = self.params
        if not groups:
            return
        if isinstance(groups[0], dict):
            iterable = (p for g in groups for p in g.get('params', []))
        else:
            iterable = iter(groups)

        for p in iterable:
            if not getattr(p, '_is_lora', False):
                continue
            w = p.data
            if mode == 'absolute':
                sigma = float(cfg.sigma)
            elif mode == 'relative':
                rms = float(w.detach().pow(2).mean().clamp_min(1e-30).sqrt())
                sigma = float(cfg.sigma) * rms
            else:
                return

            if sigma <= 0:
                continue
            noise = torch.randn_like(w) * sigma
            if do_log:
                noise_sq += float(noise.pow(2).sum())
            w.add_(noise)

        if do_log:
            self._last_weight_noise_norm = noise_sq ** 0.5

    def resolve_performance_timers(self):
        for bucket, count in self._resolution_bucket_counts.items():
            self.timer.record(f'resolution/count/{bucket}', count)
        self._resolution_bucket_counts.clear()
        for bucket, stats in self._resolution_memory_stats.items():
            for key, value in stats.items():
                self.timer.record(f'resolution/memory/{key}/{bucket}', value)
        self._resolution_memory_stats.clear()
        if not self._gpu_phase_events:
            return
        torch.cuda.synchronize(self.device_torch)
        step_count = max(1, len(self.timer.timers.get('train_loop', ())))
        for name, events in self._gpu_phase_events.items():
            seconds = sum(start.elapsed_time(end) for start, end in events) / 1000.0
            self.timer.record(f'gpu/{name}', seconds / step_count)
        self._gpu_phase_events.clear()

    def hook_train_loop(self, batch: Union[DataLoaderBatchDTO, List[DataLoaderBatchDTO]]):
        if isinstance(batch, list):
            batch_list = batch
        else:
            batch_list = [batch]
        total_loss = None
        # New step: clear the step-level peak high-water. Accumulations fold their
        # per-accumulation peak in via _finish_resolution_memory_sample.
        self._step_peak_allocated_bytes = 0
        self._step_peak_reserved_bytes = 0
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
            if len(batch_list) > 1 and self.model_config.low_vram:
                torch.cuda.empty_cache()


        if not self.is_grad_accumulation_step:
            # fix this for multi params
            if self.train_config.optimizer != 'adafactor':
                if isinstance(self.params[0], dict):
                    for i in range(len(self.params)):
                        self.accelerator.clip_grad_norm_(self.params[i]['params'], self.train_config.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.params, self.train_config.max_grad_norm)
            # only step if we are not accumulating
            with self.timer('optimizer_step'):
                with self._gpu_phase('optimizer_step'):
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                if self.adapter and isinstance(self.adapter, CustomAdapter):
                    self.adapter.post_weight_update()
            if self.ema is not None:
                with self.timer('ema_update'):
                    self.ema.update()
            self._inject_weight_noise()
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

        loss_dict = OrderedDict(
            {'loss': total_loss / len(batch_list)}
        )

        if self._last_weight_noise_norm is not None:
            loss_dict['weight_noise_norm'] = self._last_weight_noise_norm
            self._last_weight_noise_norm = None

        self.end_of_training_loop()

        return loss_dict
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
            patch_min = self._dop_patch_min()
            if patch_min > 1:
                target_h = max(patch_min, int(round(target_h / patch_min)) * patch_min)
                target_w = max(patch_min, int(round(target_w / patch_min)) * patch_min)

            # will downsample if either target dimension strictly less than current
            will_downsample = (target_h < H) or (target_w < W)
            if not will_downsample:
                return False
            # If user configured a "full resolution every N steps" schedule, do not skip the
            # full-resolution prior prediction on those scheduled steps. Avoid scheduling step 0
            # as a forced full-res run (i.e., require total > 0).
            try:
                full_every = int(getattr(self.train_config, 'diff_output_preservation_every', 10))
            except Exception:
                full_every = 10
            total = int(getattr(self, '_total_batch_count', 0))
            if full_every >= 1 and total > 0 and (total % full_every) == 0:
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
