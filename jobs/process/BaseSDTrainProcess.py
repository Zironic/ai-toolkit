import copy
import contextlib
import glob
import inspect
import json
import random
import shutil
import sys
import subprocess
import threading
import time
from collections import OrderedDict
import os
import re
import traceback
from typing import Union, List, Optional

import numpy as np
import yaml
from diffusers import T2IAdapter, ControlNetModel
from diffusers.training_utils import compute_density_for_timestep_sampling
from safetensors.torch import save_file, load_file
# from lycoris.config import PRESET
from torch.utils.data import DataLoader
import torch
import torch.backends.cuda
from huggingface_hub import HfApi, interpreter_login
from toolkit.memory_management import MemoryManager, allocator_cap, vram_budget
from toolkit.memory_management.runtime import (
    close_memory_runtime_preparation,
    get_memory_runtime,
    is_memory_managed,
    memory_runtime_owns_compile,
)

from toolkit.basic import value_map
from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.custom_adapter import CustomAdapter
from toolkit.data_loader import get_dataloader_from_datasets, trigger_dataloader_setup_epoch
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from toolkit.ema import ExponentialMovingAverage
from toolkit.embedding import Embedding
from toolkit.image_utils import show_tensors, show_latents, reduce_contrast
from toolkit.ip_adapter import IPAdapter
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lorm import convert_diffusers_unet_to_lorm, count_parameters, print_lorm_extract_details, \
    lorm_ignore_if_contains, lorm_parameter_threshold, LORM_TARGET_REPLACE_MODULE
from toolkit.lycoris_special import LycorisSpecialNetwork
from toolkit.models.decorator import Decorator
from toolkit.network_mixins import Network
from toolkit.optimizer import get_optimizer
from toolkit.paths import CONFIG_ROOT
from toolkit.progress_bar import ToolkitProgressBar
from toolkit.reference_adapter import ReferenceAdapter
from toolkit.sampler import get_sampler
from toolkit.saving import save_t2i_from_diffusers, load_t2i_model, save_ip_adapter_from_diffusers, \
    load_ip_adapter_model, load_custom_adapter_model

from toolkit.scheduler import get_lr_scheduler
from toolkit.sd_device_states_presets import get_train_sd_device_state_preset
from toolkit.stable_diffusion_model import StableDiffusion

from jobs.process import BaseTrainProcess
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors, add_base_model_info_to_meta, \
    parse_metadata_from_safetensors
from toolkit.train_tools import get_torch_dtype, LearnableSNRGamma, apply_learnable_snr_gos, apply_snr_weight
import gc

from tqdm import tqdm

from toolkit.config_modules import SaveConfig, LoggingConfig, SampleConfig, NetworkConfig, TrainConfig, ModelConfig, \
    GenerateImageConfig, EmbeddingConfig, DatasetConfig, preprocess_dataset_raw_config, AdapterConfig, GuidanceConfig, validate_configs, \
    DecoratorConfig
from toolkit.logging_aitk import create_logger
from diffusers import FluxTransformer2DModel
from toolkit.accelerator import get_accelerator, unwrap_model
from toolkit.print import print_acc
from accelerate import Accelerator
import transformers
import diffusers
import hashlib

from toolkit.util.blended_blur_noise import get_blended_blur_noise
from toolkit.util.get_model import get_model_class
from toolkit.basic import flush


def _torch_compile_backend_unavailable_reason() -> Optional[str]:
    try:
        from torch.utils._triton import has_triton
        if not bool(has_triton()):
            return "PyTorch Inductor cannot find a working Triton backend on this system."
    except Exception as e:
        return f"PyTorch Inductor Triton check failed: {e}"

    if sys.platform == "win32":
        # CUDA graphs compile through Triton. Avoid Inductor's incidental CPU
        # vector-ISA dry compile, which otherwise probes for cl.exe even when
        # no CPU kernel is present in the graph.
        from torch._inductor import config

        config.cpp.vec_isa_ok = False

    return None


def _detach_to_cpu(obj):
    """Deep-copy an optimizer state_dict onto CPU, cloning every tensor.

    The result shares no storage with the live (on-device) optimizer state, so
    it is safe to serialize from a background thread while training continues to
    mutate the originals at the next optimizer.step().
    """
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_detach_to_cpu(v) for v in obj]
        return type(obj)(seq) if not isinstance(obj, tuple) else tuple(seq)
    return obj


class _CudaDriverFreeMonitor:
    """Best-effort per-step sampler for driver-level CUDA free memory."""

    def __init__(self, device, interval_s=0.02):
        self.device = device
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self.min_free_bytes = None
        self.total_bytes = None
        self.samples = 0

    def start(self):
        if not torch.cuda.is_available():
            return self
        try:
            free_b, total_b = vram_budget.device_mem_info(self.device)
        except Exception:
            return self
        self.min_free_bytes = int(free_b)
        self.total_bytes = int(total_b)
        self.samples = 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                free_b, total_b = vram_budget.device_mem_info(self.device)
            except Exception:
                continue
            free_b = int(free_b)
            self.total_bytes = int(total_b)
            self.samples += 1
            if self.min_free_bytes is None or free_b < self.min_free_bytes:
                self.min_free_bytes = free_b

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)
        if self.min_free_bytes is None or self.total_bytes is None:
            return None
        return {
            "min_free_bytes": int(self.min_free_bytes),
            "total_bytes": int(self.total_bytes),
            "samples": int(self.samples),
        }


class BaseSDTrainProcess(BaseTrainProcess):

    def __init__(self, process_id: int, job, config: OrderedDict, custom_pipeline=None):
        super().__init__(process_id, job, config)
        self.accelerator: Accelerator = get_accelerator()
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        
        self.sd: StableDiffusion
        self.embedding: Union[Embedding, None] = None

        self.custom_pipeline = custom_pipeline
        self.step_num = 0
        self.start_step = 0
        self.epoch_num = 0
        self.last_save_step = 0
        # Off-thread, crash-atomic checkpoint writer. Created lazily on first
        # save so non-training uses of this class never spin up the thread.
        self._async_saver = None
        self._save_stager = None
        self._arena_runtime = None
        self._cleanup_started = False
        # start at 1 so we can do a sample at the start
        self.grad_accumulation_step = 1
        # if true, then we do not do an optimizer step. We are accumulating gradients
        self.is_grad_accumulation_step = False
        self.device = str(self.accelerator.device)
        self.device_torch = self.accelerator.device
        network_config = self.get_conf('network', None)
        if network_config is not None:
            self.network_config = NetworkConfig(**network_config)
        else:
            self.network_config = None
        self.train_config = TrainConfig(**self.get_conf('train', {}))
        model_config = self.get_conf('model', {})
        self.modules_being_trained: List[torch.nn.Module] = []

        # update modelconfig dtype to match train
        model_config['dtype'] = self.train_config.dtype
        self.model_config = ModelConfig(**model_config)
        from toolkit.memory_management import MemoryManager
        MemoryManager.reset_job_runtime()
        fp8_weights_configured = bool(
            self.model_config.quantize
            and self.model_config.qtype in ('qfloat8', 'float8')
        )
        if not fp8_weights_configured and any((
            self.model_config.layer_offloading_fp8_forward,
            self.model_config.layer_offloading_fp8_grad_input,
            self.model_config.layer_offloading_fp8_sampling,
        )):
            print_acc(
                "[MemoryManager] native FP8 options ignored: transformer weights "
                "are not configured as FP8"
            )
        MemoryManager.set_fp8_grad_input_enabled(
            fp8_weights_configured
            and self.model_config.layer_offloading_fp8_grad_input
        )
        self._memory_manager_fp8_weights_configured = fp8_weights_configured

        self.save_config = SaveConfig(**self.get_conf('save', {}))
        self.sample_config = SampleConfig(**self.get_conf('sample', {}))
        first_sample_config = self.get_conf('first_sample', None)
        if first_sample_config is not None:
            self.has_first_sample_requested = True
            self.first_sample_config = SampleConfig(**first_sample_config)
        else:
            self.has_first_sample_requested = False
            self.first_sample_config = self.sample_config
        self.logging_config = LoggingConfig(**self.get_conf('logging', {}))
        self.logger = create_logger(self.logging_config, config, self.save_root)
        self.performance_log_path = os.path.join(self.save_root, 'performance_log.jsonl')
        self._archive_previous_performance_log()
        # Dynamo counters are cumulative for the process; the perf log wants the
        # per-window delta, so keep the previous window's snapshot.
        self._compile_counters_prev = None
        self.timer.add_after_print_hook(self._write_performance_timing_log)
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler = None
        self.data_loader: Union[DataLoader, None] = None
        self.data_loader_reg: Union[DataLoader, None] = None
        self.trigger_word = self.get_conf('trigger_word', None)

        self.guidance_config: Union[GuidanceConfig, None] = None
        guidance_config_raw = self.get_conf('guidance', None)
        if guidance_config_raw is not None:
            self.guidance_config = GuidanceConfig(**guidance_config_raw)

        # store is all are cached. Allows us to not load vae if we don't need to
        self.is_latents_cached = True
        raw_datasets = self.get_conf('datasets', None)
        if raw_datasets is not None and len(raw_datasets) > 0:
            raw_datasets = preprocess_dataset_raw_config(raw_datasets)
        self.datasets = None
        self.datasets_reg = None
        self.dataset_configs: List[DatasetConfig] = []
        self.params = []
        
        # add dataset text embedding cache to their config
        if self.train_config.cache_text_embeddings:
            for raw_dataset in raw_datasets:
                raw_dataset['cache_text_embeddings'] = True
        
        if raw_datasets is not None and len(raw_datasets) > 0:
            for raw_dataset in raw_datasets:
                dataset = DatasetConfig(**raw_dataset)
                # handle trigger word per dataset
                if dataset.trigger_word is None and self.trigger_word is not None:
                    dataset.trigger_word = self.trigger_word
                is_caching = dataset.cache_latents or dataset.cache_latents_to_disk
                if not is_caching:
                    self.is_latents_cached = False
                if dataset.is_reg:
                    if self.datasets_reg is None:
                        self.datasets_reg = []
                    self.datasets_reg.append(dataset)
                else:
                    if self.datasets is None:
                        self.datasets = []
                    self.datasets.append(dataset)
                self.dataset_configs.append(dataset)
        
        self.is_caching_text_embeddings = any(
            dataset.cache_text_embeddings for dataset in self.dataset_configs
        )
        # set True in run() once a TE worker has cached all embeddings to disk; tells the
        # trainer to load skip_te (no text encoder) and read embeds from disk instead of
        # encoding them in-process.
        self._use_cached_te = False

        self.embed_config = None
        embedding_raw = self.get_conf('embedding', None)
        if embedding_raw is not None:
            self.embed_config = EmbeddingConfig(**embedding_raw)
        
        self.decorator_config: DecoratorConfig = None
        decorator_raw = self.get_conf('decorator', None)
        if decorator_raw is not None:
            if not self.model_config.is_flux:
                raise ValueError("Decorators are only supported for Flux models currently")
            self.decorator_config = DecoratorConfig(**decorator_raw)

        # t2i adapter
        self.adapter_config = None
        adapter_raw = self.get_conf('adapter', None)
        if adapter_raw is not None:
            self.adapter_config = AdapterConfig(**adapter_raw)
            # sdxl adapters end in _xl. Only full_adapter_xl for now
            if self.model_config.is_xl and not self.adapter_config.adapter_type.endswith('_xl'):
                self.adapter_config.adapter_type += '_xl'

        # to hold network if there is one
        self.network: Union[Network, None] = None
        self.adapter: Union[T2IAdapter, IPAdapter, ClipVisionAdapter, ReferenceAdapter, CustomAdapter, ControlNetModel, None] = None
        self.embedding: Union[Embedding, None] = None
        self.decorator: Union[Decorator, None] = None

        is_training_adapter = self.adapter_config is not None and self.adapter_config.train

        self.do_lorm = self.get_conf('do_lorm', False)
        self.lorm_extract_mode = self.get_conf('lorm_extract_mode', 'ratio')
        self.lorm_extract_mode_param = self.get_conf('lorm_extract_mode_param', 0.25)
        # 'ratio', 0.25)

        # get the device state preset based on what we are training
        self.train_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=False  # we ensure them later
        )
        
        self.get_params_device_state_preset = get_train_sd_device_state_preset(
            device=self.device_torch,
            train_unet=self.train_config.train_unet,
            train_text_encoder=self.train_config.train_text_encoder,
            cached_latents=self.is_latents_cached,
            train_lora=self.network_config is not None,
            train_adapter=is_training_adapter,
            train_embedding=self.embed_config is not None,
            train_decorator=self.decorator_config is not None,
            train_refiner=self.train_config.train_refiner,
            unload_text_encoder=self.train_config.unload_text_encoder or self.is_caching_text_embeddings,
            require_grads=True  # We check for grads when getting params
        )

        # fine_tuning here is for training actual SD network, not LoRA, embeddings, etc. it is (Dreambooth, etc)
        self.is_fine_tuning = True
        if self.network_config is not None or is_training_adapter or self.embed_config is not None or self.decorator_config is not None:
            self.is_fine_tuning = False

        self.named_lora = False
        if self.embed_config is not None or is_training_adapter:
            self.named_lora = True
        self.snr_gos: Union[LearnableSNRGamma, None] = None
        self.ema: ExponentialMovingAverage = None
        
        validate_configs(self.train_config, self.model_config, self.save_config, self.dataset_configs)
        
        do_profiler = self.get_conf('torch_profiler', False)
        self.torch_profiler = None if not do_profiler else torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        
        self.current_boundary_index = 0
        self.steps_this_boundary = 0
        self.num_consecutive_oom = 0
        self.additional_logs = {}

    def post_process_generate_image_config_list(self, generate_image_config_list: List[GenerateImageConfig]):
        # override in subclass
        return generate_image_config_list

    def sample(self, step=None, is_first=False):
        if not self.accelerator.is_main_process:
            return
        flush()
        sample_folder = os.path.join(self.save_root, 'samples')
        gen_img_config_list = []

        sample_config = self.first_sample_config if is_first else self.sample_config
        start_seed = sample_config.seed
        current_seed = start_seed

        test_image_paths = []
        if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
            test_image_path_list = self.adapter_config.test_img_path
            # divide up images so they are evenly distributed across prompts
            for i in range(len(sample_config.prompts)):
                test_image_paths.append(test_image_path_list[i % len(test_image_path_list)])

        for i in range(len(sample_config.prompts)):
            if sample_config.walk_seed:
                current_seed = start_seed + i

            step_num = ''
            if step is not None:
                # zero-pad 9 digits
                step_num = f"_{str(step).zfill(9)}"

            filename = f"[time]_{step_num}_[count].{self.sample_config.ext}"

            output_path = os.path.join(sample_folder, filename)

            prompt = sample_config.prompts[i]

            # add embedding if there is one
            # note: diffusers will automatically expand the trigger to the number of added tokens
            # ie test123 will become test123 test123_1 test123_2 etc. Do not add this yourself here
            if self.embedding is not None:
                prompt = self.embedding.inject_embedding_to_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.adapter is not None and isinstance(self.adapter, ClipVisionAdapter):
                prompt = self.adapter.inject_trigger_into_prompt(
                    prompt, expand_token=True, add_if_not_present=False
                )
            if self.trigger_word is not None:
                prompt = self.sd.inject_trigger_into_prompt(
                    prompt, self.trigger_word, add_if_not_present=False
                )

            extra_args = {}
            if self.adapter_config is not None and self.adapter_config.test_img_path is not None:
                extra_args['adapter_image_path'] = test_image_paths[i]
            
            sample_item = sample_config.samples[i]
            if sample_item.seed is not None:
                current_seed = sample_item.seed

            gen_img_config_list.append(GenerateImageConfig(
                prompt=prompt,  # it will autoparse the prompt
                width=sample_item.width,
                height=sample_item.height,
                negative_prompt=sample_item.neg,
                seed=current_seed,
                guidance_scale=sample_item.guidance_scale,
                guidance_rescale=sample_config.guidance_rescale,
                num_inference_steps=sample_item.sample_steps,
                network_multiplier=sample_item.network_multiplier,
                output_path=output_path,
                output_ext=sample_config.ext,
                adapter_conditioning_scale=sample_config.adapter_conditioning_scale,
                refiner_start_at=sample_config.refiner_start_at,
                extra_values=sample_config.extra_values,
                logger=self.logger,
                num_frames=sample_item.num_frames,
                fps=sample_item.fps,
                ctrl_img=sample_item.ctrl_img,
                ctrl_idx=sample_item.ctrl_idx,
                ctrl_img_1=sample_item.ctrl_img_1,
                ctrl_img_2=sample_item.ctrl_img_2,
                ctrl_img_3=sample_item.ctrl_img_3,
                do_cfg_norm=sample_config.do_cfg_norm,
                batch_cfg=sample_config.batch_cfg,
                **extra_args
            ))

        # post process
        gen_img_config_list = self.post_process_generate_image_config_list(gen_img_config_list)

        # if we have an ema, set it to validation mode
        if self.ema is not None:
            self.ema.eval()

        # let adapter know we are sampling
        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            self.adapter.is_sampling = True
        
        # Sampling layout mutation is opt-in and separate from native FP8
        # sampling. With it disabled, preserve the model's existing behavior.
        transformer = getattr(self.sd, 'unet', None)
        # Shape-aware cold-start reserve: models that can size their sampling
        # working set from the pending gen configs (resolution, CFG mode) hint
        # the residency planner so the first high-res sample streams enough
        # blocks up front instead of demoting mid-denoise. Optional per model;
        # a learned measured reserve replaces it after the first sample.
        estimate_fn = getattr(
            self.sd, 'estimate_sampling_working_reserve_bytes', None
        )
        cold_start_hint = (
            estimate_fn(gen_img_config_list) if estimate_fn is not None else None
        )
        sampling_context = (
            MemoryManager.inference_resident(
                transformer,
                self.device_torch,
                fp8_sampling=(
                    self._memory_manager_fp8_weights_configured
                    and self.model_config.layer_offloading_fp8_sampling
                ),
                cold_start_hint_bytes=cold_start_hint,
                working_reserve_gib=(
                    self.model_config.layer_offloading_smart_sampling_working_reserve_gb
                ),
                wddm_margin_gib=(
                    self.model_config.layer_offloading_smart_sampling_wddm_margin_gb
                ),
                wddm_hard_gib=(
                    self.model_config.layer_offloading_smart_sampling_wddm_hard_gb
                ),
            )
            if (
                self.model_config.layer_offloading
                and self.model_config.layer_offloading_smart
                and self.model_config.layer_offloading_smart_sampling
            )
            else contextlib.nullcontext()
        )
        # The arena runtime owns residency across the train<->sample boundary
        # over ONE arena: the model enters runtime.sampling_image() per image
        # (SAMPLE program: no checkpointing, forward-only streaming), and the
        # session below restores the TRAIN program once at the end. Because the
        # runtime owns residency, the legacy inference_resident sampling context
        # must NOT also re-plan the transformer -- null it out for this backend.
        arena_runtime = get_memory_runtime(self.sd.unet)
        if arena_runtime is not None:
            sampling_context = contextlib.nullcontext()

        arena_session = (
            arena_runtime.sampling_session()
            if arena_runtime is not None
            else contextlib.nullcontext()
        )

        try:
            with arena_session:
                with sampling_context:
                    self.sd.generate_images(gen_img_config_list, sampler=sample_config.sampler)
        finally:
            # Restoring offload may have moved the base transformer to CPU and back; if the LoRA
            # network rode along, make sure it's back on the training device before training resumes.
            if getattr(self, 'network', None) is not None:
                try:
                    self.network.to(self.device_torch)
                except Exception:
                    pass


        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            self.adapter.is_sampling = False

        if self.ema is not None:
            self.ema.train()

    def update_training_metadata(self):
        o_dict = OrderedDict({
            "training_info": self.get_training_info()
        })
        o_dict['ss_base_model_version'] = self.sd.get_base_model_version()

        # o_dict = add_base_model_info_to_meta(
        #     o_dict,
        #     is_v2=self.model_config.is_v2,
        #     is_xl=self.model_config.is_xl,
        # )
        o_dict['ss_output_name'] = self.job.name

        if self.trigger_word is not None:
            # just so auto1111 will pick it up
            o_dict['ss_tag_frequency'] = {
                f"1_{self.trigger_word}": {
                    f"{self.trigger_word}": 1
                }
            }

        self.add_meta(o_dict)

    def get_training_info(self):
        info = OrderedDict({
            'step': self.step_num,
            'epoch': self.epoch_num,
        })
        return info

    def clean_up_saves(self):
        if not self.accelerator.is_main_process:
            return
        # remove old saves
        # get latest saved step
        latest_item = None
        if os.path.exists(self.save_root):
            # pattern is {job_name}_{zero_filled_step} for both files and directories
            pattern = f"{self.job.name}_*"
            items = glob.glob(os.path.join(self.save_root, pattern))
            # Separate files and directories
            safetensors_files = [f for f in items if f.endswith('.safetensors')]
            pt_files = [f for f in items if f.endswith('.pt')]
            directories = [d for d in items if os.path.isdir(d) and not d.endswith('.safetensors')]
            embed_files = []
            # do embedding files
            if self.embed_config is not None:
                embed_pattern = f"{self.embed_config.trigger}_*"
                embed_items = glob.glob(os.path.join(self.save_root, embed_pattern))
                # will end in safetensors or pt
                embed_files = [f for f in embed_items if f.endswith('.safetensors') or f.endswith('.pt')]

            # check for critic files
            critic_pattern = f"CRITIC_{self.job.name}_*"
            critic_items = glob.glob(os.path.join(self.save_root, critic_pattern))

            # Sort the lists by creation time if they are not empty
            if safetensors_files:
                safetensors_files.sort(key=os.path.getctime)
            if pt_files:
                pt_files.sort(key=os.path.getctime)
            if directories:
                directories.sort(key=os.path.getctime)
            if embed_files:
                embed_files.sort(key=os.path.getctime)
            if critic_items:
                critic_items.sort(key=os.path.getctime)

            # Combine and sort the lists
            combined_items = safetensors_files + directories + pt_files
            combined_items.sort(key=os.path.getctime)
            
            num_saves_to_keep = self.save_config.max_step_saves_to_keep
            
            if hasattr(self.sd, 'max_step_saves_to_keep_multiplier'):
                num_saves_to_keep *= self.sd.max_step_saves_to_keep_multiplier

            # Use slicing with a check to avoid 'NoneType' error
            safetensors_to_remove = safetensors_files[
                                    :-num_saves_to_keep] if safetensors_files else []
            pt_files_to_remove = pt_files[:-num_saves_to_keep] if pt_files else []
            directories_to_remove = directories[:-num_saves_to_keep] if directories else []
            embeddings_to_remove = embed_files[:-num_saves_to_keep] if embed_files else []
            critic_to_remove = critic_items[:-num_saves_to_keep] if critic_items else []

            items_to_remove = safetensors_to_remove + pt_files_to_remove + directories_to_remove + embeddings_to_remove + critic_to_remove

            # remove all but the latest max_step_saves_to_keep
            # items_to_remove = combined_items[:-num_saves_to_keep]

            # remove duplicates
            items_to_remove = list(dict.fromkeys(items_to_remove))

            for item in items_to_remove:
                print_acc(f"Removing old save: {item}")
                if os.path.isdir(item):
                    shutil.rmtree(item)
                else:
                    os.remove(item)
                # see if a yaml file with same name exists
                yaml_file = os.path.splitext(item)[0] + ".yaml"
                if os.path.exists(yaml_file):
                    os.remove(yaml_file)
            if combined_items:
                latest_item = combined_items[-1]
        return latest_item

    def post_save_hook(self, save_path):
        # override in subclass
        pass
    
    def cleanup(self):
        """Release threads, pins, CUDA sidecars, and process-global job state."""
        if self._cleanup_started:
            return
        self._cleanup_started = True
        errors = []

        def attempt(label, fn):
            try:
                fn()
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")

        saver = self._async_saver
        if saver is not None:
            attempt("async saver wait", lambda: saver.wait_idle(timeout=10.0))
            attempt("async saver close", lambda: saver.close(timeout=5.0))
            self._async_saver = None

        stager = self._save_stager
        if stager is not None:
            attempt("save stager", stager.close)
            self._save_stager = None

        logger = getattr(self, "logger", None)
        if logger is not None:
            attempt("logger", logger.finish)

        dop_executor = getattr(self, "_dop_cache_executor", None)
        if dop_executor is not None:
            attempt(
                "DOP cache executor",
                lambda: dop_executor.shutdown(wait=False, cancel_futures=True),
            )
            self._dop_cache_executor = None

        db_executor = getattr(self, "thread_pool", None)
        if db_executor is not None:
            attempt(
                "UI database executor",
                lambda: db_executor.shutdown(wait=False, cancel_futures=True),
            )
            self.thread_pool = None

        runtime = self._arena_runtime
        if runtime is None:
            sd = getattr(self, "sd", None)
            runtime = get_memory_runtime(getattr(sd, "unet", None)) if sd is not None else None
        if runtime is not None:
            attempt("arena runtime", runtime.close)
            self._arena_runtime = None
        sd = getattr(self, "sd", None)
        if sd is not None:
            attempt(
                "memory runtime preparation",
                lambda: close_memory_runtime_preparation(sd),
            )

        from toolkit.memory_management import MemoryManager
        attempt("memory manager", MemoryManager.reset_job_runtime)

        host_empty_cache = getattr(torch._C, "_host_emptyCache", None)
        if host_empty_cache is not None:
            attempt("pinned host cache", host_empty_cache)
        if torch.cuda.is_available():
            attempt("CUDA cache", torch.cuda.empty_cache)

        if errors:
            raise RuntimeError("; ".join(errors))

    def done_hook(self):
        pass
    
    def end_step_hook(self):
        pass

    @property
    def async_saver(self):
        if self._async_saver is None:
            from toolkit.async_save import AsyncSaver
            self._async_saver = AsyncSaver(name=f"save-{self.job.name}")
        return self._async_saver

    @property
    def save_stager(self):
        """Pinned staging buffer for the batched checkpoint snapshot, or None
        when disabled (snapshot_buffer_mb <= 0 -> per-tensor copy path)."""
        mb = getattr(self.save_config, 'snapshot_buffer_mb', 64)
        if not mb or mb <= 0:
            return None
        if self._save_stager is None:
            from toolkit.async_save import PinnedStager
            self._save_stager = PinnedStager(cap_bytes=int(mb) * 1024 * 1024)
        return self._save_stager

    def save(self, step=None):
        if not self.accelerator.is_main_process:
            return
        _t_start = time.perf_counter()
        # Surface a failed background write from a previous save before we do more.
        if self._async_saver is not None:
            self._async_saver.wait_idle(timeout=0.0)
        flush()
        _t_flush1 = time.perf_counter()
        if self.ema is not None:
            # always save params as ema
            self.ema.eval()

        if not os.path.exists(self.save_root):
            os.makedirs(self.save_root, exist_ok=True)

        step_num = ''
        if step is not None:
            self.last_save_step = step
            # zeropad 9 digits
            step_num = f"_{str(step).zfill(9)}"

        self.update_training_metadata()
        filename = f'{self.job.name}{step_num}.safetensors'
        file_path = os.path.join(self.save_root, filename)

        save_meta = copy.deepcopy(self.meta)
        # get extra meta
        if self.adapter is not None and isinstance(self.adapter, CustomAdapter):
            additional_save_meta = self.adapter.get_additional_save_metadata()
            if additional_save_meta is not None:
                for key, value in additional_save_meta.items():
                    save_meta[key] = value

        # prepare meta
        save_meta = get_meta_for_safetensors(save_meta, self.job.name)
        if not self.is_fine_tuning and not self.train_config.merge_network_on_save:
            if self.network is not None:
                lora_name = self.job.name
                if self.named_lora:
                    # add _lora to name
                    lora_name += '_LoRA'

                filename = f'{lora_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                prev_multiplier = self.network.multiplier
                self.network.multiplier = 1.0

                # if we are doing embedding training as well, add that
                embedding_dict = self.embedding.state_dict() if self.embedding else None
                self.network.save_weights(
                    file_path,
                    dtype=get_torch_dtype(self.save_config.dtype),
                    metadata=save_meta,
                    extra_state_dict=embedding_dict,
                    writer=self.async_saver,
                    stager=self.save_stager,
                )
                self.network.multiplier = prev_multiplier
                # if we have an embedding as well, pair it with the network

            # even if added to lora, still save the trigger version
            if self.embedding is not None:
                emb_filename = f'{self.embed_config.trigger}{step_num}.safetensors'
                emb_file_path = os.path.join(self.save_root, emb_filename)
                # for combo, above will get it
                # set current step
                self.embedding.step = self.step_num
                # change filename to pt if that is set
                if self.embed_config.save_format == "pt":
                    # replace extension
                    emb_file_path = os.path.splitext(emb_file_path)[0] + ".pt"
                self.embedding.save(emb_file_path)
            
            if self.decorator is not None:
                dec_filename = f'{self.job.name}{step_num}.safetensors'
                dec_file_path = os.path.join(self.save_root, dec_filename)
                decorator_state_dict = self.decorator.state_dict()
                for key, value in decorator_state_dict.items():
                    if isinstance(value, torch.Tensor):
                        decorator_state_dict[key] = value.clone().to('cpu', dtype=get_torch_dtype(self.save_config.dtype))
                save_file(
                    decorator_state_dict,
                    dec_file_path,
                    metadata=save_meta,
                )

            if self.adapter is not None and self.adapter_config.train:
                adapter_name = self.job.name
                if self.network_config is not None or self.embedding is not None:
                    # add _lora to name
                    if self.adapter_config.type == 't2i':
                        adapter_name += '_t2i'
                    elif self.adapter_config.type == 'control_net':
                        adapter_name += '_cn'
                    elif self.adapter_config.type == 'clip':
                        adapter_name += '_clip'
                    elif self.adapter_config.type.startswith('ip'):
                        adapter_name += '_ip'
                    else:
                        adapter_name += '_adapter'

                filename = f'{adapter_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                # save adapter
                state_dict = self.adapter.state_dict()
                if self.adapter_config.type == 't2i':
                    save_t2i_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype)
                    )
                elif self.adapter_config.type == 'control_net':
                    # save in diffusers format
                    name_or_path = file_path.replace('.safetensors', '')
                    # move it to the new dtype and cpu
                    orig_device = self.adapter.device
                    orig_dtype = self.adapter.dtype
                    self.adapter = self.adapter.to(torch.device('cpu'), dtype=get_torch_dtype(self.save_config.dtype))
                    self.adapter.save_pretrained(
                        name_or_path,
                        dtype=get_torch_dtype(self.save_config.dtype),
                        safe_serialization=True
                    )
                    meta_path = os.path.join(name_or_path, 'aitk_meta.yaml')
                    with open(meta_path, 'w') as f:
                        yaml.dump(self.meta, f)
                    # move it back
                    self.adapter = self.adapter.to(orig_device, dtype=orig_dtype)
                else:
                    direct_save = False
                    if self.adapter_config.train_only_image_encoder:
                        direct_save = True
                    elif isinstance(self.adapter, CustomAdapter):
                        direct_save = self.adapter.do_direct_save
                    save_ip_adapter_from_diffusers(
                        state_dict,
                        output_file=file_path,
                        meta=save_meta,
                        dtype=get_torch_dtype(self.save_config.dtype),
                        direct_save=direct_save
                    )
        else:
            if self.network is not None and self.train_config.merge_network_on_save:
                # merge the network weights into a full model and save that.
                # torchao quantized weights can be force merged here (dequantize -> merge -> re-quantize)
                # even though can_merge_in is False (kept False so sampling never merges). quanto and
                # layer_offloading still cannot merge.
                from toolkit.util.quantize import get_torchao_config
                can_force_quantized_merge = (
                    self.model_config.quantize and not self.model_config.layer_offloading
                    and get_torchao_config(self.model_config.qtype) is not None
                )
                if not self.network.can_merge_in and not can_force_quantized_merge:
                    raise ValueError("Network cannot merge in weights. Cannot save full model.")

                print_acc("Merging network weights into full model for saving...")

                self.network.merge_in(merge_weight=self.train_config.merge_network_on_save_strength)
                # reset weights to zero
                self.network.reset_weights()
                self.network.is_merged_in = False
                
                print_acc("Done merging network weights. Saving model...")
                
            if self.save_config.save_format == "diffusers":
                # saving as a folder path
                file_path = file_path.replace('.safetensors', '')
                # convert it back to normal object
                save_meta = parse_metadata_from_safetensors(save_meta)

            if self.sd.refiner_unet and self.train_config.train_refiner:
                # save refiner
                refiner_name = self.job.name + '_refiner'
                filename = f'{refiner_name}{step_num}.safetensors'
                file_path = os.path.join(self.save_root, filename)
                self.sd.save_refiner(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )
            if self.train_config.train_unet or self.train_config.train_text_encoder:
                self.sd.save(
                    file_path,
                    save_meta,
                    get_torch_dtype(self.save_config.dtype)
                )

        # save learnable params as json if we have thim
        if self.snr_gos:
            json_data = {
                'offset_1': self.snr_gos.offset_1.item(),
                'offset_2': self.snr_gos.offset_2.item(),
                'scale': self.snr_gos.scale.item(),
                'gamma': self.snr_gos.gamma.item(),
            }
            path_to_save = file_path = os.path.join(self.save_root, 'learnable_snr.json')
            with open(path_to_save, 'w') as f:
                json.dump(json_data, f, indent=4)
        
        print_acc(f"Saved checkpoint to {file_path}")

        _t_net = time.perf_counter()

        # save optimizer
        if self.optimizer is not None:
            try:
                filename = f'optimizer.pt'
                file_path = os.path.join(self.save_root, filename)
                try:
                    state_dict = unwrap_model(self.optimizer).state_dict()
                except Exception as e:
                    state_dict = self.optimizer.state_dict()
                # The optimizer state lives on-device and the next optimizer.step
                # mutates it, so snapshot to CPU here (synchronous) before the
                # write is deferred off-thread.
                cpu_state = _detach_to_cpu(state_dict)
                from toolkit.async_save import atomic_torch_save
                self.async_saver.submit(
                    lambda cs=cpu_state, fp=file_path: atomic_torch_save(cs, fp),
                    description="optimizer",
                )
                print_acc(f"Queued optimizer save to {file_path}")
            except Exception as e:
                print_acc(e)
                print_acc("Could not save optimizer")

        _t_optim = time.perf_counter()

        self.clean_up_saves()
        self.post_save_hook(file_path)

        if self.ema is not None:
            self.ema.train()
        flush()
        _t_end = time.perf_counter()
        print_acc(
            f"[save] snapshot={_t_net - _t_flush1:.2f}s optim_snap={_t_optim - _t_net:.2f}s "
            f"flush_pre={_t_flush1 - _t_start:.2f}s flush_post={_t_end - _t_optim:.2f}s "
            f"(disk write async)"
        )

    def save_recovery_snapshot(self):
        """Frequent, cheap, latest-wins LoRA snapshot for crash recovery.

        Writes *only* the LoRA (no optimizer / adapter / embedding) to a fixed
        ``<name>.recovery.safetensors``. The device->host copy is the batched
        pinned snapshot (~0.1s on the training thread); the disk write is
        deferred to the async writer and coalesced, so snapshots can't back up.

        The filename deliberately sits outside the ``<name>_*`` glob used by
        ``clean_up_saves`` (so it is never rotated away) while still matching the
        ``<name>*`` glob in ``get_latest_save_path`` (so a resume prefers it when
        it is the newest state on disk). Step lives in the safetensors metadata,
        so resume picks up exactly where the crash happened.
        """
        if not self.accelerator.is_main_process or self.network is None:
            return
        try:
            self.update_training_metadata()
            save_meta = get_meta_for_safetensors(copy.deepcopy(self.meta), self.job.name)
            path = os.path.join(self.save_root, f'{self.job.name}.recovery.safetensors')
            prev_multiplier = self.network.multiplier
            self.network.multiplier = 1.0
            self.network.save_weights(
                path,
                dtype=get_torch_dtype(self.save_config.dtype),
                metadata=save_meta,
                writer=self.async_saver,
                stager=self.save_stager,
                coalesce_key='recovery',
            )
            self.network.multiplier = prev_multiplier
        except Exception as e:
            # Recovery snapshots are best-effort; never let one take down training.
            print_acc(f"recovery snapshot failed at step {self.step_num}: {e}")

    # Called before the model is loaded
    def hook_before_model_load(self):
        # override in subclass
        pass

    def hook_after_model_load(self):
        # override in subclass
        pass

    def hook_add_extra_train_params(self, params):
        # override in subclass
        return params

    def hook_before_train_loop(self):
        if self.accelerator.is_main_process:
            self.logger.start()
        self.prepare_accelerator()
        if self.accelerator.is_main_process:
            memory = None
            if get_memory_runtime(getattr(self.sd, 'unet', None)) is None:
                try:
                    from toolkit.memory_management import MemoryManager
                    memory = MemoryManager.training_runtime_diagnostics(
                        getattr(self.sd, 'unet', None), self.device_torch
                    )
                except Exception as error:
                    print_acc(
                        f"[MemoryManager] pre-training diagnostic failed: {error}"
                    )
            if memory is not None:
                print_acc(
                    "[MemoryManager] pre-training smart layout: "
                    f"resident={memory['planned_resident_gb']:.2f} GiB "
                    f"offloaded_cpu={memory['offloaded_cpu_gb']:.2f} GiB "
                    f"streamed_layers={memory['managed_layers']}/"
                    f"{memory['candidate_layers']} "
                    f"ring_reserve={memory['planned_ring_gb']:.2f} GiB "
                    f"reserve_space={memory['training_working_reserve_gb']:.2f} GiB "
                    f"allocated={memory['torch_allocated_gb']:.2f} GiB "
                    f"reserved={memory['torch_reserved_gb']:.2f} GiB "
                    f"device_free={memory['device_free_gb']:.2f} GiB"
                )

    def resolve_performance_timers(self):
        """Resolve model-specific asynchronous timers before logging the rolling window."""
        pass

    def _resolve_job_jsonl_path(self, requested_path, default_filename):
        """Resolve profiler artifacts under the job output folder by default."""
        if requested_path is None or requested_path is False:
            return None
        if requested_path is True:
            requested_path = default_filename
        requested_path = str(requested_path).strip()
        if not requested_path:
            return None
        if requested_path.lower() in ('1', 'true', 'yes', 'on'):
            requested_path = default_filename
        if not requested_path.lower().endswith('.jsonl'):
            requested_path = f'{requested_path}.jsonl'
        if os.path.isabs(requested_path):
            return requested_path
        return os.path.join(self.save_root, requested_path)

    def _archive_previous_jsonl(self, path, label):
        """Move an existing profiler JSONL aside into this job's logs folder."""
        if not self.accelerator.is_main_process:
            return
        try:
            if not path or not os.path.exists(path):
                return
            logs_folder = os.path.join(self.save_root, 'logs')
            os.makedirs(logs_folder, exist_ok=True)
            base = os.path.basename(path)
            num = 0
            while os.path.exists(os.path.join(logs_folder, f'{num}_{base}')):
                num += 1
            os.replace(path, os.path.join(logs_folder, f'{num}_{base}'))
        except Exception as error:
            print_acc(f"Could not archive previous {label}: {error}")

    @staticmethod
    def _compile_counter_snapshot():
        """Cumulative Dynamo work counters, or None when nothing ever compiled."""
        try:
            counters = torch._dynamo.utils.counters
        except AttributeError:
            return None
        frames = int(counters["frames"].get("total", 0) or 0)
        graphs = int(counters["stats"].get("unique_graphs", 0) or 0)
        if frames == 0 and graphs == 0:
            return None
        return {
            'frames': frames,
            'graphs': graphs,
            'graph_breaks': int(sum(counters["graph_break"].values())),
        }

    def _compile_window_counters(self):
        """Per-window compile activity.

        `new_frames` is the headline: Dynamo traced a frame this window. After the
        cold compile it should sit at 0 -- anything else is a recompile, i.e. a
        guard (usually a shape) that the compiled kernels did not cover.
        """
        now = self._compile_counter_snapshot()
        if now is None:
            return None
        prev = self._compile_counters_prev or {}
        self._compile_counters_prev = now
        return {
            'new_frames': now['frames'] - int(prev.get('frames', 0)),
            'new_graphs': now['graphs'] - int(prev.get('graphs', 0)),
            'new_graph_breaks': now['graph_breaks'] - int(prev.get('graph_breaks', 0)),
            'frames_total': now['frames'],
            'graphs_total': now['graphs'],
            'graph_breaks_total': now['graph_breaks'],
        }

    def _archive_previous_performance_log(self):
        """Move an existing performance_log.jsonl aside before a new run.

        Mirrors the UI's handling of log.txt (ui/cron/actions/startJob.ts):
        the old file is moved into a 'logs' subfolder and renamed
        '{num}_performance_log.jsonl', choosing the next free number. Doing
        it here (rather than only in the UI) also covers CLI runs.
        """
        if not self.accelerator.is_main_process:
            return
        try:
            if not os.path.exists(self.performance_log_path):
                return
            logs_folder = os.path.join(self.save_root, 'logs')
            os.makedirs(logs_folder, exist_ok=True)
            num = 0
            while os.path.exists(os.path.join(logs_folder, f'{num}_performance_log.jsonl')):
                num += 1
            os.replace(
                self.performance_log_path,
                os.path.join(logs_folder, f'{num}_performance_log.jsonl'),
            )
        except Exception as error:
            print_acc(f"Could not archive previous performance log: {error}")

    def _write_performance_timing_log(self, timing_dict):
        """Append one reconciled timing window to performance_log.jsonl."""
        if not self.accelerator.is_main_process or 'train_loop' not in timing_dict:
            return
        step_count = len(self.timer.timers.get('train_loop', ()))
        if step_count == 0:
            return

        def per_step(*names):
            return sum(sum(self.timer.timers.get(name, ())) for name in names) / step_count

        total = timing_dict['train_loop']
        data = per_step('get_batch', 'get_batch:reg', 'reset_batch', 'reset_batch:reg')
        prepare = per_step('preprocess_batch')
        sizes = (256, 512, 768, 1024)
        bucket_counts = {
            size: int(sum(self.timer.timers.get(f'resolution/count/{size}', ())))
            for size in sizes
        }
        total_bucket_samples = sum(bucket_counts.values())
        normal_forward = 0.0
        normal_backward = 0.0
        resolution_buckets = {}
        for size in sizes:
            count = bucket_counts[size]
            memory_count = int(sum(
                self.timer.timers.get(f'resolution/memory/count/{size}', ())
            ))
            gb = 1024 ** 3
            forward_part = timing_dict.get(f'gpu/normal_training_forward/{size}', 0.0)
            backward_part = timing_dict.get(f'gpu/normal_backward/{size}', 0.0)
            normal_forward += forward_part
            normal_backward += backward_part
            allocated_sum = sum(self.timer.timers.get(
                f'resolution/memory/allocated_sum/{size}', ()
            ))
            reserved_sum = sum(self.timer.timers.get(
                f'resolution/memory/reserved_sum/{size}', ()
            ))
            incremental_sum = sum(self.timer.timers.get(
                f'resolution/memory/incremental_sum/{size}', ()
            ))
            resolution_buckets[str(size)] = {
                'count': count,
                'share': count / total_bucket_samples if total_bucket_samples else 0.0,
                'normal_forward_s': forward_part * step_count / count if count else 0.0,
                'normal_backward_s': backward_part * step_count / count if count else 0.0,
                'normal_path_s': (forward_part + backward_part) * step_count / count if count else 0.0,
                'peak_allocated_gb_avg': allocated_sum / memory_count / gb if memory_count else 0.0,
                'peak_allocated_gb_max': max(self.timer.timers.get(
                    f'resolution/memory/allocated_max/{size}', (0,)
                )) / gb,
                'peak_reserved_gb_avg': reserved_sum / memory_count / gb if memory_count else 0.0,
                'peak_reserved_gb_max': max(self.timer.timers.get(
                    f'resolution/memory/reserved_max/{size}', (0,)
                )) / gb,
                'incremental_peak_allocated_gb_avg': (
                    incremental_sum / memory_count / gb if memory_count else 0.0
                ),
                'incremental_peak_allocated_gb_max': max(self.timer.timers.get(
                    f'resolution/memory/incremental_max/{size}', (0,)
                )) / gb,
            }
        dop = timing_dict.get('gpu/dop_section', 0.0)
        cache_hits = int(sum(self.timer.timers.get('dop/cache_hit', ())))
        cache_misses = int(sum(self.timer.timers.get('dop/cache_miss', ())))
        cache_lookups = cache_hits + cache_misses
        cache_lookup = per_step('dop_prior_cache_lookup')
        cache_write = per_step('dop_prior_cache_write')
        prior_generation = timing_dict.get('gpu/dop_prior_generation', 0.0)
        prior_generation_per_miss = (
            prior_generation * step_count / cache_misses if cache_misses else 0.0
        )
        dop_backward = timing_dict.get('gpu/dop_backward', 0.0)
        combined_backward = timing_dict.get('gpu/combined_backward', 0.0)
        backward = normal_backward + dop_backward + combined_backward
        optimizer = timing_dict.get('gpu/optimizer_step', 0.0)
        accounted = data + prepare + normal_forward + dop + backward + optimizer
        record = {
            'step': self.step_num,
            'window_steps': step_count,
            'entire_training_step_s': total,
            'data_loading_s': data,
            'batch_preparation_s': prepare,
            'normal_training_forward_s': normal_forward,
            'normal_path_by_resolution': resolution_buckets,
            'dop_section_s': dop,
            'dop_cache_hits': cache_hits,
            'dop_cache_misses': cache_misses,
            'dop_cache_hit_rate': cache_hits / cache_lookups if cache_lookups else None,
            'dop_cache_lookup_s': cache_lookup,
            'dop_cache_write_s': cache_write,
            'dop_prior_generation_s': prior_generation,
            'dop_prior_generation_per_miss_s': prior_generation_per_miss,
            'backward_s': backward,
            'normal_backward_s': normal_backward,
            'dop_backward_s': dop_backward,
            'combined_backward_s': combined_backward,
            'optimizer_step_s': optimizer,
            'other_overhead_s': max(0.0, total - accounted),
        }
        compile_counters = self._compile_window_counters()
        if compile_counters is not None:
            record['compile'] = compile_counters
            if compile_counters['new_frames']:
                print_acc(
                    "[compile] traced {nf} new frames ({ng} new graphs, "
                    "{nb} new graph breaks) this window; "
                    "totals frames={tf} graphs={tg}".format(
                        nf=compile_counters['new_frames'],
                        ng=compile_counters['new_graphs'],
                        nb=compile_counters['new_graph_breaks'],
                        tf=compile_counters['frames_total'],
                        tg=compile_counters['graphs_total'],
                    )
                )
        arena_runtime = get_memory_runtime(getattr(self.sd, 'unet', None))
        smart_memory = None
        if arena_runtime is None:
            try:
                from toolkit.memory_management import MemoryManager
                driver_free_sample = getattr(self, '_last_driver_free_sample', None) or {}
                smart_memory = MemoryManager.training_runtime_diagnostics(
                    getattr(self.sd, 'unet', None), self.device_torch,
                    observed_driver_free_min_bytes=driver_free_sample.get('min_free_bytes'),
                    observed_driver_total_bytes=driver_free_sample.get('total_bytes'),
                    observed_driver_free_samples=driver_free_sample.get('samples'),
                )
            except Exception as error:
                smart_memory = {'diagnostic_error': str(error)}
        if smart_memory is not None:
            record['smart_training_offload'] = smart_memory
            if 'diagnostic_error' not in smart_memory:
                peak_source = smart_memory.get('device_peak_source', 'estimate')
                peak_samples = smart_memory.get('driver_free_samples')
                if peak_source == 'observed' and peak_samples:
                    peak_source_text = f"observed/{peak_samples}"
                else:
                    peak_source_text = "estimate"
                print_acc(
                    "[MemoryManager] smart training runtime: "
                    f"resident={smart_memory['planned_resident_gb']:.2f} GiB "
                    f"offloaded_cpu={smart_memory['offloaded_cpu_gb']:.2f} GiB "
                    f"ring_peak={smart_memory.get('ring_peak_gb', smart_memory['live_ring_gb']):.2f} GiB "
                    f"ring_live={smart_memory['live_ring_gb']:.2f}/"
                    f"{smart_memory['planned_ring_gb']:.2f} GiB "
                    f"working_peak={smart_memory.get('working_reserve_peak_gb', smart_memory['working_reserve_used_gb']):.2f} GiB "
                    f"reserve_space={smart_memory['training_working_reserve_gb']:.2f} GiB "
                    f"(residual={smart_memory.get('working_reserve_residual_gb', smart_memory['working_reserve_used_gb']):.2f}) "
                    f"allocated={smart_memory['torch_allocated_gb']:.2f} GiB "
                    f"reserved={smart_memory['torch_reserved_gb']:.2f} GiB "
                    # Headline the observed driver-level PEAK footprint/free.
                    # If no sampler data exists, the source is labeled estimate.
                    f"driver_peak={smart_memory.get('device_used_peak_gb', smart_memory['device_used_gb']):.2f}/"
                    f"{smart_memory['device_total_gb']:.2f} GiB "
                    f"free_peak={smart_memory.get('device_free_peak_gb', smart_memory['device_free_gb']):.2f} GiB "
                    f"source={peak_source_text} "
                    f"(est {smart_memory.get('device_used_peak_est_gb', smart_memory.get('device_used_peak_gb', smart_memory['device_used_gb'])):.2f}/"
                    f"{smart_memory['device_total_gb']:.2f}, "
                    f"free {smart_memory.get('device_free_peak_est_gb', smart_memory.get('device_free_peak_gb', smart_memory['device_free_gb'])):.2f}; "
                    f"trough {smart_memory['device_used_gb']:.2f}/"
                    f"{smart_memory['device_total_gb']:.2f}, "
                    f"free {smart_memory['device_free_gb']:.2f})"
                )
        try:
            from toolkit.memory_management import MemoryManager
            offload_profile = MemoryManager.offload_profile_report(reset=True)
        except Exception as error:
            offload_profile = f"[OffloadProfile] report failed: {error}"
        if arena_runtime is not None:
            try:
                arena_memory = arena_runtime.diagnostics()
            except Exception as error:
                arena_memory = {'diagnostic_error': str(error)}
            record['arena_offload'] = arena_memory
            if 'diagnostic_error' not in arena_memory:
                controller = (
                    (arena_memory.get('policy') or {}).get('controller') or {}
                )
                print_acc(
                    "[ArenaOffload] policy: "
                    f"state={controller.get('state')} "
                    f"action={controller.get('last_action')} "
                    f"reason={controller.get('last_reason')} "
                    f"block={controller.get('last_block_key')} "
                    f"resident={arena_memory.get('resident_bytes', 0) / (1024 ** 3):.2f} GiB "
                    f"(singleton={arena_memory.get('singleton_resident_bytes', 0) / (1024 ** 3):.2f} "
                    f"canonical={arena_memory.get('canonical_resident_bytes', 0) / (1024 ** 3):.2f}) "
                    f"allocator_slack={float(controller.get('last_worst_shape_allocator_slack_bytes') or 0) / (1024 ** 3):.2f} GiB "
                    f"headband={float(controller.get('slack_pad_bytes') or 0) / (1024 ** 3):.2f} GiB "
                    f"bootstrap={arena_memory.get('bootstrap_budget_bytes', 0) / (1024 ** 3):.2f} GiB/"
                    f"{len(arena_memory.get('bootstrap_block_keys') or ())} blocks "
                    f"plan={arena_memory.get('plan_fingerprint')}"
                )
        if offload_profile:
            record['offload_profile'] = offload_profile
            print_acc(offload_profile)
        try:
            prefetch_report = MemoryManager.offload_prefetch_report(reset=True)
        except Exception as error:
            prefetch_report = f"[BouncePool] report failed: {error}"
        if prefetch_report:
            record['offload_prefetch'] = prefetch_report
            print_acc(prefetch_report)
        try:
            ingraph_report = MemoryManager.ingraph_fetch_report(
                reset=True,
                step_wall_ms=total * step_count * 1000.0,
            )
        except Exception as error:
            ingraph_report = f"[InGraphStream] report failed: {error}"
        if ingraph_report:
            record['ingraph_stream'] = ingraph_report
            print_acc(ingraph_report)
        os.makedirs(os.path.dirname(self.performance_log_path), exist_ok=True)
        with open(self.performance_log_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, separators=(',', ':')) + '\n')
        
    def sample_step_hook(self, img_num, total_imgs):
        pass
    
    def prepare_accelerator(self):
        # set some config
        self.accelerator.even_batches=False
        
        # # prepare all the models stuff for accelerator (hopefully we dont miss any)
        self.sd.vae = self.accelerator.prepare(self.sd.vae)
        if self.sd.unet is not None:
            self.sd.unet = self.accelerator.prepare(self.sd.unet)
            # todo always tdo it?
            self.modules_being_trained.append(self.sd.unet)
        if self.sd.text_encoder is not None and self.train_config.train_text_encoder:
            if isinstance(self.sd.text_encoder, list):
                self.sd.text_encoder = [self.accelerator.prepare(model) for model in self.sd.text_encoder]
                self.modules_being_trained.extend(self.sd.text_encoder)
            else:
                self.sd.text_encoder = self.accelerator.prepare(self.sd.text_encoder)
                self.modules_being_trained.append(self.sd.text_encoder)
        if self.sd.refiner_unet is not None and self.train_config.train_refiner:
            self.sd.refiner_unet = self.accelerator.prepare(self.sd.refiner_unet)
            self.modules_being_trained.append(self.sd.refiner_unet)
        # todo, do we need to do the network or will "unet" get it?
        if self.sd.network is not None:
            self.sd.network = self.accelerator.prepare(self.sd.network)
            self.modules_being_trained.append(self.sd.network)
        if self.adapter is not None and self.adapter_config.train:
            # todo adapters may not be a module. need to check
            self.adapter = self.accelerator.prepare(self.adapter)
            self.modules_being_trained.append(self.adapter)
        
        # prepare other things
        self.optimizer = self.accelerator.prepare(self.optimizer)
        if self.lr_scheduler is not None:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)
        # self.data_loader = self.accelerator.prepare(self.data_loader)
        # if self.data_loader_reg is not None:
        #     self.data_loader_reg = self.accelerator.prepare(self.data_loader_reg)
            

    def ensure_params_requires_grad(self, force=False):
        if self.train_config.do_paramiter_swapping and not force:
            # the optimizer will handle this if we are not forcing
            return
        for group in self.params:
            for param in group['params']:
                if isinstance(param, torch.nn.Parameter):  # Ensure it's a proper parameter
                    param.requires_grad_(True)

    def setup_ema(self):
        if self.train_config.ema_config.use_ema:
            # our params are in groups. We need them as a single iterable
            params = []
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    params.append(param)
            self.ema = ExponentialMovingAverage(
                params,
                decay=self.train_config.ema_config.ema_decay,
                use_feedback=self.train_config.ema_config.use_feedback,
                param_multiplier=self.train_config.ema_config.param_multiplier,
            )

    def before_dataset_load(self):
        pass

    def get_params(self):
        # you can extend this in subclass to get params
        # otherwise params will be gathered through normal means
        return None

    def hook_train_loop(self, batch):
        # return loss
        return 0.0
    
    def hook_after_sd_init_before_load(self):
        pass

    def get_latest_save_path(self, name=None, post='', include_pretrained_lora=True):
        if name == None:
            name = self.job.name
        # get latest saved step
        latest_path = None
        if os.path.exists(self.save_root):
            # Define patterns for both files and directories
            patterns = [
                f"{name}*{post}.safetensors",
                f"{name}*{post}.pt",
                f"{name}*{post}"
            ]
            # Search for both files and directories
            paths = []
            for pattern in patterns:
                paths.extend(glob.glob(os.path.join(self.save_root, pattern)))

            # Filter out non-existent paths and sort by creation time
            if paths:
                paths = [p for p in paths if os.path.exists(p)]
                # remove false positives
                if '_LoRA' not in name:
                    paths = [p for p in paths if '_LoRA' not in p]
                if '_refiner' not in name:
                    paths = [p for p in paths if '_refiner' not in p]
                if '_t2i' not in name:
                    paths = [p for p in paths if '_t2i' not in p]
                if '_cn' not in name:
                    paths = [p for p in paths if '_cn' not in p]

                if len(paths) > 0:
                    latest_path = max(paths, key=os.path.getctime)
        
        if include_pretrained_lora and latest_path is None and self.network_config is not None and self.network_config.pretrained_lora_path is not None:
            # set pretrained lora path as load path if we do not have a checkpoint to resume from
            if os.path.exists(self.network_config.pretrained_lora_path):
                latest_path = self.network_config.pretrained_lora_path
                print_acc(f"Using pretrained lora path from config: {latest_path}")
            else:
                # no pretrained lora found
                print_acc(f"Pretrained lora path from config does not exist: {self.network_config.pretrained_lora_path}")

        return latest_path

    def load_training_state_from_metadata(self, path):
        if not self.accelerator.is_main_process:
            return
        if path is not None and self.network_config is not None and path == self.network_config.pretrained_lora_path:
            # dont load metadata from pretrained lora
            return
        meta = None
        # if path is folder, then it is diffusers
        if os.path.isdir(path):
            meta_path = os.path.join(path, 'aitk_meta.yaml')
            # load it
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = yaml.load(f, Loader=yaml.FullLoader)
        else:
            meta = load_metadata_from_safetensors(path)
        # if 'training_info' in Orderdict keys
        if meta is not None and 'training_info' in meta and 'step' in meta['training_info'] and self.train_config.start_step is None:
            self.step_num = meta['training_info']['step']
            if 'epoch' in meta['training_info']:
                self.epoch_num = meta['training_info']['epoch']
            self.start_step = self.step_num
            print_acc(f"Found step {self.step_num} in metadata, starting from there")

    def load_weights(self, path):
        if self.network is not None:
            extra_weights = self.network.load_weights(path)
            self.load_training_state_from_metadata(path)
            return extra_weights
        else:
            print_acc("load_weights not implemented for non-network models")
            return None

    def apply_snr(self, seperated_loss, timesteps):
        if self.train_config.learnable_snr_gos:
            # add snr_gamma
            seperated_loss = apply_learnable_snr_gos(seperated_loss, timesteps, self.snr_gos)
        elif self.train_config.snr_gamma is not None and self.train_config.snr_gamma > 0.000001:
            # add snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.snr_gamma, fixed=True)
        elif self.train_config.min_snr_gamma is not None and self.train_config.min_snr_gamma > 0.000001:
            # add min_snr_gamma
            seperated_loss = apply_snr_weight(seperated_loss, timesteps, self.sd.noise_scheduler, self.train_config.min_snr_gamma)

        return seperated_loss

    def load_lorm(self):
        latest_save_path = self.get_latest_save_path()
        if latest_save_path is not None:
            # hacky way to reload weights for now
            # todo, do this
            state_dict = load_file(latest_save_path, device=self.device)
            self.sd.unet.load_state_dict(state_dict)

            meta = load_metadata_from_safetensors(latest_save_path)
            # if 'training_info' in Orderdict keys
            if 'training_info' in meta and 'step' in meta['training_info']:
                self.step_num = meta['training_info']['step']
                if 'epoch' in meta['training_info']:
                    self.epoch_num = meta['training_info']['epoch']
                self.start_step = self.step_num
                print_acc(f"Found step {self.step_num} in metadata, starting from there")

    # def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
    #     self.sd.noise_scheduler.set_timesteps(1000, device=self.device_torch)
    #     sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device_torch, dtype=dtype)
    #     schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device_torch, )
    #     timesteps = timesteps.to(self.device_torch, )
    #
    #     # step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    #     step_indices = [t for t in timesteps]
    #
    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < n_dim:
    #         sigma = sigma.unsqueeze(-1)
    #     return sigma

    def load_additional_training_modules(self, params):
        # override in subclass
        return params

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.sd.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.sd.noise_scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)

        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    def get_optimal_noise(self, latents, dtype=torch.float32):
        batch_num = latents.shape[0]
        chunks = torch.chunk(latents, batch_num, dim=0)
        noise_chunks = []
        for chunk in chunks:
            noise_samples = [torch.randn_like(chunk, device=chunk.device, dtype=dtype) for _ in range(self.train_config.optimal_noise_pairing_samples)]
            # find the one most similar to the chunk
            lowest_loss = 999999999999
            best_noise = None
            for noise in noise_samples:
                loss = torch.nn.functional.mse_loss(chunk, noise)
                if loss < lowest_loss:
                    lowest_loss = loss
                    best_noise = noise
            noise_chunks.append(best_noise)
        noise = torch.cat(noise_chunks, dim=0)
        return noise
    
    def get_consistent_noise(self, latents, batch: 'DataLoaderBatchDTO', dtype=torch.float32):
        batch_num = latents.shape[0]
        chunks = torch.chunk(latents, batch_num, dim=0)
        noise_chunks = []
        for idx, chunk in enumerate(chunks):
            # get seed from path
            file_item = batch.file_items[idx]
            img_path = file_item.path
            # add augmentors
            if file_item.flip_x:
                img_path += '_fx'
            if file_item.flip_y:
                img_path += '_fy'
            seed = int(hashlib.md5(img_path.encode()).hexdigest(), 16) & 0xffffffff
            generator = torch.Generator("cpu").manual_seed(seed)
            noise_chunk = torch.randn(chunk.shape, generator=generator).to(chunk.device, dtype=dtype)
            noise_chunks.append(noise_chunk)
        noise = torch.cat(noise_chunks, dim=0).to(dtype=dtype)
        return noise
            

    def get_noise(
        self, 
        latents, 
        batch_size, 
        dtype=torch.float32, 
        batch: 'DataLoaderBatchDTO' = None,
        timestep=None,
    ):
        if self.train_config.optimal_noise_pairing_samples > 1:
            noise = self.get_optimal_noise(latents, dtype=dtype)
        elif self.train_config.force_consistent_noise:
            if batch is None:
                raise ValueError("Batch must be provided for consistent noise")
            noise = self.get_consistent_noise(latents, batch, dtype=dtype)
        else:
            if hasattr(self.sd, 'get_latent_noise_from_latents'):
                noise = self.sd.get_latent_noise_from_latents(
                    latents,
                    noise_offset=self.train_config.noise_offset
                ).to(self.device_torch, dtype=dtype)
            else:
                # get noise
                noise = self.sd.get_latent_noise(
                    height=latents.shape[2],
                    width=latents.shape[3],
                    num_channels=latents.shape[1],
                    batch_size=batch_size,
                    noise_offset=self.train_config.noise_offset,
                ).to(self.device_torch, dtype=dtype)
        
        if self.train_config.blended_blur_noise:
            noise = get_blended_blur_noise(
                latents, noise, timestep
            )

        return noise

    def process_general_training_batch(self, batch: 'DataLoaderBatchDTO'):
        with torch.no_grad():
            with self.timer('prepare_prompt'):
                prompts = batch.get_caption_list()
                is_reg_list = batch.get_is_reg_list()

                is_any_reg = any([is_reg for is_reg in is_reg_list])

                do_double = self.train_config.short_and_long_captions and not is_any_reg

                if self.train_config.short_and_long_captions and do_double:
                    # dont do this with regs. No point

                    # double batch and add short captions to the end
                    prompts = prompts + batch.get_caption_short_list()
                    is_reg_list = is_reg_list + is_reg_list
                if self.model_config.refiner_name_or_path is not None and self.train_config.train_unet:
                    prompts = prompts + prompts
                    is_reg_list = is_reg_list + is_reg_list

                conditioned_prompts = []

                for prompt, is_reg in zip(prompts, is_reg_list):

                    # make sure the embedding is in the prompts
                    if self.embedding is not None:
                        prompt = self.embedding.inject_embedding_to_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    if self.adapter and isinstance(self.adapter, ClipVisionAdapter):
                        prompt = self.adapter.inject_trigger_into_prompt(
                            prompt,
                            expand_token=True,
                            add_if_not_present=not is_reg,
                        )

                    # make sure trigger is in the prompts if not a regularization run
                    if self.trigger_word is not None:
                        prompt = self.sd.inject_trigger_into_prompt(
                            prompt,
                            trigger=self.trigger_word,
                            add_if_not_present=not is_reg,
                        )

                    if not is_reg and self.train_config.prompt_saturation_chance > 0.0:
                        # do random prompt saturation by expanding the prompt to hit at least 77 tokens
                        if random.random() < self.train_config.prompt_saturation_chance:
                            est_num_tokens = len(prompt.split(' '))
                            if est_num_tokens < 77:
                                num_repeats = int(77 / est_num_tokens) + 1
                                prompt = ', '.join([prompt] * num_repeats)


                    conditioned_prompts.append(prompt)

            with self.timer('prepare_latents'):
                dtype = get_torch_dtype(self.train_config.dtype)
                imgs = None
                is_reg = any(batch.get_is_reg_list())
                if batch.tensor is not None:
                    imgs = batch.tensor
                    imgs = imgs.to(self.device_torch, dtype=dtype)
                    # dont adjust for regs.
                    if self.train_config.img_multiplier is not None and not is_reg:
                        # do it ad contrast
                        imgs = reduce_contrast(imgs, self.train_config.img_multiplier)
                if batch.latents is not None:
                    latents = batch.latents.to(self.device_torch, dtype=dtype)
                    batch.latents = latents
                else:
                    # normalize to
                    if self.train_config.standardize_images:
                        if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                            target_mean_list = [0.0002, -0.1034, -0.1879]
                            target_std_list = [0.5436, 0.5116, 0.5033]
                        else:
                            target_mean_list = [-0.0739, -0.1597, -0.2380]
                            target_std_list = [0.5623, 0.5295, 0.5347]
                        # Mean: tensor([-0.0739, -0.1597, -0.2380])
                        # Standard Deviation: tensor([0.5623, 0.5295, 0.5347])
                        imgs_channel_mean = imgs.mean(dim=(2, 3), keepdim=True)
                        imgs_channel_std = imgs.std(dim=(2, 3), keepdim=True)
                        imgs = (imgs - imgs_channel_mean) / imgs_channel_std
                        target_mean = torch.tensor(target_mean_list, device=self.device_torch, dtype=dtype)
                        target_std = torch.tensor(target_std_list, device=self.device_torch, dtype=dtype)
                        # expand them to match dim
                        target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                        target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                        imgs = imgs * target_std + target_mean
                        batch.tensor = imgs

                        # show_tensors(imgs, 'imgs')

                    latents = self.sd.encode_images(imgs)
                    batch.latents = latents

                if self.train_config.standardize_latents:
                    if self.sd.is_xl or self.sd.is_vega or self.sd.is_ssd:
                        target_mean_list = [-0.1075, 0.0231, -0.0135, 0.2164]
                        target_std_list = [0.8979, 0.7505, 0.9150, 0.7451]
                    else:
                        target_mean_list = [0.2949, -0.3188, 0.0807, 0.1929]
                        target_std_list = [0.8560, 0.9629, 0.7778, 0.6719]

                    latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True)
                    latents_channel_std = latents.std(dim=(2, 3), keepdim=True)
                    latents = (latents - latents_channel_mean) / latents_channel_std
                    target_mean = torch.tensor(target_mean_list, device=self.device_torch, dtype=dtype)
                    target_std = torch.tensor(target_std_list, device=self.device_torch, dtype=dtype)
                    # expand them to match dim
                    target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    latents = latents * target_std + target_mean
                    batch.latents = latents

                    # show_latents(latents, self.sd.vae, 'latents')


                if batch.unconditional_tensor is not None and batch.unconditional_latents is None:
                    unconditional_imgs = batch.unconditional_tensor
                    unconditional_imgs = unconditional_imgs.to(self.device_torch, dtype=dtype)
                    unconditional_latents = self.sd.encode_images(unconditional_imgs)
                    batch.unconditional_latents = unconditional_latents * self.train_config.latent_multiplier

                unaugmented_latents = None
                if self.train_config.loss_target == 'differential_noise':
                    # we determine noise from the differential of the latents
                    unaugmented_latents = self.sd.encode_images(batch.unaugmented_tensor)

            with self.timer('prepare_scheduler'):
                
                batch_size = len(batch.file_items)
                min_noise_steps = self.train_config.min_denoising_steps
                max_noise_steps = self.train_config.max_denoising_steps
                if self.model_config.refiner_name_or_path is not None:
                    # if we are not training the unet, then we are only doing refiner and do not need to double up
                    if self.train_config.train_unet:
                        max_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = True
                    else:
                        min_noise_steps = round(self.train_config.max_denoising_steps * self.model_config.refiner_start_at)
                        do_double = False

                num_train_timesteps = self.train_config.num_train_timesteps

                if self.train_config.noise_scheduler in ['custom_lcm']:
                    # we store this value on our custom one
                    self.sd.noise_scheduler.set_timesteps(
                        self.sd.noise_scheduler.train_timesteps, device=self.device_torch
                    )
                elif self.train_config.noise_scheduler in ['lcm']:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch, original_inference_steps=num_train_timesteps
                    )
                elif self.train_config.noise_scheduler == 'flowmatch':
                    linear_timesteps = any([
                        self.train_config.linear_timesteps,
                        self.train_config.linear_timesteps2,
                        self.train_config.timestep_type == 'linear',
                        self.train_config.timestep_type in ['one_step', 'two_step', 'four_step', 'eight_step'],
                    ])
                    
                    timestep_type = 'linear' if linear_timesteps else None
                    if timestep_type is None:
                        timestep_type = self.train_config.timestep_type
                    
                    if self.train_config.timestep_type == 'next_sample':
                        # simulate a sample
                        num_train_timesteps = self.train_config.next_sample_timesteps
                        timestep_type = 'shift'
                    
                    patch_size = 1
                    if self.sd.is_flux or 'flex' in self.sd.arch or self.sd.arch == 'zimage':
                        # flux/zimage is a patch size of 1, but latents are divided by 2, so we need to double it
                        patch_size = 2
                    elif getattr(self.sd, 'patch_size', None) is not None:
                        # models that declare their own patch size (e.g. krea2, whose transformer
                        # config exposes `patch`, not `patch_size`) — use it so image_seq_len is right
                        patch_size = self.sd.patch_size
                    elif hasattr(self.sd.unet, 'config') and hasattr(self.sd.unet.config, 'patch_size'):
                        patch_size = self.sd.unet.config.patch_size
                    
                    self.sd.noise_scheduler.set_train_timesteps(
                        num_train_timesteps,
                        device=self.device_torch,
                        timestep_type=timestep_type,
                        latents=latents,
                        patch_size=patch_size,
                    )
                else:
                    self.sd.noise_scheduler.set_timesteps(
                        num_train_timesteps, device=self.device_torch
                    )
            if self.sd.is_multistage:
                with self.timer('adjust_multistage_timesteps'):
                    # get our current sample range
                    boundaries = [1] + self.sd.multistage_boundaries
                    boundary_max, boundary_min = boundaries[self.current_boundary_index], boundaries[self.current_boundary_index + 1]
                    asc_timesteps = torch.flip(self.sd.noise_scheduler.timesteps, dims=[0])
                    lo = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_max * 1000, device=asc_timesteps.device), right=False)
                    hi = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_min * 1000, device=asc_timesteps.device), right=True)
                    first_idx = (lo - 1).item() if hi > lo else 0
                    last_idx  = (hi - 1).item() if hi > lo else 999
                    min_noise_steps = first_idx
                    max_noise_steps = last_idx

            # clip min max indicies
            min_noise_steps = max(min_noise_steps, 0)
            max_noise_steps = min(max_noise_steps, num_train_timesteps - 1)
            
                    
            with self.timer('prepare_timesteps_indices'):

                content_or_style = self.train_config.content_or_style
                if is_reg:
                    content_or_style = self.train_config.content_or_style_reg

                if self.train_config.timestep_type in ['two_step', 'four_step', 'eight_step']:
                    if self.train_config.timestep_type == 'two_step':
                        indice_choices = [0, 499]
                    elif self.train_config.timestep_type == 'four_step':
                        indice_choices = [0, 250, 500, 750]
                    elif self.train_config.timestep_type == 'eight_step':
                        indice_choices = [0, 125, 250, 375, 500, 625, 750, 875]
                    timestep_indices = torch.tensor(random.choices(indice_choices, k=batch_size), device=self.device_torch)
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type == 'next_sample':
                    timestep_indices = torch.randint(
                            0,
                            num_train_timesteps - 2, # -1 for 0 idx, -1 so we can step
                            (batch_size,),
                            device=self.device_torch
                        )
                    timestep_indices = timestep_indices.long()
                elif self.train_config.timestep_type == 'one_step':
                    timestep_indices = torch.zeros((batch_size,), device=self.device_torch, dtype=torch.long)
                elif content_or_style in ['style', 'content']:
                    # this is from diffusers training code
                    # Cubic sampling for favoring later or earlier timesteps
                    # For more details about why cubic sampling is used for content / structure,
                    # refer to section 3.4 of https://arxiv.org/abs/2302.08453

                    # for content / structure, it is best to favor earlier timesteps
                    # for style, it is best to favor later timesteps

                    orig_timesteps = torch.rand((batch_size,), device=latents.device)

                    if content_or_style == 'content':
                        timestep_indices = orig_timesteps ** 3 * self.train_config.num_train_timesteps
                    elif content_or_style == 'style':
                        timestep_indices = (1 - orig_timesteps ** 3) * self.train_config.num_train_timesteps

                    timestep_indices = value_map(
                        timestep_indices,
                        0,
                        self.train_config.num_train_timesteps - 1,
                        min_noise_steps,
                        max_noise_steps
                    )
                    timestep_indices = timestep_indices.long().clamp(
                        min_noise_steps,
                        max_noise_steps
                    )
                    
                elif content_or_style == 'balanced':
                    if min_noise_steps == max_noise_steps:
                        timestep_indices = torch.ones((batch_size,), device=self.device_torch) * min_noise_steps
                    else:
                        # todo, some schedulers use indices, otheres use timesteps. Not sure what to do here
                        min_idx = min_noise_steps + 1
                        max_idx = max_noise_steps - 1
                        if self.train_config.noise_scheduler == 'flowmatch':
                            # flowmatch uses indices, so we need to use indices
                            min_idx = min_noise_steps
                            max_idx = max_noise_steps
                        timestep_indices = torch.randint(
                            min_idx,
                            max_idx,
                            (batch_size,),
                            device=self.device_torch
                        )
                    timestep_indices = timestep_indices.long()
                else:
                    raise ValueError(f"Unknown content_or_style {content_or_style}")
            with self.timer('convert_timestep_indices_to_timesteps'):
                # convert the timestep_indices to a timestep
                timesteps = self.sd.noise_scheduler.timesteps[timestep_indices.long()]
                
            with self.timer('prepare_noise'):
                # get noise
                noise = self.get_noise(latents, batch_size, dtype=dtype, batch=batch, timestep=timesteps)

                # add dynamic noise offset. Dynamic noise is offsetting the noise to the same channelwise mean as the latents
                # this will negate any noise offsets
                if self.train_config.dynamic_noise_offset and not is_reg:
                    latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
                    # subtract channel mean to that we compensate for the mean of the latents on the noise offset per channel
                    noise = noise + latents_channel_mean

                if self.train_config.loss_target == 'differential_noise':
                    differential = latents - unaugmented_latents
                    # add noise to differential
                    # noise = noise + differential
                    noise = noise + (differential * 0.5)
                    # noise = value_map(differential, 0, torch.abs(differential).max(), 0, torch.abs(noise).max())
                    latents = unaugmented_latents

                noise_multiplier = self.train_config.noise_multiplier
                
                s = (noise.shape[0], noise.shape[1], 1, 1)
                if len(noise.shape) == 5:
                    # if we have a 5d tensor, then we need to do it on a per batch item, per channel basis, per frame
                    s = (noise.shape[0], noise.shape[1], noise.shape[2], 1, 1)
                
                noise = noise * noise_multiplier
                
                if self.train_config.do_signal_correction_noise:
                    batch_noise = latents.clone().to(noise.device, dtype=noise.dtype)
                    scn_scale = torch.randn(
                        batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                        device=batch_noise.device, 
                        dtype=batch_noise.dtype
                    ) * self.train_config.signal_correction_noise_scale
                    batch_noise = batch_noise * scn_scale
                    noise = noise + batch_noise 
                
                if self.train_config.do_batch_noise_correction:
                    if latents.shape[0] == 1:
                        # if we only have a batch size of 1, then we cant do batch noise correction, so we skip it
                        print_acc("Skipping batch noise correction because batch size is 1, increase batch size and num_repeats to use this feature")
                    else:
                        # shuffle tensors ensuring that no tensor is in the same position as before
                        batch_noise = latents.clone().roll(shifts=torch.randint(1, latents.shape[0], (1,)).item(), dims=0).to(noise.device, dtype=noise.dtype)
                        batch_noise_scale = torch.randn(
                            batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                            device=batch_noise.device,
                            dtype=batch_noise.dtype
                        ) * self.train_config.batch_noise_correction_scale
                        batch_noise = batch_noise * batch_noise_scale
                        noise = noise + batch_noise
                
                if self.train_config.random_noise_shift > 0.0:
                    # get random noise -1 to 1
                    noise_shift = torch.randn(
                        batch_size, latents.shape[1], 1, 1,
                        device=noise.device,
                        dtype=noise.dtype
                    ) * self.train_config.random_noise_shift
                    # add to noise
                    noise += noise_shift
                
                if self.train_config.random_noise_multiplier > 0.0:
                    sigma = self.train_config.random_noise_multiplier
                    noise_multiplier = torch.exp(torch.randn(s, device=noise.device, dtype=noise.dtype) * sigma)
                    noise = noise * noise_multiplier
            with self.timer('make_noisy_latents'):

                latent_multiplier = self.train_config.latent_multiplier

                # handle adaptive scaling mased on std
                if self.train_config.adaptive_scaling_factor:
                    std = latents.std(dim=(2, 3), keepdim=True)
                    normalizer = 1 / (std + 1e-6)
                    latent_multiplier = normalizer

                latents = latents * latent_multiplier
                
                if self.train_config.do_blank_stabilization:
                    # zero out latents with blank prompts
                    blank_latent = torch.zeros_like(latents)
                    for i, prompt in enumerate(conditioned_prompts):
                        if prompt.strip() == '':
                            latents[i] = blank_latent[i]
                
                batch.latents = latents

                # normalize latents to a mean of 0 and an std of 1
                # mean_zero_latents = latents - latents.mean()
                # latents = mean_zero_latents / mean_zero_latents.std()

                if batch.unconditional_latents is not None:
                    batch.unconditional_latents = batch.unconditional_latents * self.train_config.latent_multiplier


                noisy_latents = self.sd.add_noise(latents, noise, timesteps)

                # determine scaled noise
                # todo do we need to scale this or does it always predict full intensity
                # noise = noisy_latents - latents

                # https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170C17-L1171C77
                if self.train_config.loss_target == 'source' or self.train_config.loss_target == 'unaugmented':
                    sigmas = self.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                    # add it to the batch
                    batch.sigmas = sigmas
                    # todo is this for sdxl? find out where this came from originally
                    # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

            def double_up_tensor(tensor: torch.Tensor):
                if tensor is None:
                    return None
                return torch.cat([tensor, tensor], dim=0)

            if do_double:
                if self.model_config.refiner_name_or_path:
                    # apply refiner double up
                    refiner_timesteps = torch.randint(
                        max_noise_steps,
                        self.train_config.max_denoising_steps,
                        (batch_size,),
                        device=self.device_torch
                    )
                    refiner_timesteps = refiner_timesteps.long()
                    # add our new timesteps on to end
                    timesteps = torch.cat([timesteps, refiner_timesteps], dim=0)

                    refiner_noisy_latents = self.sd.noise_scheduler.add_noise(latents, noise, refiner_timesteps)
                    noisy_latents = torch.cat([noisy_latents, refiner_noisy_latents], dim=0)

                else:
                    # just double it
                    noisy_latents = double_up_tensor(noisy_latents)
                    timesteps = double_up_tensor(timesteps)

                noise = double_up_tensor(noise)
                # prompts are already updated above
                imgs = double_up_tensor(imgs)
                batch.mask_tensor = double_up_tensor(batch.mask_tensor)
                batch.control_tensor = double_up_tensor(batch.control_tensor)

            noisy_latent_multiplier = self.train_config.noisy_latent_multiplier

            if noisy_latent_multiplier != 1.0:
                noisy_latents = noisy_latents * noisy_latent_multiplier

            # remove grads for these
            noisy_latents.requires_grad = False
            noisy_latents = noisy_latents.detach()
            noise.requires_grad = False
            noise = noise.detach()

        return noisy_latents, noise, timesteps, conditioned_prompts, imgs

    def setup_adapter(self):
        # t2i adapter
        is_t2i = self.adapter_config.type == 't2i'
        is_control_net = self.adapter_config.type == 'control_net'
        if self.adapter_config.type == 't2i':
            suffix = 't2i'
        elif self.adapter_config.type == 'control_net':
            suffix = 'cn'
        elif self.adapter_config.type == 'clip':
            suffix = 'clip'
        elif self.adapter_config.type == 'reference':
            suffix = 'ref'
        elif self.adapter_config.type.startswith('ip'):
            suffix = 'ip'
        else:
            suffix = 'adapter'
        adapter_name = self.name
        if self.network_config is not None:
            adapter_name = f"{adapter_name}_{suffix}"
        latest_save_path = self.get_latest_save_path(adapter_name)
        
        if latest_save_path is not None and not self.adapter_config.train:
            # the save path is for something else since we are not training
            latest_save_path = self.adapter_config.name_or_path

        dtype = get_torch_dtype(self.train_config.dtype)
        if is_t2i:
            # if we do not have a last save path and we have a name_or_path,
            # load from that
            if latest_save_path is None and self.adapter_config.name_or_path is not None:
                self.adapter = T2IAdapter.from_pretrained(
                    self.adapter_config.name_or_path,
                    torch_dtype=get_torch_dtype(self.train_config.dtype),
                    varient="fp16",
                    # use_safetensors=True,
                )
            else:
                self.adapter = T2IAdapter(
                    in_channels=self.adapter_config.in_channels,
                    channels=self.adapter_config.channels,
                    num_res_blocks=self.adapter_config.num_res_blocks,
                    downscale_factor=self.adapter_config.downscale_factor,
                    adapter_type=self.adapter_config.adapter_type,
                )
        elif is_control_net:
            if self.adapter_config.name_or_path is None:
                raise ValueError("ControlNet requires a name_or_path to load from currently")
            load_from_path = self.adapter_config.name_or_path
            if latest_save_path is not None:
                load_from_path = latest_save_path
            self.adapter = ControlNetModel.from_pretrained(
                load_from_path,
                torch_dtype=get_torch_dtype(self.train_config.dtype),
            )
        elif self.adapter_config.type == 'clip':
            self.adapter = ClipVisionAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type == 'reference':
            self.adapter = ReferenceAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
        elif self.adapter_config.type.startswith('ip'):
            self.adapter = IPAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
            )
            if self.train_config.gradient_checkpointing:
                self.adapter.enable_gradient_checkpointing()
        else:
            self.adapter = CustomAdapter(
                sd=self.sd,
                adapter_config=self.adapter_config,
                train_config=self.train_config,
            )
        self.adapter.to(self.device_torch, dtype=dtype)
        if latest_save_path is not None and not is_control_net:
            # load adapter from path
            print_acc(f"Loading adapter from {latest_save_path}")
            if is_t2i:
                loaded_state_dict = load_t2i_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
            elif self.adapter_config.type.startswith('ip'):
                # ip adapter
                loaded_state_dict = load_ip_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype,
                    direct_load=self.adapter_config.train_only_image_encoder
                )
                self.adapter.load_state_dict(loaded_state_dict)
            else:
                # custom adapter
                loaded_state_dict = load_custom_adapter_model(
                    latest_save_path,
                    self.device,
                    dtype=dtype
                )
                self.adapter.load_state_dict(loaded_state_dict)
        if latest_save_path is not None and self.adapter_config.train:
            self.load_training_state_from_metadata(latest_save_path)
        # set trainable params
        self.sd.adapter = self.adapter

    def maybe_run_te_cache_worker(self) -> bool:
        """If this run caches text embeddings on a supported arch, ensure every embedding
        is on disk — running a throwaway TE worker subprocess if needed — so the trainer
        can load skip_te and never hold the text encoder alongside the transformer.

        Returns True if the trainer should load with skip_te and read embeds from disk.
        Raises if the worker fails or the cache is still incomplete afterward.
        """
        import sys
        import json
        import subprocess

        if not self.is_caching_text_embeddings:
            return False
        # don't recurse: the worker process sets this so it caches in-process instead
        if os.environ.get('AITK_IS_TE_WORKER', '0') == '1':
            return False
        # only archs with a te_only / skip_te load path
        if self.model_config.arch not in ('zimage', 'anima', 'ideogram4', 'krea2'):
            return False
        if not (hasattr(self, 'cache_text_encoder_outputs_to_disk') and hasattr(self, 'aux_cache_is_ready')):
            return False

        # already fully cached for this exact config and the current authoritative
        # .txt sidecars — use it, no worker needed
        if self.aux_cache_is_ready() and self._te_caption_manifest_is_current():
            print_acc("[te-worker] aux embedding cache already complete; using cached embeds (skip_te)")
            return True

        pipeline_dir = os.path.join(self.save_root, '.pipeline')
        os.makedirs(pipeline_dir, exist_ok=True)
        worker_config_path = os.path.join(pipeline_dir, 'te_worker_config.json')
        with open(worker_config_path, 'w') as f:
            json.dump(self.job.raw_config, f)

        env = dict(os.environ)
        env['AITK_IS_TE_WORKER'] = '1'

        cmd = [sys.executable, '-m', 'toolkit.te_cache_worker', worker_config_path]
        log_file = os.environ.get('AITK_LOG_FILE')
        if log_file:
            cmd += ['-l', log_file]

        print_acc("[te-worker] launching text-encoder cache worker (text encoder will not be loaded in the trainer)")
        result = subprocess.run(cmd, env=env, cwd=os.getcwd())
        if result.returncode != 0:
            raise RuntimeError(
                f"Text-encoder cache worker failed (exit code {result.returncode}). "
                f"Config: {worker_config_path}"
            )
        if not (
            self.aux_cache_is_ready()
            and self._te_caption_manifest_is_current()
        ):
            raise RuntimeError(
                "Text-encoder cache worker finished but the embedding cache or authoritative "
                ".txt caption manifest is still incomplete. Refusing to train with skip_te."
            )
        print_acc("[te-worker] cache complete; trainer will load without the text encoder (skip_te)")
        return True

    def _te_caption_source_digest(self) -> str:
        """Digest authoritative caption sidecars without constructing a dataset.

        This runs before any model is loaded. captions.json is intentionally not
        considered: image-adjacent files matching caption_ext are authoritative.
        """
        import hashlib

        digest = hashlib.sha256()
        for index, cfg in enumerate(self.dataset_configs or []):
            ext = str(getattr(cfg, 'caption_ext', '.txt') or '.txt')
            if not ext.startswith('.'):
                ext = '.' + ext
            settings = (
                index,
                ext,
                getattr(cfg, 'default_caption', None),
                getattr(cfg, 'trigger_word', None),
                tuple(getattr(cfg, 'replacements', []) or []),
                bool(getattr(cfg, 'use_short_captions', False)),
            )
            digest.update(repr(settings).encode('utf-8'))
            roots = []
            for value in (
                getattr(cfg, 'folder_path', None),
                getattr(cfg, 'dataset_path', None),
            ):
                if value and os.path.isdir(value):
                    roots.append(os.path.abspath(value))
            for root in sorted(set(roots)):
                matches = []
                for dirpath, dirnames, filenames in os.walk(root):
                    # Cache directories can be huge and never contain source captions.
                    dirnames[:] = [d for d in dirnames if d not in ('_t_e_cache', '_latent_cache')]
                    for filename in filenames:
                        if filename.lower().endswith(ext.lower()):
                            matches.append(os.path.join(dirpath, filename))
                for path in sorted(matches, key=lambda p: os.path.normcase(p)):
                    digest.update(os.path.normcase(os.path.abspath(path)).encode('utf-8'))
                    with open(path, 'rb') as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b''):
                            digest.update(chunk)
        return digest.hexdigest()

    def _te_caption_manifest_path(self):
        return os.path.join(self.save_root, '.pipeline', 'te_caption_sources.sha256')

    def _te_caption_manifest_is_current(self) -> bool:
        try:
            with open(self._te_caption_manifest_path(), 'r', encoding='utf-8') as handle:
                saved = handle.read().strip()
            return saved == self._te_caption_source_digest()
        except (FileNotFoundError, OSError):
            return False

    def _write_te_caption_manifest(self) -> None:
        path = self._te_caption_manifest_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from toolkit.cache_utils import atomic_write
        digest = self._te_caption_source_digest()
        atomic_write(
            path,
            lambda tmp: tmp.write_text(digest, encoding='utf-8'),
        )

    def run_te_cache_worker(self):
        """Entry for the throwaway text-encoder cache worker process.

        Loads the model with te_only=True (text encoder + tokenizers only, no transformer
        or VAE), encodes and persists every text embedding the trainer will need to disk
        (dataset captions, DOP, and aux blank/trigger/uncond/sample embeds), then returns
        so the process can exit and let the OS reclaim all text-encoder memory.

        Only supported on archs that implement the te_only load path and
        on process classes that implement cache_text_encoder_outputs_to_disk (SDTrainer).
        """
        BaseTrainProcess.run(self)
        self.hook_before_model_load()
        model_config_to_load = copy.deepcopy(self.model_config)

        ModelClass = get_model_class(self.model_config)
        sampler = ModelClass.get_train_scheduler() if hasattr(ModelClass, 'get_train_scheduler') else None

        self.sd = ModelClass(
            device=self.accelerator.device,
            model_config=model_config_to_load,
            dtype=self.train_config.dtype,
            custom_pipeline=self.custom_pipeline,
            noise_scheduler=sampler,
        )
        self.sd.te_only = True
        self.sd.skip_te = False

        self.hook_after_sd_init_before_load()
        from toolkit.memory_management.arena_offload import model_load_arena_session
        with model_load_arena_session(self.sd):
            self.sd.load_model()

        if not hasattr(self, 'cache_text_encoder_outputs_to_disk'):
            raise NotImplementedError(
                "This process type does not support the text-encoder cache worker."
            )
        self.cache_text_encoder_outputs_to_disk()
        self._write_te_caption_manifest()

    def _observed_input_shapes(self, layout):
        """Every (latent, text) shape combination this job can feed the trunk.

        Latent sizes come from the buckets the datasets actually built plus the
        sample resolutions -- sampling runs through the same compiled block
        kernels, so a sample size outside the training buckets would otherwise
        land outside the declared bounds. Returns None when any source cannot be
        enumerated, which leaves the hints unset rather than guessing.
        """
        from toolkit.compile_shape_bounds import ObservedInputShape

        vae_scale = int(getattr(self.sd, 'vae_scale_factor', 0) or 0)
        if vae_scale < 1:
            return None

        pixel_sizes = set()
        for loader in (self.data_loader, self.data_loader_reg):
            if loader is None:
                continue
            dataset = loader.dataset
            subsets = getattr(dataset, 'datasets', None) or [dataset]
            for subset in subsets:
                buckets = getattr(subset, 'buckets', None)
                if not buckets:
                    return None
                for bucket in buckets.values():
                    pixel_sizes.add((int(bucket.height), int(bucket.width)))

        sample_items = getattr(self.sample_config, 'samples', None) or []
        for item in sample_items:
            pixel_sizes.add((int(item.height), int(item.width)))

        if not pixel_sizes:
            return None

        text_bounds = (0, 0)
        if layout.includes_text:
            bounds = self.sd.get_text_length_bounds()
            if bounds is None:
                return None
            text_bounds = (int(bounds[0]), int(bounds[1]))

        shapes = []
        for height, width in sorted(pixel_sizes):
            if height % vae_scale or width % vae_scale:
                return None
            for text_length in set(text_bounds):
                shapes.append(
                    ObservedInputShape(
                        latent_height=height // vae_scale,
                        latent_width=width // vae_scale,
                        text_length=text_length,
                    )
                )
        return shapes

    def _apply_derived_compile_dynamic_hints(self):
        """Bound the compiled trunk's dynamic sequence dim from the job's shapes.

        Only fills in hints the config left empty: an explicit
        `compile_dynamic_hints` in the config always wins. Never touches
        `compile_dynamic` itself -- the derived hints are a bound layered on
        top of whatever dynamic mode the config already picked, gated by
        `compile_dynamic_hints_auto` since GPU measurements found the best
        setting is torch-version/hardware dependent (not always a win).
        """
        if getattr(self.model_config, 'compile_dynamic_hints', ()):
            return
        if not getattr(self.model_config, 'compile_dynamic_hints_auto', True):
            return
        runtime = get_memory_runtime(unwrap_model(self.sd.unet))
        if runtime is None or not runtime.config.compile_blocks:
            return

        from toolkit.compile_shape_bounds import estimate_hidden_sequence_bounds

        layout = self.sd.get_compile_sequence_layout()
        if layout is None:
            return
        shapes = self._observed_input_shapes(layout)
        if not shapes:
            return

        bounds = estimate_hidden_sequence_bounds(
            transformer=unwrap_model(self.sd.unet),
            observed_shapes=shapes,
            layout=layout,
        )
        if bounds is None or bounds.minimum == bounds.maximum:
            # A single shape needs no range: let it specialize.
            return

        hints = ((1, bounds.minimum, bounds.maximum),)
        runtime.set_compile_dynamic_hints(hints)
        self.model_config.compile_dynamic_hints = hints
        print_acc(
            f"Compiled blocks: sequence dim bounded to "
            f"[{bounds.minimum}, {bounds.maximum}] tokens from "
            f"{len(shapes)} observed shapes."
        )

    def run(self):
        # torch.autograd.set_detect_anomaly(True)
        # run base process run
        BaseTrainProcess.run(self)
        params = []

        ### HOOK ###
        self.hook_before_model_load()
        model_config_to_load = copy.deepcopy(self.model_config)

        if self.is_fine_tuning or self.train_config.merge_network_on_save:
            # get the latest checkpoint
            # check to see if we have a latest save
            # exclude pretrained_lora_path here so a pretrained lora is not loaded as full model
            # weights. It is loaded as the initial lora later when building the network.
            latest_save_path = self.get_latest_save_path(include_pretrained_lora=False)

            if latest_save_path is not None:
                print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                model_config_to_load.name_or_path = latest_save_path
                self.load_training_state_from_metadata(latest_save_path)

        ModelClass = get_model_class(self.model_config)
        # if the model class has get_train_scheduler static method
        if hasattr(ModelClass, 'get_train_scheduler'):
            sampler = ModelClass.get_train_scheduler()
        else:
            # get the noise scheduler
            arch = 'sd'
            if self.model_config.is_pixart:
                arch = 'pixart'
            if self.model_config.is_flux:
                arch = 'flux'
            if self.model_config.is_lumina2:
                arch = 'lumina2'
            sampler = get_sampler(
                self.train_config.noise_scheduler,
                {
                    "prediction_type": "v_prediction" if self.model_config.is_v_pred else "epsilon",
                },
                arch=arch,
            )

        if self.train_config.train_refiner and self.model_config.refiner_name_or_path is not None and self.network_config is None:
            previous_refiner_save = self.get_latest_save_path(self.job.name + '_refiner')
            if previous_refiner_save is not None:
                model_config_to_load.refiner_name_or_path = previous_refiner_save
                self.load_training_state_from_metadata(previous_refiner_save)

        # Ensure all text embeddings are cached to disk (running a throwaway TE worker if
        # needed) so the trainer can load without the text encoder.
        self._use_cached_te = self.maybe_run_te_cache_worker()
        if self._use_cached_te and not self._te_caption_manifest_is_current():
            print_acc("[te-worker] caption sources changed after cache worker; refreshing TE cache before skip_te load")
            self._use_cached_te = self.maybe_run_te_cache_worker()

        self.sd = ModelClass(
            # todo handle single gpu and multi gpu here
            # device=self.device,
            device=self.accelerator.device,
            model_config=model_config_to_load,
            dtype=self.train_config.dtype,
            custom_pipeline=self.custom_pipeline,
            noise_scheduler=sampler,
        )
        if self._use_cached_te:
            # never load the text encoder in the trainer; embeds come from disk
            self.sd.skip_te = True

        # Shape-aware cold-start hint for arena-offload attach (see
        # Krea2Model._estimate_training_working_reserve_bytes); harmless for
        # model classes that don't read it.
        self.sd.dataset_configs = self.dataset_configs
        # Arena-capable models must know checkpoint ownership before their
        # destructive canonical-storage preparation in load_model().
        self.sd.train_config = self.train_config

        self.hook_after_sd_init_before_load()
        # run base sd process run
        from toolkit.memory_management.arena_offload import model_load_arena_session
        with model_load_arena_session(self.sd):
            self.sd.load_model()

        if self._use_cached_te:
            if not self.load_cached_text_encoder_outputs_from_disk():
                raise RuntimeError(
                    "TE worker ran but cached text-encoder outputs could not be loaded "
                    "from disk. Refusing to train with a skipped text encoder."
                )
        
        self.sd.add_after_sample_image_hook(self.sample_step_hook)

        dtype = get_torch_dtype(self.train_config.dtype)

        # model is loaded from BaseSDProcess
        unet = self.sd.unet
        vae = self.sd.vae
        tokenizer = self.sd.tokenizer
        text_encoder = self.sd.text_encoder
        noise_scheduler = self.sd.noise_scheduler

        if self.train_config.xformers:
            vae.enable_xformers_memory_efficient_attention()
            unet.enable_xformers_memory_efficient_attention()
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    # if it has it
                    if hasattr(te, 'enable_xformers_memory_efficient_attention'):
                        te.enable_xformers_memory_efficient_attention()
        
        if self.train_config.attention_backend != 'native':
            if hasattr(vae, 'set_attention_backend'):
                vae.set_attention_backend(self.train_config.attention_backend)
            if hasattr(unet, 'set_attention_backend'):
                unet.set_attention_backend(self.train_config.attention_backend)
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'set_attention_backend'):
                        te.set_attention_backend(self.train_config.attention_backend)
            else:
                if hasattr(text_encoder, 'set_attention_backend'):
                    text_encoder.set_attention_backend(self.train_config.attention_backend)
        if self.train_config.sdp:
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        # # check if we have sage and is flux
        # if self.sd.is_flux:
        #     # try_to_activate_sage_attn()
        #     try:
        #         from sageattention import sageattn
        #         from toolkit.models.flux_sage_attn import FluxSageAttnProcessor2_0
        #         model: FluxTransformer2DModel = self.sd.unet
        #         # enable sage attention on each block
        #         for block in model.transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
        #         for block in model.single_transformer_blocks:
        #             processor = FluxSageAttnProcessor2_0()
        #             block.attn.set_processor(processor)
                    
        #     except ImportError:
        #         print_acc("sage attention is not installed. Using SDP instead")

        if self.train_config.gradient_checkpointing:
            # if has method enable_gradient_checkpointing
            if hasattr(unet, 'enable_gradient_checkpointing'):
                unet.enable_gradient_checkpointing()
            elif hasattr(unet, 'gradient_checkpointing'):
                unet.gradient_checkpointing = True
            else:
                print("Gradient checkpointing not supported on this model")
            if isinstance(text_encoder, list):
                for te in text_encoder:
                    if hasattr(te, 'enable_gradient_checkpointing'):
                        te.enable_gradient_checkpointing()
                    if hasattr(te, "gradient_checkpointing_enable"):
                        te.gradient_checkpointing_enable()
            else:
                if hasattr(text_encoder, 'enable_gradient_checkpointing'):
                    text_encoder.enable_gradient_checkpointing()
                if hasattr(text_encoder, "gradient_checkpointing_enable"):
                    text_encoder.gradient_checkpointing_enable()

        if self.sd.refiner_unet is not None:
            self.sd.refiner_unet.to(self.device_torch, dtype=dtype)
            self.sd.refiner_unet.requires_grad_(False)
            self.sd.refiner_unet.eval()
            if self.train_config.xformers:
                self.sd.refiner_unet.enable_xformers_memory_efficient_attention()
            if self.train_config.gradient_checkpointing:
                self.sd.refiner_unet.enable_gradient_checkpointing()

        if isinstance(text_encoder, list):
            for te in text_encoder:
                te.requires_grad_(False)
                te.eval()
        else:
            text_encoder.requires_grad_(False)
            text_encoder.eval()
        arena_runtime = get_memory_runtime(unet)
        if arena_runtime is not None:
            arena_runtime.place_permanent_modules(self.device_torch, dtype)
        else:
            unet.to(self.device_torch, dtype=dtype)
        unet.requires_grad_(False)
        unet.eval()
        vae = vae.to(torch.device('cpu'), dtype=dtype)
        vae.requires_grad_(False)
        vae.eval()
        if self.train_config.learnable_snr_gos:
            self.snr_gos = LearnableSNRGamma(
                self.sd.noise_scheduler, device=self.device_torch
            )
            # check to see if previous settings exist
            path_to_load = os.path.join(self.save_root, 'learnable_snr.json')
            if os.path.exists(path_to_load):
                with open(path_to_load, 'r') as f:
                    json_data = json.load(f)
                    if 'offset' in json_data:
                        # legacy
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset'], device=self.device_torch)
                    else:
                        self.snr_gos.offset_1.data = torch.tensor(json_data['offset_1'], device=self.device_torch)
                        self.snr_gos.offset_2.data = torch.tensor(json_data['offset_2'], device=self.device_torch)
                    self.snr_gos.scale.data = torch.tensor(json_data['scale'], device=self.device_torch)
                    self.snr_gos.gamma.data = torch.tensor(json_data['gamma'], device=self.device_torch)

        self.hook_after_model_load()
        flush()
        if not self.is_fine_tuning:
            if self.network_config is not None:
                # TODO should we completely switch to LycorisSpecialNetwork?
                network_kwargs = self.network_config.network_kwargs
                is_lycoris = False
                is_lorm = self.network_config.type.lower() == 'lorm'
                # default to LoCON if there are any conv layers or if it is named
                NetworkClass = LoRASpecialNetwork
                if self.network_config.type.lower() == 'locon' or self.network_config.type.lower() == 'lycoris':
                    NetworkClass = LycorisSpecialNetwork
                    is_lycoris = True

                if is_lorm:
                    network_kwargs['ignore_if_contains'] = lorm_ignore_if_contains
                    network_kwargs['parameter_threshold'] = lorm_parameter_threshold
                    network_kwargs['target_lin_modules'] = LORM_TARGET_REPLACE_MODULE

                # if is_lycoris:
                #     preset = PRESET['full']
                # NetworkClass.apply_preset(preset)
                
                if hasattr(self.sd, 'target_lora_modules'):
                    network_kwargs['target_lin_modules'] = self.sd.target_lora_modules

                self.network = NetworkClass(
                    text_encoder=text_encoder,
                    unet=self.sd.get_model_to_train(),
                    lora_dim=self.network_config.linear,
                    multiplier=1.0,
                    alpha=self.network_config.linear_alpha,
                    train_unet=self.train_config.train_unet,
                    train_text_encoder=self.train_config.train_text_encoder,
                    conv_lora_dim=self.network_config.conv,
                    conv_alpha=self.network_config.conv_alpha,
                    is_sdxl=self.model_config.is_xl or self.model_config.is_ssd,
                    is_v2=self.model_config.is_v2,
                    is_v3=self.model_config.is_v3,
                    is_pixart=self.model_config.is_pixart,
                    is_auraflow=self.model_config.is_auraflow,
                    is_flux=self.model_config.is_flux,
                    is_lumina2=self.model_config.is_lumina2,
                    is_ssd=self.model_config.is_ssd,
                    is_vega=self.model_config.is_vega,
                    dropout=self.network_config.dropout,
                    use_text_encoder_1=self.model_config.use_text_encoder_1,
                    use_text_encoder_2=self.model_config.use_text_encoder_2,
                    use_bias=is_lorm,
                    is_lorm=is_lorm,
                    network_config=self.network_config,
                    network_type=self.network_config.type,
                    transformer_only=self.network_config.transformer_only,
                    is_transformer=self.sd.is_transformer,
                    base_model=self.sd,
                    **network_kwargs
                )


                # todo switch everything to proper mixed precision like this
                self.network.force_to(self.device_torch, dtype=torch.float32)
                # give network to sd so it can use it
                self.sd.network = self.network
                self.network._update_torch_multiplier()

                self.network.apply_to(
                    text_encoder,
                    unet,
                    self.train_config.train_text_encoder,
                    self.train_config.train_unet
                )

                # we cannot merge in if quantized or offloading. note: torchao quantized weights can
                # still be force merged at save time for the merge-and-reset method (see save logic),
                # but we keep can_merge_in False here so sampling never merges in/out.
                if self.model_config.quantize or self.model_config.layer_offloading:
                    # todo find a way around this
                    self.network.can_merge_in = False

                if is_lorm:
                    self.network.is_lorm = True
                    # make sure it is on the right device
                    arena_runtime = get_memory_runtime(self.sd.unet)
                    if arena_runtime is not None:
                        arena_runtime.place_permanent_modules(self.sd.device, dtype)
                    else:
                        self.sd.unet.to(self.sd.device, dtype=dtype)
                    original_unet_param_count = count_parameters(self.sd.unet)
                    self.network.setup_lorm()
                    new_unet_param_count = original_unet_param_count - self.network.calculate_lorem_parameter_reduction()

                    print_lorm_extract_details(
                        start_num_params=original_unet_param_count,
                        end_num_params=new_unet_param_count,
                        num_replaced=len(self.network.get_all_modules()),
                    )

                self.network.prepare_grad_etc(text_encoder, unet)
                flush()

                # LyCORIS doesnt have default_lr
                config = {
                    'text_encoder_lr': self.train_config.lr,
                    'unet_lr': self.train_config.lr,
                }
                sig = inspect.signature(self.network.prepare_optimizer_params)
                if 'default_lr' in sig.parameters:
                    config['default_lr'] = self.train_config.lr
                if 'learning_rate' in sig.parameters:
                    config['learning_rate'] = self.train_config.lr
                params_net = self.network.prepare_optimizer_params(
                    **config
                )

                params += params_net

                if self.train_config.gradient_checkpointing:
                    self.network.enable_gradient_checkpointing()

                lora_name = self.name
                # need to adapt name so they are not mixed up
                if self.named_lora:
                    lora_name = f"{lora_name}_LoRA"

                latest_save_path = self.get_latest_save_path(lora_name)
                extra_weights = None
                if latest_save_path is not None and not self.train_config.merge_network_on_save:
                    print_acc(f"#### IMPORTANT RESUMING FROM {latest_save_path} ####")
                    print_acc(f"Loading from {latest_save_path}")
                    extra_weights = self.load_weights(latest_save_path)
                    self.network.multiplier = 1.0
                elif self.train_config.merge_network_on_save and self.network_config.pretrained_lora_path is not None:
                    # with merge_network_on_save, saved checkpoints are full models that get loaded as the
                    # base model. Only load the pretrained lora as the initial lora when we are not resuming
                    # from a saved checkpoint (otherwise it is already merged into the loaded model).
                    resume_save_path = self.get_latest_save_path(include_pretrained_lora=False)
                    if resume_save_path is None and os.path.exists(self.network_config.pretrained_lora_path):
                        print_acc(f"Loading initial lora from pretrained lora path: {self.network_config.pretrained_lora_path}")
                        extra_weights = self.load_weights(self.network_config.pretrained_lora_path)
                        self.network.multiplier = 1.0
                
                if self.network_config.layer_offloading:
                    MemoryManager.attach(
                        self.network,
                        self.device_torch
                    )

            if self.embed_config is not None:
                # we are doing embedding training as well
                self.embedding = Embedding(
                    sd=self.sd,
                    embed_config=self.embed_config
                )
                latest_save_path = self.get_latest_save_path(self.embed_config.trigger)
                # load last saved weights
                if latest_save_path is not None:
                    self.embedding.load_embedding_from_file(latest_save_path, self.device_torch)
                    if self.embedding.step > 1:
                        self.step_num = self.embedding.step
                        self.start_step = self.step_num

                # self.step_num = self.embedding.step
                # self.start_step = self.step_num
                params.append({
                    'params': list(self.embedding.get_trainable_params()),
                    'lr': self.train_config.embedding_lr
                })

                flush()
            
            if self.decorator_config is not None:
                self.decorator = Decorator(
                    num_tokens=self.decorator_config.num_tokens,
                    token_size=4096 # t5xxl hidden size for flux
                )
                latest_save_path = self.get_latest_save_path()
                # load last saved weights
                if latest_save_path is not None:
                    state_dict = load_file(latest_save_path)
                    self.decorator.load_state_dict(state_dict)
                    self.load_training_state_from_metadata(latest_save_path)
                    
                params.append({
                    'params': list(self.decorator.parameters()),
                    'lr': self.train_config.lr
                })
                
                # give it to the sd network
                self.sd.decorator = self.decorator
                self.decorator.to(self.device_torch, dtype=torch.float32)
                self.decorator.train()

                flush()

            if self.adapter_config is not None:
                self.setup_adapter()
                if self.adapter_config.train:

                    if isinstance(self.adapter, IPAdapter):
                        # we have custom LR groups for IPAdapter
                        adapter_param_groups = self.adapter.get_parameter_groups(self.train_config.adapter_lr)
                        for group in adapter_param_groups:
                            params.append(group)
                    else:
                        # set trainable params
                        params.append({
                            'params': list(self.adapter.parameters()),
                            'lr': self.train_config.adapter_lr
                        })

                if self.train_config.gradient_checkpointing:
                    self.adapter.enable_gradient_checkpointing()
                flush()

            params = self.load_additional_training_modules(params)

        else:  # no network, embedding or adapter
            # set the device state preset before getting params
            self.sd.set_device_state(self.get_params_device_state_preset)

            # params = self.get_params()
            if len(params) == 0:
                # will only return savable weights and ones with grad
                params = self.sd.prepare_optimizer_params(
                    unet=self.train_config.train_unet,
                    text_encoder=self.train_config.train_text_encoder,
                    text_encoder_lr=self.train_config.lr,
                    unet_lr=self.train_config.lr,
                    default_lr=self.train_config.lr,
                    refiner=self.train_config.train_refiner and self.sd.refiner_unet is not None,
                    refiner_lr=self.train_config.refiner_lr,
                )
            # we may be using it for prompt injections
            if self.adapter_config is not None and self.adapter is None:
                self.setup_adapter()
        flush()

        ### HOOK ###
        params = self.hook_add_extra_train_params(params)
        self.params = params
        # self.params = []

        # for param in params:
        #     if isinstance(param, dict):
        #         self.params += param['params']
        #     else:
        #         self.params.append(param)

        if self.train_config.start_step is not None:
            self.step_num = self.train_config.start_step
            self.start_step = self.step_num

        optimizer_type = self.train_config.optimizer.lower()
        
        # esure params require grad
        self.ensure_params_requires_grad(force=True)
        optimizer = get_optimizer(self.params, optimizer_type, learning_rate=self.train_config.lr,
                                  optimizer_params=self.train_config.optimizer_params)
        self.optimizer = optimizer
        
        # set it to do paramiter swapping
        if self.train_config.do_paramiter_swapping:
            # only works for adafactor, but it should have thrown an error prior to this otherwise
            self.optimizer.enable_paramiter_swapping(self.train_config.paramiter_swapping_factor)

        # check if it exists
        optimizer_state_filename = f'optimizer.pt'
        optimizer_state_file_path = os.path.join(self.save_root, optimizer_state_filename)
        if os.path.exists(optimizer_state_file_path):
            # try to load
            # previous param groups
            # previous_params = copy.deepcopy(optimizer.param_groups)
            previous_lrs = []
            for group in optimizer.param_groups:
                previous_lrs.append(group['lr'])

            load_optimizer = True
            if self.network is not None:
                if self.network.did_change_weights:
                    # do not load optimizer if the network changed, it will result in
                    # a double state that will oom.
                    load_optimizer = False

            if load_optimizer:
                try:
                    print_acc(f"Loading optimizer state from {optimizer_state_file_path}")
                    optimizer_state_dict = torch.load(optimizer_state_file_path, weights_only=True)
                    optimizer.load_state_dict(optimizer_state_dict)
                    del optimizer_state_dict
                    flush()
                except Exception as e:
                    print_acc(f"Failed to load optimizer state from {optimizer_state_file_path}")
                    print_acc(e)

            # update the optimizer LR from the params
            print_acc(f"Updating optimizer LR from params")
            if len(previous_lrs) > 0:
                for i, group in enumerate(optimizer.param_groups):
                    group['lr'] = previous_lrs[i]
                    group['initial_lr'] = previous_lrs[i]

            # Update the learning rates if they changed
            # optimizer.param_groups = previous_params

        # set up the ema now that the optimizer (and its params) are ready
        self.setup_ema()

        lr_scheduler_params = self.train_config.lr_scheduler_params

        # make sure it had bare minimum
        if 'max_iterations' not in lr_scheduler_params:
            lr_scheduler_params['total_iters'] = self.train_config.steps

        lr_scheduler = get_lr_scheduler(
            self.train_config.lr_scheduler,
            optimizer,
            **lr_scheduler_params
        )
        self.lr_scheduler = lr_scheduler

        ### HOOk ###
        self.before_dataset_load()
        if self._use_cached_te and not self._te_caption_manifest_is_current():
            raise RuntimeError(
                "Caption sidecar files changed after the text-encoder cache worker finished. "
                "Restart the job so the TE worker can encode the updated captions before "
                "the trainer loads with skip_te."
            )
        # load datasets if passed in the root process
        if self.datasets is not None:
            self.data_loader = get_dataloader_from_datasets(self.datasets, self.train_config.batch_size, self.sd)
        if self.datasets_reg is not None:
            self.data_loader_reg = get_dataloader_from_datasets(self.datasets_reg, self.train_config.batch_size,
                                                                self.sd)

        flush()
        self.last_save_step = self.step_num
        ### HOOK ###
        self.hook_before_train_loop()

        wants_torch_compile = bool(
            self.model_config.compile
            or getattr(self.model_config, 'train_compile_blocks', False)
        )
        compile_unavailable_reason = (
            _torch_compile_backend_unavailable_reason()
            if wants_torch_compile
            else None
        )
        if compile_unavailable_reason is not None:
            print_acc("WARNING: compile is disabled.")
            print_acc(compile_unavailable_reason)
            print_acc("Install a working 'triton' package to use torch.compile.")
            self.model_config.compile = False
            self.model_config.train_compile_blocks = False

        self._apply_derived_compile_dynamic_hints()

        # ============================================================
        # COMPILE
        #
        # compile: true
        #     -> whole-model torch.compile
        #
        # compile: true
        # block_compile: true
        #     -> block-level compilation
        # ============================================================
        if self.model_config.compile:
            compiled_refs = []  # (block_list, index, original_block) for rollback on failure
            try:
                inner_unet_check = unwrap_model(self.sd.unet)
                # Compile ownership is exclusive: when the memory runtime
                # compiles its own block kernels, generic block compile must not
                # also wrap them.
                immutable_compile_owner = memory_runtime_owns_compile(inner_unet_check)
                is_unet_offloaded = is_memory_managed(inner_unet_check)

                text_encoder = getattr(self.sd, "text_encoder", None)
                text_encoder_check = unwrap_model(text_encoder) if text_encoder is not None else None
                is_te_offloaded = hasattr(text_encoder_check, '_memory_manager') if text_encoder_check is not None else False

                is_unet_quantized = getattr(self.model_config, 'quantize', False)
                is_quantized = is_unet_quantized or getattr(self.model_config, 'quantize_te', False)

                if not is_unet_offloaded:
                    self.sd.unet.to(self.device_torch)

                compile_debug = getattr(self.model_config, 'compile_debug', False)
                if compile_debug:
                    # graph_breaks: where Dynamo had to split compiled regions.
                    # recompiles:   whether the same block keeps being retraced because
                    #               guards fail (shape/device/object-id changes) — the
                    #               more dangerous failure mode with 28 compiled blocks.
                    # guards:       what conditions are being checked before each call.
                    torch._logging.set_logs(
                        graph_breaks=True,
                        recompiles=True,
                        # guards=True floods output (one line per guard per block per
                        # call). Enable via TORCH_LOGS=guards only when chasing a
                        # specific recompile reason.
                    )
                    # stdout/stderr are unreachable — training runs in a subprocess
                    # and print_acc is the only channel that reaches the user.
                    # Route dynamo log records through print_acc via a custom handler.
                    import logging as _logging

                    class _PrintAccHandler(_logging.Handler):
                        def __init__(self, fn):
                            super().__init__()
                            self._fn = fn
                        def emit(self, record):
                            try:
                                self._fn(self.format(record))
                            except Exception:
                                pass

                    _dynamo_logger = _logging.getLogger("torch._dynamo")
                    if not any(isinstance(h, _PrintAccHandler) for h in _dynamo_logger.handlers):
                        _h = _PrintAccHandler(print_acc)
                        _h.setLevel(_logging.DEBUG)
                        _dynamo_logger.addHandler(_h)
                        _dynamo_logger.setLevel(_logging.DEBUG)
                    print_acc(
                        "compile_debug=True: graph_breaks and recompiles routed through "
                        "print_acc. Recompile messages every timestep = guard failures "
                        "(shape/device/object-id changed)."
                    )

                cache_size_limit = getattr(self.model_config, 'cache_size_limit', None)
                user_set_cache_limit = cache_size_limit is not None
                if user_set_cache_limit:
                    torch._dynamo.config.cache_size_limit = cache_size_limit
                # Compile failures must remain visible for every weight format.
                # In particular, quantized graphs used to suppress an actual
                # CPU subgraph failure and silently run that block eagerly.
                torch._dynamo.config.suppress_errors = False
                # torch 2.9 inductor bug: the new memory-coalescing tiling analysis
                # crashes on some dynamic-shape index expressions (sympy PowByNatural
                # "assert p >= 0", seen with Qwen Image). The analysis doesn't apply
                # to dynamic shapes anyway, so turn it off.
                if hasattr(torch._inductor.config.triton, 'coalesce_tiling_analysis'):
                    torch._inductor.config.triton.coalesce_tiling_analysis = False

                compile_mode = getattr(self.model_config, 'compile_mode', 'default')
                compile_dynamic = getattr(self.model_config, 'compile_dynamic', True)
                compile_fullgraph = getattr(self.model_config, 'compile_fullgraph', False)
                block_compile = getattr(self.model_config, 'block_compile', False)

                if is_quantized and block_compile:
                    print_acc(
                        "Quantized model detected: block-level compile requested; "
                        "avoiding model/offload wrapper code."
                    )

                if is_quantized and compile_mode == 'default':
                    print_acc(
                        "Quantized model detected: using torch.compile mode='default'. "
                        "Set compile_mode explicitly to opt into more aggressive modes."
                    )

                if is_quantized and compile_fullgraph:
                    print_acc(
                        "Quantized model detected: fullgraph=True is incompatible, "
                        "switching to fullgraph=False."
                    )
                    compile_fullgraph = False

                cache_info = ""

                # ====================================================
                # BLOCK COMPILE
                # ====================================================
                if immutable_compile_owner and block_compile:
                    print_acc(
                        "Arena dispatcher owns stateless per-block compilation; "
                        "skipping the trainer's generic module block_compile."
                    )
                elif block_compile:
                    BLOCK_LIST_ATTRS = self.sd.get_transformer_block_names()

                    if BLOCK_LIST_ATTRS is None or len(BLOCK_LIST_ATTRS) == 0:
                        BLOCK_LIST_ATTRS = [
                            'layers',
                            'transformer_blocks',
                            'single_transformer_blocks',
                            'double_stream_blocks',
                            'single_stream_blocks',
                            'double_blocks',
                            'single_blocks',
                            'blocks',
                        ]
                    inner_unet = unwrap_model(self.sd.unet)

                    compiled_block_count = 0

                    for attr_name in BLOCK_LIST_ATTRS:
                        # attr_name may be a dotted path for models that nest their
                        # blocks (e.g. hidream_o1's "model.language_model.layers").
                        block_list = inner_unet
                        for part in attr_name.split('.'):
                            block_list = getattr(block_list, part, None)
                            if block_list is None:
                                break

                        if block_list is None:
                            continue

                        if not hasattr(block_list, '__len__'):
                            continue

                        for i, block in enumerate(block_list):
                            if not isinstance(block, torch.nn.Module):
                                continue

                            if hasattr(block, '_hf_hook'):
                                continue

                            compiled_refs.append((block_list, i, block))
                            block_list[i] = torch.compile(
                                block,
                                mode=compile_mode,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )
                            compiled_block_count += 1

                    if compiled_block_count > 0:
                        if user_set_cache_limit:
                            auto_cache_limit = max(cache_size_limit, compiled_block_count * 2)
                            if auto_cache_limit != cache_size_limit:
                                torch._dynamo.config.cache_size_limit = auto_cache_limit
                                cache_info = f", cache_size_limit={auto_cache_limit} (auto)"
                            else:
                                cache_info = f", cache_size_limit={cache_size_limit}"
                        else:
                            auto_cache_limit = compiled_block_count * 2
                            torch._dynamo.config.cache_size_limit = auto_cache_limit
                            cache_info = f", cache_size_limit={auto_cache_limit} (auto)"
                        print_acc(
                            f"Compiled {compiled_block_count} transformer block(s) "
                            f"with torch.compile (mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
                        )
                        print_acc("The first forward pass will be slow during compile. This is normal.")
                        print_acc("If you are experiencing issues, disable block_compile.")
                    else:
                        print_acc(
                            f"No individual transformer blocks found; "
                            f"falling back to whole-model torch.compile "
                            f"(mode='{compile_mode}', fullgraph={compile_fullgraph}, dynamic={compile_dynamic}{cache_info})."
                        )
                        print_acc("The first forward pass will hang for a while. This is normal.")

                        if is_unet_quantized and not is_unet_offloaded and compile_fullgraph:
                            print_acc(
                                "Quantized model detected: fullgraph=True is incompatible "
                                "for whole-model compile, switching to fullgraph=False."
                            )
                            compile_fullgraph = False

                        if compile_mode == 'default':
                            self.sd.unet = torch.compile(
                                self.sd.unet,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )
                        else:
                            self.sd.unet = torch.compile(
                                self.sd.unet,
                                mode=compile_mode,
                                dynamic=compile_dynamic,
                                fullgraph=compile_fullgraph,
                            )

                # ====================================================
                # WHOLE MODEL COMPILE
                # ====================================================
                else:
                    if immutable_compile_owner:
                        print_acc(
                            "WARNING: whole-model compile around the arena dispatcher "
                            "is allowed but has unvalidated performance; dispatcher "
                            "boundaries remain outside compiled arena policy code."
                        )
                    print_acc("Compiling model with torch.compile (whole-model compile).")
                    print_acc("The first forward pass will hang for a while. This is normal.")

                    print_acc(
                        f"Using torch.compile settings: "
                        f"mode={compile_mode}, "
                        f"dynamic={compile_dynamic}, "
                        f"fullgraph={compile_fullgraph}{cache_info}"
                    )

                    if compile_fullgraph:
                        print_acc(
                            "fullgraph=True is incompatible with whole-model compile, "
                            "switching to fullgraph=False."
                        )
                        compile_fullgraph = False

                    if compile_mode == 'default':
                        self.sd.unet = torch.compile(
                            self.sd.unet,
                            dynamic=compile_dynamic,
                            fullgraph=compile_fullgraph,
                        )
                    else:
                        self.sd.unet = torch.compile(
                            self.sd.unet,
                            mode=compile_mode,
                            dynamic=compile_dynamic,
                            fullgraph=compile_fullgraph,
                        )

                if not is_unet_offloaded:
                    # once compiled, dynamo guards hold weakrefs to the params;
                    # .to() on quantized params requires swap_tensors, which fails
                    # on tensors with weakrefs. The model stays on device anyway,
                    # so make .to() a no-op.
                    unet_module = self.sd.unet
                    unet_module.to = lambda *args, **kwargs: unet_module

            except Exception as e:
                # undo any block-level compiles that happened before the failure,
                # so "continuing without compilation" is actually true
                if len(compiled_refs) > 0:
                    for block_list, i, original_block in compiled_refs:
                        block_list[i] = original_block

                if 'triton' in str(e).lower():
                    print_acc("WARNING: compile is disabled.")
                    print_acc("Triton is not available or not working on this system.")
                    print_acc("Install a working 'triton' package to use compile.")
                    print_acc("Continuing without compilation.")
                else:
                    print_acc(f"Failed to compile model: {e}")
                    print_acc("Continuing without compilation")
        arena_runtime = get_memory_runtime(self.sd.unet)
        if arena_runtime is not None:
            self._arena_runtime = arena_runtime
            # Two-phase lifecycle: the model prepared the arena (unfinalized)
            # during load_model, BEFORE LoRA. The permanent train/sample programs
            # must be FINALIZED HERE, after the network is applied, so they
            # capture the installed adapter leaves. That is exactly why
            # finalization lives in setup, not load_model.
            arena_runtime.finalize(self.network)
            info = arena_runtime.diagnostics()
            print_acc(
                "Arena offload training enabled: "
                f"{info['blocks']} block(s), "
                f"{info['resident_bytes'] / (1024 ** 3):.2f} GiB resident "
                f"sidecars; plan={info['plan_fingerprint']}; "
                f"depth={info['prefetch_depth']}. "
                "First training step will compile."
            )
            # Before step 1: if another tenant on the GPU is the reason we are
            # streaming rather than resident, say so. Otherwise the symptom is
            # just a mysteriously slow run.
            arena_runtime.report_foreign_vram_once(phase="training")

        if self.has_first_sample_requested and self.step_num <= 1 and not self.train_config.disable_sampling:
            print_acc("Generating first sample from first sample config")
            self.sample(0, is_first=True)

        # sample first
        if self.train_config.skip_first_sample or self.train_config.disable_sampling:
            print_acc("Skipping first sample due to config setting")
        elif self.step_num <= 1 or self.train_config.force_first_sample:
            print_acc("Generating baseline samples before training")
            self.sample(self.step_num)
        
        if self.accelerator.is_local_main_process:
            self.progress_bar = ToolkitProgressBar(
                total=self.train_config.steps,
                desc=self.job.name,
                leave=True,
                initial=self.step_num,
                iterable=range(0, self.train_config.steps),
            )
            self.progress_bar.pause()
        else:
            self.progress_bar = None

        if self.data_loader is not None:
            dataloader = self.data_loader
            dataloader_iterator = iter(dataloader)
        else:
            dataloader = None
            dataloader_iterator = None

        if self.data_loader_reg is not None:
            dataloader_reg = self.data_loader_reg
            dataloader_iterator_reg = iter(dataloader_reg)
        else:
            dataloader_reg = None
            dataloader_iterator_reg = None

        # zero any gradients
        optimizer.zero_grad()

        self.lr_scheduler.step(self.step_num)

        self.sd.set_device_state(self.train_device_state_preset)
        flush()
        # self.step_num = 0

        # print_acc(f"Compiling Model")
        # torch.compile(self.sd.unet, dynamic=True)

        # make sure all params require grad
        self.ensure_params_requires_grad(force=True)


        ###################################################################
        # TRAIN LOOP
        ###################################################################


        start_step_num = self.step_num
        did_first_flush = False
        flush_next = False
        for step in range(start_step_num, self.train_config.steps):
            if self.train_config.do_paramiter_swapping:
                self.optimizer.optimizer.swap_paramiters()
            self.timer.start('train_loop')
            if flush_next:
                flush()
                flush_next = False
            if self.train_config.do_random_cfg:
                self.train_config.do_cfg = True
                self.train_config.cfg_scale = value_map(random.random(), 0, 1, 1.0, self.train_config.max_cfg_scale)
            self.step_num = step
            # default to true so various things can turn it off
            self.is_grad_accumulation_step = True
            if self.train_config.free_u:
                self.sd.pipeline.enable_freeu(s1=0.9, s2=0.2, b1=1.1, b2=1.2)
            if self.progress_bar is not None:
                self.progress_bar.unpause()
            with torch.no_grad():
                # if is even step and we have a reg dataset, use that
                # todo improve this logic to send one of each through if we can buckets and batch size might be an issue
                is_reg_step = False
                is_save_step = self.save_config.save_every and self.step_num % self.save_config.save_every == 0
                is_sample_step = self.sample_config.sample_every and self.step_num % self.sample_config.sample_every == 0
                if self.train_config.disable_sampling:
                    is_sample_step = False

                batch_list = []

                for b in range(self.train_config.gradient_accumulation):
                    # keep track to alternate on an accumulation step for reg   
                    batch_step = step
                    # don't do a reg step on sample or save steps as we dont want to normalize on those
                    if batch_step % 2 == 0 and dataloader_reg is not None and not is_save_step and not is_sample_step:
                        try:
                            with self.timer('get_batch:reg'):
                                batch = next(dataloader_iterator_reg)
                        except StopIteration:
                            with self.timer('reset_batch:reg'):
                                # hit the end of an epoch, reset
                                if self.progress_bar is not None:
                                    self.progress_bar.pause()
                                dataloader_iterator_reg = iter(dataloader_reg)
                                trigger_dataloader_setup_epoch(dataloader_reg)

                            with self.timer('get_batch:reg'):
                                batch = next(dataloader_iterator_reg)
                            if self.progress_bar is not None:
                                self.progress_bar.unpause()
                        is_reg_step = True
                    elif dataloader is not None:
                        try:
                            with self.timer('get_batch'):
                                batch = next(dataloader_iterator)
                        except StopIteration:
                            with self.timer('reset_batch'):
                                # hit the end of an epoch, reset
                                if self.progress_bar is not None:
                                    self.progress_bar.pause()
                                dataloader_iterator = iter(dataloader)
                                trigger_dataloader_setup_epoch(dataloader)
                                self.epoch_num += 1
                                if self.train_config.gradient_accumulation_steps == -1:
                                    # if we are accumulating for an entire epoch, trigger a step
                                    self.is_grad_accumulation_step = False
                                    self.grad_accumulation_step = 0
                            with self.timer('get_batch'):
                                batch = next(dataloader_iterator)
                            if self.progress_bar is not None:
                                self.progress_bar.unpause()
                    else:
                        batch = None
                    batch_list.append(batch)
                    batch_step += 1

                # setup accumulation
                if self.train_config.gradient_accumulation_steps == -1:
                    # epoch is handling the accumulation, dont touch it
                    pass
                else:
                    # determine if we are accumulating or not
                    # since optimizer step happens in the loop, we trigger it a step early
                    # since we cannot reprocess it before them
                    optimizer_step_at = self.train_config.gradient_accumulation_steps
                    is_optimizer_step = self.grad_accumulation_step >= optimizer_step_at
                    self.is_grad_accumulation_step = not is_optimizer_step
                    if is_optimizer_step:
                        self.grad_accumulation_step = 0

            # flush()
            ### HOOK ###
            if self.torch_profiler is not None:
                self.torch_profiler.start()
            did_oom = False
            loss_dict = None
            offload_shape_key = MemoryManager.offload_shape_key_from_batch(
                batch_list,
                dop_enabled=bool(getattr(self.train_config, 'diff_output_preservation', False)),
                dop_resolution=getattr(
                    self.train_config, 'diff_output_preservation_resolution', None
                ),
                dop_single_backward=bool(
                    getattr(self.train_config, 'dop_single_backward', False)
                ),
                dop_prior_cache=bool(
                    getattr(self.train_config, 'dop_prior_cache', False)
                ),
                blank_preservation_enabled=bool(
                    getattr(self.train_config, 'blank_prompt_preservation', False)
                ),
                blank_preservation_resolution=getattr(
                    self.train_config, 'blank_prompt_preservation_resolution', None
                ),
                checkpoint_policy_id=(
                    getattr(self, '_checkpoint_tunable', None)._checkpoint_keep_last
                    if getattr(self, '_checkpoint_tunable', None) is not None
                    else getattr(self.model_config, 'layer_offloading_checkpoint_keep_last', 0)
                ),
                fp8_forward_enabled=bool(
                    getattr(getattr(self.sd, 'unet', None), '_memory_manager', None)
                    and getattr(
                        getattr(self.sd.unet, '_memory_manager', None),
                        '_fp8_training_layers',
                        0,
                    )
                ),
            )
            arena_runtime = get_memory_runtime(self.sd.unet)
            if arena_runtime is None:
                try:
                    MemoryManager.prepare_training_memory_for_shape(
                        getattr(self.sd, 'unet', None),
                        self.device_torch,
                        shape_key=offload_shape_key,
                    )
                except Exception as error:
                    print_acc(
                        f"[MemoryManager] manual pre-step guard failed: {error}"
                    )
            driver_free_monitor = None
            driver_free_sample = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats(self.device_torch)
                except Exception:
                    pass
                driver_free_monitor = _CudaDriverFreeMonitor(self.device_torch).start()
            step_started_at = time.perf_counter()
            MemoryManager.offload_step_begin(shape_key=offload_shape_key)
            offload_step_completed = False
            try:
                # One generic context around the whole per-batch forward AND
                # backward region. Backward must be inside: checkpoint
                # recomputation re-enters the block runtime. This is also the
                # residency controller's phase boundary (git-bug 0c577ef).
                execution_context = (
                    arena_runtime.training_step(
                        shape_key=offload_shape_key,
                        step_num=self.step_num,
                    )
                    if arena_runtime is not None
                    else contextlib.nullcontext()
                )
                with execution_context:
                    with self.accelerator.accumulate(self.modules_being_trained):
                        loss_dict = self.hook_train_loop(batch_list)
                offload_step_completed = True
            except torch.cuda.OutOfMemoryError:
                did_oom = True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    did_oom = True
                else:
                    raise  # not an OOM; surface real errors
            finally:
                if offload_step_completed:
                    MemoryManager.offload_step_end()
                else:
                    MemoryManager.offload_step_abort()
                if driver_free_monitor is not None:
                    driver_free_sample = driver_free_monitor.stop()
                self._last_driver_free_sample = driver_free_sample
            if (
                offload_step_completed
                and arena_runtime is not None
                and driver_free_sample is not None
            ):
                arena_runtime.record_training_physical_free_min(
                    driver_free_sample.get('min_free_bytes')
                )
            if did_oom:
                if arena_runtime is not None:
                    failure_event = (
                        arena_runtime.diagnostics().get('last_failure_event')
                    )
                    if failure_event is not None:
                        os.makedirs(
                            os.path.dirname(self.performance_log_path),
                            exist_ok=True,
                        )
                        with open(
                            self.performance_log_path, 'a', encoding='utf-8'
                        ) as handle:
                            handle.write(
                                json.dumps(failure_event, separators=(',', ':'))
                                + '\n'
                            )
                        print_acc(
                            "[ArenaOffload] allocation failure: "
                            f"classification={failure_event.get('classification')} "
                            f"exception={failure_event.get('exception_type')}: "
                            f"{failure_event.get('exception')} "
                            f"rollback={failure_event.get('rollback_block')} "
                            f"abandoned_fetches="
                            f"{failure_event.get('abandoned_fetch_tickets')}"
                        )
                # Legacy offload treats its allocator cap as an OOM relief
                # lever. Arena offload binds the cap only at phase boundaries,
                # so an arena OOM never widens it from this per-step path.
                cap_relieved = (
                    arena_runtime is None
                    and MemoryManager.relieve_wddm_cap_after_oom(
                        self.device_torch,
                        context=f"training step {self.step_num}",
                    )
                )
                if not cap_relieved:
                    self.num_consecutive_oom += 1
                    if self.num_consecutive_oom > 3:
                        raise RuntimeError("OOM during training step 3 times in a row, aborting training")
                optimizer.zero_grad(set_to_none=True)
                flush()
                torch.cuda.ipc_collect()
                if torch.cuda.is_available():
                    dev = self.device_torch
                    gib = 1024 ** 3
                    allocated = torch.cuda.memory_allocated(dev) / gib
                    reserved = torch.cuda.memory_reserved(dev) / gib
                    peak_allocated = torch.cuda.max_memory_allocated(dev) / gib
                    peak_reserved = torch.cuda.max_memory_reserved(dev) / gib
                    active_cap_bytes = allocator_cap.applied_cap_bytes(dev)
                    active_cap_gib = (
                        None
                        if active_cap_bytes is None
                        else active_cap_bytes / gib
                    )
                    driver_free_bytes = (driver_free_sample or {}).get(
                        'min_free_bytes'
                    )
                    driver_free_gib = (
                        None
                        if driver_free_bytes is None
                        else float(driver_free_bytes) / gib
                    )
                    arena_cap_target = None
                    if arena_runtime is not None:
                        try:
                            arena_cap_target = arena_runtime.diagnostics().get(
                                'training_cap_target_bytes'
                            )
                        except Exception:
                            arena_cap_target = None
                    cap_mode = (
                        'arena_target'
                        if arena_cap_target is not None
                        else ('cliff_bound' if active_cap_bytes is not None else 'none')
                    )
                    print_acc(
                        "[MemoryManager] OOM snapshot before ring reset: "
                        f"allocated={allocated:.2f} GiB "
                        f"reserved={reserved:.2f} GiB "
                        f"peak_allocated={peak_allocated:.2f} GiB "
                        f"peak_reserved={peak_reserved:.2f} GiB "
                        f"allocator_cap={active_cap_gib if active_cap_gib is not None else '-'} GiB "
                        f"cap_mode={cap_mode} "
                        f"driver_free={driver_free_gib if driver_free_gib is not None else '-'} GiB "
                        f"policy_owner={'arena' if arena_runtime is not None else 'legacy'}"
                    )
                    MemoryManager.recover_cuda_pipeline_after_oom()
                    # True within-step peak across accumulations (the per-accumulation
                    # sampler resets the live counter, so max it with the step-level
                    # high-water the trainer carried forward).
                    peak_alloc_override = max(
                        getattr(self, '_step_peak_allocated_bytes', 0),
                        int(torch.cuda.max_memory_allocated(dev)),
                    )
                    peak_reserved_override = max(
                        getattr(self, '_step_peak_reserved_bytes', 0),
                        int(torch.cuda.max_memory_reserved(dev)),
                    )
                    if arena_runtime is None:
                        try:
                            MemoryManager.auto_tune_training_memory(
                                getattr(self.sd, 'unet', None),
                                self.device_torch,
                                shape_key=offload_shape_key,
                                step_num=self.step_num,
                                step_time_s=time.perf_counter() - step_started_at,
                                did_oom=True,
                                peak_allocated_override=peak_alloc_override,
                                peak_reserved_override=peak_reserved_override,
                                observed_driver_free_min_bytes=(driver_free_sample or {}).get('min_free_bytes'),
                                observed_driver_total_bytes=(driver_free_sample or {}).get('total_bytes'),
                                observed_driver_free_samples=(driver_free_sample or {}).get('samples'),
                            )
                        except Exception as error:
                            print_acc(
                                f"[MemoryManager] training autotune after OOM failed: {error}"
                            )
                    torch.cuda.reset_peak_memory_stats(dev)
                # skip this step and keep going
                print_acc("")
                print_acc("################################################")
                print_acc(f"# OOM during training step, skipping batch {self.num_consecutive_oom}/3 #")
                print_acc("################################################")
                print_acc("")
            else:
                self.num_consecutive_oom = 0
                # True within-step peak across accumulations (the per-accumulation
                # sampler resets the live counter, so max it with the step-level
                # high-water the trainer carried forward).
                peak_alloc_override = None
                peak_reserved_override = None
                if torch.cuda.is_available():
                    _dev = self.device_torch
                    peak_alloc_override = max(
                        getattr(self, '_step_peak_allocated_bytes', 0),
                        int(torch.cuda.max_memory_allocated(_dev)),
                    )
                    peak_reserved_override = max(
                        getattr(self, '_step_peak_reserved_bytes', 0),
                        int(torch.cuda.max_memory_reserved(_dev)),
                    )
                if arena_runtime is None:
                    try:
                        MemoryManager.auto_tune_training_memory(
                            getattr(self.sd, 'unet', None),
                            self.device_torch,
                            shape_key=offload_shape_key,
                            step_num=self.step_num,
                            step_time_s=time.perf_counter() - step_started_at,
                            did_oom=False,
                            peak_allocated_override=peak_alloc_override,
                            peak_reserved_override=peak_reserved_override,
                            observed_driver_free_min_bytes=(driver_free_sample or {}).get('min_free_bytes'),
                            observed_driver_total_bytes=(driver_free_sample or {}).get('total_bytes'),
                            observed_driver_free_samples=(driver_free_sample or {}).get('samples'),
                        )
                    except Exception as error:
                        print_acc(
                            f"[MemoryManager] training autotune failed: {error}"
                        )
            if self.torch_profiler is not None:
                torch.cuda.synchronize()  # Make sure all CUDA ops are done
                self.torch_profiler.stop()
                
                print("\n==== Profile Results ====")
                print(self.torch_profiler.key_averages().table(sort_by="cpu_time_total", row_limit=1000))
            self.timer.stop('train_loop')
            if not did_first_flush and not did_oom:
                flush()
                did_first_flush = True
                try:
                    first_offload_profile = MemoryManager.offload_profile_report(
                        reset=False
                    )
                except Exception as error:
                    first_offload_profile = (
                        f"[OffloadProfile] first-step report failed: {error}"
                    )
                if first_offload_profile:
                    print_acc(first_offload_profile)
                # one-time CUDA memory report after the first real training step, to see
                # what is actually resident (allocated) vs what the caching allocator has
                # reserved/committed (the gap is cached/fragmented, not leaked tensors).
                # Gated to keep normal logs clean: set AITK_CUDA_MEM_REPORT=1 (or
                # DEBUG_TOOLKIT=1) to enable.
                _mem_report = (
                    os.environ.get("AITK_CUDA_MEM_REPORT", "0") == "1"
                    or os.environ.get("DEBUG_TOOLKIT", "0") == "1"
                )
                if _mem_report and torch.cuda.is_available():
                    dev = self.device_torch
                    gb = 1024 ** 3
                    alloc = torch.cuda.memory_allocated(dev) / gb
                    reserved = torch.cuda.memory_reserved(dev) / gb
                    max_alloc = torch.cuda.max_memory_allocated(dev) / gb
                    max_reserved = torch.cuda.max_memory_reserved(dev) / gb
                    free_b, total_b = vram_budget.device_mem_info(dev)
                    used_total = (total_b - free_b) / gb  # process + everything else on the device
                    print_acc("")
                    print_acc("================ CUDA memory @ first train step ================")
                    print_acc(f"  allocated (live tensors)     : {alloc:6.2f} GB")
                    print_acc(f"  reserved  (allocator pool)   : {reserved:6.2f} GB   (cached/frag: {reserved - alloc:5.2f} GB)")
                    print_acc(f"  peak allocated               : {max_alloc:6.2f} GB")
                    print_acc(f"  peak reserved                : {max_reserved:6.2f} GB")
                    print_acc(f"  device in use (nvidia-smi-ish): {used_total:6.2f} GB")
                    print_acc("================================================================")
                    print_acc("")
            # flush()
            # setup the networks to gradient checkpointing and everything works
            if self.adapter is not None and isinstance(self.adapter, ReferenceAdapter):
                self.adapter.clear_memory()

            with torch.no_grad():
                # torch.cuda.empty_cache()
                # if optimizer has get_lrs method, then use it
                learning_rate = 0.0
                if not did_oom and loss_dict is not None:
                    if hasattr(optimizer, 'get_avg_learning_rate'):
                        learning_rate = optimizer.get_avg_learning_rate()
                    elif hasattr(optimizer, 'get_learning_rates'):
                        learning_rate = optimizer.get_learning_rates()[0]
                    elif self.train_config.optimizer.lower().startswith('dadaptation') or \
                            self.train_config.optimizer.lower().startswith('prodigy'):
                        learning_rate = (
                                optimizer.param_groups[0]["d"] *
                                optimizer.param_groups[0]["lr"]
                        )
                    else:
                        learning_rate = optimizer.param_groups[0]['lr']

                    loss_dict = {
                        k: (v.item() if hasattr(v, 'item') else v)
                        for k, v in loss_dict.items()
                    }
                    prog_bar_string = f"lr: {learning_rate:.1e}"
                    for key, value in loss_dict.items():
                        prog_bar_string += f" {key}: {value:.3e}"

                    if self.progress_bar is not None:
                        self.progress_bar.set_postfix_str(prog_bar_string)

                # if the batch is a DataLoaderBatchDTO, then we need to clean it up
                if isinstance(batch, DataLoaderBatchDTO):
                    with self.timer('batch_cleanup'):
                        batch.cleanup()

                # don't do on first step
                if self.step_num != self.start_step:
                    if is_sample_step or is_save_step:
                        self.accelerator.wait_for_everyone()
                        
                    if is_save_step:
                        self.accelerator
                        # print above the progress bar
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        print_acc(f"\nSaving at step {self.step_num}")
                        self.save(self.step_num)
                        self.ensure_params_requires_grad()
                        # clear any grads
                        optimizer.zero_grad()
                        flush()
                        flush_next = True
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()
                            
                    recovery_every = getattr(self.save_config, 'recovery_every', 0)
                    if recovery_every and not is_save_step and self.step_num % recovery_every == 0:
                        self.save_recovery_snapshot()

                    if is_sample_step:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        flush()
                        # print above the progress bar
                        if self.train_config.free_u:
                            self.sd.pipeline.disable_freeu()
                        self.sample(self.step_num)
                        if self.train_config.unload_text_encoder:
                            # make sure the text encoder is unloaded
                            self.sd.text_encoder_to('cpu')
                        flush()

                        self.ensure_params_requires_grad()
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()

                    if self.logging_config.log_every and self.step_num % self.logging_config.log_every == 0:
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        with self.timer('log_to_tensorboard'):
                            # log to tensorboard
                            if self.accelerator.is_main_process:
                                if self.writer is not None:
                                    if loss_dict is not None:
                                        for key, value in loss_dict.items():
                                            self.writer.add_scalar(f"{key}", value, self.step_num)
                                        self.writer.add_scalar(f"lr", learning_rate, self.step_num)
                                if self.progress_bar is not None:
                                    self.progress_bar.unpause()
                        
                        if self.accelerator.is_main_process:
                            # log to logger
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            if loss_dict is not None:
                                for key, value in loss_dict.items():
                                    self.logger.log({
                                        f'loss/{key}': value,
                                    })
                            if self.additional_logs is not None:
                                for key, value in self.additional_logs.items():
                                    self.logger.log({
                                        key: value,
                                    })
                                self.additional_logs = {}
                    elif self.logging_config.log_every is None:
                        if self.accelerator.is_main_process:
                            # log every step
                            self.logger.log({
                                'learning_rate': learning_rate,
                            })
                            for key, value in loss_dict.items():
                                self.logger.log({
                                    f'loss/{key}': value,
                                })
                            if self.additional_logs is not None:
                                for key, value in self.additional_logs.items():
                                    self.logger.log({
                                        key: value,
                                    })
                                self.additional_logs = {}


                    if (
                        not did_oom
                        and self.performance_log_every > 0
                        and self.step_num % self.performance_log_every == 0
                    ):
                        if self.progress_bar is not None:
                            self.progress_bar.pause()
                        self.resolve_performance_timers()
                        # print the timers and clear them
                        self.timer.print()
                        self.timer.reset()
                        if self.progress_bar is not None:
                            self.progress_bar.unpause()
                
                # commit log
                if self.accelerator.is_main_process:
                    with self.timer('commit_logger'):
                        self.logger.commit(step=self.step_num)

                # sets progress bar to match out step
                if self.progress_bar is not None:
                    self.progress_bar.update(step - self.progress_bar.n)

                #############################
                # End of step
                #############################

                # update various steps
                self.step_num = step + 1
                self.grad_accumulation_step += 1
                self.end_step_hook()


        ###################################################################
        ##  END TRAIN LOOP
        ###################################################################
        self.accelerator.wait_for_everyone()
        if self.progress_bar is not None:
            self.progress_bar.close()
        if self.train_config.free_u:
            self.sd.pipeline.disable_freeu()
        if self.accelerator.is_main_process:
            self.save()
            # Block until every deferred checkpoint write has hit disk before we
            # tear down -- a daemon writer thread would otherwise be killed with
            # the final save still in flight.
            if self._async_saver is not None:
                self._async_saver.wait_idle()
                self._async_saver.close()
                self._async_saver = None
        if not self.train_config.disable_sampling:
            self.sample(self.step_num)
            self.logger.commit(step=self.step_num)
        print_acc("")
        if self.accelerator.is_main_process:
            self.logger.finish()
        self.accelerator.end_training()

        if self.accelerator.is_main_process:
            # push to hub
            if self.save_config.push_to_hub:
                if("HF_TOKEN" not in os.environ):
                    interpreter_login(new_session=False, write_permission=True)
                self.push_to_hub(
                    repo_id=self.save_config.hf_repo_id,
                    private=self.save_config.hf_private
                )
        del (
            self.sd,
            unet,
            noise_scheduler,
            optimizer,
            self.network,
            tokenizer,
            text_encoder,
        )

        flush()
        self.done_hook()

    def push_to_hub(
    self,
    repo_id: str,
    private: bool = False,
    ):  
        if not self.accelerator.is_main_process:
            return
        readme_content = self._generate_readme(repo_id)
        readme_path = os.path.join(self.save_root, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        api = HfApi()

        api.create_repo(
            repo_id,
            private=private,
            exist_ok=True
        )

        api.upload_folder(
            repo_id=repo_id,
            folder_path=self.save_root,
            ignore_patterns=["*.yaml", "*.pt"],
            repo_type="model",
        )


    def _generate_readme(self, repo_id: str) -> str:
        """Generates the content of the README.md file."""

        # Gather model info
        base_model = self.model_config.name_or_path
        instance_prompt = self.trigger_word if hasattr(self, "trigger_word") else None
        if base_model == "black-forest-labs/FLUX.1-schnell":
            license = "apache-2.0"
        elif base_model == "black-forest-labs/FLUX.1-dev":
            license = "other"
            license_name = "flux-1-dev-non-commercial-license"
            license_link = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"
        else:
            license = "creativeml-openrail-m"
        tags = [
            "text-to-image",
        ]
        if self.model_config.is_xl:
            tags.append("stable-diffusion-xl")
        if self.model_config.is_flux:
            tags.append("flux")
        if self.model_config.is_lumina2:
            tags.append("lumina2")
        if self.model_config.is_v3:
            tags.append("sd3")
        if self.network_config:
            tags.extend(
                [
                    "lora",
                    "diffusers",
                    "template:sd-lora",
                    "ai-toolkit",
                ]
            )

        # Generate the widget section
        widgets = []
        sample_image_paths = []
        samples_dir = os.path.join(self.save_root, "samples")
        if os.path.isdir(samples_dir):
            for filename in os.listdir(samples_dir):
                #The filenames are structured as 1724085406830__00000500_0.jpg
                #So here we capture the 2nd part (steps) and 3rd (index the matches the prompt)
                match = re.search(r"__(\d+)_(\d+)\.jpg$", filename)
                if match:
                    steps, index = int(match.group(1)), int(match.group(2))
                    #Here we only care about uploading the latest samples, the match with the # of steps
                    if steps == self.train_config.steps:
                        sample_image_paths.append((index, f"samples/{filename}"))

            # Sort by numeric index
            sample_image_paths.sort(key=lambda x: x[0])

            # Create widgets matching prompt with the index 
            for i, prompt in enumerate(self.sample_config.prompts):
                if i < len(sample_image_paths):
                    # Associate prompts with sample image paths based on the extracted index
                    _, image_path = sample_image_paths[i]
                    widgets.append(
                        {
                            "text": prompt,
                            "output": {
                                "url": image_path
                            },
                        }
                    )
        dtype = "torch.bfloat16" if self.model_config.is_flux else "torch.float16"
        # Construct the README content
        readme_content = f"""---
tags:
{yaml.dump(tags, indent=4).strip()}
{"widget:" if os.path.isdir(samples_dir) else ""}
{yaml.dump(widgets, indent=4).strip() if widgets else ""}
base_model: {base_model}
{"instance_prompt: " + instance_prompt if instance_prompt else ""}
license: {license}
{'license_name: ' + license_name if license == "other" else ""}
{'license_link: ' + license_link if license == "other" else ""}
---

# {self.job.name}
Model trained with [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit)
<Gallery />

## Trigger words

{"You should use `" + instance_prompt + "` to trigger the image generation." if instance_prompt else "No trigger words defined."}

## Download model and use it with ComfyUI, AUTOMATIC1111, SD.Next, Invoke AI, etc.

Weights for this model are available in Safetensors format.

[Download](/{repo_id}/tree/main) them in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained('{base_model}', torch_dtype={dtype}).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='{self.job.name}.safetensors')
image = pipeline('{instance_prompt if not widgets else self.sample_config.prompts[0]}').images[0]
image.save("my_image.png")
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

"""
        return readme_content
