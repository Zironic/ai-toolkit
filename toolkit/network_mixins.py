import json
import os
from collections import OrderedDict
from typing import Optional, Union, List, Type, TYPE_CHECKING, Dict, Any, Literal

import torch
from optimum.quanto import QTensor
from torch import nn
import weakref

from tqdm import tqdm

from toolkit.config_modules import NetworkConfig
from toolkit.lorm import extract_conv, extract_linear, count_parameters
from toolkit.metadata import add_model_hash_to_meta
from toolkit.paths import KEYMAPS_ROOT
from toolkit.saving import get_lora_keymap_from_model_keymap
from optimum.quanto import QBytesTensor

if TYPE_CHECKING:
    from toolkit.lycoris_special import LycorisSpecialNetwork, LoConSpecialModule
    from toolkit.lora_special import LoRASpecialNetwork, LoRAModule
    from toolkit.stable_diffusion_model import StableDiffusion
    from toolkit.models.DoRA import DoRAModule

Network = Union['LycorisSpecialNetwork', 'LoRASpecialNetwork']
Module = Union['LoConSpecialModule', 'LoRAModule', 'DoRAModule']

LINEAR_MODULES = [
    'Linear',
    'LoRACompatibleLinear',
    'QLinear',
    'OstrisLinear',
    # 'GroupNorm',
]
CONV_MODULES = [
    'Conv2d',
    'LoRACompatibleConv'
]

ExtractMode = Union[
    'existing'
    'fixed',
    'threshold',
    'ratio',
    'quantile',
    'percentage'
]

printed_messages = []


def print_once(msg):
    global printed_messages
    if msg not in printed_messages:
        print(msg)
        printed_messages.append(msg)


def _assistant_inverse_module_scale(module, default=1.0):
    network = module.network_ref()
    if not getattr(network, 'is_assistant_adapter', False):
        return default
    return getattr(module, 'assistant_inverse_scale', default)


def _get_network_base_model(network):
    base_ref = getattr(network, 'base_model_ref', None)
    return base_ref() if base_ref is not None else None


def _dequantized_probe_weight(module):
    if hasattr(module, 'weight'):
        weight = module.weight
        if hasattr(weight, 'dequantize'):
            return weight.dequantize().float()
        return weight.float()

    state = module.state_dict()
    if 'weight._data' in state and 'weight._scale' in state:
        return (state['weight._data'].float() * state['weight._scale'].float()).float()

    raise RuntimeError(f"Cannot dequantize probe module {module.__class__.__name__}")


def _quantized_probe_weight(fp_weight, orig_dtype, qtype, bias=False):
    from toolkit.util.quantize import get_qtype, quantize

    probe = torch.nn.Sequential(
        torch.nn.Linear(
            fp_weight.shape[1],
            fp_weight.shape[0],
            bias=bias,
            device=fp_weight.device,
            dtype=orig_dtype,
        )
    )
    probe[0].weight = torch.nn.Parameter(
        fp_weight.detach().clone().to(orig_dtype), requires_grad=False
    )
    quantize(probe, weights=get_qtype(qtype))
    return _dequantized_probe_weight(probe[0])


def _calibrate_assistant_inverse_scale(module, base_weight, merged_weight, orig_dtype):
    network = module.network_ref()
    if not getattr(network, 'is_assistant_adapter', False):
        return 1.0

    base_model = _get_network_base_model(network)
    model_config = getattr(base_model, 'model_config', None)
    qtype = getattr(model_config, 'qtype', None)
    if model_config is None or not getattr(model_config, 'quantize', False) or qtype is None:
        return 1.0

    org_module = module.org_module[0]
    if org_module.__class__.__name__ not in LINEAR_MODULES:
        return 1.0

    delta_original = (merged_weight.float() - base_weight.float()).detach()
    denom = torch.dot(delta_original.flatten(), delta_original.flatten())
    if denom <= 0:
        return 1.0

    try:
        has_bias = getattr(org_module, 'bias', None) is not None
        q_base = _quantized_probe_weight(base_weight, orig_dtype, qtype, bias=has_bias)
        q_merged = _quantized_probe_weight(merged_weight, orig_dtype, qtype, bias=has_bias)
        delta_effective = q_merged.to(delta_original.device) - q_base.to(delta_original.device)
        scale = torch.dot(delta_original.flatten(), delta_effective.flatten()) / denom
        scale = torch.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
        scale = torch.clamp(scale, min=0.0, max=2.0)
        return float(scale.item())
    except Exception as e:
        print_once(
            f"Warning: assistant LoRA inverse calibration failed for "
            f"{getattr(module, 'lora_name', '?')}: {e}"
        )
        return 1.0


def broadcast_and_multiply(tensor, multiplier):
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - multiplier.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        multiplier = multiplier.unsqueeze(-1)

    try:
        # Multiplying the broadcasted tensor with the output tensor
        result = tensor * multiplier
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(multiplier.size())
        raise e

    return result


def add_bias(tensor, bias):
    if bias is None:
        return tensor
    # add batch dim
    bias = bias.unsqueeze(0)
    bias = torch.cat([bias] * tensor.size(0), dim=0)
    # Determine the number of dimensions required
    num_extra_dims = tensor.dim() - bias.dim()

    # Unsqueezing the tensor to match the dimensionality
    for _ in range(num_extra_dims):
        bias = bias.unsqueeze(-1)

    # we may need to swap -1 for -2
    if bias.size(1) != tensor.size(1):
        if len(bias.size()) == 3:
            bias = bias.permute(0, 2, 1)
        elif len(bias.size()) == 4:
            bias = bias.permute(0, 3, 1, 2)

    # Multiplying the broadcasted tensor with the output tensor
    try:
        result = tensor + bias
    except RuntimeError as e:
        print(e)
        print(tensor.size())
        print(bias.size())
        raise e

    return result


def apply_rank_gates(tensor: torch.Tensor, gates: torch.Tensor, module_name: str = ""):
    gates = gates.to(device=tensor.device, dtype=tensor.dtype)
    if tensor.shape[-1] == gates.numel():
        return tensor * gates.view(*([1] * (tensor.ndim - 1)), -1)

    if tensor.ndim == 4 and tensor.shape[1] == gates.numel():
        return tensor * gates.view(1, -1, 1, 1)

    label = f" for {module_name}" if module_name else ""
    raise ValueError(f"Cannot apply rank gates{label}: gates={tuple(gates.shape)} tensor={tuple(tensor.shape)}")


def _get_rank_gates_for_module(network: Network, module_name: str):
    vector_gates = getattr(network, "vector_gates", None)
    if vector_gates is None:
        return None

    if isinstance(vector_gates, torch.Tensor):
        return vector_gates

    if isinstance(vector_gates, dict):
        per_module = vector_gates.get("per_module", {})
        if module_name in per_module:
            return per_module[module_name]
        for pattern, gates in per_module.items():
            if pattern in module_name:
                return gates
        return vector_gates.get("global", None)

    return vector_gates

class ExtractableModuleMixin:
    def extract_weight(
            self: Module,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        device = self.lora_down.weight.device
        weight_to_extract = self.org_module[0].weight
        if extract_mode == "existing":
            extract_mode = 'fixed'
            extract_mode_param = self.lora_dim
            
        if isinstance(weight_to_extract, QBytesTensor):
            weight_to_extract = weight_to_extract.dequantize()
        
        weight_to_extract = weight_to_extract.clone().detach().float()

        if self.org_module[0].__class__.__name__ in CONV_MODULES:
            # do conv extraction
            down_weight, up_weight, new_dim, diff = extract_conv(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device
            )

        elif self.org_module[0].__class__.__name__ in LINEAR_MODULES:
            # do linear extraction
            down_weight, up_weight, new_dim, diff = extract_linear(
                weight=weight_to_extract,
                mode=extract_mode,
                mode_param=extract_mode_param,
                device=device,
            )
        else:
            raise ValueError(f"Unknown module type: {self.org_module[0].__class__.__name__}")

        self.lora_dim = new_dim

        # inject weights into the param
        self.lora_down.weight.data = down_weight.to(self.lora_down.weight.dtype).clone().detach()
        self.lora_up.weight.data = up_weight.to(self.lora_up.weight.dtype).clone().detach()

        # copy bias if we have one and are using them
        if self.org_module[0].bias is not None and self.lora_up.bias is not None:
            self.lora_up.bias.data = self.org_module[0].bias.data.clone().detach()

        # set up alphas
        self.alpha = (self.alpha * 0) + down_weight.shape[0]
        self.scale = self.alpha / self.lora_dim

        # assign them

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            # scaler is a parameter update the value with 1.0
            self.scalar.data = torch.tensor(1.0).to(self.scalar.device, self.scalar.dtype)


class ToolkitModuleMixin:
    def __init__(
            self: Module,
            *args,
            network: Network,
            **kwargs
    ):
        self.network_ref: weakref.ref = weakref.ref(network)
        self.is_checkpointing = False
        self._multiplier: Union[float, list, torch.Tensor] = None

    def _call_forward(self: Module, x):
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return 0.0  # added to original forward

        if hasattr(self, 'lora_mid') and self.lora_mid is not None:
            lx = self.lora_mid(self.lora_down(x))
        else:
            try:
                lx = self.lora_down(x)
            except RuntimeError as e:
                print(f"Error in {self.__class__.__name__} lora_down")
                raise e

        if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
            lx = self.dropout(lx)
        # normal dropout
        elif self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        rank_gates = _get_rank_gates_for_module(self.network_ref(), self.lora_name)
        if rank_gates is not None:
            lx = apply_rank_gates(lx, rank_gates, self.lora_name)

        # rank dropout
        if self.rank_dropout is not None and self.rank_dropout > 0 and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        return lx * scale

    def lorm_forward(self: Network, x, *args, **kwargs):
        network: Network = self.network_ref()
        if not network.is_active:
            return self.org_forward(x, *args, **kwargs)
        
        orig_dtype = x.dtype
        
        if x.dtype != self.lora_down.weight.dtype:
            x = x.to(self.lora_down.weight.dtype)

        if network.lorm_train_mode == 'local':
            # we are going to predict input with both and do a loss on them
            inputs = x.detach()
            with torch.no_grad():
                # get the local prediction
                target_pred = self.org_forward(inputs, *args, **kwargs).detach()
            with torch.set_grad_enabled(True):
                # make a prediction with the lorm
                lorm_pred = self.lora_up(self.lora_down(inputs.requires_grad_(True)))

                local_loss = torch.nn.functional.mse_loss(target_pred.float(), lorm_pred.float())
                # backpropr
                local_loss.backward()

            network.module_losses.append(local_loss.detach())
            # return the original as we dont want our trainer to affect ones down the line
            return target_pred

        else:
            x = self.lora_up(self.lora_down(x))
            if x.dtype != orig_dtype:
                x = x.to(orig_dtype)

    def _memory_management_compile_fast_lora_ready(self: Module) -> bool:
        network: Network = self.network_ref()
        multiplier = getattr(network, "torch_multiplier", None)
        return bool(
            not getattr(network, "is_lorm", False)
            and getattr(network, "is_active", False)
            and not getattr(network, "is_merged_in", False)
            and getattr(network, "_multiplier", None) != 0
            and multiplier is not None
            and getattr(multiplier, "numel", lambda: 0)() == 1
            and self.__class__.__name__ not in ("DoRAModule", "LokrModule")
            and getattr(self, "module_dropout", None) is None
            and getattr(self, "rank_dropout", None) in (None, 0)
            and getattr(network, "vector_gates", None) is None
            and _assistant_inverse_module_scale(self) == 1.0
            and (getattr(self, "dropout", None) is None or isinstance(getattr(self, "dropout", None), nn.Identity))
        )

    def _memory_management_compile_fast_lora_forward(self: Module, x, *args, inner=None, **kwargs):
        base_forward = self.org_forward if inner is None else inner
        org_forwarded = base_forward(x, *args, **kwargs)
        lora_input = x.to(self.lora_down.weight.dtype)
        lora_output = self.lora_up(self.lora_down(lora_input)) * self.scale
        multiplier = self.network_ref().torch_multiplier.reshape(())
        return org_forwarded + (lora_output * multiplier).to(org_forwarded.dtype)

    def functional_forward(self: Module, inner, x, *args, **kwargs):
        if getattr(self, "_memory_management_compile_lora_fast", False):
            return self._memory_management_compile_fast_lora_forward(
                x,
                *args,
                inner=inner,
                **kwargs,
            )

        network: Network = self.network_ref()
        if network.is_lorm:
            return self.lorm_forward(x, *args, **kwargs)

        if not network.is_active or network.is_merged_in or network._multiplier == 0:
            return inner(x, *args, **kwargs)

        if self.__class__.__name__ == "LokrModule":
            return self._call_forward(x, inner=inner)

        org_forwarded = inner(x, *args, **kwargs)

        if isinstance(x, QTensor):
            x = x.dequantize()
        lora_input = x.to(self.lora_down.weight.dtype)
        lora_output = self._call_forward(lora_input)
        multiplier = self.network_ref().torch_multiplier

        lora_output_batch_size = lora_output.size(0)
        multiplier_batch_size = multiplier.size(0)
        if lora_output_batch_size != multiplier_batch_size:
            num_interleaves = lora_output_batch_size // multiplier_batch_size
            multiplier = multiplier.repeat_interleave(num_interleaves)

        module_inverse_scale = _assistant_inverse_module_scale(self)
        if module_inverse_scale != 1.0:
            lora_output = lora_output * module_inverse_scale

        scaled_lora_output = broadcast_and_multiply(lora_output, multiplier)
        scaled_lora_output = scaled_lora_output.to(org_forwarded.dtype)

        if self.__class__.__name__ == "DoRAModule":
            if isinstance(self.dropout, nn.Dropout) or isinstance(self.dropout, nn.Identity):
                lx = self.dropout(x)
            elif self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(x, p=self.dropout)
            else:
                lx = x
            lora_weight = self.lora_up.weight @ self.lora_down.weight
            scale = multiplier.mean()
            scaled_lora_weight = lora_weight * scale
            materialize = getattr(inner, "materialized_weight", None)
            base_weight = (
                None
                if materialize is None
                else materialize(dtype=scaled_lora_weight.dtype)
            )
            scaled_lora_output = scaled_lora_output + self.apply_dora(
                lx,
                scaled_lora_weight,
                base_weight=base_weight,
            ).to(org_forwarded.dtype)

        try:
            return org_forwarded + scaled_lora_output
        except RuntimeError as exc:
            print(exc)
            print(org_forwarded.size())
            print(scaled_lora_output.size())
            raise

    def forward(self: Module, x, *args, **kwargs):
        return self.functional_forward(self.org_forward, x, *args, **kwargs)

    def enable_gradient_checkpointing(self: Module):
        self.is_checkpointing = True

    def disable_gradient_checkpointing(self: Module):
        self.is_checkpointing = False

    def _get_base_qtype(self: Module):
        # the qtype string the base model was quantized with (so we can re-quantize after merging), or None
        network = self.network_ref()
        base_ref = getattr(network, 'base_model_ref', None)
        base = base_ref() if base_ref is not None else None
        return getattr(getattr(base, 'model_config', None), 'qtype', None)

    @torch.no_grad()
    def merge_out(self: Module, merge_out_weight=1.0):
        # make sure it is positive
        merge_out_weight = abs(merge_out_weight)
        # merging out is just merging in the negative of the weight
        self.merge_in(merge_weight=-merge_out_weight)

    @torch.no_grad()
    def merge_in(self: Module, merge_weight=1.0):
        if not self.can_merge_in:
            return
        # get up/down weight
        if self.full_rank:
            up_weight = None
        else:
            up_weight = self.lora_up.weight.clone().float()
        down_weight = self.lora_down.weight.clone().float()

        # extract weight from org_module
        org_sd = self.org_module[0].state_dict()
        # todo find a way to merge in weights when doing quantized model
        if 'weight._data' in org_sd:
            # quantized weight
            return

        weight_key = "weight"
        from toolkit.util.quantize import is_quantized_tensor
        org_weight = self.org_module[0].weight
        is_ao_quantized = is_quantized_tensor(org_weight)
        orig_dtype = org_weight.dtype
        # dequantize torchao weights so the delta can be merged in full precision
        base_weight = org_weight.dequantize() if is_ao_quantized else org_weight
        weight = base_weight.float()

        multiplier = merge_weight
        scale = self.scale
        # handle trainable scaler method locon does
        if hasattr(self, 'scalar'):
            scale = scale * self.scalar

        weight_device = weight.device
        if weight.device != down_weight.device:
            weight = weight.to(down_weight.device)
        if scale.device != down_weight.device:
            scale = scale.to(down_weight.device)
        # merge weight
        if self.full_rank:
            weight = weight + multiplier * down_weight * scale
        elif len(weight.size()) == 2:
            # linear
            weight = weight + multiplier * (up_weight @ down_weight) * scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                    weight
                    + multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # print(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + multiplier * conved * scale

        if getattr(self.network_ref(), 'is_assistant_adapter', False):
            self.assistant_inverse_scale = _calibrate_assistant_inverse_scale(
                self, base_weight, weight, orig_dtype
            )

        # write the merged weight back, re-quantizing if the original was torchao quantized so the
        # model stays quantized across continuous merge/reset cycles
        if is_ao_quantized:
            from toolkit.util.quantize import get_torchao_config, requantize_module_weight
            config = get_torchao_config(self._get_base_qtype())
            if config is None:
                print_once(f"Warning: merging into quantized layer {getattr(self, 'lora_name', '?')} "
                           f"without a known qtype; it will be left dequantized")
            requantize_module_weight(self.org_module[0], weight.to(weight_device), orig_dtype, config)
        else:
            org_sd[weight_key] = weight.to(weight_device, orig_dtype)
            self.org_module[0].load_state_dict(org_sd)

    def reset_weights(self: Module):
        # reset the weights to zero
        org_sd = self.state_dict()
        for key in org_sd.keys():
            # only reset lora up
            if 'lora_up' in key:
                org_sd[key] = torch.zeros_like(org_sd[key])
        self.load_state_dict(org_sd)

    def setup_lorm(self: Module, state_dict: Optional[Dict[str, Any]] = None):
        # LoRM (Low Rank Middle) is a method reduce the number of parameters in a module while keeping the inputs and
        # outputs the same. It is basically a LoRA but with the original module removed

        # if a state dict is passed, use those weights instead of extracting
        # todo load from state dict
        network: Network = self.network_ref()
        lorm_config = network.network_config.lorm_config.get_config_for_module(self.lora_name)

        extract_mode = lorm_config.extract_mode
        extract_mode_param = lorm_config.extract_mode_param
        parameter_threshold = lorm_config.parameter_threshold
        self.extract_weight(
            extract_mode=extract_mode,
            extract_mode_param=extract_mode_param
        )


class ToolkitNetworkMixin:
    def __init__(
            self: Network,
            *args,
            train_text_encoder: Optional[bool] = True,
            train_unet: Optional[bool] = True,
            is_sdxl=False,
            is_v2=False,
            is_ssd=False,
            is_vega=False,
            network_config: Optional[NetworkConfig] = None,
            is_lorm=False,
            **kwargs
    ):
        self.train_text_encoder = train_text_encoder
        self.train_unet = train_unet
        self.is_checkpointing = False
        self._multiplier: float = 1.0
        self.is_active: bool = False
        self.is_sdxl = is_sdxl
        self.is_ssd = is_ssd
        self.is_vega = is_vega
        self.is_v2 = is_v2
        self.is_v1 = not is_v2 and not is_sdxl and not is_ssd and not is_vega
        self.is_merged_in = False
        self.is_lorm = is_lorm
        self.network_config: NetworkConfig = network_config
        self.module_losses: List[torch.Tensor] = []
        self.lorm_train_mode: Literal['local', None] = None
        self.can_merge_in = not is_lorm
        # will prevent optimizer from loading as it will have double states
        self.did_change_weights = False
        self.vector_gates = None

    def get_keymap(self: Network, force_weight_mapping=False):
        use_weight_mapping = False

        if self.is_ssd:
            keymap_tail = 'ssd'
            use_weight_mapping = True
        elif self.is_vega:
            keymap_tail = 'vega'
            use_weight_mapping = True
        elif self.is_sdxl:
            keymap_tail = 'sdxl'
        elif self.is_v2:
            keymap_tail = 'sd2'
        else:
            keymap_tail = 'sd1'
            # todo double check this
            # use_weight_mapping = True

        if force_weight_mapping:
            use_weight_mapping = True

        # load keymap
        keymap_name = f"stable_diffusion_locon_{keymap_tail}.json"
        if use_weight_mapping:
            keymap_name = f"stable_diffusion_{keymap_tail}.json"

        keymap_path = os.path.join(KEYMAPS_ROOT, keymap_name)

        keymap = None
        # check if file exists
        if os.path.exists(keymap_path):
            with open(keymap_path, 'r') as f:
                keymap = json.load(f)['ldm_diffusers_keymap']

        if use_weight_mapping and keymap is not None:
            # get keymap from weights
            keymap = get_lora_keymap_from_model_keymap(keymap)

        # upgrade keymaps for DoRA
        if self.network_type.lower() == 'dora':
            if keymap is not None:
                new_keymap = {}
                for ldm_key, diffusers_key in keymap.items():
                    ldm_key = ldm_key.replace('.alpha', '.magnitude')
                    # ldm_key = ldm_key.replace('.lora_down.weight', '.lora_down')
                    # ldm_key = ldm_key.replace('.lora_up.weight', '.lora_up')

                    diffusers_key = diffusers_key.replace('.alpha', '.magnitude')
                    # diffusers_key = diffusers_key.replace('.lora_down.weight', '.lora_down')
                    # diffusers_key = diffusers_key.replace('.lora_up.weight', '.lora_up')

                    new_keymap[ldm_key] = diffusers_key

                keymap = new_keymap

        return keymap
    
    def get_state_dict(self: Network, extra_state_dict=None, dtype=torch.float16, stager=None):
        keymap = self.get_keymap()

        save_keymap = {}
        if keymap is not None:
            for ldm_key, diffusers_key in keymap.items():
                #  invert them
                save_keymap[diffusers_key] = ldm_key

        state_dict = self.state_dict()

        if stager is not None:
            # Batched device->host copy through a capped pinned buffer: one CUDA
            # sync per chunk instead of one per tensor. Output is identical to the
            # per-tensor loop below; only the copy path differs.
            items = [
                (save_keymap.get(key, key), state_dict[key])
                for key in list(state_dict.keys())
            ]
            save_dict = stager.snapshot(items, out_dtype=dtype)
            state_dict.clear()
        else:
            save_dict = OrderedDict()
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_key = save_keymap[key] if key in save_keymap else key
                save_dict[save_key] = v
                del state_dict[key]

        if extra_state_dict is not None:
            # add extra items to state dict
            for key in list(extra_state_dict.keys()):
                v = extra_state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                save_dict[key] = v

        if self.peft_format:
            # lora_down = lora_A
            # lora_up = lora_B
            # no alpha

            new_save_dict = {}
            for key, value in save_dict.items():
                # lokr needs alpha
                if key.endswith('.alpha') and self.network_type.lower() != "lokr":
                    continue
                new_key = key
                new_key = new_key.replace('lora_down', 'lora_A')
                new_key = new_key.replace('lora_up', 'lora_B')
                # replace all $$ with .
                new_key = new_key.replace('$$', '.')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
                
        if self.network_type.lower() == "lokr" and self.use_old_lokr_format:
            new_save_dict = {}
            for key, value in save_dict.items():
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                new_key = key
                new_key = new_key.replace('lora_transformer_', 'lycoris_')
                new_save_dict[new_key] = value

            save_dict = new_save_dict
        
        if self.base_model_ref is not None:
            save_dict = self.base_model_ref().convert_lora_weights_before_save(save_dict)
        return save_dict

    def save_weights(
            self: Network,
            file, dtype=torch.float16,
            metadata=None,
            extra_state_dict: Optional[OrderedDict] = None,
            writer=None,
            stager=None,
            coalesce_key=None,
    ):
        # get_state_dict does the device->host copy: it must run here, on the
        # calling (training) thread, so the snapshot is consistent before the
        # next optimizer.step mutates the live weights. Only the CPU-side hash +
        # disk write below can be deferred to ``writer`` (an AsyncSaver).
        save_dict = self.get_state_dict(extra_state_dict=extra_state_dict, dtype=dtype, stager=stager)

        if metadata is not None and len(metadata) == 0:
            metadata = None

        if metadata is None:
            metadata = OrderedDict()

        # let the model handle the saving
        if self.base_model_ref is not None and hasattr(self.base_model_ref(), 'save_lora'):
            # call the base model save lora method (kept synchronous: model owns the format)
            metadata = add_model_hash_to_meta(save_dict, metadata)
            self.base_model_ref().save_lora(save_dict, file, metadata)
            return

        is_safetensors = os.path.splitext(file)[1] == ".safetensors"

        def _write():
            # Hashing walks the (CPU) tensors -- offload it to the writer thread too.
            md = add_model_hash_to_meta(save_dict, metadata)
            if is_safetensors:
                from toolkit.async_save import atomic_save_file
                atomic_save_file(save_dict, file, md)
            else:
                from toolkit.async_save import atomic_torch_save
                atomic_torch_save(save_dict, file)

        if writer is not None:
            writer.submit(_write, coalesce_key=coalesce_key,
                          description=f"lora:{os.path.basename(file)}")
        else:
            _write()

    def load_weights(self: Network, file, force_weight_mapping=False):
        # allows us to save and load to and from ldm weights
        keymap = self.get_keymap(force_weight_mapping)
        keymap = {} if keymap is None else keymap

        if isinstance(file, str):
            if self.base_model_ref is not None and hasattr(self.base_model_ref(), 'load_lora'):
                # call the base model load lora method
                weights_sd = self.base_model_ref().load_lora(file)
            else:
                if os.path.splitext(file)[1] == ".safetensors":
                    from safetensors.torch import load_file
                    weights_sd = load_file(file)
                else:
                    weights_sd = torch.load(file, map_location="cpu")
        else:
            # probably a state dict
            weights_sd = file
        
        if self.base_model_ref is not None:
            weights_sd = self.base_model_ref().convert_lora_weights_before_load(weights_sd)

        load_sd = OrderedDict()
        for key, value in weights_sd.items():
            load_key = keymap[key] if key in keymap else key
            # replace old double __ with single _
            if self.is_pixart:
                load_key = load_key.replace('__', '_')

            if self.peft_format:
                # lora_down = lora_A
                # lora_up = lora_B
                # no alpha
                if load_key.endswith('.alpha') and self.network_type.lower() != "lokr":
                    continue
                load_key = load_key.replace('lora_A', 'lora_down')
                load_key = load_key.replace('lora_B', 'lora_up')
                # replace all . with $$
                load_key = load_key.replace('.', '$$')
                load_key = load_key.replace('$$lora_down$$', '.lora_down.')
                load_key = load_key.replace('$$lora_up$$', '.lora_up.')
                # full weight modules store their delta as `.diff` / `.diff_b` (anchored at the
                # end so this is a no-op for any non-full-weight key)
                if load_key.endswith('$$diff'):
                    load_key = load_key[:-len('$$diff')] + '.diff'
                elif load_key.endswith('$$diff_b'):
                    load_key = load_key[:-len('$$diff_b')] + '.diff_b'

                # patch lokr, not sure why we need to but whatever
                if self.network_type.lower() == "lokr":
                    load_key = load_key.replace('$$lokr_w1', '.lokr_w1')
                    load_key = load_key.replace('$$lokr_w2', '.lokr_w2')
                    if load_key.endswith('$$alpha'):
                        load_key = load_key[:-7] + '.alpha'
            
            if self.network_type.lower() == "lokr":
                # lora_transformer_transformer_blocks_7_attn_to_v.lokr_w1 to lycoris_transformer_blocks_7_attn_to_v.lokr_w1
                load_key = load_key.replace('lycoris_', 'lora_transformer_')

            load_sd[load_key] = value

        # extract extra items from state dict
        current_state_dict = self.state_dict()
        extra_dict = OrderedDict()
        to_delete = []
        for key in list(load_sd.keys()):
            if key not in current_state_dict:
                extra_dict[key] = load_sd[key]
                to_delete.append(key)
            elif "lora_down" in key or "lora_up" in key:
                # handle expanding/shrinking LoRA (linear only)
                if len(load_sd[key].shape) == 2:
                    load_value = load_sd[key]                 # from checkpoint
                    blank_val = current_state_dict[key]       # shape we need in the target model
                    tgt_h, tgt_w = blank_val.shape
                    src_h, src_w = load_value.shape

                    if (src_h, src_w) == (tgt_h, tgt_w):
                        # shapes already match: keep original
                        pass

                    elif "lora_down" in key and src_h < tgt_h:
                        print_once(f"Expanding {key} from {load_value.shape} to {blank_val.shape}")
                        new_val = torch.zeros((tgt_h, tgt_w), device=load_value.device, dtype=load_value.dtype)
                        new_val[:src_h, :src_w] = load_value  # src_w should already match
                        load_sd[key] = new_val
                        self.did_change_weights = True

                    elif "lora_up" in key and src_w < tgt_w:
                        print_once(f"Expanding {key} from {load_value.shape} to {blank_val.shape}")
                        new_val = torch.zeros((tgt_h, tgt_w), device=load_value.device, dtype=load_value.dtype)
                        new_val[:src_h, :src_w] = load_value  # src_h should already match
                        load_sd[key] = new_val
                        self.did_change_weights = True

                    elif "lora_down" in key and src_h > tgt_h:
                        print_once(f"Shrinking {key} from {load_value.shape} to {blank_val.shape}")
                        load_sd[key] = load_value[:tgt_h, :tgt_w]
                        self.did_change_weights = True

                    elif "lora_up" in key and src_w > tgt_w:
                        print_once(f"Shrinking {key} from {load_value.shape} to {blank_val.shape}")
                        load_sd[key] = load_value[:tgt_h, :tgt_w]
                        self.did_change_weights = True

                    else:
                        # unexpected mismatch (e.g., both dims differ in a way that doesn't match lora_up/down semantics)
                        raise ValueError(f"Unhandled LoRA shape change for {key}: src={load_value.shape}, tgt={blank_val.shape}")

        for key in to_delete:
            del load_sd[key]

        print(f"Missing keys: {to_delete}")
        if len(to_delete) > 0 and self.is_v1 and not force_weight_mapping and not (
                len(to_delete) == 1 and 'emb_params' in to_delete):
            print(" Attempting to load with forced keymap")
            return self.load_weights(file, force_weight_mapping=True)

        info = self.load_state_dict(load_sd, False)
        if len(extra_dict.keys()) == 0:
            extra_dict = None
        return extra_dict

    @torch.no_grad()
    def _update_torch_multiplier(self: Network):
        # builds a tensor for fast usage in the forward pass of the network modules
        # without having to set it in every single module every time it changes
        multiplier = self._multiplier
        # get first module
        try:
            first_module = self.get_all_modules()[0]
        except IndexError:
            raise ValueError("There are not any lora modules in this network. Check your config and try again")
        
        if hasattr(first_module, 'lora_down'):
            device = first_module.lora_down.weight.device
            dtype = first_module.lora_down.weight.dtype
            if hasattr(first_module.lora_down, '_memory_management_device'):
                device = first_module.lora_down._memory_management_device
        elif hasattr(first_module, 'lokr_w1'):
            device = first_module.lokr_w1.device
            dtype = first_module.lokr_w1.dtype
            if hasattr(first_module.lokr_w1, '_memory_management_device'):
                device = first_module.lokr_w1._memory_management_device
        elif hasattr(first_module, 'lokr_w1_a'):
            device = first_module.lokr_w1_a.device
            dtype = first_module.lokr_w1_a.dtype
            if hasattr(first_module.lokr_w1_a, '_memory_management_device'):
                device = first_module.lokr_w1_a._memory_management_device
        elif hasattr(first_module, 'diff'):
            # full weight module
            device = first_module.diff.device
            dtype = first_module.diff.dtype
        else:
            raise ValueError("Unknown module type")
        with torch.no_grad():
            tensor_multiplier = None
            if isinstance(multiplier, int) or isinstance(multiplier, float):
                tensor_multiplier = torch.tensor((multiplier,)).to(device, dtype=dtype)
            elif isinstance(multiplier, list):
                tensor_multiplier = torch.tensor(multiplier).to(device, dtype=dtype)
            elif isinstance(multiplier, torch.Tensor):
                tensor_multiplier = multiplier.clone().detach().to(device, dtype=dtype)

            self.torch_multiplier = tensor_multiplier.clone().detach()

    @property
    def multiplier(self) -> Union[float, List[float], List[List[float]]]:
        return self._multiplier

    @multiplier.setter
    def multiplier(self, value: Union[float, List[float], List[List[float]]]):
        # it takes time to update all the multipliers, so we only do it if the value has changed
        if self._multiplier == value:
            return
        # if we are setting a single value but have a list, keep the list if every item is the same as value
        self._multiplier = value
        self._update_torch_multiplier()

    # called when the context manager is entered
    # ie: with network:
    def __enter__(self: Network):
        self.is_active = True

    def __exit__(self: Network, exc_type, exc_value, tb):
        self.is_active = False

    def force_to(self: Network, device, dtype):
        self.to(device, dtype)
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        for lora in loras:
            lora.to(device, dtype)

        if self.vector_gates is not None:
            self.set_vector_gates(self.vector_gates, device=device, dtype=dtype)

    def set_vector_gates(self: Network, gates, device=None, dtype=None):
        def to_tensor(value):
            if value is None or isinstance(value, torch.Tensor):
                tensor = value
            else:
                tensor = torch.tensor(value)
            if tensor is not None and (device is not None or dtype is not None):
                tensor = tensor.to(
                    device=device if device is not None else tensor.device,
                    dtype=dtype if dtype is not None else tensor.dtype,
                )
            return tensor

        if gates is None:
            self.vector_gates = None
            return

        if isinstance(gates, dict):
            converted = {}
            if "global" in gates:
                converted["global"] = to_tensor(gates["global"])
            if "per_module" in gates:
                converted["per_module"] = {
                    name: to_tensor(value)
                    for name, value in gates["per_module"].items()
                }
            self.vector_gates = converted
        else:
            self.vector_gates = to_tensor(gates)

    def clear_vector_gates(self: Network):
        self.vector_gates = None

    def get_all_modules(self: Network) -> List[Module]:
        loras = []
        if hasattr(self, 'unet_loras'):
            loras += self.unet_loras
        if hasattr(self, 'text_encoder_loras'):
            loras += self.text_encoder_loras
        return loras

    def _update_checkpointing(self: Network):
        for module in self.get_all_modules():
            if self.is_checkpointing:
                module.enable_gradient_checkpointing()
            else:
                module.disable_gradient_checkpointing()

    def enable_gradient_checkpointing(self: Network):
        # not supported
        self.is_checkpointing = True
        self._update_checkpointing()

    def disable_gradient_checkpointing(self: Network):
        # not supported
        self.is_checkpointing = False
        self._update_checkpointing()
    
    def reset_weights(self: Network):
        for module in self.get_all_modules():
            module.reset_weights()

    def merge_in(self, merge_weight=1.0):
        if self.network_type.lower() == 'dora':
            return
        self.is_merged_in = True
        for module in self.get_all_modules():
            module.merge_in(merge_weight)

    def merge_out(self: Network, merge_weight=1.0):
        if not self.is_merged_in:
            return
        self.is_merged_in = False
        for module in self.get_all_modules():
            module.merge_out(merge_weight)

    def extract_weight(
            self: Network,
            extract_mode: ExtractMode = "existing",
            extract_mode_param: Union[int, float] = None,
    ):
        if extract_mode_param is None:
            raise ValueError("extract_mode_param must be set")
        for module in tqdm(self.get_all_modules(), desc="Extracting weights"):
            module.extract_weight(
                extract_mode=extract_mode,
                extract_mode_param=extract_mode_param
            )

    def setup_lorm(self: Network, state_dict: Optional[Dict[str, Any]] = None):
        for module in tqdm(self.get_all_modules(), desc="Extracting LoRM"):
            module.setup_lorm(state_dict=state_dict)

    def calculate_lorem_parameter_reduction(self):
        params_reduced = 0
        for module in self.get_all_modules():
            num_orig_module_params = count_parameters(module.org_module[0])
            num_lorem_params = count_parameters(module.lora_down) + count_parameters(module.lora_up)
            params_reduced += (num_orig_module_params - num_lorem_params)

        return params_reduced
