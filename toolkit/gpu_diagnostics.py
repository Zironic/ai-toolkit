import os
import sys
import subprocess
import gc
from typing import Dict, Any, Iterable

import torch
from toolkit.print import print_acc


def _format_bytes(n: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def _module_bytes(module: Any, device_index: int = None) -> int:
    total = 0
    try:
        for p in module.parameters():
            try:
                if p is None:
                    continue
                if not torch.is_tensor(p):
                    continue
                if device_index is None or (hasattr(p, 'device') and getattr(p.device, 'index', None) == device_index):
                    total += p.numel() * p.element_size()
            except Exception:
                continue
        for b in module.buffers():
            try:
                if b is None:
                    continue
                if not torch.is_tensor(b):
                    continue
                if device_index is None or (hasattr(b, 'device') and getattr(b.device, 'index', None) == device_index):
                    total += b.numel() * b.element_size()
            except Exception:
                continue
    except Exception:
        return 0
    return total


def dump_vram_map(root_modules: Dict[str, Any], deep_scan: bool = False, include_nvidia_smi: bool = True) -> str:
    """Return a human-readable VRAM map for the current process.

    root_modules: mapping of name -> module/object (e.g., {'sd.unet': sd.unet, 'sd.vae': sd.vae})
    deep_scan: if True, scan Python GC for CUDA tensors (costly)
    include_nvidia_smi: if True, attempt to collect per-process memory from nvidia-smi
    """
    lines = []
    lines.append("=== VRAM DIAGNOSTIC SNAPSHOT ===")

    try:
        cuda_available = torch.cuda.is_available()
        num_devices = torch.cuda.device_count() if cuda_available else 0
        pid = os.getpid()

        # Compute global module sizes (device-agnostic) so we can report on CPU-only setups
        global_module_sizes = {}
        for mname, m in root_modules.items():
            try:
                global_module_sizes[mname] = _module_bytes(m, device_index=None)
            except Exception:
                global_module_sizes[mname] = 0

        if cuda_available and num_devices > 0:
            for dev in range(num_devices):
                try:
                    name = torch.cuda.get_device_name(dev)
                except Exception:
                    name = f"cuda:{dev}"
                allocated = torch.cuda.memory_allocated(dev)
                reserved = torch.cuda.memory_reserved(dev)
                lines.append(f"Device {dev}: {name} | allocated={_format_bytes(allocated)} reserved={_format_bytes(reserved)}")

                # Summarize provided modules on this device
                module_summaries = []
                tot = 0
                for mname, m in root_modules.items():
                    try:
                        b = _module_bytes(m, device_index=dev)
                        if b > 0:
                            module_summaries.append((mname, b))
                            tot += b
                    except Exception:
                        continue
                if module_summaries:
                    lines.append("  Modules on this device:")
                    for mname, b in sorted(module_summaries, key=lambda x: x[1], reverse=True):
                        lines.append(f"    - {mname}: {_format_bytes(b)}")
                    lines.append(f"  Module total: {_format_bytes(tot)}")

                    # Device-local role breakdown (best-effort)
                    try:
                        role_totals = {'model_unet': 0, 'vae': 0, 'text_encoder': 0, 'controlnet_adapter': 0, 'other_modules': 0}
                        lora_total = 0
                        for mname, m in root_modules.items():
                            b = _module_bytes(m, device_index=dev)
                            name_lower = mname.lower()
                            try:
                                cls_name = m.__class__.__name__.lower()
                            except Exception:
                                cls_name = ''

                            if ('unet' in name_lower) or ('unet' in cls_name) or ('model' in name_lower and 'control' not in name_lower):
                                role_totals['model_unet'] += b
                            elif ('vae' in name_lower) or ('vae' in cls_name):
                                role_totals['vae'] += b
                            elif ('text' in name_lower) or ('encoder' in name_lower) or ('text' in cls_name):
                                role_totals['text_encoder'] += b
                            elif ('control' in name_lower) or ('controlnet' in name_lower) or ('adapter' in name_lower) or ('control' in cls_name):
                                role_totals['controlnet_adapter'] += b
                            else:
                                role_totals['other_modules'] += b

                            try:
                                for pname, p in m.named_parameters(recurse=True):
                                    if p is None:
                                        continue
                                    pname_lower = pname.lower()
                                    if 'lora' in pname_lower or 'lokr' in pname_lower or 'locon' in pname_lower or 'lorm' in pname_lower or ('rank' in pname_lower and 'lora' in pname_lower):
                                        try:
                                            if p.is_cuda and p.device.index == dev:
                                                lora_total += p.numel() * p.element_size()
                                        except Exception:
                                            continue
                            except Exception:
                                pass

                        # Optimizer state on this device
                        optimizer_bytes = 0
                        if 'optimizer' in root_modules:
                            try:
                                opt = root_modules['optimizer']
                                for state in opt.state.values():
                                    if isinstance(state, dict):
                                        for v in state.values():
                                            try:
                                                if torch.is_tensor(v) and v.is_cuda and (v.device.index == dev):
                                                    optimizer_bytes += v.numel() * v.element_size()
                                            except Exception:
                                                continue
                            except Exception:
                                optimizer_bytes = 0

                        lines.append("  Role breakdown (best-effort):")
                        lines.append(f"    - Model/UNet: {_format_bytes(role_totals['model_unet'])}")
                        lines.append(f"    - VAE: {_format_bytes(role_totals['vae'])}")
                        lines.append(f"    - TextEncoder: {_format_bytes(role_totals['text_encoder'])}")
                        lines.append(f"    - ControlNet/Adapters: {_format_bytes(role_totals['controlnet_adapter'])}")
                        lines.append(f"    - LoRA (subset of model): {_format_bytes(lora_total)}")
                        lines.append(f"    - Optimizer state: {_format_bytes(optimizer_bytes)}")
                        other_role = max(0, tot - (role_totals['model_unet'] + role_totals['vae'] + role_totals['text_encoder'] + role_totals['controlnet_adapter']))
                        lines.append(f"    - Other modules: {_format_bytes(other_role)}")

                        # Device-local param/submodule breakdown (only count params resident on this device)
                        try:
                            param_sizes_dev = []
                            submodule_totals_dev = {}
                            for mname, m in root_modules.items():
                                try:
                                    for pname, p in m.named_parameters(recurse=True):
                                        try:
                                            if p is None:
                                                continue
                                            if not torch.is_tensor(p):
                                                continue
                                            # Only include parameters that are CUDA tensors resident on this device
                                            try:
                                                if not p.is_cuda or getattr(p.device, 'index', None) != dev:
                                                    continue
                                            except Exception:
                                                continue
                                            sz = p.numel() * p.element_size()
                                            full_name = f"{mname}.{pname}"
                                            param_sizes_dev.append((full_name, sz))
                                            # Record to submodule prefix (take first two segments)
                                            prefix = full_name.split('.')
                                            if len(prefix) >= 2:
                                                sub = '.'.join(prefix[:2])
                                            else:
                                                sub = prefix[0]
                                            submodule_totals_dev[sub] = submodule_totals_dev.get(sub, 0) + sz
                                        except Exception:
                                            continue
                                except Exception:
                                    continue

                            if param_sizes_dev:
                                # Aggregate third-tier groups for per-device listing as well
                                group_totals_dev = {}
                                for full, sz in param_sizes_dev:
                                    parts = full.split('.')
                                    if len(parts) >= 3:
                                        grp = '.'.join(parts[:3])
                                    else:
                                        grp = full
                                    group_totals_dev[grp] = group_totals_dev.get(grp, 0) + sz

                                param_sizes_sorted_dev = sorted(param_sizes_dev, key=lambda x: x[1], reverse=True)
                                lines.append("  Top parameter groups on this device (desc, best-effort):")
                                for name, b in sorted(group_totals_dev.items(), key=lambda x: x[1], reverse=True)[:20]:
                                    lines.append(f"    - {name}: {_format_bytes(b)}")

                                if submodule_totals_dev:
                                    lines.append("  Top submodules on this device:")
                                    for name, b in sorted(submodule_totals_dev.items(), key=lambda x: x[1], reverse=True)[:20]:
                                        lines.append(f"    - {name}: {_format_bytes(b)}")
                        except Exception:
                            pass
                    except Exception:
                        pass
        else:
            lines.append("No CUDA devices available — emitting CPU-only module size summary.")

        # Global sorted module list (device-agnostic)
        try:
            if global_module_sizes:
                lines.append("All modules by size (descending):")
                for mname, b in sorted(global_module_sizes.items(), key=lambda x: x[1], reverse=True):
                    if b > 0:
                        lines.append(f"  - {mname}: {_format_bytes(b)}")
        except Exception:
            pass

        # More detailed per-parameter and submodule breakdown (best-effort, CPU-safe)
        try:
            # Build a flat list of (full_param_name, size) across all modules
            param_sizes = []
            submodule_totals = {}
            for mname, m in root_modules.items():
                try:
                    for pname, p in m.named_parameters(recurse=True):
                        try:
                            if p is None:
                                continue
                            sz = p.numel() * p.element_size()
                            full_name = f"{mname}.{pname}"
                            param_sizes.append((full_name, sz))
                            # Record to submodule prefix (take first two segments)
                            prefix = full_name.split('.')
                            if len(prefix) >= 2:
                                sub = '.'.join(prefix[:2])
                            else:
                                sub = prefix[0]
                            submodule_totals[sub] = submodule_totals.get(sub, 0) + sz
                        except Exception:
                            continue
                except Exception:
                    continue

            # Aggregate by third-tier group (e.g., adapter.inner.noise_refiner)
            try:
                if param_sizes:
                    group_totals = {}
                    for full, sz in param_sizes:
                        parts = full.split('.')
                        if len(parts) >= 3:
                            grp = '.'.join(parts[:3])
                        else:
                            grp = full
                        group_totals[grp] = group_totals.get(grp, 0) + sz

                    # Report aggregated third-tier groups for readability
                    if group_totals:
                        lines.append("Top parameter groups by aggregated size (3-tier, best-effort):")
                        for name, b in sorted(group_totals.items(), key=lambda x: x[1], reverse=True)[:20]:
                            lines.append(f"  - {name}: {_format_bytes(b)}")

                    # Also print top submodules by aggregated parameter size
                    if submodule_totals:
                        lines.append("Top submodules by aggregated param size:")
                        for name, b in sorted(submodule_totals.items(), key=lambda x: x[1], reverse=True)[:20]:
                            lines.append(f"  - {name}: {_format_bytes(b)}")

            except Exception:
                pass
        except Exception:
            pass
        # Optional: run nvidia-smi to get process memory usage if available
        if include_nvidia_smi:
            try:
                cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,nounits,noheader"]
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
                out = p.stdout.strip()
                if out:
                    lines.append("nvidia-smi process usage:")
                    for line in out.splitlines():
                        parts = [x.strip() for x in line.split(',')]
                        if len(parts) >= 3:
                            try:
                                lpid = int(parts[0])
                                if lpid == pid:
                                    lines.append(f"  {parts[1]} (pid={lpid}) used_memory={parts[2]}MiB")
                            except Exception:
                                continue
            except Exception:
                # best-effort, don't fail the diagnostic
                pass

        if deep_scan:
            # costly GC scan for CUDA tensors
            try:
                tensor_count = 0
                bytes_total = 0
                by_type = {}
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            tensor_count += 1
                            sz = obj.numel() * obj.element_size()
                            bytes_total += sz
                            tname = str(type(obj)).split("'")[1]
                            by_type[tname] = by_type.get(tname, 0) + sz
                    except Exception:
                        continue
                lines.append(f"GC CUDA tensors: count={tensor_count} total={_format_bytes(bytes_total)}")
                for tname, b in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
                    lines.append(f"  {tname}: {_format_bytes(b)}")
            except Exception:
                pass

    except Exception as e:
        lines.append(f"Failed to compute VRAM map: {e}")

    lines.append("=== END VRAM DIAGNOSTIC ===")
    return "\n".join(lines)
