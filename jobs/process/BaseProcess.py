import copy
import json
from collections import OrderedDict
from toolkit.print import print_acc
from toolkit.timer import Timer
# Optional VRAM diagnostic utility (opt-in, disabled by default)
from toolkit.gpu_diagnostics import dump_vram_map


class BaseProcess(object):

    def __init__(
            self,
            process_id: int,
            job: 'BaseJob',
            config: OrderedDict
    ):
        self.process_id = process_id
        self.meta: OrderedDict
        self.job = job
        self.config = config
        self.raw_process_config = config
        self.name = self.get_conf('name', self.job.name)
        self.meta = copy.deepcopy(self.job.meta)
        self.timer: Timer = Timer(f'{self.name} Timer')
        # Allow opt-in GPU timing for diagnostics (disabled by default).
        self.timer.gpu_timing_enabled = self.get_conf('performance.precise_gpu_timing', False)
        # Add a composite hook that prints aggregated ControlNet vs Model timings
        # so users can quickly see how heavy ControlNet work is relative to model work.
        self.timer.add_after_print_hook(self._print_timer_composites)
        # Add an optional VRAM diagnostic hook to print a lightweight VRAM map
        # alongside the PERF SUMMARY. Controlled by `performance.vram_diagnostics` config.
        self._vram_diag_printed = False
        self.timer.add_after_print_hook(self._print_vram_diagnostics)
        self.performance_log_every = self.get_conf('performance_log_every', 0)

        print(json.dumps(self.config, indent=4))

    def _print_timer_composites(self, timing_dict):
        """Print a concise composite summary for ControlNet and Model timings.
        timing_dict maps timer_name -> average_seconds.
        """

        control_keys = [
            'get_adapter_images', 'encode_adapter', 'encode_adapter_embeds', 'use_precomputed_control_residuals', 'get_mask_multiplier', 'controlnet_forward', 'controlnet_forward_io', 'controlnet_zimage_forward'
        ]
        model_keys = [
            'predict_unet', 'encode_images', 'to_device', 'cpu_transfer', 'calculate_loss', 'backward', 'optimizer_step', 'ema_update'
        ]
        control_total = sum([timing_dict.get(k, 0.0) for k in control_keys])
        model_total = sum([timing_dict.get(k, 0.0) for k in model_keys])

        # Exclude aggregator timers (like the overall 'train_loop') so they don't dominate 'Other'
        aggregator_keys = ['train_loop', 'train_epoch', 'train_step']
        aggregator_total = sum([timing_dict.get(k, 0.0) for k in aggregator_keys])

        # Other is everything except control/model/aggregators
        other_total = max(0.0, sum(timing_dict.values()) - control_total - model_total - aggregator_total)

        # Determine top-5 timers contributing to "Other" and include them in the summary
        excluded_keys = set(control_keys + model_keys + aggregator_keys)
        other_timers = [(k, v) for k, v in timing_dict.items() if k not in excluded_keys]
        other_timers = sorted(other_timers, key=lambda x: x[1], reverse=True)
        top_others = other_timers[:5]
        if top_others:
            top_others_str = ', '.join([f"{name}:{val:.4f}s" for name, val in top_others])
            other_extra = f" (top: {top_others_str})"
        else:
            other_extra = ''

        # Also show the overall train_loop time (if measured) and how much of it is unaccounted for
        train_loop_total = timing_dict.get('train_loop', None)
        unaccounted = None
        if train_loop_total is not None:
            # measured_sum is the sum of all non-aggregator timers (i.e., explicit measured parts)
            measured_sum = sum([v for k, v in timing_dict.items() if k not in aggregator_keys])
            # how much of the train loop time is not accounted for by measured timers
            unaccounted = max(0.0, train_loop_total - measured_sum)

        # Build the parts list dynamically so we don't print ControlNet for non-ControlNet jobs
        parts = []
        if train_loop_total is not None:
            parts.append(f"total/train_loop: {train_loop_total:.4f}s avg")

        if any(k in timing_dict for k in control_keys):
            parts.append(f"ControlNet: {control_total:.4f}s avg")

        if any(k in timing_dict for k in model_keys):
            parts.append(f"Model: {model_total:.4f}s avg")

        parts.append(f"Other: {other_total:.4f}s avg{other_extra}")

        if train_loop_total is not None:
            parts.append(f"Unaccounted: {unaccounted:.4f}s avg")

        summary = ' | '.join(parts)
        print_acc(f"PERF SUMMARY: {summary}")

    def _print_vram_diagnostics(self, timing_dict):
        """Optional hook to print a lightweight VRAM diagnostic at the same time as PERF SUMMARY.
        Controlled by these config keys (all optional):
          - performance.vram_diagnostics.enabled (bool, default False)
          - performance.vram_diagnostics.once_per_run (bool, default True)
          - performance.vram_diagnostics.deep_scan (bool, default False)
          - performance.vram_diagnostics.include_nvidia_smi (bool, default False)
        """
        try:
            enabled = self.get_conf('performance.vram_diagnostics.enabled', False)
            if not enabled:
                return

            # Print at most once per run by default to avoid heavy output
            once_per_run = self.get_conf('performance.vram_diagnostics.once_per_run', True)
            if once_per_run and getattr(self, '_vram_diag_printed', False):
                return

            deep_scan = self.get_conf('performance.vram_diagnostics.deep_scan', False)
            include_nvidia_smi = self.get_conf('performance.vram_diagnostics.include_nvidia_smi', False)

            # Build a minimal root_modules map if present (best-effort, non-fatal)
            root_modules = {}
            try:
                # `self` is often a process subclass with attributes like `sd`, `adapter`
                sd = getattr(self, 'sd', None)
                if sd is not None:
                    if hasattr(sd, 'unet'):
                        root_modules['sd.unet'] = sd.unet
                    if hasattr(sd, 'vae'):
                        root_modules['sd.vae'] = sd.vae
                    # text_encoder can be a list or single module
                    te = getattr(sd, 'text_encoder', None)
                    if te is not None:
                        if isinstance(te, (list, tuple)):
                            for i, t in enumerate(te):
                                root_modules[f'sd.text_encoder[{i}]'] = t
                        else:
                            root_modules['sd.text_encoder'] = te
            except Exception:
                # best effort; do not fail the perf printing if introspection fails
                pass

            # Include adapter (if present) for adapter-heavy jobs
            try:
                adapter = getattr(self, 'adapter', None)
                if adapter is not None:
                    root_modules['adapter'] = adapter
            except Exception:
                pass

            # Run the diagnostic (best-effort; it returns a string)
            try:
                diag = dump_vram_map(root_modules, deep_scan=deep_scan, include_nvidia_smi=include_nvidia_smi)
                print_acc(diag)
                self._vram_diag_printed = True
            except Exception as e:
                print_acc(f"[VRAM-DIAG] Failed to compute VRAM diagnostic: {e}")
        except Exception:
            # Never let diagnostic fail the timer print
            pass

        
    def on_error(self, e: Exception):
        pass

    def get_conf(self, key, default=None, required=False, as_type=None):
        # split key by '.' and recursively get the value
        keys = key.split('.')

        # see if it exists in the config
        value = self.config
        for subkey in keys:
            if subkey in value:
                value = value[subkey]
            else:
                value = None
                break

        if value is not None:
            if as_type is not None:
                value = as_type(value)
            return value
        elif required:
            raise ValueError(f'config file error. Missing "config.process[{self.process_id}].{key}" key')
        else:
            if as_type is not None and default is not None:
                return as_type(default)
            return default

    def run(self):
        # implement in child class
        # be sure to call super().run() first incase something is added here
        pass

    def add_meta(self, additional_meta: OrderedDict):
        self.meta.update(additional_meta)


from jobs import BaseJob
