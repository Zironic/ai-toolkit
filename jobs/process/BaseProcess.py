import copy
import json
from collections import OrderedDict
from toolkit.print import print_acc
from toolkit.timer import Timer


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
        # Add a composite hook that prints aggregated ControlNet vs Model timings
        # so users can quickly see how heavy ControlNet work is relative to model work.
        self.timer.add_after_print_hook(self._print_timer_composites)
        self.performance_log_every = self.get_conf('performance_log_every', 0)

        print(json.dumps(self.config, indent=4))

    def _print_timer_composites(self, timing_dict):
        """Print a concise composite summary for ControlNet and Model timings.
        timing_dict maps timer_name -> average_seconds.
        """

        control_keys = [
            'get_adapter_images', 'encode_adapter', 'encode_adapter_embeds', 'use_precomputed_control_residuals', 'get_mask_multiplier'
        ]
        model_keys = [
            'predict_unet', 'calculate_loss', 'backward', 'optimizer_step', 'ema_update'
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
