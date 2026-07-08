import importlib
import logging


def force_hf_hub_progress_bars():
    hf_tqdm = importlib.import_module('huggingface_hub.utils.tqdm')

    original_is_tqdm_disabled = hf_tqdm.is_tqdm_disabled
    if getattr(original_is_tqdm_disabled, '_aitk_forced_progress', False):
        return

    def is_tqdm_disabled(log_level):
        disabled = original_is_tqdm_disabled(log_level)
        # UI jobs log to files, so stderr is not a TTY. Keep HF download bars
        # visible while preserving normal carriage-return progress updates.
        if disabled is None and log_level != logging.NOTSET:
            return False
        return disabled

    is_tqdm_disabled._aitk_forced_progress = True
    hf_tqdm.is_tqdm_disabled = is_tqdm_disabled


def _apply_sdpa_gqa_patch():
    # Late import: sdpa_patch pulls in torch, and this package init runs for
    # every toolkit consumer. See toolkit/sdpa_patch.py for the rationale
    # (enable_gqa silently dispatching to the MATH SDPA backend on builds
    # without Flash, e.g. all Windows torch wheels).
    from toolkit.sdpa_patch import apply_sdpa_gqa_patch
    apply_sdpa_gqa_patch()


force_hf_hub_progress_bars()
_apply_sdpa_gqa_patch()
