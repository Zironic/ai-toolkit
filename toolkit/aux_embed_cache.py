"""Disk cache for the small set of "auxiliary" text embeddings that are otherwise
held only in memory by the trainer: the blank/unconditional embed, the trigger-word
embed, and the per-sample conditional/unconditional embeds used for inline sampling.

Dataset *caption* embeddings already persist to disk via the dataloader cache
(`dataloader_mixins.cache_text_embeddings`). These aux embeds did not, which meant a
separate text-encoder worker process could not hand them off to the trainer. This module
provides that handoff: the TE worker writes them here, the trainer loads them here, and
neither needs the text encoder resident at the same time as the transformer.

Layout (under a per-run save_root):

    <save_root>/.te_cache/
        manifest.json            # config hash + which entries exist + sample count
        blank.safetensors        # PromptEmbeds (optional)
        trigger.safetensors      # PromptEmbeds (optional)
        unconditional.safetensors# PromptEmbeds (optional)
        sample_000_cond.safetensors
        sample_000_uncond.safetensors
        ...

The manifest's `config_hash` invalidates the cache when the relevant config changes
(prompts, trigger word, model id, etc.). A stale or missing manifest is treated as a miss.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from toolkit.cache_utils import atomic_write, compute_param_digest
from toolkit.prompt_utils import PromptEmbeds

CACHE_DIR_NAME = ".te_cache"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


class SampleEmbedPair(TypedDict):
    conditional: PromptEmbeds
    unconditional: PromptEmbeds


class AuxEmbeds(TypedDict, total=False):
    blank: Optional[PromptEmbeds]
    trigger: Optional[PromptEmbeds]
    unconditional: Optional[PromptEmbeds]
    samples: List[SampleEmbedPair]


def aux_cache_dir(save_root: str) -> Path:
    return Path(save_root) / CACHE_DIR_NAME


def compute_aux_config_hash(params: dict) -> str:
    """Stable hash of the config inputs that determine the aux embeds.

    Pass anything that, if changed, should invalidate the cache: the model id/path,
    the trigger word, the unconditional prompt, and the ordered list of sample prompts
    (+ negatives). Reuses the project's deterministic digest helper.
    """
    return compute_param_digest(params, length=16)


def _manifest_path(save_root: str) -> Path:
    return aux_cache_dir(save_root) / MANIFEST_NAME


def _entry_path(save_root: str, name: str) -> Path:
    return aux_cache_dir(save_root) / f"{name}.safetensors"


def _sample_paths(save_root: str, index: int) -> tuple[Path, Path]:
    d = aux_cache_dir(save_root)
    return (
        d / f"sample_{index:03d}_cond.safetensors",
        d / f"sample_{index:03d}_uncond.safetensors",
    )


def save_aux_embeds(
    save_root: str,
    config_hash: str,
    *,
    blank: Optional[PromptEmbeds] = None,
    trigger: Optional[PromptEmbeds] = None,
    unconditional: Optional[PromptEmbeds] = None,
    samples: Optional[List[SampleEmbedPair]] = None,
) -> None:
    """Persist the aux embeds and a manifest. Writes are atomic per file; the manifest
    is written last so a reader only sees the cache as complete once everything is on disk.
    """
    samples = samples or []
    os.makedirs(aux_cache_dir(save_root), exist_ok=True)

    if blank is not None:
        blank.save(str(_entry_path(save_root, "blank")))
    if trigger is not None:
        trigger.save(str(_entry_path(save_root, "trigger")))
    if unconditional is not None:
        unconditional.save(str(_entry_path(save_root, "unconditional")))

    for i, pair in enumerate(samples):
        cond_path, uncond_path = _sample_paths(save_root, i)
        pair["conditional"].save(str(cond_path))
        pair["unconditional"].save(str(uncond_path))

    manifest = {
        "version": MANIFEST_VERSION,
        "config_hash": config_hash,
        "has_blank": blank is not None,
        "has_trigger": trigger is not None,
        "has_unconditional": unconditional is not None,
        "sample_count": len(samples),
    }

    def _writer(p: Path) -> None:
        with open(p, "w") as f:
            json.dump(manifest, f)

    # manifest is the completion marker — write it atomically and last
    atomic_write(_manifest_path(save_root), _writer)


def read_manifest(save_root: str) -> Optional[dict]:
    path = _manifest_path(save_root)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def aux_cache_is_complete(save_root: str, config_hash: str, expected_sample_count: int) -> bool:
    """True only if a manifest exists, its hash matches, the sample count matches, and
    every file the manifest claims is actually present on disk."""
    manifest = read_manifest(save_root)
    if manifest is None:
        return False
    if manifest.get("version") != MANIFEST_VERSION:
        return False
    if manifest.get("config_hash") != config_hash:
        return False
    if manifest.get("sample_count") != expected_sample_count:
        return False

    if manifest.get("has_blank") and not _entry_path(save_root, "blank").exists():
        return False
    if manifest.get("has_trigger") and not _entry_path(save_root, "trigger").exists():
        return False
    if manifest.get("has_unconditional") and not _entry_path(save_root, "unconditional").exists():
        return False
    for i in range(expected_sample_count):
        cond_path, uncond_path = _sample_paths(save_root, i)
        if not cond_path.exists() or not uncond_path.exists():
            return False
    return True


def load_aux_embeds(save_root: str, config_hash: str, expected_sample_count: int) -> Optional[AuxEmbeds]:
    """Load aux embeds if the cache is complete and valid for `config_hash`, else None."""
    if not aux_cache_is_complete(save_root, config_hash, expected_sample_count):
        return None
    manifest = read_manifest(save_root)
    assert manifest is not None  # guaranteed by aux_cache_is_complete

    result: AuxEmbeds = {
        "blank": None,
        "trigger": None,
        "unconditional": None,
        "samples": [],
    }
    if manifest.get("has_blank"):
        result["blank"] = PromptEmbeds.load(str(_entry_path(save_root, "blank")))
    if manifest.get("has_trigger"):
        result["trigger"] = PromptEmbeds.load(str(_entry_path(save_root, "trigger")))
    if manifest.get("has_unconditional"):
        result["unconditional"] = PromptEmbeds.load(str(_entry_path(save_root, "unconditional")))

    samples: List[SampleEmbedPair] = []
    for i in range(expected_sample_count):
        cond_path, uncond_path = _sample_paths(save_root, i)
        samples.append({
            "conditional": PromptEmbeds.load(str(cond_path)),
            "unconditional": PromptEmbeds.load(str(uncond_path)),
        })
    result["samples"] = samples
    return result
