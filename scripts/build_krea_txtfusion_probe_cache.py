import argparse
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoTokenizer, Qwen2TokenizerFast, Qwen3VLForConditionalGeneration

from extensions_built_in.diffusion_models.krea2.krea2 import QWEN3_VL_PATH
from extensions_built_in.diffusion_models.krea2.src.text_encoder import SELECT_LAYERS, encode_krea_prompt


DEFAULT_PROMPTS = Path("loras/krea_vector_explore/txtfusion_probe/prompts.json")
DEFAULT_OUTPUT_DIR = Path("loras/krea_vector_explore/txtfusion_probe/cache")
BLOCKED_PROMPT_TERMS = (
    "child", "children", "minor", "teen", "teenage", "young", "girl", "boy",
    "schoolgirl", "schoolboy", "school", "student", "childlike", "underage",
)


def dtype_from_name(name: str):
    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp16", "float16", "half"):
        return torch.float16
    if normalized in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def validate_adult_prompt(prompt: str):
    lowered = prompt.lower()
    if "adult" not in lowered and "age " not in lowered:
        raise ValueError(f"Prompt must explicitly specify adult age: {prompt!r}")
    blocked = [term for term in BLOCKED_PROMPT_TERMS if term in lowered]
    if blocked:
        raise ValueError(f"Prompt contains blocked ambiguous/youth term(s) {blocked}: {prompt!r}")


def load_prompt_library(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    phrases = data.get("phrases", [])
    pairs = data.get("contrast_pairs", [])
    if not phrases:
        raise ValueError(f"No phrases found in {path}")

    seen = set()
    for item in phrases:
        if "id" not in item or "prompt" not in item:
            raise ValueError(f"Phrase entries must have id and prompt: {item!r}")
        if item["id"] in seen:
            raise ValueError(f"Duplicate phrase id: {item['id']}")
        seen.add(item["id"])
        if item["id"] == "blank" and item["prompt"] == "":
            continue
        validate_adult_prompt(item["prompt"])

    for pair in pairs:
        for key in ("negative", "positive"):
            if pair.get(key) not in seen:
                raise ValueError(f"Contrast pair {pair.get('id', '?')} references unknown {key}: {pair.get(key)!r}")

    return data


def summarize_features(features: torch.Tensor) -> OrderedDict:
    # encode_krea_prompt returns (tokens, 12, hidden). Summarize over tokens and hidden.
    f = features.float()
    slot_mean = f.mean(dim=(0, 2))
    slot_rms = torch.sqrt(torch.mean(f * f, dim=(0, 2)))
    token_count, slot_count, hidden_dim = f.shape
    return OrderedDict([
        ("shape", [int(token_count), int(slot_count), int(hidden_dim)]),
        ("slot_mean", [float(x) for x in slot_mean.tolist()]),
        ("slot_rms", [float(x) for x in slot_rms.tolist()]),
    ])


def vector_diff(pos: List[float], neg: List[float]) -> List[float]:
    return [float(p - n) for p, n in zip(pos, neg)]


def build_contrast_summaries(prompt_summaries: Dict[str, OrderedDict], pairs: List[dict]) -> List[OrderedDict]:
    out = []
    for pair in pairs:
        neg = prompt_summaries[pair["negative"]]
        pos = prompt_summaries[pair["positive"]]
        out.append(OrderedDict([
            ("id", pair["id"]),
            ("label", pair.get("label", f"{pair['negative']} -> {pair['positive']}")),
            ("negative", pair["negative"]),
            ("positive", pair["positive"]),
            ("slot_mean_direction", vector_diff(pos["slot_mean"], neg["slot_mean"])),
            ("slot_rms_direction", vector_diff(pos["slot_rms"], neg["slot_rms"])),
        ]))
    return out


def load_text_encoder(path: str, dtype: torch.dtype, device: torch.device, max_length: int, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(path, max_length=max_length, local_files_only=local_files_only)
    processor = Qwen2TokenizerFast.from_pretrained(path, max_length=max_length, local_files_only=local_files_only)
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype, local_files_only=local_files_only)
    if getattr(text_encoder.model, "visual", None) is not None:
        text_encoder.model.visual = None
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    return tokenizer, processor, text_encoder


def main():
    parser = argparse.ArgumentParser(description="Build a Krea2 Qwen text-fusion probe cache.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--text-encoder-path", default=QWEN3_VL_PATH)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true", help="Validate prompt set without loading the TE.")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    prompt_library = load_prompt_library(args.prompts)
    phrases = prompt_library["phrases"]
    pairs = prompt_library.get("contrast_pairs", [])

    if args.dry_run:
        print(f"Validated {len(phrases)} adult-only prompts and {len(pairs)} contrast pairs from {args.prompts}")
        return

    dtype = dtype_from_name(args.dtype)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, processor, text_encoder = load_text_encoder(
        args.text_encoder_path,
        dtype=dtype,
        device=device,
        max_length=args.max_length,
        local_files_only=args.local_files_only,
    )

    features_by_id = {}
    prompt_summaries = OrderedDict()
    with torch.no_grad():
        for item in phrases:
            prompt_id = item["id"]
            prompt = item["prompt"]
            print(f"Encoding {prompt_id}: {prompt}")
            features = encode_krea_prompt(
                text_encoder,
                tokenizer,
                processor,
                prompt,
                max_length=args.max_length,
                select_layers=SELECT_LAYERS,
            ).detach().to("cpu")
            features_by_id[prompt_id] = features.to(torch.float16)
            prompt_summaries[prompt_id] = OrderedDict([
                ("prompt", prompt),
                *summarize_features(features).items(),
            ])

    contrast_summaries = build_contrast_summaries(prompt_summaries, pairs)
    summary = OrderedDict([
        ("schema", "krea_txtfusion_probe_cache.v1"),
        ("text_encoder_path", args.text_encoder_path),
        ("select_layers", list(SELECT_LAYERS)),
        ("prompt_source", str(args.prompts)),
        ("prompts", prompt_summaries),
        ("contrast_pairs", contrast_summaries),
    ])

    torch.save(features_by_id, args.output_dir / "features.pt")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_dir / 'features.pt'}")
    print(f"Wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
