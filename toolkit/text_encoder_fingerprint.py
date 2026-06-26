# toolkit/text_encoder_fingerprint.py

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any, Iterable


_TEXT_ENCODER_SIDE_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "sentencepiece.bpe.model",
    "spiece.model",
    "merges.txt",
    "vocab.json",
}


_TEXT_ENCODER_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 16) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def _safetensors_header(path: Path) -> dict[str, Any] | None:
    """Read only the safetensors header.

    This gives tensor names, dtypes, shapes, and metadata without loading weights.
    It is useful for diagnostics, but it is not enough for correctness by itself:
    two different checkpoints can have identical tensor metadata.
    """
    try:
        with path.open("rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) != 8:
                return None

            header_len = struct.unpack("<Q", header_len_bytes)[0]
            header_json = f.read(header_len)
            return json.loads(header_json)
    except Exception:
        return None


def _stable_json_hash(payload: Any, length: int = 16) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")

    return hashlib.sha256(blob).hexdigest()[:length]


def _file_payload(path: Path, *, full_hash: bool = True) -> dict[str, Any]:
    stat = path.stat()

    payload: dict[str, Any] = {
        "path_name": path.name,
        "suffix": path.suffix.lower(),
        "size": stat.st_size,
    }

    if path.suffix.lower() == ".safetensors":
        header = _safetensors_header(path)
        if header is not None:
            payload["safetensors_header_hash"] = _stable_json_hash(header, length=32)

            metadata = header.get("__metadata__")
            if metadata:
                payload["safetensors_metadata"] = metadata

    if full_hash:
        payload["sha256"] = _sha256_file(path)
    else:
        # Fast but weaker. Fine for diagnostics; not ideal for correctness.
        payload["mtime_ns"] = stat.st_mtime_ns

    return payload


def _iter_text_encoder_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if not path.is_dir():
        return

    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue

        name = child.name
        suffix = child.suffix.lower()

        if name in _TEXT_ENCODER_SIDE_FILES or suffix in _TEXT_ENCODER_WEIGHT_SUFFIXES:
            yield child


def fingerprint_text_encoder_paths(
    paths: str | os.PathLike | Iterable[str | os.PathLike],
    *,
    full_hash: bool = True,
) -> str:
    """Fingerprint the actual text encoder files used for cached text embeddings.

    Pass the resolved TE file/path(s) used by the model loader.

    ``full_hash=True`` is the correctness mode. It detects same-name, same-shape,
    different-weight swaps. It costs one sequential read per file, but the result
    can be cached by the surrounding aux-cache metadata.

    ``full_hash=False`` is faster but only uses metadata/header/stat info.
    """
    if isinstance(paths, (str, os.PathLike)):
        raw_paths = [paths]
    else:
        raw_paths = list(paths)

    resolved_paths = [Path(p).expanduser().resolve() for p in raw_paths]

    payload: list[dict[str, Any]] = []

    for root in resolved_paths:
        if not root.exists():
            payload.append(
                {
                    "root": str(root),
                    "exists": False,
                }
            )
            continue

        files = list(_iter_text_encoder_files(root))

        payload.append(
            {
                "root": str(root),
                "exists": True,
                "is_file": root.is_file(),
                "is_dir": root.is_dir(),
                "files": [
                    {
                        "relative": str(file.relative_to(root.parent if root.is_file() else root)),
                        **_file_payload(file, full_hash=full_hash),
                    }
                    for file in files
                ],
            }
        )

    return f"te:{_stable_json_hash(payload, length=24)}"