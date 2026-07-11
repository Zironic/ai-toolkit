from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMMUTABLE = ROOT / "extensions_built_in/diffusion_models/krea2/src/immutable_arena.py"
MMDIT = ROOT / "extensions_built_in/diffusion_models/krea2/src/mmdit.py"
KREA2 = ROOT / "extensions_built_in/diffusion_models/krea2/krea2.py"
EAGER_TEST = ROOT / "tests/test_krea2_immutable_arena.py"
SELF = Path(__file__).resolve()
WORKFLOW = ROOT / ".github/workflows/gpt-experiments-cleanup.yml"


class CleanupError(RuntimeError):
    pass


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, source: str) -> None:
    ast.parse(source, filename=str(path))
    path.write_text(source, encoding="utf-8", newline="\n")


def remove_top_level_class(source: str, class_name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            lines = source.splitlines(keepends=True)
            start = node.lineno - 1
            end = node.end_lineno
            while end < len(lines) and not lines[end].strip():
                end += 1
            return "".join(lines[:start] + lines[end:])
    raise CleanupError(f"missing class {class_name}")


def remove_class_methods(source: str, class_name: str, method_names: set[str]) -> str:
    tree = ast.parse(source)
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ),
        None,
    )
    if target is None:
        raise CleanupError(f"missing class {class_name}")
    nodes = [
        node
        for node in target.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    found = {node.name for node in nodes}
    if found != method_names:
        raise CleanupError(f"missing methods: {sorted(method_names - found)}")
    lines = source.splitlines(keepends=True)
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        start = node.lineno - 1
        end = node.end_lineno
        while end < len(lines) and not lines[end].strip():
            end += 1
        del lines[start:end]
    return "".join(lines)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise CleanupError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def cleanup_immutable() -> None:
    source = read(IMMUTABLE)
    source = remove_top_level_class(source, "KreaImmutableArenaAdapter")
    source = replace_once(
        source,
        "``KreaImmutableArenaAdapter`` is the Slice 4 eager path (one adapter per\n"
        "residency-plan fingerprint). ``KreaImmutablePlanExecutor`` is the Slice 5\n"
        "compiled path: separate train/sample callables over ONE canonical arena,\n",
        "``KreaImmutablePlanExecutor`` provides separate train/sample callables over ONE\n"
        "canonical arena,\n",
        "immutable module docstring",
    )
    write(IMMUTABLE, source)


def cleanup_mmdit() -> None:
    source = read(MMDIT)
    source = remove_class_methods(
        source,
        "SingleStreamDiT",
        {"enable_immutable_arena_eager", "disable_immutable_arena_eager"},
    )
    source = replace_once(
        source,
        "        # Slice 4 eager-only immutable-arena functional execution. Compile and\n"
        "        # train/sample phase-callable caching are intentionally deferred to\n"
        "        # Slice 5; this adapter captures exactly one residency fingerprint.\n"
        "        self._immutable_arena_adapter = None\n",
        "",
        "eager adapter state",
    )
    source = replace_once(
        source,
        "        immutable_adapter = self._immutable_arena_adapter\n"
        "        if (\n"
        "            immutable_adapter is not None\n"
        "            and torch.is_grad_enabled()\n"
        "            and not reference_mode\n"
        "        ):\n"
        "            return immutable_adapter.forward_blocks(combined, tvec, freqs, mask)\n",
        "",
        "eager adapter routing",
    )
    source = source.replace(
        "        # Slice 5 (IMMUTABLE_TRANSFER_ARENA_PLAN.md): compiled train/sample\n"
        "        # plan executor over one canonical arena. Takes precedence over the\n"
        "        # eager adapter in _blocks_trunk when both exist.\n",
        "        # Immutable compiled train/sample executor over one canonical arena.\n",
    )
    write(MMDIT, source)


def cleanup_krea2_attach() -> None:
    source = read(KREA2)
    source = source.replace(
        '        """Build Krea\'s canonical block arena and attach singleton streaming.\n\n'
        "        Sequencing is strict: load/quantize -> freeze -> canonicalize -> legacy\n"
        "        singleton attach. LoRA/optimizer construction happens later in the\n",
        '        """Build Krea\'s canonical block arena with permanent singletons.\n\n'
        "        Sequencing is strict: load/quantize -> freeze -> canonicalize. Noncanonical\n"
        "        layers remain ordinary permanent residents; only canonical transformer\n"
        "        leaves participate in runtime residency. LoRA/optimizer construction happens later in the\n",
    )
    marker = (
        "        canonical_modules = []\n"
        "        for entries in entries_by_block.values():\n"
        "            for _name, child in entries:\n"
        "                child._mm_canonical_leaf = True\n"
        "                canonical_modules.append(child)\n\n"
    )
    replacement = marker + (
        "        canonical_ids = {id(child) for child in canonical_modules}\n"
        "        permanent_modules = [\n"
        "            child\n"
        "            for child in transformer.modules()\n"
        "            if id(child) not in canonical_ids\n"
        "            and isinstance(child, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d))\n"
        "        ]\n"
        "        immutable_ignore_modules = list(\n"
        "            dict.fromkeys([*(ignore_modules or ()), *permanent_modules])\n"
        "        )\n\n"
    )
    source = replace_once(source, marker, replacement, "permanent singleton collection")
    source = replace_once(
        source,
        "                ignore_modules=ignore_modules,\n",
        "                ignore_modules=immutable_ignore_modules,\n",
        "immutable ignore modules",
    )
    source = replace_once(
        source,
        "        transformer._mm_immutable_canonical_modules = tuple(canonical_modules)\n"
        "        transformer._mm_immutable_backend = True\n",
        "        transformer._mm_immutable_canonical_modules = tuple(canonical_modules)\n"
        "        transformer._mm_immutable_permanent_modules = tuple(permanent_modules)\n"
        "        transformer._mm_immutable_backend = True\n",
        "permanent singleton diagnostics",
    )
    write(KREA2, source)


def main() -> None:
    cleanup_immutable()
    cleanup_mmdit()
    cleanup_krea2_attach()
    if EAGER_TEST.exists():
        EAGER_TEST.unlink()
    if WORKFLOW.exists():
        WORKFLOW.unlink()
    SELF.unlink()


if __name__ == "__main__":
    main()
