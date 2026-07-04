import gc
import glob
import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import weakref

import torch
from safetensors.torch import load_file

from jobs.process import BaseExtensionProcess
from toolkit.config_modules import GenerateImageConfig, ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.train_tools import get_torch_dtype
from toolkit.util.get_model import get_model_class


DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight")
ALPHA_SUFFIXES = (".alpha",)
DIFF_SUFFIXES = (".diff", ".diff_b")


def _strip_suffix(key: str, suffixes: Tuple[str, ...]) -> Optional[str]:
    for suffix in suffixes:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    return None


def _rank_from_down(weight: torch.Tensor) -> int:
    return int(weight.shape[0])


def _rank_from_up(weight: torch.Tensor) -> int:
    if weight.ndim < 2:
        raise ValueError(f"LoRA up weight must have at least 2 dims, got {tuple(weight.shape)}")
    return int(weight.shape[1])


def _rank_norms_down(weight: torch.Tensor) -> List[float]:
    return torch.linalg.vector_norm(weight.float().reshape(weight.shape[0], -1), dim=1).tolist()


def _rank_norms_up(weight: torch.Tensor) -> List[float]:
    moved = weight.float().movedim(1, 0).reshape(weight.shape[1], -1)
    return torch.linalg.vector_norm(moved, dim=1).tolist()


def _canonical_projector_name(name: str) -> str:
    for prefix in ("diffusion_model.", "transformer."):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("text_fusion.", "txtfusion.")


def _dequantize_weight_if_needed(weight: torch.Tensor) -> torch.Tensor:
    if hasattr(weight, "dequantize"):
        return weight.dequantize()
    return weight


def _cosine(a, b) -> Optional[float]:
    a = torch.tensor(a, dtype=torch.float32).reshape(-1)
    b = torch.tensor(b, dtype=torch.float32).reshape(-1)
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom <= 0:
        return None
    return float(torch.dot(a, b) / denom)


def _vector_stats(vector) -> OrderedDict:
    t = torch.tensor(vector, dtype=torch.float32).reshape(-1)
    abs_t = t.abs()
    top = torch.argsort(abs_t, descending=True)[: min(6, t.numel())]
    return OrderedDict([
        ("numel", int(t.numel())),
        ("norm", float(torch.linalg.vector_norm(t))),
        ("mean", float(t.mean())),
        ("max_abs", float(abs_t.max()) if t.numel() else 0.0),
        ("top_slots", [
            OrderedDict([
                ("slot", int(i.item()) + 1),
                ("name", f"V{int(i.item()) + 1}"),
                ("value", float(t[i])),
                ("abs", float(abs_t[i])),
            ])
            for i in top
        ]),
    ])


BLOCKED_PROMPT_TERMS = (
    "child", "children", "minor", "teen", "teenage", "young", "girl", "boy",
    "schoolgirl", "schoolboy", "school", "student", "childlike", "underage",
)


def _validate_adult_prompt(prompt: str):
    lowered = prompt.lower()
    if "adult" not in lowered and "age " not in lowered:
        raise ValueError(f"Probe prompt must explicitly specify adult age: {prompt!r}")
    blocked = [term for term in BLOCKED_PROMPT_TERMS if term in lowered]
    if blocked:
        raise ValueError(f"Probe prompt contains blocked ambiguous/youth term(s) {blocked}: {prompt!r}")


def _slot_basis_axes(dim: int = 12) -> List[OrderedDict]:
    axes = []
    for idx in range(dim):
        vector = [0.0] * dim
        vector[idx] = 1.0
        axes.append(OrderedDict([
            ("label", f"slot V{idx + 1}"),
            ("source", "slot_basis"),
            ("vector", vector),
        ]))
    return axes


def extract_projector_vectors(state_dict: Dict[str, torch.Tensor]) -> List[OrderedDict]:
    vectors = []
    for key, value in state_dict.items():
        base = _strip_suffix(key, DIFF_SUFFIXES)
        if base is not None and value.shape == (1, 12):
            vectors.append(OrderedDict([
                ("source", key),
                ("module", _canonical_projector_name(base)),
                ("vector", [float(x) for x in value.float().reshape(-1).tolist()]),
            ]))

    down_by_base = {}
    up_by_base = {}
    for key, value in state_dict.items():
        base = _strip_suffix(key, DOWN_SUFFIXES)
        if base is not None:
            down_by_base[base] = (key, value)
            continue
        base = _strip_suffix(key, UP_SUFFIXES)
        if base is not None:
            up_by_base[base] = (key, value)

    for base, (down_key, down) in down_by_base.items():
        if base not in up_by_base:
            continue
        up_key, up = up_by_base[base]
        if _canonical_projector_name(base) != "txtfusion.projector":
            continue
        if down.ndim == 2 and up.ndim == 2:
            baked = up.float() @ down.float()
            if baked.shape == (1, 12):
                vectors.append(OrderedDict([
                    ("source", f"{up_key} @ {down_key}"),
                    ("module", "txtfusion.projector"),
                    ("vector", [float(x) for x in baked.reshape(-1).tolist()]),
                ]))

    return vectors


class ProjectorDiffController:
    def __init__(self, projector: torch.nn.Module):
        self.projector_ref = weakref.ref(projector)
        self.original_forward = projector.forward
        self.vector = None
        self.strength = 0.0
        projector.forward = self.forward

    def set_vector(self, vector, strength: float):
        self.vector = None if vector is None else torch.tensor(vector, dtype=torch.float32)
        self.strength = float(strength)

    def forward(self, x):
        projector = self.projector_ref()
        if projector is None or self.vector is None or self.strength == 0.0:
            return self.original_forward(x)

        weight = _dequantize_weight_if_needed(projector.weight)
        vector = self.vector.to(device=weight.device, dtype=weight.dtype).view_as(weight)
        effective_weight = (weight + vector * self.strength).to(device=x.device, dtype=x.dtype)
        bias = getattr(projector, "bias", None)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.linear(x, effective_weight, bias)

    def restore(self):
        projector = self.projector_ref()
        if projector is not None:
            projector.forward = self.original_forward


def inspect_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    modules = OrderedDict()
    alphas = {}

    for key, value in state_dict.items():
        base = _strip_suffix(key, DOWN_SUFFIXES)
        if base is not None:
            modules.setdefault(base, OrderedDict())["down_key"] = key
            modules[base]["down_shape"] = list(value.shape)
            modules[base]["rank"] = _rank_from_down(value)
            modules[base]["down_norms"] = _rank_norms_down(value)
            continue

        base = _strip_suffix(key, UP_SUFFIXES)
        if base is not None:
            modules.setdefault(base, OrderedDict())["up_key"] = key
            modules[base]["up_shape"] = list(value.shape)
            modules[base]["up_rank"] = _rank_from_up(value)
            modules[base]["up_norms"] = _rank_norms_up(value)
            continue

        base = _strip_suffix(key, ALPHA_SUFFIXES)
        if base is not None:
            try:
                alphas[base] = float(value.float().item())
            except Exception:
                alphas[base] = None
            continue

        base = _strip_suffix(key, DIFF_SUFFIXES)
        if base is not None:
            flat = value.float().reshape(-1)
            modules.setdefault(base, OrderedDict())["diff_key"] = key
            modules[base]["diff_shape"] = list(value.shape)
            modules[base]["diff_numel"] = int(value.numel())
            modules[base]["diff_norm"] = float(torch.linalg.vector_norm(flat))
            if value.numel() <= 64:
                modules[base]["diff_values"] = [float(x) for x in flat.tolist()]
            continue

    ranks = {}
    complete = 0
    diff_modules = 0
    for base, info in modules.items():
        if base in alphas:
            info["alpha"] = alphas[base]
        if "diff_key" in info:
            diff_modules += 1
        if "rank" in info and "up_rank" in info:
            complete += 1
            if info["rank"] != info["up_rank"]:
                info["rank_mismatch"] = True
            rank = int(info["rank"])
            ranks[str(rank)] = ranks.get(str(rank), 0) + 1
            down_norms = info.get("down_norms", [])
            up_norms = info.get("up_norms", [])
            info["component_norms"] = [d * u for d, u in zip(down_norms, up_norms)]

    projector_vectors = extract_projector_vectors(state_dict)
    return OrderedDict([
        ("module_count", len(modules)),
        ("complete_module_count", complete),
        ("diff_module_count", diff_modules),
        ("projector_vector_count", len(projector_vectors)),
        ("projector_vectors", projector_vectors),
        ("rank_histogram", ranks),
        ("modules", modules),
    ])


class LoraVectorExploreProcess(BaseExtensionProcess):
    def __init__(self, process_id: int, job, config: OrderedDict):
        super().__init__(process_id, job, config)
        self.lora_path = self.get_conf("lora.path", self.get_conf("lora_path", None), required=True)
        self.output_dir = self.get_conf("output.dir", self.get_conf("output_folder", "output/lora_vector_explore"))
        self.inspect_only = self.get_conf("inspect_only", default=False, as_type=bool)
        self.device = self.get_conf("device", getattr(self.job, "device", "cuda"))
        self.dtype = self.get_conf("dtype", "float16")
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.model_config_raw = self.get_conf("model", None)
        self.sample_config = self.get_conf("sample", None)
        self.prompt_config = self.get_conf("prompt", None)
        self.network_config_raw = self.get_conf("network", None)
        self.vector_recipe = self.get_conf("vector_recipe", None)
        self.recipes = self.get_conf("recipes", None)
        self.sweep = self.get_conf("sweep", None)
        self.projector_probe = self.get_conf("projector_probe", None)
        self.txtfusion_probe = self.get_conf("txtfusion_probe", None)

    def _projector_probe_config(self):
        if self.projector_probe is None:
            return {"enabled": False}
        if isinstance(self.projector_probe, bool):
            return {"enabled": self.projector_probe}
        return dict(self.projector_probe)

    def _load_axis_lora_vectors(self, patterns: List[str]) -> List[OrderedDict]:
        axes = []
        seen = set()
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                if path in seen or not path.endswith(".safetensors"):
                    continue
                seen.add(path)
                try:
                    vectors = extract_projector_vectors(load_file(path))
                except Exception as error:
                    axes.append(OrderedDict([
                        ("label", os.path.basename(path)),
                        ("source", path),
                        ("error", str(error)),
                    ]))
                    continue
                for idx, item in enumerate(vectors):
                    label = os.path.splitext(os.path.basename(path))[0]
                    if len(vectors) > 1:
                        label = f"{label}#{idx}"
                    axes.append(OrderedDict([
                        ("label", label),
                        ("source", item["source"]),
                        ("path", path),
                        ("vector", item["vector"]),
                    ]))
        return axes

    def _configured_projector_axes(self, config: Dict) -> List[OrderedDict]:
        axes = []
        if config.get("include_slot_basis", True):
            axes.extend(_slot_basis_axes())

        for item in config.get("axes", []) or []:
            if "vector" not in item:
                continue
            axes.append(OrderedDict([
                ("label", item.get("label", item.get("name", "configured_axis"))),
                ("source", item.get("source", "config")),
                ("vector", [float(x) for x in item["vector"]]),
            ]))

        patterns = config.get("axis_loras", None)
        if patterns is None and config.get("include_lora_folder_axes", True):
            patterns = ["loras/krea_vector_explore/*.safetensors"]
        if isinstance(patterns, str):
            patterns = [patterns]
        if patterns:
            axes.extend(self._load_axis_lora_vectors(patterns))
        return axes

    def _run_projector_probe(self, state_dict: Dict[str, torch.Tensor]) -> Optional[OrderedDict]:
        config = self._projector_probe_config()
        if not config.get("enabled", False):
            return None

        vectors = extract_projector_vectors(state_dict)
        if not vectors:
            raise ValueError("projector_probe requested, but no txtfusion.projector vector was found")

        strengths = config.get("strengths", None)
        if strengths is None:
            if self.sweep is not None and self.sweep.get("parameter", "global_strength") == "global_strength":
                strengths = self.sweep.get("values", [1.0])
            else:
                strengths = [config.get("strength", 1.0)]
        strengths = [float(x) for x in strengths]

        axes = self._configured_projector_axes(config)
        top_k = int(config.get("top_k", 12))
        source_path = os.path.normpath(self.lora_path)
        report_vectors = []

        for vector_info in vectors:
            vector = [float(x) for x in vector_info["vector"]]
            similarities = []
            for axis in axes:
                if "vector" not in axis:
                    similarities.append(axis)
                    continue
                axis_path = os.path.normpath(axis.get("path", "")) if axis.get("path") else None
                if axis_path == source_path and axis.get("source") == vector_info.get("source"):
                    continue
                cosine = _cosine(vector, axis["vector"])
                if cosine is None:
                    continue
                similarities.append(OrderedDict([
                    ("label", axis["label"]),
                    ("source", axis.get("source")),
                    ("path", axis.get("path")),
                    ("cosine", cosine),
                ]))

            similarities = [s for s in similarities if "cosine" in s]
            similarities.sort(key=lambda item: abs(item["cosine"]), reverse=True)

            strength_reports = []
            for strength in strengths:
                effective_delta = [x * strength for x in vector]
                strength_reports.append(OrderedDict([
                    ("strength", strength),
                    ("effective_delta", effective_delta),
                    ("stats", _vector_stats(effective_delta)),
                ]))

            report_vectors.append(OrderedDict([
                ("source", vector_info["source"]),
                ("module", vector_info["module"]),
                ("vector", vector),
                ("stats", _vector_stats(vector)),
                ("strengths", strength_reports),
                ("nearest_axes", similarities[:top_k]),
            ]))

        report = OrderedDict([
            ("lora_path", self.lora_path),
            ("axis_count", len([axis for axis in axes if "vector" in axis])),
            ("vectors", report_vectors),
        ])
        output_path = os.path.join(self.output_dir, "projector_probe.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {output_path}")
        return report

    def _print_inspection(self, inspection: OrderedDict):
        print(f"LoRA modules: {inspection['complete_module_count']} complete / {inspection['module_count']} total")
        print(f"Full-diff modules: {inspection.get('diff_module_count', 0)}")
        print(f"Projector vectors: {inspection.get('projector_vector_count', 0)}")
        print(f"Rank histogram: {dict(inspection['rank_histogram'])}")
        for name, info in list(inspection["modules"].items())[:50]:
            if "diff_key" in info:
                print(f"{name}: diff_shape={info.get('diff_shape')} diff_norm={info.get('diff_norm')}")
                continue
            rank = info.get("rank", "?")
            alpha = info.get("alpha", None)
            print(f"{name}: rank={rank} alpha={alpha} down={info.get('down_shape')} up={info.get('up_shape')}")
        if inspection["module_count"] > 50:
            print(f"... {inspection['module_count'] - 50} more modules in inspection.json")

    def _default_rank(self, inspection: OrderedDict) -> int:
        histogram = inspection["rank_histogram"]
        if not histogram:
            raise ValueError("No complete LoRA up/down module pairs found")
        return int(max(histogram.items(), key=lambda item: item[1])[0])

    def _recipe_gates(self, recipe):
        if recipe is None:
            return None
        if "values" in recipe:
            return {"global": recipe["values"]}
        if "global" in recipe or "per_module" in recipe:
            return {
                key: value
                for key, value in recipe.items()
                if key in ("global", "per_module")
            }
        return recipe

    def _recipe_projector_vector(self, source_vectors, recipe):
        if recipe is not None and recipe.get("mode") in ("projector_diff", "projector_vector") and "values" in recipe:
            return [float(x) for x in recipe["values"]]
        if not source_vectors:
            return None
        return source_vectors[0]["vector"]

    def _expanded_recipes(self):
        if self.recipes is not None:
            return self.recipes

        base = self.vector_recipe or {}
        has_recipe_values = "values" in base or "global" in base or "per_module" in base
        if not has_recipe_values and self.sweep is None:
            return [{"name": "ungated", "network_multiplier": 1.0, "gates": None}]

        values = [1.0]
        parameter = "global_strength"
        if self.sweep is not None:
            parameter = self.sweep.get("parameter", parameter)
            values = self.sweep.get("values", values)

        out = []
        for value in values:
            recipe = {
                "name": f"{parameter}_{value}",
                "network_multiplier": float(value) if parameter == "global_strength" else 1.0,
                "gates": self._recipe_gates(base),
            }
            out.append(recipe)
        return out

    def _find_projector(self, sd):
        target_model = getattr(sd, "unet", None) or getattr(sd, "model", None)
        if target_model is None:
            raise ValueError("Loaded model does not expose unet/model for projector vector exploration")
        target = target_model
        for part in "txtfusion.projector".split("."):
            if not hasattr(target, part):
                raise ValueError("Loaded model does not have txtfusion.projector")
            target = getattr(target, part)
        return target

    def _build_network(self, sd, rank: int, state_dict: Dict[str, torch.Tensor]):
        raw = dict(self.network_config_raw or {})
        raw.setdefault("type", "lora")
        raw.setdefault("linear", rank)
        raw.setdefault("linear_alpha", rank)
        raw.setdefault("transformer_only", getattr(sd, "is_transformer", False))
        network_config = NetworkConfig(**raw)

        network_kwargs = dict(getattr(network_config, "network_kwargs", {}) or {})
        if hasattr(sd, "target_lora_modules"):
            network_kwargs.setdefault("target_lin_modules", sd.target_lora_modules)

        network = LoRASpecialNetwork(
            text_encoder=getattr(sd, "text_encoder", None),
            unet=getattr(sd, "unet", None) or getattr(sd, "model", None),
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=getattr(sd, "is_transformer", False),
            base_model=sd,
            **network_kwargs,
        )
        target_model = getattr(sd, "unet", None) or getattr(sd, "model", None)
        network.apply_to(getattr(sd, "text_encoder", None), target_model, apply_text_encoder=False, apply_unet=True)
        network.force_to(torch.device(self.device), dtype=self.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(state_dict)
        network.can_merge_in = False
        sd.network = network
        return network

    def _sample_settings(self):
        raw = self.sample_config or {}
        prompt_raw = self.prompt_config or {}
        prompts = raw.get("prompts", None)
        if prompts is None and "samples" in raw:
            prompts = [sample.get("prompt", "") for sample in raw["samples"]]
        if prompts is None:
            text = prompt_raw.get("text", "")
            prompts = [text] if isinstance(text, str) else text
        if isinstance(prompts, str):
            prompts = [prompts]
        return {
            "prompts": prompts,
            "negative_prompt": raw.get("neg", prompt_raw.get("negative", "")),
            "width": int(raw.get("width", prompt_raw.get("width", 1024))),
            "height": int(raw.get("height", prompt_raw.get("height", 1024))),
            "steps": int(raw.get("sample_steps", prompt_raw.get("steps", 20))),
            "guidance_scale": float(raw.get("guidance_scale", prompt_raw.get("cfg", 3.5))),
            "guidance_rescale": float(raw.get("guidance_rescale", prompt_raw.get("guidance_rescale", 0.0))),
            "seed": int(raw.get("seed", prompt_raw.get("seed", 0))),
            "sampler": raw.get("sampler", prompt_raw.get("sampler", "ddpm")),
            "ext": raw.get("format", raw.get("ext", "png")),
            "batch_cfg": bool(raw.get("batch_cfg", prompt_raw.get("batch_cfg", False))),
        }

    def _build_image_configs(self, settings, recipe_idx: int, recipe_name: str, network_multiplier: float):
        configs = []
        records = []
        for prompt_idx, prompt in enumerate(settings["prompts"]):
            output_path = os.path.join(self.output_dir, f"{recipe_idx:03d}_{prompt_idx:03d}_{recipe_name}.{settings['ext']}")
            configs.append(GenerateImageConfig(
                prompt=prompt,
                width=settings["width"],
                height=settings["height"],
                negative_prompt=settings["negative_prompt"],
                seed=settings["seed"],
                guidance_scale=settings["guidance_scale"],
                guidance_rescale=settings["guidance_rescale"],
                num_inference_steps=settings["steps"],
                network_multiplier=network_multiplier,
                output_path=output_path,
                output_ext=settings["ext"],
                batch_cfg=settings["batch_cfg"],
            ))
            records.append({
                "recipe": recipe_name,
                "prompt": prompt,
                "seed": settings["seed"],
                "network_multiplier": network_multiplier,
                "image_path": output_path,
            })
        return configs, records

    def _generate(self, inspection: OrderedDict, state_dict: Dict[str, torch.Tensor]):
        if self.model_config_raw is None or (self.sample_config is None and self.prompt_config is None):
            return []

        model_config = ModelConfig(**dict(self.model_config_raw))
        model_config.lora_path = None
        model_config.inference_lora_path = None
        ModelClass = get_model_class(model_config)
        sd = ModelClass(device=self.device, model_config=model_config, dtype=self.dtype)

        print("Loading model for LoRA vector exploration...")
        controller = None
        try:
            with torch.no_grad():
                sd.load_model()
                target_model = getattr(sd, "unet", None) or getattr(sd, "model", None)
                if target_model is not None:
                    target_model.eval()
                    target_model.to(torch.device(self.device), dtype=self.torch_dtype)

                settings = self._sample_settings()
                records = []
                source_vectors = extract_projector_vectors(state_dict)

                if source_vectors:
                    projector = self._find_projector(sd)
                    controller = ProjectorDiffController(projector)
                    for recipe_idx, recipe in enumerate(self._expanded_recipes()):
                        recipe_name = str(recipe.get("name", f"recipe_{recipe_idx}"))
                        strength = float(recipe.get("network_multiplier", 1.0))
                        vector = self._recipe_projector_vector(source_vectors, self.vector_recipe or recipe)
                        controller.set_vector(vector, strength)
                        configs, recipe_records = self._build_image_configs(settings, recipe_idx, recipe_name, 0.0)
                        for record in recipe_records:
                            record["projector_strength"] = strength
                            record["projector_vector"] = vector
                            record["projector_source"] = source_vectors[0]["source"]
                        records.extend(recipe_records)
                        sd.generate_images(configs, sampler=settings["sampler"])
                    controller.set_vector(None, 0.0)
                    return records

                rank = self._default_rank(inspection)
                if hasattr(sd, "convert_lora_weights_before_load"):
                    state_dict = sd.convert_lora_weights_before_load(state_dict)
                network = self._build_network(sd, rank, state_dict)

                for recipe_idx, recipe in enumerate(self._expanded_recipes()):
                    gates = recipe.get("gates", None)
                    network.set_vector_gates(gates, device=torch.device(self.device), dtype=self.torch_dtype)
                    recipe_name = str(recipe.get("name", f"recipe_{recipe_idx}"))
                    network_multiplier = float(recipe.get("network_multiplier", 1.0))
                    configs, recipe_records = self._build_image_configs(settings, recipe_idx, recipe_name, network_multiplier)
                    for record in recipe_records:
                        record["gates"] = gates
                    records.extend(recipe_records)
                    sd.generate_images(configs, sampler=settings["sampler"])

                return records
        finally:
            if controller is not None:
                controller.restore()
            del sd
            gc.collect()
            torch.cuda.empty_cache()

    def run(self):
        super().run()
        os.makedirs(self.output_dir, exist_ok=True)
        state_dict = load_file(self.lora_path)
        inspection = inspect_lora_state_dict(state_dict)

        inspection_path = os.path.join(self.output_dir, "inspection.json")
        with open(inspection_path, "w", encoding="utf-8") as f:
            json.dump(inspection, f, indent=2)
        self._print_inspection(inspection)
        print(f"Wrote {inspection_path}")
        self._run_projector_probe(state_dict)

        records = []
        if not self.inspect_only:
            records = self._generate(inspection, state_dict)

        if records:
            results_path = os.path.join(self.output_dir, "results.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump({"lora_path": self.lora_path, "results": records}, f, indent=2)
            print(f"Wrote {results_path}")