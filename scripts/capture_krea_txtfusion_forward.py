import argparse
import json
import sys
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import SimpleModulation, DoubleSharedModulation
from extensions_built_in.diffusion_models.krea2.src.pipeline import pad_text_features, predict_velocity
from sweep_skc3vo_blank_vector import extract_projector_vector
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush


DEFAULT_MODEL = "krea/Krea-2-Raw"
DEFAULT_IMAGE = Path("loras/krea_vector_explore/test.png")
DEFAULT_OUTPUT = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_forward_capture.pt")
DEFAULT_SUMMARY = Path("loras/krea_vector_explore/txtfusion_probe/captures/krea_txtfusion_forward_capture_summary.json")
DEFAULT_FEATURES = Path("loras/krea_vector_explore/txtfusion_probe/cache/features.pt")
DEFAULT_LORA = Path("loras/krea_vector_explore/skc3vo.safetensors")
DEFAULT_PROMPT = "adult woman, age 30, fully exposed, wide shot"


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def tensor_stats(tensor: torch.Tensor) -> OrderedDict:
    t = tensor.detach().float().cpu()
    return OrderedDict(
        [
            ("shape", list(t.shape)),
            ("dtype", dtype_name(tensor.dtype)),
            ("mean", float(t.mean())),
            ("std", float(t.std(unbiased=False))),
            ("rms", float(torch.sqrt(torch.mean(t * t)))),
            ("min", float(t.min())),
            ("max", float(t.max())),
        ]
    )


def load_image_tensor(path: Path, width: int, height: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    tensor = transforms.ToTensor()(image)
    return tensor * 2.0 - 1.0


def build_model(args) -> Krea2Model:
    model_config = ModelConfig(
        name_or_path=args.model_path,
        arch="krea2",
        dtype=args.dtype,
        vae_dtype=args.vae_dtype,
        te_dtype=args.dtype,
        quantize=args.quantize,
        quantize_te=args.quantize_te,
        qtype=args.qtype,
        qtype_te=args.qtype_te,
        low_vram=args.low_vram,
        layer_offloading=args.layer_offloading,
        layer_offloading_transformer_percent=args.layer_offloading_transformer_percent,
        layer_offloading_text_encoder_percent=args.layer_offloading_text_encoder_percent,
        model_kwargs={
            "max_text_length": args.max_text_length,
            **({"checkpoint_filename": args.checkpoint_filename} if args.checkpoint_filename else {}),
            **({"text_encoder_path": args.text_encoder_path} if args.text_encoder_path else {}),
            **({"vae_path": args.vae_path} if args.vae_path else {}),
            **({"schedule_mu": args.schedule_mu} if getattr(args, "schedule_mu", None) is not None else {}),
            **({"schedule_y1": args.schedule_y1} if getattr(args, "schedule_y1", None) is not None else {}),
            **({"schedule_y2": args.schedule_y2} if getattr(args, "schedule_y2", None) is not None else {}),
            **({"schedule_min_res": args.schedule_min_res} if getattr(args, "schedule_min_res", None) is not None else {}),
            **({"schedule_max_res": args.schedule_max_res} if getattr(args, "schedule_max_res", None) is not None else {}),
        },
    )
    model = Krea2Model(device=args.device, model_config=model_config, dtype=args.dtype)
    if args.cached_prompt:
        transformer = model._load_transformer()
        # The cached_prompt path bypasses load_model(), which is normally where
        # the offload attach runs. Without this, layer offloading is silently a
        # no-op for the probe scripts (managed=0, whole quantized model resident).
        # Mirror load_model()'s percent-path attach here, before the .to(device)
        # so managed weights get pinned to CPU and only unmanaged move to GPU.
        mc = model.model_config
        if (
            mc.layer_offloading
            and not mc.layer_offloading_smart
            and mc.layer_offloading_transformer_percent > 0
        ):
            ignore_modules = [
                m for m in transformer.modules()
                if isinstance(m, (SimpleModulation, DoubleSharedModulation))
            ]
            MemoryManager.attach(
                transformer,
                model.device_torch,
                offload_percent=mc.layer_offloading_transformer_percent,
                ignore_modules=ignore_modules,
            )
        if model.model_config.low_vram:
            transformer.to("cpu")
        elif model.model_config.quantize:
            transformer.to(model.device_torch)
        else:
            transformer.to(model.device_torch, dtype=model.torch_dtype)
        vae = None
        if not args.skip_vae:
            vae = model._load_vae()
            vae.to(model.vae_device_torch, dtype=model.vae_torch_dtype)
        model.model = transformer
        model.vae = vae
        model.text_encoder = None
        model.tokenizer = None
        model.processor = None
        model.noise_scheduler = Krea2Model.get_train_scheduler(model.model_config)
        model.pipeline = None
    else:
        model.load_model()
    model.model.eval()
    model.model.requires_grad_(False)
    if model.vae is not None:
        model.vae.eval()
        model.vae.requires_grad_(False)
    if model.text_encoder is not None:
        model.text_encoder.eval()
        model.text_encoder.requires_grad_(False)
    return model


class ProjectorCapture:
    def __init__(self, projector: torch.nn.Module):
        self.projector = projector
        self.records = []
        self.handle = None

    def __enter__(self):
        def hook(module, inputs, output):
            x = inputs[0].detach().cpu()
            y = output.detach().cpu()
            self.records.append({"input": x, "output": y})

        self.handle = self.projector.register_forward_hook(hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@contextmanager
def projector_delta_forward(projector: torch.nn.Module, vector: Optional[torch.Tensor], strength: float):
    if vector is None or strength == 0.0:
        yield
        return

    original_forward = projector.forward
    v_cpu = vector.detach().float().view(1, 1, -1)

    def forward_with_delta(x):
        base = original_forward(x)
        v = v_cpu.to(device=x.device, dtype=x.dtype)
        delta = (x * v).sum(dim=-1, keepdim=True) * float(strength)
        return base + delta

    projector.forward = forward_with_delta
    try:
        yield
    finally:
        projector.forward = original_forward


def encode_prompt(model: Krea2Model, prompt: str):
    embeds = model.get_prompt_embeds(prompt)
    return pad_text_features(embeds.text_embeds, model.device_torch, model.torch_dtype)


def load_cached_prompt_features(path: Path, prompt_id: str, device: torch.device, dtype: torch.dtype):
    features_by_id = torch.load(path, map_location="cpu")
    if prompt_id not in features_by_id:
        known = ", ".join(sorted(features_by_id))
        raise ValueError(f"Prompt id {prompt_id!r} not in {path}. Known ids: {known}")
    features = features_by_id[prompt_id]
    if features.dim() != 3:
        raise ValueError(f"Expected cached features shaped (tokens, 12, hidden), got {tuple(features.shape)}")
    flattened = features.reshape(features.shape[0], -1)
    return pad_text_features([flattened], device, dtype)


def run_forward(model: Krea2Model, latents: torch.Tensor, t: torch.Tensor, context, text_mask, projector_vector, lora_strength):
    projector = model.model.txtfusion.projector
    with projector_delta_forward(projector, projector_vector, lora_strength):
        with ProjectorCapture(projector) as capture:
            with torch.no_grad():
                velocity = predict_velocity(
                    model.model,
                    latents.to(model.device_torch, model.torch_dtype),
                    t.to(model.device_torch, model.torch_dtype),
                    context,
                    text_mask,
                ).detach().cpu()
    if len(capture.records) != 1:
        raise RuntimeError(f"Expected one projector call, captured {len(capture.records)}")
    return velocity, capture.records[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture Krea2 txtfusion.projector inputs/outputs from one real noisy-latent forward.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-id", default="exposed_wide")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--cached-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--lora", type=Path, default=DEFAULT_LORA)
    parser.add_argument("--lora-strength", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--noise-t", type=float, default=0.5)
    parser.add_argument("--latent-mode", choices=["image", "random"], default="image")
    parser.add_argument("--skip-vae", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--vae-dtype", default="bf16")
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quantize-te", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qtype", default="qfloat8")
    parser.add_argument("--qtype-te", default="qfloat8")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layer-offloading-transformer-percent", type=float, default=1.0)
    parser.add_argument("--layer-offloading-text-encoder-percent", type=float, default=1.0)
    parser.add_argument("--max-text-length", type=int, default=2000)
    parser.add_argument("--checkpoint-filename", default=None)
    parser.add_argument("--text-encoder-path", default=None)
    parser.add_argument("--vae-path", default=None)
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(args.image)
    if args.lora and not args.lora.exists():
        raise FileNotFoundError(args.lora)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = build_model(args)
    device = model.device_torch

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t_value = float(args.noise_t)
    if args.latent_mode == "image":
        if model.vae is None:
            raise ValueError("--latent-mode image requires VAE; remove --skip-vae or pass a working --vae-path")
        image_tensor = load_image_tensor(args.image, args.width, args.height)
        with torch.no_grad():
            clean_latents = model.encode_images([image_tensor], device=model.vae_device_torch, dtype=model.vae_torch_dtype).detach()
        noise = torch.randn(clean_latents.shape, generator=generator, dtype=torch.float32).to(clean_latents.device)
        noisy_latents = (1.0 - t_value) * clean_latents.float() + t_value * noise.float()
    else:
        latent_channels = model.model.config.channels
        clean_latents = torch.zeros(1, latent_channels, args.height // model.vae_scale_factor, args.width // model.vae_scale_factor, dtype=torch.float32)
        noise = torch.randn(clean_latents.shape, generator=generator, dtype=torch.float32)
        noisy_latents = noise.float()
    noisy_latents = noisy_latents.to(device, dtype=model.torch_dtype)
    t = torch.full((noisy_latents.shape[0],), t_value, device=device, dtype=model.torch_dtype)

    if args.cached_prompt:
        context, text_mask = load_cached_prompt_features(args.features, args.prompt_id, model.device_torch, model.torch_dtype)
        uncond_context = uncond_mask = None
    else:
        context, text_mask = encode_prompt(model, args.prompt)
        uncond_context = uncond_mask = None
        if args.negative_prompt is not None:
            uncond_context, uncond_mask = encode_prompt(model, args.negative_prompt)

    projector_vector = None
    vector_source = None
    if args.lora:
        vector_source, projector_vector = extract_projector_vector(load_file(str(args.lora)))

    base_velocity, base_capture = run_forward(
        model,
        noisy_latents,
        t,
        context,
        text_mask,
        projector_vector=None,
        lora_strength=0.0,
    )
    lora_velocity = None
    lora_capture = None
    if projector_vector is not None and args.lora_strength != 0.0:
        lora_velocity, lora_capture = run_forward(
            model,
            noisy_latents,
            t,
            context,
            text_mask,
            projector_vector=projector_vector,
            lora_strength=args.lora_strength,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = OrderedDict(
        [
            ("schema", "krea_txtfusion_forward_capture.v1"),
            ("prompt", args.prompt),
            ("prompt_id", args.prompt_id),
            ("cached_prompt", args.cached_prompt),
            ("features", str(args.features)),
            ("negative_prompt", args.negative_prompt),
            ("model_path", args.model_path),
            ("image", str(args.image)),
            ("width", args.width),
            ("height", args.height),
            ("noise_t", t_value),
            ("latent_mode", args.latent_mode),
            ("skip_vae", args.skip_vae),
            ("seed", args.seed),
            ("dtype", args.dtype),
            ("quantize", args.quantize),
            ("qtype", args.qtype),
            ("lora", str(args.lora) if args.lora else None),
            ("lora_strength", args.lora_strength),
            ("projector_vector_source", vector_source),
            ("projector_vector", projector_vector.cpu() if projector_vector is not None else None),
            ("clean_latents", clean_latents.detach().cpu()),
            ("noise", noise.detach().cpu()),
            ("noisy_latents", noisy_latents.detach().cpu()),
            ("context", context.detach().cpu()),
            ("text_mask", text_mask.detach().cpu()),
            ("uncond_context", uncond_context.detach().cpu() if uncond_context is not None else None),
            ("uncond_mask", uncond_mask.detach().cpu() if uncond_mask is not None else None),
            ("base_projector_input", base_capture["input"]),
            ("base_projector_output", base_capture["output"]),
            ("base_velocity", base_velocity),
            ("lora_projector_input", lora_capture["input"] if lora_capture is not None else None),
            ("lora_projector_output", lora_capture["output"] if lora_capture is not None else None),
            ("lora_velocity", lora_velocity),
        ]
    )
    torch.save(payload, args.output)

    summary = OrderedDict(
        [
            ("schema", "krea_txtfusion_forward_capture_summary.v1"),
            ("capture", str(args.output)),
            ("prompt", args.prompt),
            ("prompt_id", args.prompt_id),
            ("cached_prompt", args.cached_prompt),
            ("features", str(args.features)),
            ("model_path", args.model_path),
            ("image", str(args.image)),
            ("width", args.width),
            ("height", args.height),
            ("noise_t", t_value),
            ("latent_mode", args.latent_mode),
            ("skip_vae", args.skip_vae),
            ("seed", args.seed),
            ("dtype", args.dtype),
            ("quantize", args.quantize),
            ("qtype", args.qtype),
            ("lora", str(args.lora) if args.lora else None),
            ("lora_strength", args.lora_strength),
            ("projector_vector_source", vector_source),
            ("projector_vector", [float(x) for x in projector_vector.tolist()] if projector_vector is not None else None),
            ("clean_latents", tensor_stats(clean_latents)),
            ("noise", tensor_stats(noise)),
            ("noisy_latents", tensor_stats(noisy_latents)),
            ("context", tensor_stats(context)),
            ("base_projector_input", tensor_stats(base_capture["input"])),
            ("base_projector_output", tensor_stats(base_capture["output"])),
            ("base_velocity", tensor_stats(base_velocity)),
        ]
    )
    if lora_capture is not None and lora_velocity is not None:
        projector_output_delta = lora_capture["output"].float() - base_capture["output"].float()
        velocity_delta = lora_velocity.float() - base_velocity.float()
        summary["lora_projector_input"] = tensor_stats(lora_capture["input"])
        summary["lora_projector_output"] = tensor_stats(lora_capture["output"])
        summary["lora_projector_output_delta"] = tensor_stats(projector_output_delta)
        summary["lora_velocity"] = tensor_stats(lora_velocity)
        summary["lora_velocity_delta"] = tensor_stats(velocity_delta)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")
    print(f"Projector input shape: {list(base_capture['input'].shape)}")
    print(f"Base velocity RMS: {summary['base_velocity']['rms']:.6f}")
    if "lora_velocity_delta" in summary:
        print(f"LoRA projector delta RMS: {summary['lora_projector_output_delta']['rms']:.6f}")
        print(f"LoRA velocity delta RMS: {summary['lora_velocity_delta']['rms']:.6f}")
    flush()


if __name__ == "__main__":
    main()
