import torch
from torchvision.transforms import Resize


def encode_images_anima(vae, image_list: list, device, dtype) -> torch.Tensor:
    """Encode a list of (C, H, W) image tensors using the Cosmos/Qwen VAE.

    The Cosmos VAE expects 5D input (B, C, T, H, W). For single-frame images T=1.
    Normalization: normalized = (raw - latents_mean) / latents_std
    Returns 4D latents (B, C, H, W) with the time dimension squeezed for
    compatibility with the rest of the training pipeline; the training forward
    pass re-expands to 5D before the transformer call.
    """
    vae_scale = 8
    image_list = [img.to(device, dtype=dtype) for img in image_list]

    for i, img in enumerate(image_list):
        h, w = img.shape[-2], img.shape[-1]
        if h % vae_scale != 0 or w % vae_scale != 0:
            image_list[i] = Resize((h // vae_scale * vae_scale, w // vae_scale * vae_scale))(img)

    # Stack to (B, C, H, W) then add time dimension → (B, C, 1, H, W)
    images = torch.stack(image_list).unsqueeze(2)

    raw_latents = vae.encode(images).latent_dist.sample()

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(raw_latents)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(raw_latents)
    normalized = (raw_latents - latents_mean) / latents_std

    # Squeeze time dim so latents are (B, C, H, W) — consistent with other archs
    return normalized.squeeze(2).to(device, dtype=dtype)
