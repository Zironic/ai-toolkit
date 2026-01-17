from typing import Type, List, Union, TypedDict
import math
import torch


class BucketResolution(TypedDict):
    width: int
    height: int


# resolutions SDXL was trained on with a 1024x1024 base resolution
resolutions_1024: List[BucketResolution] = [
    # SDXL Base resolution
    {"width": 1024, "height": 1024},
    # SDXL Resolutions, widescreen
    {"width": 2048, "height": 512},
    {"width": 1984, "height": 512},
    {"width": 1920, "height": 512},
    {"width": 1856, "height": 512},
    {"width": 1792, "height": 576},
    {"width": 1728, "height": 576},
    {"width": 1664, "height": 576},
    {"width": 1600, "height": 640},
    {"width": 1536, "height": 640},
    {"width": 1472, "height": 704},
    {"width": 1408, "height": 704},
    {"width": 1344, "height": 704},
    {"width": 1344, "height": 768},
    {"width": 1280, "height": 768},
    {"width": 1216, "height": 832},
    {"width": 1152, "height": 832},
    {"width": 1152, "height": 896},
    {"width": 1088, "height": 896},
    {"width": 1088, "height": 960},
    {"width": 1024, "height": 960},
    # SDXL Resolutions, portrait
    {"width": 960, "height": 1024},
    {"width": 960, "height": 1088},
    {"width": 896, "height": 1088},
    {"width": 896, "height": 1152},  # 2:3
    {"width": 832, "height": 1152},
    {"width": 832, "height": 1216},
    {"width": 768, "height": 1280},
    {"width": 768, "height": 1344},
    {"width": 704, "height": 1408},
    {"width": 704, "height": 1472},
    {"width": 640, "height": 1536},
    {"width": 640, "height": 1600},
    {"width": 576, "height": 1664},
    {"width": 576, "height": 1728},
    {"width": 576, "height": 1792},
    {"width": 512, "height": 1856},
    {"width": 512, "height": 1920},
    {"width": 512, "height": 1984},
    {"width": 512, "height": 2048},
    # extra wides
    {"width": 8192, "height": 128},
    {"width": 128, "height": 8192},
]

def get_bucket_sizes(resolution: int = 512, divisibility: int = 8) -> List[BucketResolution]:
    # determine scaler form 1024 to resolution
    scaler = resolution / 1024

    bucket_size_list = []
    for bucket in resolutions_1024:
        # must be divisible by 8
        width = int(bucket["width"] * scaler)
        height = int(bucket["height"] * scaler)
        if width % divisibility != 0:
            width = width - (width % divisibility)
        if height % divisibility != 0:
            height = height - (height % divisibility)
        bucket_size_list.append({"width": width, "height": height})

    return bucket_size_list


def get_resolution(width, height):
    num_pixels = width * height
    # determine same number of pixels for square image
    square_resolution = int(num_pixels ** 0.5)
    return square_resolution


def get_bucket_for_image_size(
        width: int,
        height: int,
        bucket_size_list: List[BucketResolution] = None,
        resolution: Union[int, None] = None,
        divisibility: int = 8
) -> BucketResolution:

    if bucket_size_list is None and resolution is None:
        # get resolution from width and height
        resolution = get_resolution(width, height)
    if bucket_size_list is None:
        # if real resolution is smaller, use that instead
        real_resolution = get_resolution(width, height)
        resolution = min(resolution, real_resolution)
        bucket_size_list = get_bucket_sizes(resolution=resolution, divisibility=divisibility)

    # Check for exact match first
    for bucket in bucket_size_list:
        if bucket["width"] == width and bucket["height"] == height:
            return bucket

    # If exact match not found, find the closest bucket
    closest_bucket = None
    min_removed_pixels = float("inf")

    for bucket in bucket_size_list:
        scale_w = bucket["width"] / width
        scale_h = bucket["height"] / height

        # To minimize pixels, we use the larger scale factor to minimize the amount that has to be cropped.
        scale = max(scale_w, scale_h)

        new_width = int(width * scale)
        new_height = int(height * scale)

        removed_pixels = (new_width - bucket["width"]) * new_height + (new_height - bucket["height"]) * new_width

        if removed_pixels < min_removed_pixels:
            min_removed_pixels = removed_pixels
            closest_bucket = bucket

    if closest_bucket is None:
        raise ValueError("No suitable bucket found")

    return closest_bucket


# New helper: perform the same resize-and-center-crop logic used by the dataset
# loader but operate on tensors. This ensures dataset images and control images
# can be processed *by the exact same algorithm* so their final pixel dims match.
def resize_tensor_to_bucket_exact(tensor: 'torch.Tensor', target_w: int, target_h: int) -> 'torch.Tensor':
    """Resize a tensor image [B,C,H,W] or [C,H,W] so that it is scaled preserving
    aspect and then center-cropped to exactly (target_h, target_w).

    Logic mirrors the dataset's PIL-based resize+CenterCrop sequence:
    - If target width > target height: scale each image so height == target_h then
      compute width accordingly and center crop to target_w.
    - Else: scale so width == target_w then center crop to target_h.

    Returns a tensor of shape [B,C,target_h,target_w] or [C,target_h,target_w].
    """
    import torch.nn.functional as F

    was_batched = True
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
        was_batched = False
    B, C, H, W = tensor.shape

    # Determine scaling: ensure the scaled image is at least as large as the
    # requested (target_h, target_w) in both dimensions using the larger scale
    # factor. This mirrors the dataset's PIL-based resize+CenterCrop behavior
    # and avoids undersized scaled images that would result in smaller-than-
    # expected final crops.
    scale_w = target_w / float(W)
    scale_h = target_h / float(H)
    scale = max(scale_w, scale_h)

    new_w = max(1, int(math.ceil(W * scale)))
    new_h = max(1, int(math.ceil(H * scale)))

    # Resize using bilinear (dataset used PIL.BICUBIC; this preserves algorithmic parity for scale+crop)
    scaled = F.interpolate(tensor.to(torch.float32), size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Center crop to exact target (scaled dims are guaranteed >= target dims)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    cropped = scaled[:, :, top:top + target_h, left:left + target_w]

    if not was_batched:
        return cropped.squeeze(0)
    return cropped