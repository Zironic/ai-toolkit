"""
Z-Image Turbo VRAM calculator.
Estimates resolution-dependent VRAM consumption for perceptual LoKr training.
Fixed model weights (~11.5 GB quantised transformer + ~500 MB misc) are NOT included.

Usage:
    python scripts/vram_calc_zimage.py
    python scripts/vram_calc_zimage.py --batch 2 --resolutions 512 768 1024 1280
"""

import argparse

# ── Z-Image Turbo architecture ────────────────────────────────────────────────
VAE_SPATIAL     = 8      # AutoencoderKL: image → latent spatial downscale
LATENT_CH       = 16     # latent channels
PATCH_SIZE      = 2      # transformer patch size (in latent space)
TOKEN_STRIDE    = VAE_SPATIAL * PATCH_SIZE   # = 16: pixels per token edge
DIM             = 3840   # transformer hidden dim
N_HEADS         = 30
N_LAYERS        = 30     # main layers (+ 2 refiner layers, minor)
HEAD_DIM        = DIM // N_HEADS   # = 128

# VAE anchor encoder (Flux 2 VAE encoder used as perceptual discriminator)
# Multi-scale features captured at pixel-space resolutions.
# ch_mult = [1,2,4,4], base_ch = 128 → channels: 128, 256, 512, 512, 512(mid)
# Spatial: H, H/2, H/4, H/8, H/8
VAE_ANCHOR_LEVELS = [
    (128, 1),    # level 0: 128ch at full pixel res
    (256, 2),    # level 1: 256ch at H/2
    (512, 4),    # level 2: 512ch at H/4
    (512, 8),    # level 3: 512ch at H/8
    (512, 8),    # mid block: 512ch at H/8
]

BF16  = 2   # bytes
FP32  = 4   # bytes

def mb(n_bytes): return n_bytes / 1024**2
def gb(n_bytes): return n_bytes / 1024**3
def fmt(n_bytes):
    if n_bytes >= 1024**3 * 0.1:
        return f"{gb(n_bytes):.3f} GB"
    return f"{mb(n_bytes):.0f} MB"


def calc(H: int, W: int, batch: int, grad_ckpt: bool = True, perceptual: bool = True):
    results = {}

    # ── Latent ───────────────────────────────────────────────────────────────
    lH, lW   = H // VAE_SPATIAL, W // VAE_SPATIAL
    latent_b = batch * LATENT_CH * lH * lW * BF16
    results["latent"] = latent_b

    # ── Sequence length ───────────────────────────────────────────────────────
    seq_len = (lH // PATCH_SIZE) * (lW // PATCH_SIZE)
    results["seq_len"] = seq_len

    # ── Transformer activations ───────────────────────────────────────────────
    # With flash attention: Q,K,V stored per layer (no score matrix).
    # Per-layer boundary tensor: batch × seq_len × DIM (bf16).
    # With grad ckpt: store only segment inputs, recompute internals during bwd.
    # Without grad ckpt: store Q,K,V + FFN intermediate (~4×DIM) per layer.
    layer_boundary = batch * seq_len * DIM * BF16

    if grad_ckpt:
        # Only the input to each layer is kept; internals are recomputed.
        # + 2× overhead for recompute scratch during backward (rough).
        attn_b = N_LAYERS * layer_boundary
    else:
        # Q, K, V (3×DIM) + FFN intermediate (4×DIM) per layer = 7×DIM
        attn_b = N_LAYERS * batch * seq_len * (7 * DIM) * BF16

    results["transformer_activations"] = attn_b

    # ── VAE decode → pixel tensor (for perceptual loss) ───────────────────────
    if perceptual:
        pixel_b = batch * 3 * H * W * FP32
        results["vae_pixel_decode"] = pixel_b

        # ── VAE anchor encoder features (multi-scale, float32) ────────────────
        anchor_b = 0
        for ch, div in VAE_ANCHOR_LEVELS:
            anchor_b += batch * ch * (H // div) * (W // div) * FP32
        results["vae_anchor_features"] = anchor_b
    else:
        results["vae_pixel_decode"]    = 0
        results["vae_anchor_features"] = 0

    # ── Gradients for LoKr weights (same size as the LoKr params, ~324 MB) ───
    # Fixed regardless of resolution — not included here (it's in the snapshot).

    total = sum(v for k, v in results.items() if k != "seq_len")
    results["total_scalable"] = total
    return results


def print_table(resolutions, batch, grad_ckpt, perceptual):
    header_cols = ["Resolution", "Tokens", "Latent", "Transformer acts",
                   "VAE decode", "Anchor feats", "Total scalable"]
    col_w = [12, 8, 10, 18, 12, 14, 16]

    def row(*cells):
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, col_w))

    print()
    print(f"Z-Image Turbo VRAM — batch={batch}, "
          f"grad_ckpt={'on' if grad_ckpt else 'off'}, "
          f"perceptual={'on' if perceptual else 'off'}")
    print("Fixed model weights NOT included (~11.5 GB quantised transformer + misc)")
    print()
    print(row(*header_cols))
    print("  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1)))

    for res in resolutions:
        if isinstance(res, int):
            H = W = res
        else:
            H, W = res

        r = calc(H, W, batch, grad_ckpt, perceptual)
        print(row(
            f"{W}×{H}",
            f"{r['seq_len']:,}",
            fmt(r['latent']),
            fmt(r['transformer_activations']),
            fmt(r['vae_pixel_decode']),
            fmt(r['vae_anchor_features']),
            fmt(r['total_scalable']),
        ))
    print()


def parse_resolution(s):
    """Accept '1024', '1024x768', or '1024×768'."""
    s = s.replace("×", "x")
    if "x" in s:
        w, h = s.split("x")
        return (int(h), int(w))
    return int(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VRAM calculator for Z-Image Turbo perceptual training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Resolution formats: 1024  or  1024x768  (WxH)"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--resolutions", type=str, nargs="+",
                        default=["512", "640", "768", "896", "1024", "1280", "1536"])
    parser.add_argument("--no-grad-ckpt", action="store_true")
    parser.add_argument("--no-perceptual", action="store_true")
    args = parser.parse_args()

    resolutions = [parse_resolution(r) for r in args.resolutions]
    grad_ckpt   = not args.no_grad_ckpt
    perceptual  = not args.no_perceptual

    print_table(resolutions=resolutions, batch=args.batch,
                grad_ckpt=grad_ckpt, perceptual=perceptual)

    # Batch scaling at the first listed resolution
    ref = resolutions[0]
    H, W = (ref, ref) if isinstance(ref, int) else ref
    label = f"{W}×{H}"
    print(f"Batch scaling at {label}:")
    for b in [1, 2, 4, 8]:
        r = calc(H, W, b, grad_ckpt, perceptual)
        print(f"  batch={b}:  tokens={r['seq_len']:,}  "
              f"transformer={gb(r['transformer_activations']):.2f} GB  "
              f"anchor={gb(r['vae_anchor_features']):.2f} GB  "
              f"total_scalable={gb(r['total_scalable']):.2f} GB")
    print()
