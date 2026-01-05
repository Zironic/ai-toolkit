"""CLI to run adapter diagnostics without launching a full job.

Usage examples:
  # inspect adapter by model name
  python scripts/diagnose_adapter.py --model alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1 --latents-shape 1,16,112,84 --ctrl-shape 1,3,512,512 --log logs/diag_adapter.log

This script loads the model in inference mode and runs the diagnostics added in
`toolkit.control_diagnostics` to print the adapter conv channels, inferred expected_in,
and adaptation steps applied to synthetic latents and control images.
"""
import argparse
import torch
from toolkit.model_utils import load_model_for_inference
from toolkit.control_diagnostics import diagnose_adapter
from toolkit.print import setup_log_to_file, print_acc


def parse_shape(s: str):
    return tuple(int(x) for x in s.split(','))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model name_or_path to load (used to obtain adapter)')
    parser.add_argument('--latents-shape', default='1,16,112,84', help='comma-separated shape for synthetic latents')
    parser.add_argument('--ctrl-shape', default='1,3,512,512', help='comma-separated shape for synthetic control_images')
    parser.add_argument('--log', default=None, help='optional log file to capture output')
    parser.add_argument('--dtype', default='bf16', choices=['bf16','float32','float16'], help='dtype for loading model/adapter')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'], help='device for loading model - use cpu to avoid GPU allocation')
    args = parser.parse_args()

    if args.log:
        setup_log_to_file(args.log)

    print_acc(f"[DIAG-CLI] Loading model {args.model} (dtype={args.dtype}, device={args.device})")
    # Load model in requested device (default cpu) to avoid GPU allocation when not desired
    sd = load_model_for_inference(args.model, device=args.device, dtype=args.dtype, apply_lora=False)

    adapter = getattr(sd, 'controlnet', None)
    if adapter is None:
        print_acc("[DIAG-CLI] model has no controlnet adapter attached; exiting")
        return

    lat_shape = parse_shape(args.latents_shape)
    ctrl_shape = parse_shape(args.ctrl_shape)

    lat = torch.zeros(lat_shape)
    ctrl = torch.zeros(ctrl_shape)

    print_acc("[DIAG-CLI] Running diagnose_adapter...")
    diagnose_adapter(adapter, lat, ctrl)

if __name__ == '__main__':
    main()
