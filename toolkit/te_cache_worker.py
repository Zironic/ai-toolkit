"""Throwaway text-encoder cache worker.

Loads a training job's model with te_only=True (text encoder + tokenizers only — no
transformer, no VAE), encodes and persists every text embedding the trainer will need to
disk (dataset captions, DOP, and the aux blank/trigger/uncond/sample embeds), then exits.
Because it runs as its own OS process, all text-encoder VRAM/host RAM is reclaimed on exit,
so the trainer process never has the text encoder and the transformer resident at once.

Invoked by the trainer (see BaseSDTrainProcess.run) as:

    python -m toolkit.te_cache_worker <config_file> [-n NAME] [-l LOG]

Env (forwarded by the trainer): AITK_JOB_ID, CUDA_VISIBLE_DEVICES, HF_TOKEN, SEED.
"""
import os
import sys

from dotenv import load_dotenv

# Mirror run.py's pre-torch environment setup so behavior/allocator tuning matches.
load_dotenv()
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "1")
# Generous HF download read timeout (default 10s) so a flaky/slow Qwen3-VL pull in the
# worker doesn't die on a transient ReadTimeout. Overridable via .env.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
sys.path.insert(0, os.getcwd())
os.environ["DISABLE_TELEMETRY"] = "YES"
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8"
)

import argparse  # noqa: E402

from toolkit.job import get_job  # noqa: E402
from toolkit.print import print_acc, setup_log_to_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Text-encoder cache worker")
    parser.add_argument("config_file", type=str, help="Path to the job config (yaml/json)")
    parser.add_argument("-n", "--name", type=str, default=None)
    parser.add_argument("-l", "--log", type=str, default=None)
    args = parser.parse_args()

    if args.log is not None:
        setup_log_to_file(args.log)

    job = get_job(args.config_file, args.name)

    # Z-Image / Anima trainers run as 'extension' jobs, not 'train' — gate on whether the
    # process supports TE caching rather than on the job type.
    processes = getattr(job, "process", None) or []
    supported = [p for p in processes if hasattr(p, "run_te_cache_worker")]
    if not supported:
        raise ValueError(
            f"te_cache_worker requires a training process that supports TE caching "
            f"(job type '{job.job}' has none)."
        )
    for process in supported:
        process.run_te_cache_worker()

    print_acc("[te-worker] done")
    job.cleanup()


if __name__ == "__main__":
    main()
