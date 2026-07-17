# Zironic fork Docker image

This repository builds a Linux amd64 NVIDIA image from an exact
`Zironic/ai-toolkit` commit. Local Docker Engine with NVIDIA GPU passthrough is
the acceptance environment. The image uses only standard OCI, NVIDIA runtime,
port, and volume behavior so it can also be used by compatible container hosts.

## Prerequisites

- Docker Engine running Linux containers;
- NVIDIA container GPU support;
- enough Docker disk space for the CUDA development base, Torch, Toolkit
  dependencies, and model caches.

Verify GPU passthrough before building Toolkit:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
```

## Build an exact commit

The source commit must exist in `https://github.com/Zironic/ai-toolkit.git`.
Use the full 40-character SHA:

```bash
docker build \
  --file docker/Dockerfile \
  --tag zironic/ai-toolkit:local \
  --build-arg SOURCE_REF=<full-commit-sha> \
  --build-arg SOURCE_CREATED=<commit-iso-8601-time> \
  .
```

The build fails if the checked-out source does not match the requested full
SHA or if the dependency manifests in the build context do not match that
source. `ALLOW_PACKAGE_LOCK_MISMATCH=1` exists only for a local development
build while repairing a lockfile; never use it for a published image.

Inspect the source identity:

```bash
docker run --rm --entrypoint git zironic/ai-toolkit:local \
  -C /app/ai-toolkit rev-parse HEAD
```

## Run locally

```bash
docker volume create ai-toolkit-workspace
docker run --detach \
  --name ai-toolkit \
  --gpus all \
  --publish 8675:8675 \
  --volume ai-toolkit-workspace:/workspace \
  --env AI_TOOLKIT_AUTH=<strong-password> \
  zironic/ai-toolkit:local
```

Open `http://localhost:8675`. Omit `AI_TOOLKIT_AUTH` only on a trusted local
network. Supply `PUBLIC_KEY` and publish TCP 22 only when SSH is actually
needed.

The entrypoint initializes and preserves these paths under `/workspace`:

| Toolkit state | Persistent path |
| --- | --- |
| Outputs and compile cache artifacts | `/workspace/output` |
| Datasets | `/workspace/datasets` |
| Job configuration | `/workspace/config` |
| UI and worker database | `/workspace/aitk_db.db` |
| Hugging Face cache | `/workspace/.cache/huggingface` |
| Torch cache | `/workspace/.cache/torch` |
| TorchInductor cache | `/workspace/.cache/torchinductor` |

Replacing the container does not remove the named volume. Reuse the same
volume with the new immutable image tag when updating or rolling back.

## Docker Compose

Set an exact source SHA and start the service:

```bash
SOURCE_REF=<full-commit-sha> docker compose up --build --detach
```

PowerShell:

```powershell
$env:SOURCE_REF = "<full-commit-sha>"
docker compose up --build --detach
```

Compose uses the `ai-toolkit-workspace` named volume and exposes port 8675. Set
`AI_TOOLKIT_IMAGE` to use a prebuilt image instead of the local tag.

## Publish to GHCR

Run the `Publish fork container` workflow manually on the desired Git ref. It
publishes:

- `ghcr.io/zironic/ai-toolkit:sha-<12-character-sha>` as the immutable tag;
- a lowercase branch or tag alias for convenience.

The workflow records the pushed digest in its job summary. Make the GHCR
package public before relying on anonymous pulls. Deploy and roll back by the
immutable SHA tag or digest, not only by the mutable branch alias.

No RunPod deployment or paid cloud acceptance run is required. A downstream
host needs the same contract: linux/amd64, an NVIDIA GPU runtime, HTTP 8675,
and persistent storage mounted at `/workspace`.

## Docker Scout warning

Docker Scout is extremely poorly behaved with this roughly 20 GiB CUDA image.
Automatic analysis can consume substantial CPU, disk I/O, memory, and network
resources while indexing the large CUDA and PyTorch layers, independently of a
normal build or container run. Scout is not part of this image's acceptance
criteria. Disable automatic/background Scout analysis for this image and run it
only as a deliberate, monitored operation when its report is actually needed.
