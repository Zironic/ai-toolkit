# Zironic fork Docker image plan

> **git-bug:** `c1e17c7` (open) - mutable build/run state and validation
> evidence belong in the ticket.

> **Type:** implementation plan for a reproducible Linux NVIDIA container that
> is accepted on local Docker Engine and remains portable to RunPod and other
> standard NVIDIA Docker hosts.

## Outcome

Build and publish a commit-addressable container for this repository that is
proven on the user's local Docker Engine with NVIDIA GPU access. Keep the image
contract portable so the same image should run on RunPod or another ordinary
NVIDIA Docker host without provider-specific changes.

When complete:

- the image contains `Zironic/ai-toolkit` at an exact `faster-dop` commit, not
  `ostris/ai-toolkit`;
- `git rev-parse HEAD` inside the image returns the source revision used in the
  image tag and OCI metadata;
- the UI and worker start on port 8675, with optional SSH when a public key is
  supplied;
- outputs, datasets, job config/database state, Hugging Face caches, Torch
  caches, and compile caches can live under a mounted `/workspace` volume;
- a public GHCR image can be used with an ordinary
  `docker run --gpus all` command;
- local Docker Engine proves CUDA visibility, Toolkit imports, one focused
  synthetic CUDA path, UI startup, and persistence across a container replace;
- immutable commit tags provide rollback and evidence identity.

The first useful deliverable is one linux/amd64 NVIDIA image for standard Docker
Engine. Serverless workers, Kubernetes, multi-architecture images,
provider-specific images, and a paid cloud validation run are not required.

## Verified starting point

The repository already has Docker/RunPod pieces, but they do not currently
produce this fork's image:

- `docker/Dockerfile` clones `https://github.com/ostris/ai-toolkit.git` and
  defaults `GIT_COMMIT=main`;
- `build_and_push_docker*` and `docker-compose.yml` publish or pull
  `ostris/aitoolkit`;
- the current image uses Ubuntu 24.04, CUDA 12.8.1, Python 3.12, and pinned
  Torch 2.9.1/cu128;
- `docker/start.sh` starts the UI/worker and optionally SSH, but does not map
  RunPod's persistent `/workspace` into Toolkit state;
- no `.dockerignore` exists on this branch, while `output/` and `datasets/` are
  Windows junctions to large shared storage and must never enter a Docker build
  context;
- the local Docker Engine is a Linux amd64 engine and is the primary build and
  acceptance environment; NVIDIA container access still needs a focused probe.

Useful prior work exists in history and should be adapted rather than recreated:

- `eab447f9` added a bounded `.dockerignore` and built fork source from context;
- `dba092fc` preserved a real git repository inside the image;
- `f3369e05` and `perceptual/runpod-qol` added `/workspace` cache/output
  persistence;
- `perceptual/runpod-qol:.github/workflows/build.yml` publishes commit-derived
  fork images to GHCR with Buildx caching.

No earlier current Markdown plan or git-bug ticket covered a Zironic/faster-dop
Docker image; this plan and ticket `c1e17c7` now provide that durable state.

RunPod accepts images from Docker Hub or another registry, supports private
registry credentials, conventionally mounts persistent storage at
`/workspace`, and exposes configured HTTP/TCP ports. GitHub Actions can publish
to GHCR with the repository `GITHUB_TOKEN` and `packages: write` permission.

References:

- [RunPod custom Pod templates](https://docs.runpod.io/pods/templates/manage-templates)
- [RunPod storage and `/workspace`](https://docs.runpod.io/pods/storage/types)
- [GitHub container publishing](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)
- [Docker Buildx cache in GitHub Actions](https://docs.docker.com/build/ci/github-actions/cache/)

## Requirements, assumptions, and non-goals

### Explicit requirements

- Build and publish the user's own fork image.
- Accept it on the user's local Docker Engine with `--gpus all`.
- Keep the image provider-neutral enough for RunPod and other NVIDIA Docker
  hosts without requiring a cloud-host validation run.
- Preserve this fork's source identity and runtime behavior.

### Working assumptions

- Registry: public `ghcr.io/zironic/ai-toolkit` so RunPod needs no registry
  secret. Change this before implementation if the package must remain private.
- Source branch: `faster-dop`; every published build also gets an immutable
  commit-SHA tag.
- Runtime: Linux amd64, NVIDIA GPU through local `docker run --gpus all`,
  UI/worker as the default command, HTTP 8675, optional TCP 22, persistent
  volume at `/workspace`.
- First image: retain the Dockerfile's current CUDA/Torch baseline while proving
  the container path. Do not combine the first publish with an unrelated Torch
  migration.

### Non-goals

- no WSL path;
- no paid cloud acceptance run or required RunPod template/deployment;
- no automatic real training job during image build;
- no baked model weights, datasets, Hugging Face tokens, registry credentials,
  SSH keys, or UI auth token;
- no publication of dirty/uncommitted source;
- no `latest`-only deployment or mutable tag as the sole evidence identity;
- no requirement that WDDM-only DXGI behavior work on Linux. Linux fallbacks
  must work, while WDDM-specific policies remain a Windows concern.

## Design

### 1. Build exact fork source and retain identity

Keep the existing dependency-first layering, but parameterize and verify the
source clone:

- `SOURCE_REPO=https://github.com/Zironic/ai-toolkit.git`;
- `SOURCE_REF` supplied as the full GitHub Actions commit SHA for published
  images, with `faster-dop` only as a local convenience default;
- clone/checkout that ref in detached mode and retain its shallow `.git`
  directory in `/app/ai-toolkit`;
- remove the timestamp `CACHEBUST`; the immutable source SHA is the cache key;
- add OCI source/revision/version labels and a small build-info JSON containing
  source URL, SHA, image build time, Python, Torch, and CUDA versions;
- fail the build if the checked-out HEAD does not equal the requested full SHA.

The Docker build context still supplies dependency manifests and Docker assets.
Add `.dockerignore` before any build so it excludes at least `.git`, local venvs,
node_modules, `.agent`, `.codex`, caches, `output`, `datasets`, masks, temp files,
and local databases. This is mandatory because the canonical output/dataset
paths are junctions in this checkout.

### 2. Make runtime state explicitly persistent

Use `/app/ai-toolkit` for immutable application code and `/workspace` for
mutable state. The entrypoint should idempotently prepare and link:

```text
/app/ai-toolkit/output          -> /workspace/output
/app/ai-toolkit/datasets        -> /workspace/datasets
/app/ai-toolkit/config          -> /workspace/config
/app/ai-toolkit/aitk_db.db      -> /workspace/aitk_db.db
HF_HOME                         =  /workspace/.cache/huggingface
TORCH_HOME                      =  /workspace/.cache/torch
TORCHINDUCTOR_CACHE_DIR         =  /workspace/.cache/torchinductor
AI Toolkit compile/MegaCache    =  documented persistent path under /workspace
```

Do not rely on "link only when the image path is absent": tracked `.gitkeep`
files make some directories exist in a clean clone. Instead, bake seed config
and database state separately, initialize an empty volume once, then install
deterministic symlinks. Never overwrite non-empty user state.

Run the Prisma schema update against the persistent database at container start
before launching the worker/UI. End the script with `exec npm run start` so
signals reach the service process. Keep SSH optional and do not print secrets.

### 3. Publish through GHCR with reproducible tags

Add a manual-first GitHub Actions workflow based on the already-proven
`perceptual/runpod-qol` workflow:

- checkout the selected commit;
- set up Buildx;
- log into GHCR using `GITHUB_TOKEN` with `contents: read` and
  `packages: write`;
- build linux/amd64 from `docker/Dockerfile` with `SOURCE_REF=${{ github.sha }}`;
- use the GitHub Actions BuildKit cache;
- publish `sha-<shortsha>` and a branch alias such as `faster-dop`;
- publish version/release aliases only from an explicit tag or release;
- record the pushed digest in the job summary.

Pin third-party workflow actions to reviewed commit SHAs. Keep the image public
for simple pulls. All validation commands and downstream deployments should use
an immutable SHA tag or digest so rollback and evidence identity are explicit.

### 4. Validate the image on local Docker Engine

#### Build-time inspection

The local build must prove, with CI reproducing the same checks when publishing:

- source HEAD and OCI revision equal the workflow SHA;
- `python`, Torch, Toolkit imports, Node, Prisma client, UI build, and worker
  bundle exist;
- no local junction payload, token, venv, `.agent`, or `.codex` content entered
  the image;
- image size and major layers are reported, not silently allowed to grow.

#### Container startup without a model

Start the built image with a temporary `/workspace` volume and verify:

- the UI answers on port 8675;
- the worker stays running;
- auth works when `AI_TOOLKIT_AUTH` is supplied;
- the database and runtime links point into `/workspace`;
- stop/restart preserves a marker, database row, and output file.

#### Local Docker Engine GPU acceptance

On the user's local Linux Docker Engine:

1. build the exact local SHA, then pull the published immutable SHA tag/digest
   when registry publishing is being verified;
2. verify `nvidia-smi`, `torch.cuda.is_available()`, GPU name/capability, Torch,
   CUDA, and fork SHA;
3. run the narrowest existing synthetic CUDA smoke that exercises a fork seam
   without downloading a full model;
4. open the UI through HTTP port 8675 and confirm the worker can see the GPU;
5. write state under the ordinary Toolkit paths, replace the container while
   reusing the volume, and prove that output, database, and caches persist;
6. only then run a short full-model smoke if the container acceptance question
   requires checkpoint loading or arena behavior.

Do not use a real dataset training job as a generic container test. Record the
host GPU, driver, Docker Engine and NVIDIA runtime versions, image digest,
source SHA, commands, timings, and failures on the ticket.

### 5. Document the portable Docker contract

Add a concise container section that contains:

- GHCR image and immutable tag convention;
- a provider-neutral `docker run --gpus all` example;
- the portable image contract: volume mount `/workspace`, HTTP `8675`, optional
  TCP `22`, adequate container/volume disk, and required secrets;
- what persists and what is ephemeral;
- update/rollback procedure using immutable tags;
- the distinction between Linux container behavior and native Windows/WDDM.

Do not create or validate a RunPod template in the first slice. Compatibility is
an expected consequence of the standard OCI image, NVIDIA runtime, port, and
volume contract; provider-specific templates can be added later if actually
needed.

## CUPTI monitor follow-up

The Linux image creates a credible place to revisit closed ticket `2d6b39c`,
but the experimental profiler must not hold the base container hostage.

After the base image passes:

1. create a separate diagnostic image/tag or explicitly approved Torch 2.13
   update;
2. pin a compatible CUDA/Torch/cupti-python set;
3. add the opt-in training-smoke backend and run the stock-versus-monitor trace;
4. keep the profiler only if it produces useful evidence without destabilizing
   the supported Linux container runtime.

Do not silently move the general-purpose image to Torch 2.13 merely to acquire
an experimental diagnostic.

## Acceptance gates

The ticket can close when:

- GHCR contains a public immutable image for a known `faster-dop` SHA;
- the image reports that exact fork SHA internally;
- the local Linux Docker Engine starts the UI and worker from the image;
- one focused CUDA smoke passes through local `docker run --gpus all`;
- `/workspace` preserves output, database, config, and caches across container
  replacement;
- no secrets or local junction payloads are present in the image;
- the documented launch and rollback commands work from the published digest;
- mutable validation state and evidence are recorded on the git-bug ticket.

The CUPTI monitor is a separate follow-up decision, not a close gate for the
Docker image.
