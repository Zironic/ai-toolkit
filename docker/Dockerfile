FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG SOURCE_REPO=https://github.com/Zironic/ai-toolkit.git
ARG SOURCE_REF=faster-dop
ARG SOURCE_CREATED=unknown

LABEL authors="jaret"

# Set noninteractive to avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive

# ref https://en.wikipedia.org/wiki/CUDA
ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    git \
    curl \
    build-essential \
    cmake \
    wget \
    python3.12 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    ffmpeg \
    tmux \
    htop \
    nvtop \
    python3-opencv \
    openssh-client \
    openssh-server \
    openssl \
    rsync \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install nodejs
WORKDIR /tmp
RUN curl -sL https://deb.nodesource.com/setup_23.x -o nodesource_setup.sh && \
    bash nodesource_setup.sh && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set aliases for python and pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install pytorch before cache bust to avoid redownloading pytorch
RUN pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 --break-system-packages

WORKDIR /app/ai-toolkit

# ---------------------------------------------------------------------------- #
# Dependency layers come BEFORE the source clone so they are only rebuilt (and
# only need to be re-pulled by servers) when the dependency manifests change,
# not on every code change.
# ---------------------------------------------------------------------------- #

# Install Python dependencies (only re-runs when the requirements files change)
COPY requirements.txt requirements_base.txt /app/ai-toolkit/
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt && \
    pip install setuptools==69.5.1 --no-cache-dir --break-system-packages

# Install Node dependencies (only re-runs when package.json / package-lock.json change)
COPY ui/package.json ui/package-lock.json /app/ai-toolkit/ui/
RUN cd /app/ai-toolkit/ui && npm ci

# ---------------------------------------------------------------------------- #
# Source code comes LAST. Only this layer (plus the UI build below) is rebuilt
# on a code change, so servers only re-pull the (small) source, not the deps.
# Clone to a temp dir and rsync the source in, preserving the dependency dirs
# already populated above (ui/node_modules), the manifests already used, and
# the shallow .git directory that identifies the image source.
# ---------------------------------------------------------------------------- #
LABEL org.opencontainers.image.source="${SOURCE_REPO}" \
      org.opencontainers.image.revision="${SOURCE_REF}" \
      org.opencontainers.image.created="${SOURCE_CREATED}"

ARG ALLOW_PACKAGE_LOCK_MISMATCH=0
RUN git init /tmp/ai-toolkit-src && \
    cd /tmp/ai-toolkit-src && \
    git remote add origin "${SOURCE_REPO}" && \
    git fetch --depth 1 origin "${SOURCE_REF}" && \
    git checkout --detach FETCH_HEAD && \
    actual_revision="$(git rev-parse HEAD)" && \
    if [ "${#SOURCE_REF}" -eq 40 ]; then test "${actual_revision}" = "${SOURCE_REF}"; fi && \
    diff --brief --strip-trailing-cr requirements.txt /app/ai-toolkit/requirements.txt && \
    diff --brief --strip-trailing-cr requirements_base.txt /app/ai-toolkit/requirements_base.txt && \
    diff --brief --strip-trailing-cr ui/package.json /app/ai-toolkit/ui/package.json && \
    if ! diff --brief --strip-trailing-cr ui/package-lock.json /app/ai-toolkit/ui/package-lock.json; then \
        test "${ALLOW_PACKAGE_LOCK_MISMATCH}" = "1"; \
        echo "WARNING: using a local package-lock repair for dependency installation"; \
    fi && \
    rsync -a --delete \
        --exclude 'ui/node_modules' \
        /tmp/ai-toolkit-src/ /app/ai-toolkit/ && \
    python_version="$(python --version 2>&1 | awk '{print $2}')" && \
    torch_version="$(python -c 'import torch; print(torch.__version__)')" && \
    cuda_version="$(python -c 'import torch; print(torch.version.cuda)')" && \
    node_version="$(node --version)" && \
    printf '{"source":"%s","revision":"%s","created":"%s","python":"%s","torch":"%s","cuda":"%s","node":"%s","package_lock_mismatch_allowed":%s}\n' \
        "${SOURCE_REPO}" "${actual_revision}" "${SOURCE_CREATED}" \
        "${python_version}" "${torch_version}" "${cuda_version}" "${node_version}" \
        "${ALLOW_PACKAGE_LOCK_MISMATCH}" \
        > /etc/ai-toolkit-image.json && \
    rm -rf /tmp/ai-toolkit-src

# Build UI (re-runs on code change, but reuses the cached node_modules above).
# update_db runs first because it does `prisma generate`, which creates the
# @prisma/client types the TS build needs. In the old layout generate happened
# as a side effect of npm install seeing the schema; now the source arrives
# after npm ci, so run it explicitly before the build.
RUN cd /app/ai-toolkit/ui && \
    npm run update_db && \
    npm run build

# Keep immutable seeds outside the application tree. The entrypoint installs
# workspace links at container start so the image's Git worktree stays clean.
RUN mkdir -p /opt/ai-toolkit-seed && \
    cp -a /app/ai-toolkit/config /opt/ai-toolkit-seed/config && \
    cp /app/ai-toolkit/aitk_db.db /opt/ai-toolkit-seed/aitk_db.db && \
    rm -f /app/ai-toolkit/aitk_db.db

# Expose the Toolkit UI port.
EXPOSE 8675

VOLUME ["/workspace"]

WORKDIR /

COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
