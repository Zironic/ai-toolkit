#!/bin/bash
set -euo pipefail

# Keep optional SSH support for remote Docker hosts without making the local
# Docker path depend on provider-specific environment variables.
setup_ssh() {
    if [[ -z "${PUBLIC_KEY:-}" ]]; then
        return
    fi

    echo "Setting up SSH..."
    install -d -m 700 /root/.ssh
    printf '%s\n' "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    ssh-keygen -A
    service ssh start
}

# Preserve RunPod variables in interactive SSH shells when they are present.
# This is inert on an ordinary local Docker Engine.
export_runpod_env() {
    if ! printenv | grep -q '^RUNPOD_'; then
        return
    fi

    echo "Exporting RunPod environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' \
        | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' \
        > /etc/rp_environment
    grep -qF 'source /etc/rp_environment' /root/.bashrc \
        || echo 'source /etc/rp_environment' >> /root/.bashrc
}

# Initialize persistent state without overwriting anything the user already
# placed in /workspace. Application paths are linked to these paths at runtime.
prepare_workspace() {
    mkdir -p \
        /workspace/output \
        /workspace/datasets \
        /workspace/config \
        /workspace/.cache/huggingface \
        /workspace/.cache/torch \
        /workspace/.cache/torchinductor

    if [[ -z "$(find /workspace/config -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        cp -a /opt/ai-toolkit-seed/config/. /workspace/config/
    fi

    if [[ ! -e /workspace/aitk_db.db ]]; then
        cp /opt/ai-toolkit-seed/aitk_db.db /workspace/aitk_db.db
    fi

    link_workspace_directory output
    link_workspace_directory datasets
    link_workspace_directory config
    link_workspace_file aitk_db.db

    export HF_HOME=/workspace/.cache/huggingface
    export TORCH_HOME=/workspace/.cache/torch
    export TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor
    export XDG_CACHE_HOME=/workspace/.cache
}

link_workspace_directory() {
    local name="$1"
    local target="/app/ai-toolkit/${name}"

    if mountpoint -q "${target}"; then
        echo "Using direct mount at ${target}"
        return
    fi

    if [[ -L "${target}" ]]; then
        rm -f "${target}"
    elif [[ -e "${target}" ]]; then
        rm -rf "${target}"
    fi
    ln -s "/workspace/${name}" "${target}"
}

link_workspace_file() {
    local name="$1"
    local target="/app/ai-toolkit/${name}"

    if mountpoint -q "${target}"; then
        echo "Using direct mount at ${target}"
        return
    fi

    rm -f "${target}"
    ln -s "/workspace/${name}" "${target}"
}

echo "AI Toolkit container started"

setup_ssh
export_runpod_env
prepare_workspace

cd /app/ai-toolkit/ui
npm run update_db
echo "Starting AI Toolkit UI..."
exec npm run start
