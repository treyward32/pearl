#!/usr/bin/env bash
#
# Pearl miner setup for a fresh RunPod Ubuntu 22.04 instance with one H100.
# Installs toolchains, clones the repo, builds the blockchain binaries, and
# builds the vLLM miner Docker image. Does NOT create a wallet or start mining.
#
# Run as root (RunPod default). Review and edit PEARL_REPO_URL before running.

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. User-editable settings
# ---------------------------------------------------------------------------
PEARL_REPO_URL="https://github.com/treyward32/pearl.git"
PEARL_BRANCH="master"                              # branch to check out
WORKDIR="/workspace"                               # RunPod's persistent volume
GO_VERSION="1.26.0"                                # Go 1.26+ required by README

# ---------------------------------------------------------------------------
# 1. Base system packages
# ---------------------------------------------------------------------------
# Update apt and install the build essentials Pearl needs:
#   - build-essential / clang: C compiler for the XMSS library
#   - git, curl, wget, ca-certificates: cloning + downloading toolchains
#   - pkg-config, libssl-dev: common Rust crate build deps
#   - software-properties-common: needed to add the deadsnakes PPA for Python 3.12
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential clang \
    git curl wget ca-certificates \
    pkg-config libssl-dev \
    software-properties-common \
    xz-utils unzip

# ---------------------------------------------------------------------------
# 2. Verify NVIDIA driver, Docker, and the NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
# RunPod images normally ship with all three already configured. We don't
# install them here; we just fail fast if anything is missing so you don't
# discover it after a 30-minute Docker build.
echo "==> Checking NVIDIA driver (nvidia-smi)..."
nvidia-smi

echo "==> Checking Docker..."
docker --version
docker info >/dev/null

echo "==> Checking NVIDIA Container Toolkit (GPU visible inside a container)..."
# This pulls a tiny CUDA image and runs nvidia-smi inside it. If this fails,
# the toolkit isn't wired up and the vLLM miner container will not see the GPU.
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# ---------------------------------------------------------------------------
# 3. Install Go 1.26+
# ---------------------------------------------------------------------------
# Ubuntu 22.04 apt only has older Go, so we install the official tarball into
# /usr/local/go and add it to PATH for this script and future shells.
if ! command -v go >/dev/null || [[ "$(go version 2>/dev/null)" != *"go${GO_VERSION%.*}"* ]]; then
    echo "==> Installing Go ${GO_VERSION}..."
    cd /tmp
    wget -q "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz"
    rm -rf /usr/local/go
    tar -C /usr/local -xzf "go${GO_VERSION}.linux-amd64.tar.gz"
    rm "go${GO_VERSION}.linux-amd64.tar.gz"
fi
export PATH="/usr/local/go/bin:${PATH}"
# Persist Go on PATH for future SSH sessions.
grep -q '/usr/local/go/bin' /root/.bashrc || \
    echo 'export PATH=/usr/local/go/bin:$PATH' >> /root/.bashrc
go version

# ---------------------------------------------------------------------------
# 4. Install Rust (rustup, stable toolchain)
# ---------------------------------------------------------------------------
# Needed by zk-pow, pearl-blake3, plonky2, and the py-pearl-mining PyO3 bindings.
if ! command -v rustc >/dev/null; then
    echo "==> Installing Rust via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"
rustc --version
cargo --version

# ---------------------------------------------------------------------------
# 5. Install Python 3.12 via the deadsnakes PPA
# ---------------------------------------------------------------------------
# Ubuntu 22.04 ships Python 3.10; the miner workspace requires 3.12.
if ! command -v python3.12 >/dev/null; then
    echo "==> Installing Python 3.12..."
    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev
fi
python3.12 --version

# ---------------------------------------------------------------------------
# 6. Install uv (Python package manager used by the miner workspace)
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv installs to ~/.local/bin by default.
export PATH="$HOME/.local/bin:${PATH}"
grep -q '.local/bin' /root/.bashrc || \
    echo 'export PATH=$HOME/.local/bin:$PATH' >> /root/.bashrc
uv --version

# ---------------------------------------------------------------------------
# 7. Install the Task runner
# ---------------------------------------------------------------------------
# The Pearl Taskfile.yml drives the build (`task build:blockchain`, etc).
if ! command -v task >/dev/null; then
    echo "==> Installing Task runner..."
    sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin
fi
task --version

# ---------------------------------------------------------------------------
# 8. Clone your forked Pearl repo
# ---------------------------------------------------------------------------
mkdir -p "${WORKDIR}"
cd "${WORKDIR}"
if [[ ! -d "${WORKDIR}/pearl/.git" ]]; then
    echo "==> Cloning ${PEARL_REPO_URL}..."
    git clone "${PEARL_REPO_URL}" pearl
fi
cd "${WORKDIR}/pearl"
# Fetch and check out the requested branch (creates a local tracking branch
# if it doesn't exist yet).
git fetch origin
git checkout "${PEARL_BRANCH}" 2>/dev/null || git checkout -b "${PEARL_BRANCH}" "origin/${PEARL_BRANCH}"
# Initialize any submodules referenced by .gitmodules.
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# 9. Build the blockchain binaries (pearld, oyster, prlctl)
# ---------------------------------------------------------------------------
# Outputs go to ./bin/ inside the repo. This does NOT build the miner Python
# packages or run any wallet/mining commands.
echo "==> Running task build:blockchain..."
cd "${WORKDIR}/pearl"
task build:blockchain
ls -lh "${WORKDIR}/pearl/bin"

# ---------------------------------------------------------------------------
# 10. Build the vLLM miner Docker image
# ---------------------------------------------------------------------------
# The Dockerfile expects the full monorepo as build context, so this runs
# from the repo root. The resulting image is tagged `vllm_miner:latest`.
# This step is long (CUDA kernel compilation + vLLM install) and uses a lot
# of disk; make sure your RunPod volume has plenty of free space.
echo "==> Building vllm_miner Docker image (this takes a while)..."
cd "${WORKDIR}/pearl"
docker buildx build -t vllm_miner -f miner/vllm-miner/Dockerfile .

# ---------------------------------------------------------------------------
# Done. Next steps (NOT performed by this script):
#   - Generate a wallet on a TRUSTED machine (not this rental) with oyster.
#   - Run pearld and pearl-gateway, then launch the vllm_miner container.
# ---------------------------------------------------------------------------
echo "==> Setup complete."
echo "Repo:       ${WORKDIR}/pearl"
echo "Binaries:   ${WORKDIR}/pearl/bin/{pearld,oyster,prlctl}"
echo "Docker tag: vllm_miner:latest"
