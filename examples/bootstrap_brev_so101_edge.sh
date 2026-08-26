#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
# Fresh-box bootstrap for Cosmos3-Edge action-policy SFT (cosmos-framework fork, branch experiment/so101-edge).
# Idempotent; run as the box user. Env: DATA_ROOT (default /data/cosmos3), CUDA_GROUP (cu128-train | cu130-train; auto by driver).
# Needs ~/.env with HF_TOKEN (sapanostic), SMR_HF_TOKEN (private datasets), WANDB_API_KEY.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
DATA_ROOT="${DATA_ROOT:-/data/cosmos3}"
BRANCH="${BRANCH:-experiment/so101-edge}"
DS=smr_dataset_so101_pick_place_test_tube_on_the_helicopter_300eps_cosmos3
set -a; source ~/.env; set +a

# --- driver -> CUDA group: torch cu130 needs driver >= 580; else cu128 (needs >= 525) ---
DRV_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ -z "${CUDA_GROUP:-}" ]; then
    if [ "$DRV_MAJOR" -ge 580 ]; then CUDA_GROUP=cu130-train; else CUDA_GROUP=cu128-train; fi
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "CUDA_GROUP=$CUDA_GROUP"

# --- system packages ---
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git-lfs ffmpeg libxcb1 libgl1 libglib2.0-0 python3-dev
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv self update || true
uv --version

# --- data root ---
sudo mkdir -p "$DATA_ROOT"/{checkpoints,outputs}
sudo chown -R "$USER:$USER" "$(dirname "$DATA_ROOT")" 2>/dev/null || sudo chown -R "$USER:$USER" "$DATA_ROOT"
df -h "$DATA_ROOT" | tail -1

# --- repo + env ---
[ -d ~/cosmos-framework/.git ] || git clone --branch "$BRANCH" https://github.com/Rebis-IvLabs/cosmos-framework.git ~/cosmos-framework
cd ~/cosmos-framework
git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH"
export GIT_LFS_SKIP_SMUDGE=1
uv sync --all-extras --group="$CUDA_GROUP"
source .venv/bin/activate
export LD_LIBRARY_PATH=
python -c "import torch, cosmos_framework; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ok', torch.cuda.is_available())"

# --- assets ---
export HF_HUB_ENABLE_HF_TRANSFER=1
[ -f "$DATA_ROOT/checkpoints/wan22_vae/Wan2.2_VAE.pth" ] || uvx hf@latest download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$DATA_ROOT/checkpoints/wan22_vae"
[ -d "$DATA_ROOT/checkpoints/Cosmos3-Edge" ] || python -m cosmos_framework.scripts.convert_model_to_dcp -o "$DATA_ROOT/checkpoints/Cosmos3-Edge" --checkpoint-path Cosmos3-Edge
for split in train val; do
    [ -f "$DATA_ROOT/${DS}_${split}/meta/info.json" ] || uvx hf@latest download "selfmaderesearchers/${DS}_${split}" --repo-type dataset --local-dir "$DATA_ROOT/${DS}_${split}" --token "$SMR_HF_TOKEN"
done
python - "$DATA_ROOT/${DS}_train" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/meta/info.json"))
assert m["codebase_version"] == "v3.0" and m["total_episodes"] == 270, m
print("dataset OK:", m["total_episodes"], "eps", m["total_frames"], "frames", m["fps"], "fps")
PY
ls "$DATA_ROOT/checkpoints/Cosmos3-Edge" | head -5
echo BOOTSTRAP_DONE
