#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Cosmos3-Edge WRIST-ONLY action-policy SFT on the SO-ARM101 pick-place dataset (experiment
# `action_policy_so101_edge_wrist`). Single-GPU by default (NPROC_PER_NODE=1).
#
#   export SO101_ROOT=/data/cosmos3/smr_dataset_so101_pick_place_test_tube_on_the_helicopter_300eps_cosmos3_train
#   export BASE_CHECKPOINT_PATH=examples/checkpoints/Cosmos3-Edge   # DCP dir (convert_model_to_dcp)
#   NPROC_PER_NODE=1 bash examples/launch_sft_action_policy_so101_edge_wrist.sh
#
# Smoke test:
#   EXTRA_TAIL_OVERRIDES="job.wandb_mode=disabled trainer.max_iter=10 checkpoint.save_iter=10" \
#     NPROC_PER_NODE=1 bash examples/launch_sft_action_policy_so101_edge_wrist.sh

TOML_FILE="examples/toml/sft_config/action_policy_so101_edge_wrist.toml"
: "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}"
export SO101_ROOT="${SO101_ROOT:-}"
EXTRA_DATASET_CHECK='[[ -f "$SO101_ROOT/meta/info.json" ]] || { echo "ERROR: SO101_ROOT must be a local LeRobot v3.0 dir containing meta/info.json (got: '\''$SO101_ROOT'\''). Build it with rebisvla flywheel/01_data_collection_and_curation/06_to_cosmos3_policy.py" >&2; exit 1; }'
TAIL_OVERRIDES=(
    ${EXTRA_TAIL_OVERRIDES:-}
)
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
# torchcodec (cu130 build) dlopens libnppicc.so.13 / libnvrtc.so.13 at the first video decode;
# torch does not preload NPP/NVRTC, so put the venv's CUDA-13 runtime libs on the loader path
# (they come from the `nvidia-npp` / `nvidia-cuda-nvrtc` wheels of the cu130-train group).
_CU13_LIB="$(python -c 'import nvidia, os, glob; print(":".join(sorted({os.path.dirname(f) for p in nvidia.__path__ for f in glob.glob(p + "/**/libnppicc.so.13", recursive=True) + glob.glob(p + "/**/libnvrtc.so.13", recursive=True)})))' 2>/dev/null || true)"
if [[ -n "$_CU13_LIB" ]]; then export LD_LIBRARY_PATH="${_CU13_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"
