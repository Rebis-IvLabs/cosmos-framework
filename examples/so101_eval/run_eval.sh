#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Evaluate one or more DCP checkpoints of `action_policy_so101_edge` on the SO101 val split:
# per checkpoint, start the LIBERO-style policy server, run eval_so101_val.py, stop the server.
#
#   RUN_DIR=/data/cosmos3/outputs/cosmos3_so101/action_sft/action_policy_so101_edge \
#   VAL_ROOT=/data/cosmos3/smr_dataset_so101_pick_place_test_tube_on_the_helicopter_300eps_cosmos3_val \
#   EPISODES=0-7 bash examples/so101_eval/run_eval.sh iter_000002500 iter_000001500
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RUN_DIR:?}" "${VAL_ROOT:?}"
STATS="${STATS:-cosmos_framework/data/generator/action/normalizer_stats/so101_native_frame_wise_relative_rot6d.json}"
WAN_VAE_PATH="${WAN_VAE_PATH:-/data/cosmos3/checkpoints/wan22_vae/Wan2.2_VAE.pth}"
OUT="${OUT:-/data/cosmos3/eval}"
PORT="${PORT:-8000}"
EPISODES="${EPISODES:-}"
NUM_STEPS="${NUM_STEPS:-30}"
GUIDANCE="${GUIDANCE:-1.0}"

source .venv/bin/activate
# HF_TOKEN (gated nvidia/Cosmos-Guardrail1 + processor downloads) and WANDB_API_KEY live in ~/.env on the box.
if [[ -f ~/.env ]]; then set -a; source ~/.env; set +a; fi
_CU13_LIB="$(python -c 'import nvidia, os, glob; print(":".join(sorted({os.path.dirname(f) for p in nvidia.__path__ for f in glob.glob(p + "/**/libnppicc.so.13", recursive=True)})))' 2>/dev/null || true)"
export LD_LIBRARY_PATH="${_CU13_LIB}" PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

for CKPT in "$@"; do
    CKPT_PATH="$RUN_DIR/checkpoints/$CKPT"
    [[ -d "$CKPT_PATH" ]] || { echo "missing $CKPT_PATH" >&2; exit 1; }
    echo ">>> $(date -u +%H:%M:%S) server for $CKPT"
    python -m cosmos_framework.scripts.action_policy_server_libero \
        --experiment action_policy_so101_edge \
        --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
        --checkpoint-path "$CKPT_PATH" \
        --action-normalization quantile_rot --action-stats-path "$STATS" \
        --raw-action-dim 10 --fps 15 --port "$PORT" \
        --num-steps "$NUM_STEPS" --guidance "$GUIDANCE" \
        > "$OUT/logs/server_${CKPT}.log" 2>&1 &
    SERVER_PID=$!
    for i in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/info" >/dev/null 2>&1 && break
        kill -0 $SERVER_PID 2>/dev/null || { echo "server died; see $OUT/logs/server_${CKPT}.log" >&2; exit 1; }
        sleep 5
    done
    curl -sf "http://127.0.0.1:$PORT/info" >/dev/null || { echo "server never became ready" >&2; kill $SERVER_PID; exit 1; }
    echo ">>> $(date -u +%H:%M:%S) eval $CKPT"
    python examples/so101_eval/eval_so101_val.py --root "$VAL_ROOT" --stats "$STATS" \
        --server "http://127.0.0.1:$PORT" --checkpoint-name "$CKPT" --out "$OUT" \
        ${EPISODES:+--episodes "$EPISODES"} 2>&1 | tee "$OUT/logs/eval_${CKPT}.log"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo ">>> $(date -u +%H:%M:%S) done $CKPT"
done
echo EVAL_DONE
