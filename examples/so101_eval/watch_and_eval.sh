#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Evaluate each DCP checkpoint of a running (or finished) action-policy job as soon as
# it lands, so the val curve is ready when training is.
#
# Only checkpoints named by `checkpoints/latest_checkpoint.txt` are eaten — the trainer
# writes that file after a save completes, so a half-written directory is never picked
# up. Each checkpoint is evaluated once; the marker file makes restarts idempotent.
#
# The eval server shares the GPU with training (~11 GB on top of training's ~48 GB of a
# 96 GB card). Both slow down somewhat; training keeps priority simply by already
# holding its allocation.
#
#   RUN_DIR=/path/to/outputs/<project>/<group>/<name> \
#   VAL_ROOT=/path/to/..._cosmos3_val \
#   EXPERIMENT=action_policy_so101_edge_wrist CAMERA_MODE=wrist_image \
#   STOP_AFTER=iter_000010000 bash examples/so101_eval/watch_and_eval.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${RUN_DIR:?}" "${VAL_ROOT:?}"
OUT="${OUT:-/data/cosmos3/eval}"
POLL_S="${POLL_S:-300}"
STOP_AFTER="${STOP_AFTER:-}"          # stop once this checkpoint has been evaluated
MAX_WAIT_S="${MAX_WAIT_S:-86400}"
DONE_FILE="${DONE_FILE:-$OUT/.evaluated}"

mkdir -p "$OUT"
touch "$DONE_FILE"
start=$(date +%s)

while true; do
    latest="$(cat "$RUN_DIR/checkpoints/latest_checkpoint.txt" 2>/dev/null | tr -d '[:space:]')"
    if [[ -n "$latest" ]] && ! grep -qxF "$latest" "$DONE_FILE"; then
        ckpt="$RUN_DIR/checkpoints/$latest"
        if [[ -d "$ckpt/model" ]]; then
            echo ">>> $(date -u +%H:%M:%S) evaluating $latest"
            if bash examples/so101_eval/run_eval.sh "$latest"; then
                echo "$latest" >> "$DONE_FILE"
                echo ">>> $(date -u +%H:%M:%S) done $latest"
            else
                echo "!!! $(date -u +%H:%M:%S) eval FAILED for $latest — will retry next poll" >&2
            fi
        fi
    fi

    if [[ -n "$STOP_AFTER" ]] && grep -qxF "$STOP_AFTER" "$DONE_FILE"; then
        echo ">>> $(date -u +%H:%M:%S) $STOP_AFTER evaluated; watcher exiting"
        break
    fi
    if (( $(date +%s) - start > MAX_WAIT_S )); then
        echo "!!! watcher timed out after ${MAX_WAIT_S}s" >&2
        break
    fi
    sleep "$POLL_S"
done
echo WATCH_EVAL_DONE
