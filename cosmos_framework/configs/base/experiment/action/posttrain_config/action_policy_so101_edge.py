# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_so101_edge`` — Cosmos3-Edge action-policy SFT on the Rebis-IvLabs SO-ARM101
pick-place dataset (single arm, wrist + top cameras, frame-wise-relative rot6d + gripper).

Derived from ``action_policy_libero_nano`` (same loader, optimizer, scheduler, trainer and
checkpoint blocks) with two deltas:

* model config = ``EDGE_MODEL_CONFIG`` (Nemotron-2B-Dense-VL backbone, Cosmos3-Edge native 480p,
  ``action_gen=True``) instead of ``NANO_MODEL_CONFIG`` — the base is ``nvidia/Cosmos3-Edge``
  converted to DCP. There is no shipped Edge action-policy recipe; this is the first.
* dataset = ``LIBEROLeRobotDataset`` pointed at a LeRobot v3.0 dataset produced by the rebisvla
  flywheel (``06_to_cosmos3_policy.py``): ``observation.images.image`` (top) +
  ``observation.images.wrist_image`` (wrist), ``action`` = 7-D per-frame delta of the achieved
  EEF pose ``[dxyz, axis-angle, gripper]`` (Cosmos ``backward_framewise``), 15 fps, task string
  "Pick up the test-tube and place it in the holder on the helicopter". ``embodiment_type="so101"``
  selects domain id 25 (own ``action2llm``/``llm2action`` slot) and the bundled stats file
  ``normalizer_stats/so101_native_frame_wise_relative_rot6d.json`` (``quantile_rot``).

No proprioceptive state is used (the LIBERO loader never prepends a state row), matching the
vision-only GR00T exp 021 baseline this is compared against. Env: ``SO101_ROOT``,
``BASE_CHECKPOINT_PATH`` (Edge DCP), ``WAN_VAE_PATH``, ``IMAGINAIRE_OUTPUT_ROOT``.
Run-level scalars (parallelism, batch, iters, lr) live in
``examples/toml/sft_config/action_policy_so101_edge.toml``.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_libero_nano import (
    action_policy_libero_nano,
)
from cosmos_framework.configs.base.experiment.sft.models.edge_model_config import EDGE_MODEL_CONFIG
from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_libero_sft_dataset
from cosmos_framework.utils.lazy_config import LazyCall as L
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()


def _action_policy_so101_edge_model_config() -> dict:
    """Edge model config for action-policy SFT: same knobs the LIBERO recipe applies to Nano
    (packed-token cap, fresh diffusion-expert init, 10x vision flow-matching loss, exact VAE
    durations for 17-frame policy clips). Edge is already ``activation_checkpointing.mode="full"``."""
    cfg = copy.deepcopy(EDGE_MODEL_CONFIG)  # action_gen=True, max_action_dim=64, resolution="480"
    cfg["max_num_tokens_after_packing"] = 45056
    cfg["diffusion_expert_config"]["load_weights_from_pretrained"] = False  # weights come from the Edge DCP
    cfg["rectified_flow_training_config"]["loss_scale"] = 10.0
    cfg["rectified_flow_training_config"]["image_loss_scale"] = None
    cfg["tokenizer"]["encode_exact_durations"] = [17, 61, 73]  # Cosmos3-Edge.yaml value; chunk 16 -> 17 frames
    return cfg


_base = copy.deepcopy(action_policy_libero_nano)
_base["job"] = dict(
    project="cosmos3_so101",
    group="action_sft",
    name="action_policy_so101_edge",
    wandb_mode="disabled",
)
_base["model"] = dict(config=_action_policy_so101_edge_model_config())
_base["dataloader_train"]["dataset_name"] = "action_so101"
_base["dataloader_train"]["dataloader"]["datasets"] = dict(
    so101=dict(
        ratio=1,
        dataset=L(get_action_libero_sft_dataset)(
            root="${oc.env:SO101_ROOT}",
            fps=15,  # metadata only; the loader reads the native fps (15) from info.json
            chunk_length=16,  # 16 x 1/15 s ~ 1.07 s horizon, same as GR00T exp 021
            image_size=256,  # concat_view -> 256x512 (top | wrist)
            mode="wam",
            camera_mode="concat_view",
            action_space="frame_wise_relative",
            rotation_space="6d",
            pose_coordinate_frame="native",
            action_normalization="quantile_rot",
            action_stats_path="so101_native_frame_wise_relative_rot6d.json",  # resolved under normalizer_stats/
            embodiment_type="so101",
            val_ratio=0.01,
            iterable_shuffle=True,
            episode_shuffle_seed=42,
            resolution=None,
            max_action_dim="${model.config.max_action_dim}",
            cfg_dropout_rate=0.1,
            format_prompt_as_json=True,
            tokenizer_config="${model.config.vlm_config.tokenizer}",
        ),
    ),
)

action_policy_so101_edge = LazyDict(_base, flags={"allow_objects": True})

cs.store(group="experiment", package="_global_", name="action_policy_so101_edge", node=action_policy_so101_edge)
