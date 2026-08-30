# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""``action_policy_so101_edge_wrist`` — Cosmos3-Edge action-policy SFT on the SO-ARM101
pick-place dataset, conditioned on the WRIST camera alone.

Sibling of ``action_policy_so101_edge``, which concatenates the third-person and wrist
views into a 256x512 canvas. The only change here is ``camera_mode="wrist_image"``: the
loader selects the wrist view alone, so the observation is a single 256x256 image and
the prompt's viewpoint tag becomes ``wrist_view`` instead of ``concat_view``
(``_VIEWPOINT_BY_CAMERA``). The action space, normalization, chunk length, fps, base
model and schedule are untouched.

No dataset rebuild is needed — the published dataset carries both views
(``observation.images.image`` = the SO101 top camera, ``observation.images.wrist_image``)
and the loader simply reads one of them. The action statistics are camera-independent,
so ``so101_native_frame_wise_relative_rot6d.json`` is reused unchanged.

Why wrist-only matters here: every GR00T experiment on this task (016 onward) is
wrist-only, so this variant is the like-for-like comparison against them; and the live
SO-ARM101 rig has only a wrist camera wired, so a wrist-only checkpoint is the one that
can actually be closed-loop evaluated on hardware without adding a second camera.

As with the concat-view recipe, no proprioceptive state is used: the LIBERO-style loader
reads no ``observation.state`` column at all, so the policy is vision + language only.

Env: ``SO101_ROOT``, ``BASE_CHECKPOINT_PATH`` (Edge DCP), ``WAN_VAE_PATH``,
``IMAGINAIRE_OUTPUT_ROOT``. Run-level scalars live in
``examples/toml/sft_config/action_policy_so101_edge_wrist.toml``.
"""

import copy

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_so101_edge import (
    action_policy_so101_edge,
)
from cosmos_framework.utils.lazy_config import LazyDict

cs = ConfigStore.instance()

_base = copy.deepcopy(action_policy_so101_edge)
_base["job"] = dict(
    project="cosmos3_so101",
    group="action_sft",
    name="action_policy_so101_edge_wrist",
    wandb_mode="disabled",
)
_base["dataloader_train"]["dataset_name"] = "action_so101_wrist"
_dataset = _base["dataloader_train"]["dataloader"]["datasets"]["so101"]["dataset"]
_dataset["camera_mode"] = "wrist_image"
# Fail loudly rather than silently training the concat-view recipe again if the
# parent's config layout ever moves.
assert _dataset["camera_mode"] == "wrist_image", "camera_mode override did not take"

action_policy_so101_edge_wrist = LazyDict(_base, flags={"allow_objects": True})

cs.store(
    group="experiment",
    package="_global_",
    name="action_policy_so101_edge_wrist",
    node=action_policy_so101_edge_wrist,
)
