# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Open-loop episode evaluation of a Cosmos3 SO-ARM101 action policy against the val split.

For every validation episode the episode is cut into consecutive chunk-length windows (stride =
chunk length, last window clamped to the episode end). Each window's first frame (the training
``concat_view`` image: third-person | wrist) is sent to a running
``action_policy_server_libero`` ``/predict`` endpoint, which returns the denormalised 10-D
frame-wise action chunk ``[dxyz, rot6d, gripper]`` and the generated future frames. The predicted
deltas are integrated from the ground-truth pose at the window start (``T_{i+1} = T_i @ Δ_i``),
so the stitched trajectory is what a closed-loop controller would have commanded, re-anchored at
every chunk boundary.

Outputs per (checkpoint, episode): ``ep<NNN>.mp4`` (ground-truth frames on top, generated frames
below, a trajectory / gripper HUD on the right) and ``ep<NNN>.json`` (per-frame GT and predicted
pose, per-window metrics). ``summary.json`` aggregates ADE / FDE / rotation / gripper errors.

Usage (venv active, server up):
    python examples/so101_eval/eval_so101_val.py --root <val v3.0 dir> --stats <so101 stats json>
        --server http://127.0.0.1:8000 --checkpoint-name iter_000002500 --out /data/cosmos3/eval
        --episodes 0-7
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import LIBEROLeRobotDataset

TASK = "Pick up the test-tube and place it in the holder on the helicopter"
CHUNK = 16
GT_COLOR = (67, 118, 168)  # gantry blue
PR_COLOR = (214, 140, 52)  # bronze
HUD_W = 272


# ----------------------------------------------------------------------------- pose helpers
def eef9d_to_T(e: np.ndarray) -> np.ndarray:
    """Flywheel eef9d [t, R[0,:], R[1,:]] -> 4x4 (rows Gram-Schmidt)."""
    r0 = e[3:6] / np.linalg.norm(e[3:6])
    r1 = e[6:9] - np.dot(r0, e[6:9]) * r0
    r1 /= np.linalg.norm(r1)
    T = np.eye(4)
    T[0, :3], T[1, :3], T[2, :3] = r0, r1, np.cross(r0, r1)
    T[:3, 3] = e[0:3]
    return T


def rot6d_cols_to_R(v: np.ndarray) -> np.ndarray:
    """Cosmos rot6d (first two COLUMNS of R, column-major) -> 3x3."""
    c0 = v[0:3] / np.linalg.norm(v[0:3])
    c1 = v[3:6] - np.dot(c0, v[3:6]) * c0
    c1 /= np.linalg.norm(c1)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=1)


def action10_to_delta_T(a: np.ndarray) -> np.ndarray:
    D = np.eye(4)
    D[:3, :3] = rot6d_cols_to_R(a[3:9])
    D[:3, 3] = a[0:3]
    return D


def integrate(T0: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """T0 + [k,10] frame-wise deltas -> [k+1,4,4] absolute poses (pose 0 = T0)."""
    out = [T0]
    T = T0.copy()
    for a in actions:
        T = T @ action10_to_delta_T(a)
        out.append(T)
    return np.stack(out)


def rot_err_deg(Ta: np.ndarray, Tb: np.ndarray) -> float:
    Rrel = Ta[:3, :3].T @ Tb[:3, :3]
    return float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))


def rpy_deg(T: np.ndarray) -> np.ndarray:
    return R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)


def denormalize(z: np.ndarray, stats: dict) -> np.ndarray:
    """Inverse of the quantile / quantile_rot affine (server-side formula) using global_raw."""
    q01 = np.asarray(stats["global_raw"]["q01"])
    q99 = np.asarray(stats["global_raw"]["q99"])
    return z * (q99 - q01) / 2 + (q99 + q01) / 2


# ----------------------------------------------------------------------------- data access
def load_episode_poses(root: Path) -> dict[int, dict[str, np.ndarray]]:
    """episode_index -> {'eef9d': [n,9], 'gripper': [n], 'row0': int} from the v3.0 data files."""
    table = pq.read_table(
        sorted((root / "data").glob("chunk-*/file-*.parquet")),
        columns=["index", "episode_index", "observation.state.eef9d", "action.gripper"],
    ).to_pandas()
    out = {}
    for ep, df in table.groupby("episode_index"):
        df = df.sort_values("index")
        out[int(ep)] = {
            "eef9d": np.stack(df["observation.state.eef9d"].to_numpy()).astype(np.float64),
            "gripper": np.stack(df["action.gripper"].to_numpy()).reshape(-1).astype(np.float64),
            "row0": int(df["index"].iloc[0]),
        }
    return out


def png_b64(img_hwc: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_hwc).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_png_to_np(s: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))


# ----------------------------------------------------------------------------- rendering
def _font(size: int):
    for cand in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frame(
    gt_img: np.ndarray,
    gen_img: np.ndarray,
    gt_T: np.ndarray,
    pr_T: np.ndarray,
    gt_grip: np.ndarray,
    pr_grip: np.ndarray,
    step: int,
    ep_xy_bounds: tuple[float, float, float, float],
    label: str,
    pos_err_cm: float,
) -> np.ndarray:
    """Compose GT (top) | generated (bottom) | HUD (right) into one 768x512 frame."""
    H, W = 256, 512
    canvas = Image.new("RGB", (W + HUD_W, 2 * H), (18, 20, 22))
    canvas.paste(Image.fromarray(gt_img).resize((W, H)), (0, 0))
    canvas.paste(Image.fromarray(gen_img).resize((W, H)), (0, H))
    d = ImageDraw.Draw(canvas, "RGBA")
    f_small, f_big = _font(12), _font(15)
    d.text((6, 4), "ground truth", font=f_small, fill=(230, 230, 230))
    d.text((6, H + 4), "generated by the policy", font=f_small, fill=(230, 230, 230))

    # chunk path inset on the generated frame (bottom-right corner)
    ix, iy, iw, ih = W - 132, 2 * H - 132, 124, 124
    d.rectangle([ix, iy, ix + iw, iy + ih], fill=(0, 0, 0, 140), outline=(90, 90, 90, 200))
    # chunk-local bounds (episode-wide bounds collapse a 1 s chunk to a dot); >= 3 cm extent
    pts_xy = np.concatenate([gt_T[:, :2, 3], pr_T[:, :2, 3]])
    cx_, cy_ = pts_xy.mean(0)
    half = max(0.015, (pts_xy.max(0) - pts_xy.min(0)).max() / 2 + 0.004)
    xmin, ymin = cx_ - half, cy_ - half
    s = (iw - 12) / (2 * half)

    def to_inset(p):
        return (ix + 6 + (p[0] - xmin) * s, iy + ih - 6 - (p[1] - ymin) * s)

    for traj, col in ((gt_T, GT_COLOR), (pr_T, PR_COLOR)):
        pts = [to_inset(T[:3, 3]) for T in traj]
        if len(pts) > 1:
            d.line(pts, fill=col + (230,), width=2)
    d.ellipse([to_inset(pr_T[step][:3, 3])[0] - 3, to_inset(pr_T[step][:3, 3])[1] - 3, to_inset(pr_T[step][:3, 3])[0] + 3, to_inset(pr_T[step][:3, 3])[1] + 3], fill=PR_COLOR)
    d.text((ix + 4, iy + 2), f"chunk xy  ({2 * half * 100:.0f} cm box)", font=f_small, fill=(200, 200, 200))

    # HUD panel
    hx = W + 10
    d.text((hx, 8), label, font=f_big, fill=(235, 235, 235))
    d.text((hx, 30), f"step {step:2d}/{len(gt_T) - 1}   pos err {pos_err_cm:5.1f} cm", font=f_small, fill=(200, 200, 200))
    # z over the chunk
    zx, zy, zw, zh = hx, 60, HUD_W - 20, 120
    d.rectangle([zx, zy, zx + zw, zy + zh], outline=(80, 80, 80))
    d.text((zx + 2, zy + 2), "z (m) over chunk", font=f_small, fill=(170, 170, 170))
    zs = np.concatenate([gt_T[:, 2, 3], pr_T[:, 2, 3]])
    zmin, zmax = zs.min() - 0.005, zs.max() + 0.005
    for traj, col in ((gt_T, GT_COLOR), (pr_T, PR_COLOR)):
        pts = [(zx + 4 + i * (zw - 8) / max(len(traj) - 1, 1), zy + zh - 4 - (T[2, 3] - zmin) / (zmax - zmin) * (zh - 24)) for i, T in enumerate(traj)]
        d.line(pts, fill=col, width=2)
    cx = zx + 4 + step * (zw - 8) / max(len(gt_T) - 1, 1)
    d.line([(cx, zy + 18), (cx, zy + zh - 2)], fill=(200, 200, 200, 120), width=1)
    # gripper bars
    gy = 200
    d.text((hx, gy), "gripper  (0 = closed)", font=f_small, fill=(170, 170, 170))
    for i, (val, col, name) in enumerate(((gt_grip[step], GT_COLOR, "gt"), (pr_grip[step], PR_COLOR, "pred"))):
        by = gy + 20 + i * 26
        d.rectangle([hx, by, hx + HUD_W - 20, by + 16], outline=(80, 80, 80))
        d.rectangle([hx, by, hx + (HUD_W - 20) * float(np.clip(val / 0.65, 0, 1)), by + 16], fill=col)
        d.text((hx + HUD_W - 16, by + 1), name, font=f_small, fill=(200, 200, 200), anchor="ra")
    # rpy
    ry = 290
    gr, pr_ = rpy_deg(gt_T[step]), rpy_deg(pr_T[step])
    d.text((hx, ry), "roll / pitch / yaw (deg)", font=f_small, fill=(170, 170, 170))
    d.text((hx, ry + 18), f"gt   {gr[0]:6.1f} {gr[1]:6.1f} {gr[2]:6.1f}", font=f_small, fill=GT_COLOR)
    d.text((hx, ry + 34), f"pred {pr_[0]:6.1f} {pr_[1]:6.1f} {pr_[2]:6.1f}", font=f_small, fill=PR_COLOR)
    # xyz
    xy = 350
    d.text((hx, xy), "xyz (m)", font=f_small, fill=(170, 170, 170))
    g, p = gt_T[step][:3, 3], pr_T[step][:3, 3]
    d.text((hx, xy + 18), f"gt   {g[0]:6.3f} {g[1]:6.3f} {g[2]:6.3f}", font=f_small, fill=GT_COLOR)
    d.text((hx, xy + 34), f"pred {p[0]:6.3f} {p[1]:6.3f} {p[2]:6.3f}", font=f_small, fill=PR_COLOR)
    d.text((hx, 2 * H - 40), "blue = ground truth", font=f_small, fill=GT_COLOR)
    d.text((hx, 2 * H - 24), "bronze = predicted, re-anchored per chunk", font=f_small, fill=PR_COLOR)
    return np.asarray(canvas)


def encode_mp4(frames: list[np.ndarray], out: Path, fps: int) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="so101eval_"))
    try:
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(tmp / f"{i:05d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "%05d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "27", "-movflags", "+faststart", str(out)],
            check=True,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------------------------------------------------------- main
def parse_episodes(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return [e for e in out if e < n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--checkpoint-name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", default=None, help="e.g. 0-7 or 0,3,5 (positions in the split)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--camera-mode",
        default="concat_view",
        choices=("concat_view", "wrist_image", "image"),
        help="must match what the checkpoint trained on (action_policy_so101_edge = concat_view, "
        "action_policy_so101_edge_wrist = wrist_image)",
    )
    args = ap.parse_args()

    stats = json.load(open(args.stats))
    ds = LIBEROLeRobotDataset(
        root=str(args.root), image_size=args.image_size, chunk_length=CHUNK, fps=args.fps, mode="wam",
        split="full", val_ratio=0.5, seed=0, camera_mode=args.camera_mode, action_space="frame_wise_relative",
        rotation_space="6d", pose_coordinate_frame="native", action_normalization="quantile_rot",
        action_stats_path=str(args.stats), embodiment_type="so101",
    )
    poses = load_episode_poses(args.root)
    out_dir = args.out / args.checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)
    info = requests.get(f"{args.server}/info", timeout=30).json()
    print("server:", {k: info[k] for k in list(info)[:8]}, flush=True)

    n_eps = len(ds._ep_vals)
    summary = {"checkpoint": args.checkpoint_name, "server_info": info, "episodes": {}}
    for ep_pos in parse_episodes(args.episodes, n_eps):
        ep_index = int(ds._ep_vals[ep_pos])
        first_idx = int(ds._valid_cum[ep_pos - 1]) if ep_pos > 0 else 0
        n_valid = int(ds._valid_cum[ep_pos]) - first_idx  # starts with a full 17-frame window
        ep_len = n_valid + CHUNK
        P = poses[ep_index]
        assert len(P["eef9d"]) == ep_len, (ep_index, len(P["eef9d"]), ep_len)
        starts = list(range(0, n_valid, CHUNK))
        if starts[-1] != n_valid - 1:
            starts.append(n_valid - 1)  # clamp the last window to the episode end
        xs, ys = P["eef9d"][:, 0], P["eef9d"][:, 1]
        bounds = (xs.min() - 0.01, xs.max() + 0.01, ys.min() - 0.01, ys.max() + 0.01)
        frames_out: list[np.ndarray] = []
        per_frame: dict[str, list] = {k: [] for k in ("t", "frame", "window", "gt_xyz", "pr_xyz", "gt_rpy", "pr_rpy", "gt_grip", "pr_grip", "pos_err_cm", "rot_err_deg")}
        windows = []
        t_ep = time.time()
        for w, st in enumerate(starts):
            sample = ds[first_idx + st]
            video = sample["video"].permute(1, 2, 3, 0).numpy()  # [17,H,W,3] uint8
            cond = video[0]
            t0 = time.time()
            resp = requests.post(
                f"{args.server}/predict",
                json={"image": png_b64(cond), "prompt": TASK, "domain_name": "so101", "image_size": args.image_size},
                timeout=600,
            ).json()
            if resp.get("error"):
                raise RuntimeError(resp["error"])
            pred = np.asarray(resp["action"], dtype=np.float64)[:CHUNK, :10]
            gen = [b64_png_to_np(s) for s in resp.get("video", [])]
            dt = time.time() - t0
            gt_act = denormalize(sample["action"].numpy()[:CHUNK, :10].astype(np.float64), stats)
            T0 = eef9d_to_T(P["eef9d"][st])
            gt_T = np.stack([eef9d_to_T(e) for e in P["eef9d"][st : st + CHUNK + 1]])
            pr_T = integrate(T0, pred)
            gt_grip = P["gripper"][st : st + CHUNK + 1]
            pr_grip = np.concatenate([[gt_grip[0]], pred[:, 9]])
            pos_err = np.linalg.norm(gt_T[:, :3, 3] - pr_T[:, :3, 3], axis=1) * 100
            rot_err = np.array([rot_err_deg(a, b) for a, b in zip(gt_T, pr_T)])
            grip_err = np.abs(gt_grip - pr_grip) * 100
            act_mse = float(np.mean((gt_act - pred) ** 2))
            windows.append({
                "start": st, "ade_cm": float(pos_err[1:].mean()), "fde_cm": float(pos_err[-1]),
                "rot_ade_deg": float(rot_err[1:].mean()), "rot_fde_deg": float(rot_err[-1]),
                "grip_mae_pct": float(grip_err[1:].mean()), "action_mse": act_mse, "infer_s": dt, "gen_frames": len(gen),
            })
            print(f"  ep {ep_index:3d} window {w:2d}/{len(starts)} start {st:3d}  ADE {pos_err[1:].mean():5.2f} cm  FDE {pos_err[-1]:5.2f} cm  rot {rot_err[1:].mean():4.1f}°  grip {grip_err[1:].mean():4.1f}%  {dt:4.1f}s  gen={len(gen)}", flush=True)
            # steps 1..16 of this window (step 0 is the conditioning frame, drawn only for the first window)
            step_range = range(0, CHUNK + 1) if w == 0 else range(1, CHUNK + 1)
            last_frame_written = per_frame["frame"][-1] if per_frame["frame"] else -1
            for k in step_range:
                frame_idx = st + k
                if frame_idx <= last_frame_written:
                    continue  # overlap from the clamped last window
                gen_img = gen[min(k, len(gen) - 1)] if gen else video[k]
                if not args.no_video:
                    frames_out.append(render_frame(video[k], gen_img, gt_T, pr_T, gt_grip, pr_grip, k, bounds,
                                                   f"{args.checkpoint_name}  ep {ep_index}  w{w}", float(pos_err[k])))
                per_frame["t"].append(frame_idx / args.fps)
                per_frame["frame"].append(frame_idx)
                per_frame["window"].append(w)
                per_frame["gt_xyz"].append(gt_T[k][:3, 3].round(4).tolist())
                per_frame["pr_xyz"].append(pr_T[k][:3, 3].round(4).tolist())
                per_frame["gt_rpy"].append(rpy_deg(gt_T[k]).round(2).tolist())
                per_frame["pr_rpy"].append(rpy_deg(pr_T[k]).round(2).tolist())
                per_frame["gt_grip"].append(round(float(gt_grip[k]), 4))
                per_frame["pr_grip"].append(round(float(pr_grip[k]), 4))
                per_frame["pos_err_cm"].append(round(float(pos_err[k]), 3))
                per_frame["rot_err_deg"].append(round(float(rot_err[k]), 2))
        if not args.no_video:
            encode_mp4(frames_out, out_dir / f"ep{ep_index:03d}.mp4", args.fps)
        ep_summary = {
            "episode_index": ep_index, "n_frames": ep_len, "n_windows": len(starts),
            "ade_cm": float(np.mean([w["ade_cm"] for w in windows])),
            "fde_cm": float(np.mean([w["fde_cm"] for w in windows])),
            "rot_ade_deg": float(np.mean([w["rot_ade_deg"] for w in windows])),
            "grip_mae_pct": float(np.mean([w["grip_mae_pct"] for w in windows])),
            "action_mse": float(np.mean([w["action_mse"] for w in windows])),
            "eval_s": time.time() - t_ep,
        }
        json.dump({"checkpoint": args.checkpoint_name, "episode": ep_summary, "windows": windows, "frames": per_frame},
                  open(out_dir / f"ep{ep_index:03d}.json", "w"))
        summary["episodes"][str(ep_index)] = ep_summary
        print(f"episode {ep_index}: ADE {ep_summary['ade_cm']:.2f} cm  FDE {ep_summary['fde_cm']:.2f} cm  rot {ep_summary['rot_ade_deg']:.1f}°  grip {ep_summary['grip_mae_pct']:.1f}%  ({ep_summary['eval_s']:.0f}s)", flush=True)
    eps = list(summary["episodes"].values())
    summary["mean"] = {k: float(np.mean([e[k] for e in eps])) for k in ("ade_cm", "fde_cm", "rot_ade_deg", "grip_mae_pct", "action_mse")} if eps else {}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    print("summary:", summary["mean"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
