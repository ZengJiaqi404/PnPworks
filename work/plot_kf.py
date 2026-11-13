#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""滤波效果对比图。

两种数据来源：
  1) 实测 CSV：./run.sh --tool plot --csv docs/run_log.csv
  2) 仿真真值：./run.sh --tool plot --synthetic      （有真值，能定量算误差，最适合写进报告）
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_csv(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        keys = [k for k in (rd.fieldnames or []) if k != "cls"]
        for row in rd:
            item = {}
            for k in keys:
                try:
                    item[k] = float(row[k])
                except (TypeError, ValueError):
                    item[k] = np.nan
            rows.append(item)
    return {k: np.array([r.get(k, np.nan) for r in rows]) for k in keys}


def simulate(n: int = 180, fps: float = 30.0, sigma: float = 0.02, seed: int = 0):
    """仿真一个靠近相机并横移的目标，返回 (t, truth, meas, est, pred)。"""
    from kf import ArmorKalman

    rng = np.random.default_rng(seed)
    dt = 1 / fps
    t = np.arange(n) * dt
    # 真实运动：Z 从 3.0m 匀速逼近到 1.2m，同时 X 横移，Y 有轻微起伏
    truth = np.stack([
        0.6 * np.sin(1.6 * t),
        0.03 * np.sin(4.0 * t),
        3.0 - 1.6 * t / t[-1],
    ], 1)
    meas = truth + rng.normal(0, sigma, truth.shape)

    kf = ArmorKalman(pos_std=0.04, acc_std=3.0)
    est, pred = [], []
    for i in range(n):
        e, p = kf.update(meas[i], i * dt)
        est.append(e)
        pred.append(p)
    return t, truth, meas, np.array(est), np.array(pred)


def plot(t, raw, kf, pred, out: Path, truth=None, title_suffix: str = ""):
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.34, wspace=0.3)

    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    ax3.plot(raw[:, 0], raw[:, 1], raw[:, 2], "o", ms=3, color="#ff6b6b", alpha=0.55, label="PnP raw")
    ax3.plot(kf[:, 0], kf[:, 1], kf[:, 2], "-", lw=2, color="#1f77b4", label="Kalman")
    if truth is not None:
        ax3.plot(truth[:, 0], truth[:, 1], truth[:, 2], "k--", lw=1.4, label="ground truth")
    ax3.set_xlabel("X (m)"); ax3.set_ylabel("Y (m)"); ax3.set_zlabel("Z (m)")
    ax3.set_title("3D trajectory")
    ax3.legend(fontsize=8)
    ax3.view_init(elev=20, azim=-62)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(raw[:, 0], raw[:, 2], "o", ms=3, color="#ff6b6b", alpha=0.55, label="raw")
    ax2.plot(kf[:, 0], kf[:, 2], "-", lw=2, color="#1f77b4", label="filtered")
    if truth is not None:
        ax2.plot(truth[:, 0], truth[:, 2], "k--", lw=1.4, label="truth")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Z (m)"); ax2.set_title("Top view X-Z")
    ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

    axd = fig.add_subplot(gs[0, 2])
    axd.plot(t, np.linalg.norm(raw, axis=1), ".", ms=3, color="#ff6b6b", alpha=0.55, label="raw")
    axd.plot(t, np.linalg.norm(kf, axis=1), "-", lw=2, color="#1f77b4", label="filtered")
    if truth is not None:
        axd.plot(t, np.linalg.norm(truth, axis=1), "k--", lw=1.4, label="truth")
    axd.set_xlabel("t (s)"); axd.set_ylabel("distance (m)"); axd.set_title("Distance to camera")
    axd.grid(alpha=0.3); axd.legend(fontsize=8)

    for i, axis in enumerate(("x", "y", "z")):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(t, raw[:, i], ".", ms=3, color="#ff6b6b", alpha=0.5, label="raw")
        ax.plot(t, kf[:, i], "-", lw=2, color="#1f77b4", label="filtered")
        ax.plot(t, pred[:, i], "--", lw=1.1, color="#2ca02c", alpha=0.9, label="predicted")
        if truth is not None:
            ax.plot(t, truth[:, i], "k:", lw=1.2, label="truth")
        ax.set_xlabel("t (s)"); ax.set_ylabel(f"{axis.upper()} (m)")
        ax.set_title(f"{axis.upper()}: raw / filtered / predicted")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    lines = [f"samples {len(raw)}"]
    if truth is not None:
        raw_rmse = np.sqrt(((raw - truth) ** 2).mean()) * 1000
        kf_rmse = np.sqrt(((kf - truth) ** 2).mean()) * 1000
        pred_err = np.sqrt(((pred[:-1] - truth[1:]) ** 2).mean()) * 1000
        lines += [f"raw  RMSE {raw_rmse:6.2f} mm",
                  f"kf   RMSE {kf_rmse:6.2f} mm   ({(1 - kf_rmse / raw_rmse) * 100:.0f}% better)",
                  f"pred 1-step err {pred_err:6.2f} mm"]
    else:
        delta = np.linalg.norm(raw - kf, axis=1)
        lines += [f"|raw-kf| mean {delta.mean() * 1000:.1f} mm",
                  f"|pred-kf| mean {np.linalg.norm(pred - kf, axis=1).mean() * 1000:.1f} mm"]
    fig.text(0.99, 0.34, "\n".join(lines), ha="right", va="top", family="monospace",
             fontsize=8.5, bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))

    fig.suptitle("Armor PnP + Kalman: raw vs filtered vs predicted" + title_suffix, fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"[plot] 已保存 {out}\n" + "\n".join(lines))
    return lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--synthetic", action="store_true", help="用仿真真值生成对比图")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.synthetic or args.csv is None:
        t, truth, raw, kf, pred = simulate()
        out = args.out or HERE / "docs" / "kf_synthetic.png"
        plot(t, raw, kf, pred, out, truth=truth, title_suffix=" (simulated, with ground truth)")
        return 0

    if not args.csv.is_file():
        print(f"[plot] 找不到 {args.csv}")
        return 2
    d = load_csv(args.csv)
    ok = ~np.isnan(d["raw_x"])
    if ok.sum() < 5:
        print("[plot] 有效点太少")
        return 2
    t = d["t"][ok] - d["t"][ok][0]
    raw = np.stack([d["raw_x"][ok], d["raw_y"][ok], d["raw_z"][ok]], 1)
    kf = np.stack([d["kf_x"][ok], d["kf_y"][ok], d["kf_z"][ok]], 1)
    pred = np.stack([d["pred_x"][ok], d["pred_y"][ok], d["pred_z"][ok]], 1)
    plot(t, raw, kf, pred, args.out or HERE / "docs" / "kf_video.png", title_suffix=" (real video, demo intrinsics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
