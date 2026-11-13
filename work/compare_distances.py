#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比不同距离下的定位结果：实测距离 vs 卷尺距离、稳定性、轨迹。

用法：./run.sh --tool compare --near docs/dist_70cm.csv --far docs/dist_120cm.csv
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


def load(path: Path):
    t, d, xyz = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    t0 = float(rows[0]["t"]) if rows else 0.0
    for r in rows:
        if not r.get("dist"):
            continue
        t.append(float(r["t"]) - t0)
        d.append(float(r["dist"]))
        xyz.append([float(r["raw_x"]), float(r["raw_y"]), float(r["raw_z"])])
    return np.array(t), np.array(d), np.array(xyz)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--near", type=Path, default=HERE / "docs" / "dist_70cm.csv")
    p.add_argument("--near-mm", type=float, default=700.0, help="近档卷尺距离 mm")
    p.add_argument("--far", type=Path, default=HERE / "docs" / "dist_120cm.csv")
    p.add_argument("--far-mm", type=float, default=1200.0, help="远档卷尺距离 mm")
    p.add_argument("--out", type=Path, default=HERE / "docs" / "distance_compare.png")
    args = p.parse_args(argv)

    tn, dn, pn = load(args.near)
    tf, df, pf = load(args.far)
    if len(dn) < 5 or len(df) < 5:
        print("[compare] 数据不足")
        return 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    ax.plot(tn, dn, lw=0.8, color="#2ca02c", label=f"near (median {np.median(dn):.3f} m)")
    ax.plot(tf, df, lw=0.8, color="#1f77b4", label=f"far (median {np.median(df):.3f} m)")
    ax.axhline(args.near_mm / 1000, ls="--", lw=1, color="#2ca02c", alpha=0.6,
               label=f"tape {args.near_mm/1000:.2f} m")
    ax.axhline(args.far_mm / 1000, ls="--", lw=1, color="#1f77b4", alpha=0.6,
               label=f"tape {args.far_mm/1000:.2f} m")
    ax.set_xlabel("t (s)"); ax.set_ylabel("distance (m)")
    ax.set_title("Measured distance over time")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(pn[:, 0], pn[:, 2], ".", ms=3, color="#2ca02c", alpha=0.6, label="near")
    ax.plot(pf[:, 0], pf[:, 2], ".", ms=3, color="#1f77b4", alpha=0.6, label="far")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("Top view X-Z (raw PnP)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    names = ["near\n(tape 0.70)", "far\n(tape 1.20)"]
    measured = [np.median(dn), np.median(df)]
    nominal = [args.near_mm / 1000, args.far_mm / 1000]
    x = np.arange(2); w = 0.35
    b1 = ax.bar(x - w / 2, nominal, w, label="tape", color="#bbbbbb")
    b2 = ax.bar(x + w / 2, measured, w, label="PnP median", color=["#2ca02c", "#1f77b4"])
    for i, (n, m) in enumerate(zip(nominal, measured)):
        err = (m - n) / n * 100
        ax.text(i, max(n, m) + 0.05, f"{err:+.1f}%", ha="center", fontsize=10, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("distance (m)")
    ax.set_title("Tape vs PnP distance")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, max(nominal + measured) * 1.25)

    fig.suptitle("Armor PnP at two distances (same camera, same calibration)", fontsize=12)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[compare] 已保存 {args.out}")
    for name, d, nom in (("near", dn, args.near_mm / 1000), ("far", df, args.far_mm / 1000)):
        print(f"  {name}: 样本 {len(d):3d}  中位 {np.median(d):.3f} m  std {d.std()*1000:5.1f} mm  "
              f"卷尺 {nom:.2f} m  偏差 {(np.median(d)-nom)/nom*100:+.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
