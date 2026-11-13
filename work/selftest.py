#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自检：在没有相机的情况下验证标定与 PnP 的数学正确性。

做两件事：
  1) 用已知内参合成 12 张棋盘格视图 -> 跑 calibrate.run_calibration -> 比较恢复的内参；
  2) 用已知位姿合成装甲板四个角点（加亚像素噪声）-> 跑 solve_pnp -> 比较位置/角度误差。

运行：  ./run.sh --tool selftest
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from armor import ArmorDetector, Detection, order_corners, reprojection_error, solve_pnp  # noqa: E402
from calibrate import run_calibration  # noqa: E402


# --------------------------------------------------------------------------- #
def make_board_image(cols: int, rows: int, px_per_square: int = 160) -> np.ndarray:
    """生成一张 (cols+1) x (rows+1) 格的棋盘格图片。"""
    n_w, n_h = cols + 1, rows + 1
    s = px_per_square
    img = np.full((n_h * s, n_w * s), 255, np.uint8)
    for r in range(n_h):
        for c in range(n_w):
            if (r + c) % 2 == 0:
                img[r * s:(r + 1) * s, c * s:(c + 1) * s] = 20
    return img


def synth_chessboard_views(out_dir: Path, cols: int, rows: int, square_mm: float,
                           K: np.ndarray, dist: np.ndarray, size: tuple[int, int],
                           n_views: int = 12, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    w, h = size
    square = square_mm / 1000.0
    board_img = make_board_image(cols, rows)

    # 棋盘格黑格角点在物理平面上的坐标（以格子为单位，再换算成米）
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    # 棋盘格图片的四个外角（-0.5 格 到 cols+0.5 格）
    bw, bh = board_img.shape[1], board_img.shape[0]
    src = np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]])

    out_dir.mkdir(parents=True, exist_ok=True)
    grays = []
    for i in range(n_views):
        # 随机位姿：距离 0.4~0.9 m，倾斜 0~45°
        rvec = rng.normal(0, 0.35, 3)
        tvec = np.array([rng.uniform(-0.12, 0.12), rng.uniform(-0.09, 0.09),
                         rng.uniform(0.45, 0.9)])

        # 物理平面四角：( -0.5sq, -0.5sq ) ~ ( (cols+0.5)sq, (rows+0.5)sq )
        plane = np.float32([
            [-0.5 * square, -0.5 * square, 0],
            [(cols + 0.5) * square, -0.5 * square, 0],
            [(cols + 0.5) * square, (rows + 0.5) * square, 0],
            [-0.5 * square, (rows + 0.5) * square, 0],
        ])
        proj, _ = cv2.projectPoints(plane, rvec, tvec, K, dist)
        dst = proj.reshape(-1, 2).astype(np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        canvas = cv2.warpPerspective(board_img, H, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=90)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.6)
        canvas = np.clip(canvas.astype(np.int16) + rng.normal(0, 2.0, canvas.shape), 0, 255).astype(np.uint8)
        grays.append(canvas)
        cv2.imwrite(str(out_dir / f"synth_{i:02d}.png"), canvas)
    return grays


# --------------------------------------------------------------------------- #
def test_calibration(tmp: Path) -> bool:
    cols, rows, square_mm = 9, 6, 25.0
    size = (1280, 960)
    K_true = np.array([[1150.0, 0, 640.0], [0, 1152.0, 480.0], [0, 0, 1]])
    dist_true = np.array([[-0.12, 0.06, 0.0005, -0.0004, 0.0]])

    print("=" * 68)
    print("[1] 标定自检：合成 12 张棋盘格视图")
    grays = synth_chessboard_views(tmp / "chess", cols, rows, square_mm, K_true, dist_true, size)
    print(f"    已生成 {len(grays)} 张到 {tmp / 'chess'}")

    res = run_calibration(grays, (cols, rows), square_mm, tmp / "calib_synth.yaml")
    K = res["K"]
    fx_err = abs(K[0, 0] - K_true[0, 0]) / K_true[0, 0] * 100
    fy_err = abs(K[1, 1] - K_true[1, 1]) / K_true[1, 1] * 100
    cx_err = abs(K[0, 2] - K_true[0, 2])
    cy_err = abs(K[1, 2] - K_true[1, 2])
    print(f"    恢复 fx={K[0, 0]:.1f} (真值 {K_true[0, 0]:.1f}, 误差 {fx_err:.2f}%)")
    print(f"    恢复 fy={K[1, 1]:.1f} (真值 {K_true[1, 1]:.1f}, 误差 {fy_err:.2f}%)")
    print(f"    主点误差: cx {cx_err:.1f}px, cy {cy_err:.1f}px")
    ok = fx_err < 3.0 and fy_err < 3.0 and cx_err < 15 and cy_err < 15
    print(f"    结论: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def test_pnp() -> bool:
    print("=" * 68)
    print("[2] PnP 自检：已知位姿 -> 投影 -> 反解")
    K = np.array([[1150.0, 0, 640.0], [0, 1152.0, 480.0], [0, 0, 1]])
    dist = np.zeros((1, 5))
    det = ArmorDetector.__new__(ArmorDetector)  # 只为拿 object_points，不加载模型
    w, h = 0.135, 0.125
    det.object_points = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0],
                                  [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float64)

    cases = [
        ("正对 1.5m", (0, 0, 0), (0.0, 0.0, 1.5)),
        ("偏 0.3m/1.2m", (0.1, 0.15, 0.05), (0.3, -0.1, 1.2)),
        ("大角度 3m", (0.5, -0.4, 0.25), (-0.4, 0.2, 3.0)),
    ]
    all_ok = True
    for name, rv_deg, tv in cases:
        rvec = np.radians(np.array(rv_deg, float)).reshape(3, 1)
        tvec = np.array(tv, float).reshape(3, 1)
        proj, _ = cv2.projectPoints(det.object_points, rvec, tvec, K, dist)
        pts = order_corners(proj.reshape(-1, 2))
        d = Detection(cls_id=0, cls_name="B1", conf=1.0, box=[0, 0, 1, 1], corners=pts)
        r = solve_pnp(d, det.object_points, K, dist)
        if r is None:
            print(f"    {name}: 解算失败")
            all_ok = False
            continue
        pos_err = np.linalg.norm(r.tvec.ravel() - tvec.ravel()) * 1000
        R1, _ = cv2.Rodrigues(rvec)
        R2, _ = cv2.Rodrigues(r.rvec)
        ang = np.degrees(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))
        print(f"    {name}: 位置误差 {pos_err:.3f} mm, 角度误差 {ang:.3f}°, "
              f"距离 {r.distance:.3f} m, reproj {reprojection_error(r, det.object_points, K, dist):.3f} px")
        all_ok &= pos_err < 1.0 and ang < 0.5
    print(f"    结论: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        ok1 = test_calibration(tmp)
        ok2 = test_pnp()
    print("=" * 68)
    print(f"总体: {'全部通过' if ok1 and ok2 else '有失败项'}")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    sys.exit(main())
