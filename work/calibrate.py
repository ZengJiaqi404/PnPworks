#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机标定：棋盘格 -> 内参矩阵 K、畸变系数 dist。

三种用法：
  1) 用 MVS 相机现场采图（推荐）
       ./run.sh --tool calibrate --cols 9 --rows 6 --square 25
     对着棋盘格从不同角度拍摄：空格=存图（检测到角点才存），回车=开始标定，ESC=放弃
  2) 用已经拍好的图片
       ./run.sh --tool calibrate --images assets/calib_frames
  3) 用 USB 摄像头
       ./run.sh --tool calibrate --webcam 0

注意：--cols/--rows 是「内角点」数量（黑白棋盘内部交点），不是格子数。
常用 9x6（宽 9 个内角点、高 6 个）；方格边长按你板子的实际尺寸填 mm。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
def find_corners(gray: np.ndarray, pattern: tuple[int, int], sb: bool = True,
                 fast: bool = False):
    """优先用 SB 检测器（亚像素、鲁棒），失败再退回经典检测器。

    fast=True 用于视频抽帧扫描（SB 不开 exhaustive，快 5~10 倍），
    最终标定仍用 fast=False 的高精度检测。
    """
    if sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = 0 if fast else (cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
        if ok:
            return True, corners.astype(np.float32)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not ok:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def collect_from_camera(pattern, source=None, camera_index=0, exposure=0.0, gain=0.0,
                        out_dir: Path | None = None, cols=9, rows=6) -> tuple[list, tuple[int, int]]:
    """现场采图，返回 (图像灰度列表, (w,h))。"""
    from camera import LatestFrame, open_source

    buf = LatestFrame()
    grab = open_source(buf, source=source, camera_index=camera_index,
                       exposure_us=exposure, gain=gain)
    grab.start()

    frames: list[np.ndarray] = []
    shape = None
    win = "calibrate: SPACE=save  ENTER=finish  ESC=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    seq = -1
    print("[calib] 把棋盘格放到画面里，不同位置/角度各拍一张，至少 8 张。", flush=True)
    try:
        while True:
            got = buf.get(seq, timeout=1.0)
            if got is None:
                if grab.error:
                    print(f"[calib][error] {grab.error}")
                    break
                if not grab.is_alive():
                    break
                continue
            seq, frame, _ = got
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            shape = gray.shape
            ok, corners = find_corners(gray, pattern)
            vis = frame.copy()
            if ok:
                cv2.drawChessboardCorners(vis, pattern, corners, ok)
            tip = f"saved {len(frames)} imgs | {'FOUND' if ok else 'no board'}"
            cv2.putText(vis, tip, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0) if ok else (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):  # Enter
                break
            elif key == 27:  # ESC
                frames = []
                break
            elif key == 32 and ok:  # SPACE
                frames.append(gray.copy())
                if out_dir:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_dir / f"calib_{len(frames):02d}.png"), frame)
                print(f"[calib] 已保存第 {len(frames)} 张", flush=True)
    finally:
        grab.stop()
        cv2.destroyAllWindows()
    return frames, shape


def collect_from_video(video: Path, pattern, stride: int = 8,
                      max_samples: int = 40) -> list[np.ndarray]:
    """从视频里自动抽帧：隔 stride 帧检测一次棋盘格，保留能用且姿态有差异的帧。

    比手动按空格拍图省事：拿相机对着棋盘格走一圈录一段视频即可。
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"[calib] 打不开视频 {video}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[calib] 视频 {video.name}: {total} 帧 @ {fps:.1f} fps，"
          f"每 {stride} 帧检测一次，最多保留 {max_samples} 帧")

    kept: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern, fast=True)
            if found:
                c = corners.reshape(-1, 2).mean(axis=0)
                # 去重：和已保留的样本位置太近就跳过（保证视角多样）
                if all(np.linalg.norm(c - pc) > 18 for pc in centers):
                    kept.append(gray.copy())
                    centers.append(c)
                    print(f"  帧 {idx:5d}  角点 {len(corners):2d}  已保留 {len(kept):2d}")
                    if len(kept) >= max_samples:
                        break
        idx += 1
    cap.release()

    if len(kept) >= 3:
        cs = np.array(centers)
        spread = cs.max(axis=0) - cs.min(axis=0)
        print(f"[calib] 样本位置覆盖范围: {spread[0]:.0f} x {spread[1]:.0f} px"
              f"（越大越好，建议覆盖画面 1/3 以上）")
        if min(spread) < 120:
            print("[calib][warn] 棋盘格位置变化太小，标定结果会不稳；建议录视频时让板子铺满画面不同区域")
    return kept


# --------------------------------------------------------------------------- #
def run_calibration(grays: list[np.ndarray], pattern: tuple[int, int], square_mm: float,
                    out_path: Path) -> dict:
    cols, rows = pattern
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)

    obj_points, img_points = [], []
    shape = None
    for i, gray in enumerate(grays, 1):
        shape = gray.shape
        ok, corners = find_corners(gray, pattern)
        if not ok:
            print(f"  [{i}/{len(grays)}] 未找到棋盘格，跳过")
            continue
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  [{i}/{len(grays)}] 角点 {corners.shape[0]} 个 OK")

    if len(obj_points) < 3:
        raise SystemExit(f"[calib] 有效图片只有 {len(obj_points)} 张，至少需要 3 张（建议 ≥10）")

    h, w = shape

    def fit(objs, imgs, flags):
        return cv2.calibrateCamera(objs, imgs, (w, h), None, None, flags=flags)

    def view_errors(objs, imgs, K, dist, rvecs, tvecs):
        out = []
        for i in range(len(objs)):
            proj, _ = cv2.projectPoints(objs[i], rvecs[i], tvecs[i], K, dist)
            out.append(float(np.linalg.norm(imgs[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1).mean()))
        return np.array(out)

    # 先用 4 参数（fix k3，更稳），再做异常样本剔除后重算
    flags = 0 if len(obj_points) >= 15 else cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = fit(obj_points, img_points, flags)
    errs = view_errors(obj_points, img_points, K, dist, rvecs, tvecs)

    for _ in range(3):
        thr = max(0.8, float(np.median(errs)) * 2.0)
        keep = errs <= thr
        if keep.all() or keep.sum() < 8:
            break
        dropped = np.where(~keep)[0]
        print(f"[calib] 剔除 {len(dropped)} 个异常样本（误差 > {thr:.2f} px，"
              f"索引 {[int(i) + 1 for i in dropped]}）")
        obj_points = [o for o, k in zip(obj_points, keep) if k]
        img_points = [p_ for p_, k in zip(img_points, keep) if k]
        flags = 0 if len(obj_points) >= 15 else cv2.CALIB_FIX_K3
        rms, K, dist, rvecs, tvecs = fit(obj_points, img_points, flags)
        errs = view_errors(obj_points, img_points, K, dist, rvecs, tvecs)

    for i, e in enumerate(errs, 1):
        print(f"  image {i:2d}: reproj err = {e:.4f} px")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    fov_x = 2 * np.degrees(np.arctan(w / (2 * fx)))
    fov_y = 2 * np.degrees(np.arctan(h / (2 * fy)))

    print("\n===== 标定结果 =====")
    print(f"  图像尺寸: {w}x{h}   最终样本: {len(obj_points)}")
    print(f"  RMS 重投影误差: {rms:.4f} px（<0.5 算好，<1.0 可接受）")
    print(f"  逐图误差: 中位数 {np.median(errs):.3f} px，最大 {errs.max():.3f} px")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={K[0, 2]:.2f}  cy={K[1, 2]:.2f}")
    print(f"  f/w = {fx / w:.3f}   （约 1.0~2.5 属正常，越小视场越广）")
    print(f"  视场角: {fov_x:.1f}° x {fov_y:.1f}°")
    print(f"  畸变系数: {dist.ravel()[:5].round(5).tolist()}")
    if abs(fx - fy) / max(fx, fy) > 0.05:
        print("  [warn] fx 与 fy 差 >5%，像素可能不是方形或有畸变未建模")
    if abs(K[0, 2] - w / 2) > 0.1 * w or abs(K[1, 2] - h / 2) > 0.1 * h:
        print("  [warn] 主点偏离画面中心较多，检查标定样本覆盖是否偏一侧")

    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms", float(rms))
    fs.write("pattern_cols", cols)
    fs.write("pattern_rows", rows)
    fs.write("square_mm", float(square_mm))
    fs.write("samples", len(obj_points))
    fs.release()
    print(f"\n[calib] 已保存 {out_path}")
    return {"K": K, "dist": dist, "rms": rms, "size": (w, h)}


# --------------------------------------------------------------------------- #
def load_calibration(path: Path) -> dict | None:
    """读取 calib.yaml；不存在返回 None。"""
    if not path.is_file():
        return None
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    w = int(fs.getNode("image_width").real())
    h = int(fs.getNode("image_height").real())
    rms = float(fs.getNode("rms").real())
    fs.release()
    return {"K": K, "dist": dist, "size": (w, h), "rms": rms}


def demo_intrinsics(width: int, height: int) -> dict:
    """没有标定文件时的兜底内参（只用于跑通流程，精度不可用于验收）。"""
    f = 1.2 * width
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], np.float64)
    dist = np.zeros((1, 5), np.float64)
    return {"K": K, "dist": dist, "size": (width, height), "rms": float("nan"), "demo": True}


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="棋盘格相机标定")
    p.add_argument("--cols", type=int, default=9, help="内角点列数")
    p.add_argument("--rows", type=int, default=6, help="内角点行数")
    p.add_argument("--square", type=float, default=25.0, help="方格边长 mm")
    p.add_argument("--images", type=Path, default=None, help="用已有图片目录")
    p.add_argument("--video", type=Path, default=None, help="用一段录像抽帧标定（推荐）")
    p.add_argument("--video-stride", type=int, default=8, help="视频每 N 帧检测一次")
    p.add_argument("--max-samples", type=int, default=40, help="最多保留多少帧样本")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--webcam", type=int, default=None, help="改用 USB 摄像头序号")
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--gain", type=float, default=0.0)
    p.add_argument("--output", type=Path, default=HERE / "calib.yaml")
    p.add_argument("--save-frames", action="store_true", help="采集时把原图存到 assets/calib_frames")
    args = p.parse_args(argv)

    pattern = (args.cols, args.rows)

    if args.video:
        grays = collect_from_video(args.video, pattern, args.video_stride, args.max_samples)
        if len(grays) < 3:
            print(f"[calib] 只抽到 {len(grays)} 帧有效棋盘格，太小了；"
                  f"请让板子完整入镜、对焦清晰，再录一段")
            return 3
        print(f"[calib] 抽到 {len(grays)} 帧有效样本")
    elif args.images:
        files = sorted([f for f in args.images.iterdir()
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        if not files:
            print(f"[calib] {args.images} 里没有图片")
            return 2
        grays = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2GRAY) for f in files]
        print(f"[calib] 读取 {len(grays)} 张图片")
    else:
        src = args.webcam if args.webcam is not None else None
        save_dir = HERE / "assets" / "calib_frames" if args.save_frames else None
        grays, _ = collect_from_camera(
            pattern, source=src, camera_index=args.camera_index,
            exposure=args.exposure, gain=args.gain, out_dir=save_dir,
        )
        if not grays:
            print("[calib] 没有采集到图片，退出")
            return 3

    run_calibration(grays, pattern, args.square, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
