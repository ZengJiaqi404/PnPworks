#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板识别 + 四角点排序 + PnP 位姿解算。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# 小装甲板外框尺寸（米）。用卡尺量你的实物，不一致就用 --armor-w/--armor-h 改。
ARMOR_W_DEFAULT = 0.224
ARMOR_H_DEFAULT = 0.054


# --------------------------------------------------------------------------- #
def order_corners(pts: np.ndarray) -> np.ndarray:
    """把 4 个点整理成固定顺序：左上、右上、右下、左下。

    不依赖模型输出顺序（该权重的关键点顺序并不稳定），而是按几何位置排序：
    先按绕质心角度排一圈，再从最接近左上的点开始，保证是 TL→TR→BR→BL 的顺时针。
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)
    q = pts[idx]

    # 从 TL 开始：x+y 最小的点
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    q = np.roll(q, -start, axis=0)

    # 保证第二个点是“右上”（在第四点右侧）
    if q[1, 0] < q[3, 0]:
        q = q[[0, 3, 2, 1]]
    return q.astype(np.float32)


def quad_is_valid(q: np.ndarray, min_area: float = 300.0) -> bool:
    area = cv2.contourArea(q.reshape(-1, 1, 2).astype(np.float32))
    if area < min_area:
        return False
    # 凸性检查
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.int32)):
        return False
    # 边长比例 sanity：任意一边不能太短
    d = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    return min(d) > 5.0


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    box: list[float]          # x1,y1,x2,y2
    corners: np.ndarray       # 4x2, TL TR BR BL
    rvec: np.ndarray = field(default=None)
    tvec: np.ndarray = field(default=None)
    distance: float = 0.0
    euler: tuple[float, float, float] = (0.0, 0.0, 0.0)  # yaw, pitch, roll (deg)


class ArmorDetector:
    def __init__(self, weights: str | Path, armor_w: float = ARMOR_W_DEFAULT,
                 armor_h: float = ARMOR_H_DEFAULT, use_gpu: bool = True,
                 imgsz: int = 640, conf: float = 0.25, iou: float = 0.7,
                 quantize16: bool = False) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        if self.model.task != "pose":
            raise SystemExit(f"[armor] 权重任务类型是 {self.model.task}，需要 pose 模型")
        self.names = self.model.names
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = "0" if use_gpu else "cpu"
        self.quantize = 16 if quantize16 else None
        self.kpt_shape = getattr(self.model.model, "kpt_shape", (4, 2))

        w, h = armor_w, armor_h
        self.object_points = np.array([
            [-w / 2, -h / 2, 0],   # TL
            [+w / 2, -h / 2, 0],   # TR
            [+w / 2, +h / 2, 0],   # BR
            [-w / 2, +h / 2, 0],   # BL
        ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    def detect(self, img: np.ndarray) -> list[Detection]:
        res = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, quantize=self.quantize,
                                 verbose=False)[0]
        out: list[Detection] = []
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return out

        kpts_all = res.keypoints.xy.cpu().numpy()
        for box, kpts in zip(res.boxes, kpts_all):
            if len(kpts) != 4:
                continue
            corners = order_corners(kpts)
            if not quad_is_valid(corners):
                continue
            out.append(Detection(
                cls_id=int(box.cls),
                cls_name=str(self.names[int(box.cls)]),
                conf=float(box.conf),
                box=[float(v) for v in box.xyxy[0].tolist()],
                corners=corners,
            ))
        return out

    # ------------------------------------------------------------------ #
    def solve_pnp(self, det: Detection, K: np.ndarray, dist: np.ndarray,
                  prev_tvec: np.ndarray | None = None) -> Detection | None:
        return solve_pnp(det, self.object_points, K, dist, prev_tvec=prev_tvec)



def solve_pnp(det: Detection, object_points: np.ndarray, K: np.ndarray,
              dist: np.ndarray, prev_tvec: np.ndarray | None = None,
              max_reproj_px: float = 6.0) -> Detection | None:
    """平面 PnP：IPPE 给出两个解 -> 各自 LM 精化 -> 按重投影误差 + 时间连续性择优。

    平面靶标（装甲板）天然存在「翻转二义性」：远看时正反两个姿态都能解释同一组角点。
    只取第一个解会出现距离在 1m/3.5m 之间来回跳的锯齿，所以这里把两个解都算出来，
    优先选重投影误差小的；若两者误差接近（<1.5px），选与上一帧位姿更连续的那个。
    """
    img_pts = det.corners.astype(np.float64).reshape(4, 1, 2)
    out = cv2.solvePnPGeneric(object_points, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE)
    n_solutions, rvecs, tvecs = int(out[0]), out[1], out[2]
    if not n_solutions or not rvecs:
        return None

    cands = []
    for i in range(min(n_solutions, len(rvecs))):
        rvec, tvec = rvecs[i].copy(), tvecs[i].copy()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(object_points, img_pts, K, dist, rvec, tvec)
        except cv2.error:
            pass
        if tvec[2, 0] <= 0:
            continue
        proj, _ = cv2.projectPoints(object_points, rvec, tvec, K, dist)
        err = float(np.linalg.norm(det.corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1).mean())
        if err > max_reproj_px:
            continue
        cands.append((err, rvec, tvec))

    if not cands:
        return None

    cands.sort(key=lambda c: c[0])
    err0, rvec, tvec = cands[0]
    if prev_tvec is not None and len(cands) > 1:
        err1, rvec1, tvec1 = cands[1]
        if err1 - err0 < 1.5:  # 两个解都能解释角点时，选与上一帧更接近的
            d0 = float(np.linalg.norm(tvec.ravel() - prev_tvec.ravel()))
            d1 = float(np.linalg.norm(tvec1.ravel() - prev_tvec.ravel()))
            if d1 < d0:
                rvec, tvec = rvec1, tvec1

    det.rvec, det.tvec = rvec, tvec
    det.distance = float(np.linalg.norm(tvec))
    det.euler = _euler_from_R(cv2.Rodrigues(rvec)[0])
    return det


def reprojection_error(det: Detection, object_points: np.ndarray, K: np.ndarray,
                       dist: np.ndarray) -> float:
    if det.rvec is None:
        return float("nan")
    proj, _ = cv2.projectPoints(object_points, det.rvec, det.tvec, K, dist)
    return float(np.linalg.norm(det.corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1).mean())


def _euler_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> (yaw, pitch, roll)，单位度。相机系：x 右、y 下、z 前。"""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return tuple(float(np.degrees(v)) for v in (y, x, z))


def draw_pose(img: np.ndarray, det: Detection, K: np.ndarray, dist: np.ndarray,
              axis_len: float = 0.08) -> np.ndarray:
    """画四角点顺序、检测框与三维坐标轴。"""
    out = img
    q = det.corners.astype(np.int32)
    cv2.polylines(out, [q.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(q):
        cv2.circle(out, (int(x), int(y)), 4, (0, 200, 255), -1, cv2.LINE_AA)
        cv2.putText(out, str(i), (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 255), 1, cv2.LINE_AA)

    if det.rvec is not None:
        axes = np.float64([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        proj, _ = cv2.projectPoints(axes, det.rvec, det.tvec, K, dist)
        p = proj.reshape(-1, 2).astype(int)
        o = tuple(p[0])
        cv2.line(out, o, tuple(p[1]), (0, 0, 255), 2, cv2.LINE_AA)   # X 红
        cv2.line(out, o, tuple(p[2]), (0, 255, 0), 2, cv2.LINE_AA)   # Y 绿
        cv2.line(out, o, tuple(p[3]), (255, 0, 0), 2, cv2.LINE_AA)   # Z 蓝
    return out
