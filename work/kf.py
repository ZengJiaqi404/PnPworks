#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""卡尔曼滤波：对 PnP 解算出的三维位置做匀速（CV）模型滤波与短时预测。

状态向量 x = [X, Y, Z, Vx, Vy, Vz]^T（米、米/秒）
观测向量 z = [X, Y, Z]^T（PnP 直接测量）

    x(k+1) = F(dt) x(k) + w,   F = [[I, dt*I], [0, I]]
    z(k)   = H x(k) + v,       H = [I 0]

注意：cv2.KalmanFilter.predict() 会推进内部状态，所以「预测未来位置」必须用
纯函数 predict_at(dt) 计算（p + v*dt），不能多调一次 predict()，否则状态会被
多推进一帧、速度被重复积分。
"""

from __future__ import annotations

import cv2
import numpy as np


class ArmorKalman:
    def __init__(self, pos_std: float = 0.04, acc_std: float = 3.0,
                 gate: float = 0.6, max_dt: float = 0.5) -> None:
        self.kf = cv2.KalmanFilter(6, 3)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0]], np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * (pos_std ** 2)
        self.pos_std = pos_std
        self.acc_std = acc_std
        self.gate = gate
        self.max_dt = max_dt
        self.initialized = False
        self.last_t: float | None = None
        self.resets = 0
        self.updates = 0

    # ------------------------------------------------------------------ #
    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        """加速度白噪声模型：Q = G q G^T, G = [dt²/2·I; dt·I]。"""
        q = self.acc_std ** 2
        G = np.zeros((6, 3), dtype=np.float32)
        G[0:3, 0:3] = np.eye(3) * (dt ** 2 / 2.0)
        G[3:6, 0:3] = np.eye(3) * dt
        return G @ (np.eye(3, dtype=np.float32) * q) @ G.T

    # ------------------------------------------------------------------ #
    def init(self, pos: np.ndarray, t: float) -> None:
        x = np.zeros((6, 1), np.float32)
        x[0:3, 0] = np.asarray(pos, np.float32).ravel()
        self.kf.statePost = x
        self.kf.statePre = x.copy()
        self.initialized = True
        self.last_t = t

    def reset(self, pos: np.ndarray, t: float) -> None:
        self.init(pos, t)
        self.resets += 1

    def predict_at(self, dt: float) -> np.ndarray:
        """纯预测：按当前状态外推 dt 秒后的位置，不改动滤波器内部状态。"""
        p = self.position
        v = self.velocity
        return p + v * float(dt)

    # ------------------------------------------------------------------ #
    def update(self, pos: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """返回 (滤波后位置, 下一帧预测位置)。首帧/跳变过大时自动初始化。"""
        pos = np.asarray(pos, np.float32).ravel()
        if not self.initialized:
            self.init(pos, t)
            self.updates += 1
            return pos.copy(), self.predict_at(1.0 / 30.0)

        dt = t - (self.last_t or t)
        if dt <= 0 or dt > self.max_dt:
            self.reset(pos, t)
            return pos.copy(), self.predict_at(1.0 / 30.0)

        self.kf.transitionMatrix = self._F(dt)
        self.kf.processNoiseCov = self._Q(dt)
        pred = self.kf.predict()

        # 门限：测量与预测相差过大（误检/换目标）→ 重置
        if float(np.linalg.norm(pos - pred[0:3, 0])) > self.gate:
            self.reset(pos, t)
            return pos.copy(), self.predict_at(1.0 / 30.0)

        self.kf.correct(pos.reshape(3, 1).astype(np.float32))
        self.last_t = t
        self.updates += 1
        est = self.kf.statePost[0:3, 0].copy()
        return est, self.predict_at(1.0 / 30.0)

    # ------------------------------------------------------------------ #
    @property
    def velocity(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3, np.float32)
        return self.kf.statePost[3:6, 0].copy()

    @property
    def position(self) -> np.ndarray:
        if not self.initialized:
            return np.zeros(3, np.float32)
        return self.kf.statePost[0:3, 0].copy()
