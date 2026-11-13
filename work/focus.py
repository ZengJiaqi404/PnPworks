#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对焦辅助：实时显示清晰度评分，拧对焦环把分数拧到最大。

用法：
    ./run.sh --tool focus
    ./run.sh --tool focus --exposure 12000 --gain 18

窗口快捷键：q 退出 · s 保存当前帧 · b 保存目前最清晰的一帧
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from camera import LatestFrame, open_source  # noqa: E402


def sharpness(gray: np.ndarray) -> float:
    """Laplacian 方差：越大越清晰（通常 >150 算锐，<50 明显虚）。"""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="对焦辅助")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--source", default=None)
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--gain", type=float, default=0.0)
    p.add_argument("--roi", type=float, default=0.5, help="只统计画面中心多大区域（0~1）")
    p.add_argument("--save-dir", type=Path, default=HERE / "captures")
    args = p.parse_args(argv)

    buf = LatestFrame()
    g = open_source(buf, source=args.source, camera_index=args.camera_index,
                    exposure_us=args.exposure, gain=args.gain)
    g.start()
    t0 = time.time()
    while g.error is None and g.frames == 0 and time.time() - t0 < 10:
        time.sleep(0.05)
    if g.error:
        print(f"[focus][error] {g.error}")
        return 3

    win = "focus: 拧对焦环让 sharp 最大  |  q quit  s save  b save-best"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    seq, best, best_img = -1, 0.0, None
    print("[focus] 调整对焦环，让画面里 sharp 数值尽可能大（>150 算锐利）", flush=True)

    try:
        while True:
            got = buf.get(seq, timeout=1.0)
            if got is None:
                if g.error or not g.is_alive():
                    break
                continue
            seq, frame, _ = got
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            r = max(0.1, min(1.0, args.roi))
            cy, cx = h // 2, w // 2
            half_h, half_w = int(h * r / 2), int(w * r / 2)
            roi = gray[cy - half_h:cy + half_h, cx - half_w:cx + half_w]
            sc = sharpness(roi)
            bright = float(roi.mean())
            if sc > best:
                best, best_img = sc, frame.copy()

            vis = frame.copy()
            cv2.rectangle(vis, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), (0, 255, 0), 1)
            color = (0, 255, 0) if sc > 150 else ((0, 200, 255) if sc > 50 else (0, 0, 255))
            cv2.putText(vis, f"sharp {sc:7.1f}", (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
            cv2.putText(vis, f"bright {bright:5.1f}  best {best:7.1f}", (14, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
            tip = "TURN FOCUS RING -> maximize sharp" if sc < 150 else "SHARP OK"
            cv2.putText(vis, tip, (14, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            tw = 1000
            if vis.shape[1] != tw:
                vis = cv2.resize(vis, (tw, int(vis.shape[0] * tw / vis.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imshow(win, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                args.save_dir.mkdir(parents=True, exist_ok=True)
                pth = args.save_dir / time.strftime("focus_%H%M%S.jpg")
                cv2.imwrite(str(pth), frame)
                print(f"[focus] 已保存 {pth}", flush=True)
            if key == ord("b") and best_img is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                pth = args.save_dir / "focus_best.jpg"
                cv2.imwrite(str(pth), best_img)
                print(f"[focus] 已保存最清晰帧 {pth}（sharp={best:.1f}）", flush=True)
    finally:
        g.stop()
        try:
            g.join(timeout=3)
        except RuntimeError:
            pass
        cv2.destroyAllWindows()
        if best_img is not None:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            pth = args.save_dir / "focus_best.jpg"
            cv2.imwrite(str(pth), best_img)
            print(f"[focus] 最佳清晰度 {best:.1f}，已存 {pth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
