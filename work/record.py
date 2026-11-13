#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""录一段原始画面（不加任何叠加），用于棋盘格标定。

用法：
    ./run.sh --tool record --seconds 30            # 录 30 秒到 assets/chessboard.mp4
    ./run.sh --tool record --seconds 30 --preview  # 同时开窗口看着调焦

录的时候让棋盘格铺满画面的不同区域/角度（近的、远的、倾斜的），对焦要清晰。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from camera import LatestFrame, open_source  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="录制原始画面")
    p.add_argument("--seconds", type=float, default=30.0, help="录制时长（秒）")
    p.add_argument("--out", type=Path, default=HERE / "assets" / "chessboard.mp4")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--source", default=None, help="改用视频/摄像头（调试用）")
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--gain", type=float, default=0.0)
    p.add_argument("--fps", type=float, default=30.0, help="编码帧率（写文件用）")
    p.add_argument("--preview", action="store_true", help="开窗口实时预览")
    args = p.parse_args(argv)

    buf = LatestFrame()
    grab = open_source(buf, source=args.source, camera_index=args.camera_index,
                       exposure_us=args.exposure, gain=args.gain)
    grab.start()

    t0 = time.time()
    while grab.error is None and grab.frames == 0 and time.time() - t0 < 8:
        time.sleep(0.05)
    if grab.error:
        print(f"[record][error] {grab.error}")
        return 3

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    seq = -1
    saved = 0
    t_start = time.time()
    print(f"[record] 开始录制 {args.seconds:.0f}s -> {args.out}"
          f"（棋盘格铺满不同区域，对焦清晰）", flush=True)

    try:
        while True:
            got = buf.get(seq, timeout=1.0)
            if got is None:
                if grab.error:
                    print(f"[record][error] {grab.error}")
                    break
                if not grab.is_alive():
                    break
                continue
            seq, frame, _ = got
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (w, h))
                print(f"[record] 分辨率 {w}x{h}，编码 {args.fps:.0f} fps", flush=True)

            writer.write(frame)
            saved += 1

            if args.preview:
                vis = frame.copy()
                cv2.putText(vis, f"REC {saved} / {args.seconds:.0f}s", (10, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("record (q to stop early)", vis)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break

            el = time.time() - t_start
            if el >= args.seconds:
                break
            if saved % 100 == 0:
                print(f"  ... {el:.1f}s / {saved} 帧", flush=True)
    finally:
        grab.stop()
        if writer is not None:
            writer.release()
        if args.preview:
            cv2.destroyAllWindows()

    el = time.time() - t_start
    print(f"[record] 完成：{saved} 帧 / {el:.1f}s（{saved / el if el else 0:.1f} FPS）-> {args.out}")
    print(f"[record] 下一步： ./run.sh --tool calibrate --video {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
