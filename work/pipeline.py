#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""装甲板实时定位主程序：相机采集 -> 装甲板识别 -> 角点提取 -> PnP -> 卡尔曼滤波 -> 运动预测。

运行：
    ./run.sh                                   # 默认：MVS 相机 + calib.yaml
    ./run.sh --source videos/demo.mp4          # 没有相机时用视频调试
    ./run.sh --exposure 8000 --gain 12         # 暗环境固定曝光
    ./run.sh --device cpu                      # 用 CPU

窗口快捷键：q 退出 · s 存图 · r 录制 · p 暂停 · h HUD · k 轨迹
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from armor import ArmorDetector, draw_pose  # noqa: E402
from calibrate import demo_intrinsics, load_calibration  # noqa: E402
from kf import ArmorKalman  # noqa: E402

WEIGHT_CANDIDATES = [
    Path("weights/best.pt"),
    Path.home() / "桌面" / "noname" / "armor-inference" / "weights" / "best.pt",
]


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="装甲板 PnP 定位 + 卡尔曼滤波实时程序",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("输入")
    g.add_argument("--source", default=None, help="视频文件/摄像头序号；空=用 MVS 相机")
    g.add_argument("--camera-index", type=int, default=0)
    g.add_argument("--exposure", type=float, default=0.0, help="固定曝光 us，0=保持相机设置")
    g.add_argument("--gain", type=float, default=0.0, help="增益 dB，0=保持相机设置")

    g = p.add_argument_group("模型与标定")
    g.add_argument("--weights", type=Path, default=None,
                   help="默认依次找 weights/best.pt、环境变量 ARMOR_WEIGHTS、家目录里的课程权重")
    g.add_argument("--calib", type=Path, default=HERE / "calib.yaml")
    g.add_argument("--armor-w", type=float, default=0.224, help="装甲板宽（米，四角点横向跨度）")
    g.add_argument("--armor-h", type=float, default=0.054, help="装甲板高（米，四角点纵向跨度=灯条长度）")
    g.add_argument("--device", default="auto", help="auto / cpu / 0")
    g.add_argument("--imgsz", type=int, default=640)
    g.add_argument("--conf", type=float, default=0.35)
    g.add_argument("--iou", type=float, default=0.7)
    g.add_argument("--half", action="store_true", help="GPU 上 FP16")

    g = p.add_argument_group("滤波")
    g.add_argument("--kf-pos-std", type=float, default=0.04, help="观测噪声标准差（米）")
    g.add_argument("--kf-acc-std", type=float, default=3.0, help="过程噪声（加速度标准差，m/s²）")
    g.add_argument("--kf-gate", type=float, default=0.6, help="跳变重置门限（米）")
    g.add_argument("--no-kf", action="store_true", help="关闭卡尔曼，只看原始 PnP")

    g = p.add_argument_group("输出")
    g.add_argument("--record", default=None, help="录制结果视频")
    g.add_argument("--csv", default=None, help="逐帧数据 CSV（原始/滤波/预测）")
    g.add_argument("--report", default=None, help="运行统计 JSON")
    g.add_argument("--snapshot-dir", type=Path, default=HERE / "captures")
    g.add_argument("--headless", action="store_true")
    g.add_argument("--max-frames", type=int, default=0)
    g.add_argument("--display-width", type=int, default=1280)
    p.add_argument("--list-devices", action="store_true")
    return p.parse_args(argv)


def resolve_weights(explicit: Path | None) -> Path | None:
    if explicit:
        return Path(explicit)
    env = os.environ.get("ARMOR_WEIGHTS")
    if env and Path(env).is_file():
        return Path(env)
    for cand in WEIGHT_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def resolve_device(name: str) -> str:
    import torch

    if name == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return name


# --------------------------------------------------------------------------- #
def hud(img: np.ndarray, lines: list[str]) -> np.ndarray:
    pad, lh, sc = 10, 24, 0.58
    w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0] for t in lines) + pad * 2
    h = pad * 2 + lh * len(lines)
    ov = img.copy()
    cv2.rectangle(ov, (0, 0), (w, h), (18, 18, 18), -1)
    out = cv2.addWeighted(ov, 0.62, img, 0.38, 0)
    for i, t in enumerate(lines):
        cv2.putText(out, t, (pad, pad + lh * (i + 1) - 7), cv2.FONT_HERSHEY_SIMPLEX, sc,
                    (120, 255, 120) if i == 0 else (235, 235, 235), 1, cv2.LINE_AA)
    return out


def project_points(pts3d: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if len(pts3d) == 0:
        return np.zeros((0, 2), np.int32)
    pts = np.asarray(pts3d, np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, dist)
    return proj.reshape(-1, 2).astype(np.int32)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_devices:
        from camera import list_mvs_cameras

        try:
            cams = list_mvs_cameras()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {exc}")
            return 3
        print(f"找到 {len(cams)} 台 MVS 相机")
        for c in cams:
            print(f"  [{c['index']}] {c['layer']:4s} {c['model']:<30s} SN={c['serial']:<14s} IP={c['ip']}")
        return 0

    args.weights = resolve_weights(args.weights)
    if args.weights is None or not args.weights.is_file():
        print("[error] 找不到权重文件。把 best.pt 放到 weights/best.pt，"
              "或用 --weights /绝对路径，或设环境变量 ARMOR_WEIGHTS")
        return 2

    device = resolve_device(args.device)
    det = ArmorDetector(args.weights, armor_w=args.armor_w, armor_h=args.armor_h,
                        use_gpu=device != "cpu", imgsz=args.imgsz, conf=args.conf,
                        iou=args.iou, quantize16=args.half)
    print(f"[info] 模型 {Path(args.weights).name}  task={det.model.task}  "
          f"names={det.names}  kpt_shape={det.kpt_shape}  device={device}", flush=True)
    print(f"[info] 装甲板尺寸 {args.armor_w * 1000:.0f} x {args.armor_h * 1000:.0f} mm", flush=True)

    calib = load_calibration(args.calib)
    if calib is None:
        print(f"[warn] 没找到 {args.calib}，先用演示内参跑通流程（距离数值不可用于验收）", flush=True)
        calib = None

    from camera import LatestFrame, open_source

    buf = LatestFrame()
    grab = open_source(buf, source=args.source, camera_index=args.camera_index,
                       exposure_us=args.exposure, gain=args.gain)
    grab.start()

    t0 = time.time()
    while grab.error is None and grab.frames == 0 and time.time() - t0 < 8:
        time.sleep(0.05)
    if grab.error:
        print(f"[error] {grab.error}")
        print("排查：MVS 客户端是否占用相机 / USB3 线 / 网段；没有相机可加 --source 视频文件")
        return 3
    if calib is None:
        calib = demo_intrinsics(grab.width or 1440, grab.height or 1080)
    else:
        if grab.width and (grab.width != calib["size"][0] or grab.height != calib["size"][1]):
            print(f"[warn] 标定分辨率 {calib['size']} 与当前画面 {grab.width}x{grab.height} 不一致，"
                  f"内参会失真！请用同分辨率重新标定", flush=True)
    K, dist = calib["K"], calib["dist"]

    kf = ArmorKalman(pos_std=args.kf_pos_std, acc_std=args.kf_acc_std, gate=args.kf_gate)
    win = "Armor PnP + Kalman  (q quit | s shot | r rec | p pause | h hud | k trail)"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    writer = None
    rec_size = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict] = []
    trail: list[np.ndarray] = []
    api_rows = [] if args.csv else []
    show_hud, show_trail = True, True
    conf = args.conf
    paused = False
    recording = bool(args.record)
    last_cls: str | None = None
    prev_tvec: np.ndarray | None = None
    lost_t: float | None = None
    seq = -1
    frame_i = 0
    n_det_total = 0
    lat_ema = fps_ema = 0.0
    t_start = time.time()

    print("[info] 开始运行（窗口需获得焦点）", flush=True)
    try:
        while True:
            got = buf.get(seq, timeout=1.0)
            if got is None:
                if grab.error:
                    print(f"[error] {grab.error}")
                    break
                if not grab.is_alive():
                    print("[info] 采集结束")
                    break
                continue
            seq, frame, _ts = got

            if paused:
                k = cv2.waitKey(30) & 0xFF if not args.headless else 0xFF
                if k == ord("p"):
                    paused = False
                elif k in (ord("q"), 27):
                    break
                continue

            t1 = time.perf_counter()
            dets = det.detect(frame)
            lat = (time.perf_counter() - t1) * 1000.0
            frame_i += 1
            dt_all = time.time() - t_start
            fps_ema = frame_i / dt_all if dt_all > 0 else 0
            lat_ema = lat if lat_ema == 0 else lat_ema * 0.9 + lat * 0.1

            kept, prev = [], prev_tvec
            for d in dets:
                if det.solve_pnp(d, K, dist, prev_tvec=prev) is not None:
                    kept.append(d)
                    prev = d.tvec
            dets = kept
            n_det_total += len(dets)
            best = max(dets, key=lambda d: d.conf) if dets else None

            raw = filt = pred = None
            vel = np.zeros(3, np.float32)
            if best is not None:
                raw = best.tvec.ravel().copy()
                if args.no_kf:
                    filt, pred = raw.copy(), raw.copy()
                else:
                    if last_cls is not None and best.cls_name != last_cls:
                        kf.initialized = False
                    filt, pred = kf.update(raw, time.time())
                    vel = kf.velocity
                    if args.kf_gate and float(np.linalg.norm(raw - filt)) > args.kf_gate:
                        kf.reset(raw, time.time())
                        filt, pred = raw.copy(), raw.copy()
                last_cls = best.cls_name
                prev_tvec = best.tvec.copy()
                lost_t = None
                if np.all(np.isfinite(filt)) and filt[2] > 0.05:
                    trail.append(filt.copy())
                if len(trail) > 90:
                    trail.pop(0)
            else:
                if lost_t is None:
                    lost_t = time.time()
                elif time.time() - lost_t > 0.15:
                    kf.initialized = False
                    trail.clear()
                    last_cls = None
                    prev_tvec = None

            # ---------------- 绘制 ----------------
            vis = frame.copy()
            if show_trail and len(trail) > 1:
                pts = project_points(np.array(trail), K, dist)
                cv2.polylines(vis, [pts.reshape(-1, 1, 2)], False, (0, 200, 255), 1, cv2.LINE_AA)

            for d in dets:
                vis = draw_pose(vis, d, K, dist)
                x, y = d.corners[0]
                cv2.putText(vis, f"{d.cls_name} {d.conf:.2f} {d.distance:.2f}m",
                            (int(x), int(y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 2, cv2.LINE_AA)

            if best is not None and filt is not None:
                p_pred = project_points(np.array([pred]), K, dist)[0]
                p_filt = project_points(np.array([filt]), K, dist)[0]
                cv2.drawMarker(vis, tuple(p_filt), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                if not args.no_kf:
                    cv2.circle(vis, tuple(p_pred), 9, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis, "pred", (p_pred[0] + 10, p_pred[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

            if show_hud:
                lines = [
                    f"PnP+KF  {fps_ema:5.1f} FPS   infer {lat_ema:4.1f} ms   "
                    f"{'KF ON' if not args.no_kf else 'KF OFF'}",
                ]
                if best is not None and filt is not None:
                    lines += [
                        f"target {best.cls_name}  conf {best.conf:.2f}  dist {best.distance:.3f} m",
                        f"raw    X {raw[0]:+.3f}  Y {raw[1]:+.3f}  Z {raw[2]:+.3f} m",
                        f"filter X {filt[0]:+.3f}  Y {filt[1]:+.3f}  Z {filt[2]:+.3f} m",
                        f"pred   X {pred[0]:+.3f}  Y {pred[1]:+.3f}  Z {pred[2]:+.3f} m",
                        f"vel    {vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f} m/s",
                        f"yaw {best.euler[0]:+.1f}  pitch {best.euler[1]:+.1f}  "
                        f"roll {best.euler[2]:+.1f} deg",
                    ]
                else:
                    lines.append("no target")
                vis = hud(vis, lines)

            if recording:
                if writer is None:
                    tw = int(args.display_width)
                    rec_size = (tw, int(round(vis.shape[0] * tw / vis.shape[1])))
                    Path(args.record).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             30.0, rec_size)
                    print(f"[info] 录制中 -> {args.record} {rec_size}", flush=True)
                if vis.shape[1] != rec_size[0]:
                    writer.write(cv2.resize(vis, rec_size, interpolation=cv2.INTER_AREA))
                else:
                    writer.write(vis)

            if args.csv:
                api_rows.append({
                    "t": round(time.time() - t_start, 4),
                    "frame": frame_i,
                    "cls": best.cls_name if best else "",
                    "conf": round(best.conf, 4) if best else "",
                    "raw_x": round(float(raw[0]), 5) if raw is not None else "",
                    "raw_y": round(float(raw[1]), 5) if raw is not None else "",
                    "raw_z": round(float(raw[2]), 5) if raw is not None else "",
                    "kf_x": round(float(filt[0]), 5) if filt is not None else "",
                    "kf_y": round(float(filt[1]), 5) if filt is not None else "",
                    "kf_z": round(float(filt[2]), 5) if filt is not None else "",
                    "pred_x": round(float(pred[0]), 5) if pred is not None else "",
                    "pred_y": round(float(pred[1]), 5) if pred is not None else "",
                    "pred_z": round(float(pred[2]), 5) if pred is not None else "",
                    "dist": round(best.distance, 4) if best else "",
                    "vx": round(float(vel[0]), 4),
                    "vy": round(float(vel[1]), 4),
                    "vz": round(float(vel[2]), 4),
                })

            if not args.headless:
                tw = int(args.display_width)
                if abs(vis.shape[1] - tw) > 8:
                    sc = tw / vis.shape[1]
                    vis_show = cv2.resize(vis, (tw, int(vis.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                else:
                    vis_show = vis
                cv2.imshow(win, vis_show)

            if args.max_frames and frame_i >= args.max_frames:
                print(f"[info] 达到 --max-frames {args.max_frames}")
                break

            key = 0xFF if args.headless else cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                args.snapshot_dir.mkdir(parents=True, exist_ok=True)
                p = args.snapshot_dir / time.strftime("shot_%H%M%S.jpg")
                cv2.imwrite(str(p), vis)
                print(f"[info] 已存图 {p}", flush=True)
            elif key == ord("r"):
                recording = not recording
                if recording and args.record is None:
                    args.record = str(Path("captures") / time.strftime("rec_%H%M%S.mp4"))
                if not recording and writer is not None:
                    writer.release()
                    writer = None
                print(f"[info] 录制 {'开始' if recording else '停止'}", flush=True)
            elif key == ord("h"):
                show_hud = not show_hud
            elif key == ord("k"):
                show_trail = not show_trail
            elif key == ord("p"):
                paused = True
    finally:
        grab.stop()
        try:
            grab.join(timeout=3.0)   # 等采集线程退出，避免解释器关闭时 C++ 线程析构崩溃
        except RuntimeError:
            pass
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        if args.csv and api_rows:
            Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(api_rows[0].keys()))
                wr.writeheader()
                wr.writerows(api_rows)
            print(f"[info] CSV 已保存 {args.csv}")

    elapsed = time.time() - t_start
    stats = {
        "frames": frame_i, "elapsed_s": round(elapsed, 2),
        "fps": round(frame_i / elapsed, 2) if elapsed else 0,
        "infer_ms_mean": round(lat_ema, 2),
        "detections": n_det_total,
        "device": device, "imgsz": args.imgsz, "conf": args.conf,
        "armor_mm": [args.armor_w * 1000, args.armor_h * 1000],
        "kf": None if args.no_kf else {"pos_std": args.kf_pos_std, "acc_std": args.kf_acc_std,
                                       "gate": args.kf_gate, "resets": kf.resets},
        "calib": "demo" if calib.get("demo") else str(args.calib),
    }
    print("\n===== 运行统计 =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[info] 用户中断")
        sys.exit(130)
