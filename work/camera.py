#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""取帧统一接口：海康 MVS 工业相机 / USB 摄像头 / 视频文件。

对外只暴露 `open_source(...)`，返回一个 Grabber 线程对象：
    grabber.start()                       # 开始采集
    grabber.get(last_seq, timeout)        # 取最新一帧 (seq, bgr, ts)
    grabber.stop()                        # 结束
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

MVS_PYTHON = "/opt/MVS/Samples/64/Python"
MVS_LIB = "/opt/MVS/lib/64"


# --------------------------------------------------------------------------- #
# 只保留最新帧的缓冲（相机快、处理慢时丢旧帧，保证低延迟）
# --------------------------------------------------------------------------- #
class LatestFrame:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._img: np.ndarray | None = None
        self._ts = 0.0
        self._seq = 0
        self._closed = False

    def put(self, img: np.ndarray, ts: float) -> None:
        with self._cv:
            self._img = img
            self._ts = ts
            self._seq += 1
            self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def get(self, last_seq: int, timeout: float = 1.0):
        with self._cv:
            if not self._cv.wait_for(lambda: self._seq != last_seq or self._closed, timeout):
                return None
            if self._img is None or self._seq == last_seq:
                return None
            return self._seq, self._img, self._ts


# --------------------------------------------------------------------------- #
# 基类
# --------------------------------------------------------------------------- #
class Grabber(threading.Thread):
    desc = ""

    def __init__(self, buf: LatestFrame) -> None:
        super().__init__(daemon=True)
        self.buf = buf
        self.error: str | None = None
        self.frames = 0
        self.params: dict = {}
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()


# --------------------------------------------------------------------------- #
# MVS 相机
# --------------------------------------------------------------------------- #
def _mvs():
    if MVS_LIB not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = MVS_LIB + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    if MVS_PYTHON not in sys.path:
        sys.path.insert(0, MVS_PYTHON)
    from MvImport.MvCameraControl_class import (  # noqa: F401
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_CC_PIXEL_CONVERT_PARAM,
        MV_FRAME_OUT,
        MVCC_ENUMVALUE,
        MVCC_FLOATVALUE,
        MvCamera,
        MV_GIGE_DEVICE,
        MV_OK,
        MV_TRIGGER_MODE_OFF,
        MV_USB_DEVICE,
        PixelType_Gvsp_BGR8_Packed,
    )

    return dict(
        MV_CC_DEVICE_INFO=MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST=MV_CC_DEVICE_INFO_LIST,
        MV_CC_PIXEL_CONVERT_PARAM=MV_CC_PIXEL_CONVERT_PARAM,
        MV_FRAME_OUT=MV_FRAME_OUT,
        MVCC_ENUMVALUE=MVCC_ENUMVALUE,
        MVCC_FLOATVALUE=MVCC_FLOATVALUE,
        MvCamera=MvCamera,
        MV_GIGE_DEVICE=MV_GIGE_DEVICE,
        MV_OK=MV_OK,
        MV_TRIGGER_MODE_OFF=MV_TRIGGER_MODE_OFF,
        MV_USB_DEVICE=MV_USB_DEVICE,
        PixelType_Gvsp_BGR8_Packed=PixelType_Gvsp_BGR8_Packed,
    )


def hexret(v: int) -> str:
    return f"0x{v & 0xFFFFFFFF:08X}"


def list_mvs_cameras() -> list[dict]:
    m = _mvs()
    from ctypes import POINTER, cast

    dev_list = m["MV_CC_DEVICE_INFO_LIST"]()
    ret = m["MvCamera"].MV_CC_EnumDevices(m["MV_GIGE_DEVICE"] | m["MV_USB_DEVICE"], dev_list)
    if ret != m["MV_OK"]:
        raise RuntimeError(f"枚举失败 ret={hexret(ret)}")
    out = []
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(m["MV_CC_DEVICE_INFO"])).contents
        layer = "GigE" if info.nTLayerType == m["MV_GIGE_DEVICE"] else "USB3"
        if layer == "GigE":
            g = info.SpecialInfo.stGigEInfo
            model = bytes(g.chModelName).split(b"\x00")[0].decode("utf-8", "ignore")
            sn = bytes(g.chSerialNumber).split(b"\x00")[0].decode("utf-8", "ignore")
            ip = g.nCurrentIp
            addr = f"{(ip >> 24) & 0xFF}.{(ip >> 16) & 0xFF}.{(ip >> 8) & 0xFF}.{ip & 0xFF}"
        else:
            u = info.SpecialInfo.stUsb3VInfo
            model = bytes(u.chModelName).split(b"\x00")[0].decode("utf-8", "ignore")
            sn = bytes(u.chSerialNumber).split(b"\x00")[0].decode("utf-8", "ignore")
            addr = "-"
        out.append({"index": i, "layer": layer, "model": model, "serial": sn, "ip": addr})
    return out


class MvsGrabber(Grabber):
    def __init__(self, buf: LatestFrame, index: int = 0, exposure_us: float = 0.0,
                 gain_db: float = 0.0, fps: float = 0.0, swap_rb: bool = False) -> None:
        super().__init__(buf)
        self.index = index
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.fps = fps
        self.swap_rb = swap_rb
        self.cam = None
        self._dst = None
        self._dst_len = 0
        self.width = self.height = 0

    def _open(self) -> None:
        from ctypes import POINTER, cast

        m = self._m = _mvs()
        dev_list = m["MV_CC_DEVICE_INFO_LIST"]()
        ret = m["MvCamera"].MV_CC_EnumDevices(m["MV_GIGE_DEVICE"] | m["MV_USB_DEVICE"], dev_list)
        if ret != m["MV_OK"]:
            raise RuntimeError(f"枚举相机失败 ret={hexret(ret)}")
        if dev_list.nDeviceNum == 0:
            raise RuntimeError("没有找到 MVS 相机（检查 USB3 线 / 网段 / 是否被 MVS 客户端占用）")
        if self.index >= dev_list.nDeviceNum:
            raise RuntimeError(f"相机索引 {self.index} 越界（共 {dev_list.nDeviceNum} 台）")

        dev = cast(dev_list.pDeviceInfo[self.index], POINTER(m["MV_CC_DEVICE_INFO"])).contents
        self.cam = m["MvCamera"]()
        if self.cam.MV_CC_CreateHandle(dev) != m["MV_OK"]:
            raise RuntimeError("MV_CC_CreateHandle 失败")
        ret = self.cam.MV_CC_OpenDevice()
        if ret != m["MV_OK"]:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"MV_CC_OpenDevice 失败 ret={hexret(ret)}；相机被其它程序（MVS 客户端/另一个实例）占用，"
                f"或 IP 不在同一网段"
            )

        if dev.nTLayerType == m["MV_GIGE_DEVICE"]:
            pkt = self.cam.MV_CC_GetOptimalPacketSize()
            if pkt > 0:
                self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

        self.cam.MV_CC_SetEnumValue("TriggerMode", m["MV_TRIGGER_MODE_OFF"])

        if self.exposure_us > 0:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(self.exposure_us))
        if self.gain_db > 0:
            self.cam.MV_CC_SetEnumValue("GainAuto", 0)
            self.cam.MV_CC_SetFloatValue("Gain", float(self.gain_db))
        if self.fps > 0:
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps))

        self._read_params(m)

        if self.cam.MV_CC_StartGrabbing() != m["MV_OK"]:
            raise RuntimeError("MV_CC_StartGrabbing 失败")

        name = "GigE" if dev.nTLayerType == m["MV_GIGE_DEVICE"] else "USB3"
        self.desc = f"MVS {name} #{self.index}"

    def _read_params(self, m) -> None:
        p = {"exposure": "?", "gain": "?", "fps": "?", "pixel": "?"}
        try:
            st = m["MVCC_FLOATVALUE"]()
            if self.cam.MV_CC_GetFloatValue("ExposureTime", st) == m["MV_OK"]:
                p["exposure"] = f"{st.fCurValue:.0f}us"
            if self.cam.MV_CC_GetFloatValue("Gain", st) == m["MV_OK"]:
                p["gain"] = f"{st.fCurValue:.1f}dB"
            if self.cam.MV_CC_GetFloatValue("ResultingFrameRate", st) == m["MV_OK"]:
                p["fps"] = f"{st.fCurValue:.1f}"
            st_e = m["MVCC_ENUMVALUE"]()
            if self.cam.MV_CC_GetEnumValue("PixelFormat", st_e) == m["MV_OK"]:
                p["pixel"] = str(st_e.nCurValue)
        except Exception:  # noqa: BLE001
            pass
        self.params = p
        print(f"[camera] 参数：曝光 {p['exposure']}  增益 {p['gain']}  "
              f"帧率 {p['fps']}  pixelFormat {p['pixel']}", flush=True)
        if p["exposure"].endswith("us") and float(p["exposure"][:-2]) > 33000:
            print(f"[camera][warn] 曝光 {p['exposure']} 过长，帧率会很低；"
                  f"可用 --exposure 8000 --gain 15 固定曝光", flush=True)

    def run(self) -> None:
        from ctypes import POINTER, byref, c_ubyte, cast, memset, sizeof

        try:
            self._open()
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            self.buf.close()
            return

        m, cam = self._m, self.cam
        st = m["MV_FRAME_OUT"]()
        memset(byref(st), 0, sizeof(st))
        fails = 0

        while not self._stop_evt.is_set():
            ret = cam.MV_CC_GetImageBuffer(st, 1000)
            if ret != 0:
                fails += 1
                continue
            fails = 0
            try:
                fi = st.stFrameInfo
                w, h = int(fi.nWidth), int(fi.nHeight)
                need = w * h * 3
                if self._dst is None or self._dst_len < need:
                    self._dst = (c_ubyte * need)()
                    self._dst_len = need
                    self.width, self.height = w, h

                cvt = m["MV_CC_PIXEL_CONVERT_PARAM"]()
                memset(byref(cvt), 0, sizeof(cvt))
                cvt.nWidth, cvt.nHeight = w, h
                cvt.enSrcPixelType = fi.enPixelType
                cvt.pSrcData = cast(st.pBufAddr, POINTER(c_ubyte))
                cvt.nSrcDataLen = int(fi.nFrameLen)
                cvt.enDstPixelType = m["PixelType_Gvsp_BGR8_Packed"]
                cvt.pDstBuffer = cast(self._dst, POINTER(c_ubyte))
                cvt.nDstBufferSize = self._dst_len
                if cam.MV_CC_ConvertPixelType(cvt) != m["MV_OK"]:
                    continue

                img = np.frombuffer(self._dst, dtype=np.uint8, count=need).reshape(h, w, 3).copy()
                if self.swap_rb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.frames += 1
                self.buf.put(img, time.time())
            finally:
                cam.MV_CC_FreeImageBuffer(st)

        self._close()

    def _close(self) -> None:
        if self.cam is None:
            return
        for fn in (self.cam.MV_CC_StopGrabbing, self.cam.MV_CC_CloseDevice,
                   self.cam.MV_CC_DestroyHandle):
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        self.cam = None


# --------------------------------------------------------------------------- #
# OpenCV 源（视频文件 / USB 摄像头），用于没有 MVS 相机时调试与演示
# --------------------------------------------------------------------------- #
class CvGrabber(Grabber):
    def __init__(self, buf: LatestFrame, source) -> None:
        super().__init__(buf)
        self.source = source
        self._loop = isinstance(source, (str, Path)) and Path(str(source)).is_file()
        self.desc = f"file {Path(str(source)).name}" if self._loop else f"webcam {source}"
        self.fps_native = 30.0

    def run(self) -> None:
        src = int(self.source) if str(self.source).isdigit() else str(self.source)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            self.error = f"打不开输入源 {self.source}"
            self.buf.close()
            return
        self.fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        period = 1.0 / self.fps_native if self.fps_native > 1 else 0.0
        t_next = time.time()
        try:
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    if not self._loop:
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self.frames += 1
                self.buf.put(frame, time.time())
                if period:
                    t_next += period
                    delay = t_next - time.time()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        t_next = time.time()
        finally:
            cap.release()
            self.buf.close()


# --------------------------------------------------------------------------- #
def open_source(buf: LatestFrame, *, source=None, camera_index: int = 0,
                exposure_us: float = 0.0, gain: float = 0.0, fps: float = 0.0,
                swap_rb: bool = False) -> Grabber:
    """source 为空 -> 用 MVS 相机；否则用 cv2 源（视频文件/摄像头序号）。"""
    if source is None:
        return MvsGrabber(buf, index=camera_index, exposure_us=exposure_us,
                          gain_db=gain, fps=fps, swap_rb=swap_rb)
    return CvGrabber(buf, source)
