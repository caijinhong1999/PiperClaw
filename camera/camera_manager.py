# -*- coding: utf-8 -*-
"""
camera_manager.py

当前版本目标：
1. 稳定跑通 Orbbec DaBai 的 RGB
2. 优先使用用户指定的彩色 profile（例如 640x480@30）
3. 默认不启用 depth，避免 USB2.0 + depth profile 不匹配导致失败
4. 为后续单独调通 depth 保留接口
5. 对内参获取失败做兜底，避免程序中断
6. latest_only=True 时由后台线程独占 wait_for_frames，只保留最新帧，减轻「队列满丢帧」

运行：
    python camera/camera_manager.py
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import cv2

try:
    from pyorbbecsdk import (
        Context,
        Pipeline,
        Config,
        OBSensorType,
        OBFormat,
    )
except Exception:
    Context = None
    Pipeline = None
    Config = None
    OBSensorType = None
    OBFormat = None


def _safe_print(*args, **kwargs) -> None:
    """
    尽量避免 SSH / 终端编码导致中文乱码或报错。
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(x) for x in args)
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="ignore"))
        sys.stdout.flush()


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FrameBundle:
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    timestamp: float
    rgb_timestamp: Optional[float] = None
    depth_timestamp: Optional[float] = None


class CameraError(Exception):
    pass


class CameraManager:
    def __init__(
        self,
        color_width: int = 640,
        color_height: int = 480,
        color_fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = False,   # 默认先关掉 depth
        align_to_color: bool = False,
        device_index: int = 0,
        auto_start: bool = False,
        latest_only: bool = False,
    ) -> None:
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.align_to_color = align_to_color
        self.device_index = device_index
        self._latest_only = latest_only

        self._ctx = None
        self._pipeline = None
        self._config = None

        self._started = False
        self._lock = threading.Lock()
        self._latest_lock = threading.Lock()
        self._latest_bundle: Optional[FrameBundle] = None
        self._grab_stop = threading.Event()
        self._grab_thread: Optional[threading.Thread] = None

        self._cached_intrinsics: Optional[CameraIntrinsics] = None
        self._depth_scale: Optional[float] = None
        self._active_color_profile = None
        self._active_depth_profile = None

        if auto_start:
            self.start()

    # =========================
    # 生命周期
    # =========================
    def start(self) -> None:
        with self._lock:
            if self._started:
                return

            if Pipeline is None or Config is None:
                raise CameraError("未检测到 pyorbbecsdk，请先确认环境安装正确。")

            try:
                self._ctx = Context() if Context is not None else None
                self._pipeline = Pipeline()
                self._config = Config()

                if self.enable_color:
                    color_profile = self._select_video_profile(
                        pipeline=self._pipeline,
                        sensor_type=OBSensorType.COLOR_SENSOR,
                        width=self.color_width,
                        height=self.color_height,
                        fmt=self._get_color_format_preference(),
                        fps=self.color_fps,
                    )
                    self._active_color_profile = color_profile
                    self._config.enable_stream(color_profile)

                if self.enable_depth:
                    depth_profile = self._select_video_profile(
                        pipeline=self._pipeline,
                        sensor_type=OBSensorType.DEPTH_SENSOR,
                        width=self.depth_width,
                        height=self.depth_height,
                        fmt=self._get_depth_format_preference(),
                        fps=self.depth_fps,
                    )
                    self._active_depth_profile = depth_profile
                    self._config.enable_stream(depth_profile)

                # 某些设备不支持 frame sync，这里只尝试，不致命
                if self.align_to_color:
                    try:
                        self._pipeline.enable_frame_sync()
                    except Exception as e:
                        _safe_print(f"[WARN] 当前设备不支持 frame sync，已忽略: {e}")

                self._pipeline.start(self._config)
                self._started = True

                if self._latest_only:
                    self._grab_stop.clear()
                    self._latest_bundle = None
                    self._grab_thread = threading.Thread(
                        target=self._grab_loop,
                        name="OrbbecLatestGrab",
                        daemon=True,
                    )
                    self._grab_thread.start()
                else:
                    self._warmup_and_cache()

            except Exception as e:
                self._started = False
                try:
                    if self._pipeline is not None:
                        self._pipeline.stop()
                except Exception:
                    pass
                finally:
                    self._pipeline = None
                    self._config = None
                    self._ctx = None

                raise CameraError(f"启动相机失败: {e}") from e

    def stop(self) -> None:
        t = self._grab_thread
        if t is not None:
            self._grab_stop.set()
            self._grab_thread = None
            t.join(timeout=3.0)

        with self._lock:
            if not self._started and self._pipeline is None:
                return

            try:
                if self._pipeline is not None:
                    self._pipeline.stop()
            except Exception:
                pass
            finally:
                self._started = False
                self._pipeline = None
                self._config = None
                self._ctx = None
                self._cached_intrinsics = None
                self._depth_scale = None
                self._active_color_profile = None
                self._active_depth_profile = None
                with self._latest_lock:
                    self._latest_bundle = None

    def is_started(self) -> bool:
        return self._started

    # =========================
    # 取帧
    # =========================
    def get_frame(self, timeout_ms: int = 1000) -> FrameBundle:
        if not self._started or self._pipeline is None:
            raise CameraError("相机尚未启动，请先调用 start()")

        if self._latest_only:
            deadline = time.time() + timeout_ms / 1000.0
            while time.time() < deadline:
                with self._latest_lock:
                    if self._latest_bundle is not None:
                        # latest_only 下避免对整帧做 numpy copy
                        # 调用方会对 rgb 做 frame.copy() 用于绘制，因此返回共享快照引用即可降低延迟
                        return self._latest_bundle
                time.sleep(0.001)
            raise CameraError("获取帧超时：尚无可用帧")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                raise CameraError("获取帧失败：wait_for_frames 返回空")
            return self._process_frameset(frames)

        except Exception as e:
            if isinstance(e, CameraError):
                raise
            raise CameraError(f"获取相机帧失败: {e}") from e

    def _grab_loop(self) -> None:
        while not self._grab_stop.is_set():
            if not self._started or self._pipeline is None:
                break
            try:
                frames = self._pipeline.wait_for_frames(200)
                if frames is None:
                    continue
                bundle = self._process_frameset(frames)
                with self._latest_lock:
                    self._latest_bundle = bundle
            except Exception:
                if self._grab_stop.is_set():
                    break
                time.sleep(0.001)

    def _copy_frame_bundle(self, bundle: FrameBundle) -> FrameBundle:
        rgb = bundle.rgb.copy() if bundle.rgb is not None else None
        depth = bundle.depth.copy() if bundle.depth is not None else None
        return FrameBundle(
            rgb=rgb,
            depth=depth,
            timestamp=bundle.timestamp,
            rgb_timestamp=bundle.rgb_timestamp,
            depth_timestamp=bundle.depth_timestamp,
        )

    def _process_frameset(self, frames: Any) -> FrameBundle:
        color_image = None
        depth_image = None
        rgb_ts = None
        depth_ts = None

        if self.enable_color:
            try:
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    color_image = self._convert_color_frame_to_bgr(color_frame)
                    rgb_ts = self._safe_get_timestamp(color_frame)
            except Exception as e:
                raise CameraError(f"解析彩色帧失败: {e}") from e

        if self.enable_depth:
            try:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    depth_image = self._convert_depth_frame_to_meters(depth_frame)
                    depth_ts = self._safe_get_timestamp(depth_frame)
            except Exception as e:
                raise CameraError(f"解析深度帧失败: {e}") from e

        return FrameBundle(
            rgb=color_image,
            depth=depth_image,
            timestamp=time.time(),
            rgb_timestamp=rgb_ts,
            depth_timestamp=depth_ts,
        )

    # =========================
    # 内参与深度
    # =========================
    def get_intrinsics(self) -> CameraIntrinsics:
        """
        多级兜底获取相机内参：
        1. 优先从 active profile 获取
        2. 再尝试从当前 frame 的 stream profile 获取
        3. 最后返回一个近似内参，保证流程先跑通
        """
        if self._cached_intrinsics is not None:
            return self._cached_intrinsics

        if not self._started:
            raise CameraError("相机尚未启动，无法获取内参")

        profile = self._active_color_profile if self.enable_color else self._active_depth_profile
        if profile is None:
            raise CameraError("当前没有活动的 stream profile，无法获取内参")

        # 方案1：从启动时的 profile 直接取
        try:
            intrinsic = profile.get_intrinsic()
            intr = CameraIntrinsics(
                width=int(intrinsic.width),
                height=int(intrinsic.height),
                fx=float(intrinsic.fx),
                fy=float(intrinsic.fy),
                cx=float(intrinsic.cx),
                cy=float(intrinsic.cy),
            )
            self._cached_intrinsics = intr
            return intr
        except Exception:
            pass

        # 方案2：从当前 frame 对应的 profile 取（latest_only 时由后台线程独占 wait_for_frames，此处不再抢帧）
        if not self._latest_only:
            try:
                frames = self._pipeline.wait_for_frames(1000)
                if frames is not None:
                    frame = None
                    if self.enable_color:
                        frame = frames.get_color_frame()
                    elif self.enable_depth:
                        frame = frames.get_depth_frame()

                    if frame is not None:
                        try:
                            stream_profile = frame.get_stream_profile()
                            intrinsic = stream_profile.get_intrinsic()
                            intr = CameraIntrinsics(
                                width=int(intrinsic.width),
                                height=int(intrinsic.height),
                                fx=float(intrinsic.fx),
                                fy=float(intrinsic.fy),
                                cx=float(intrinsic.cx),
                                cy=float(intrinsic.cy),
                            )
                            self._cached_intrinsics = intr
                            return intr
                        except Exception:
                            pass
            except Exception:
                pass

        # 方案3：兜底近似内参
        try:
            width = int(profile.get_width())
            height = int(profile.get_height())
        except Exception:
            width = int(self.color_width if self.enable_color else self.depth_width)
            height = int(self.color_height if self.enable_color else self.depth_height)

        # 这里是为了保证后续流程不崩，不是真实标定值
        fx = width * 0.9
        fy = height * 0.9
        cx = width / 2.0
        cy = height / 2.0

        intr = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )
        self._cached_intrinsics = intr
        _safe_print("[WARN] SDK 未返回真实内参，当前使用近似内参兜底：", intr)
        return intr

    def get_depth_scale(self) -> float:
        if self._depth_scale is not None:
            return self._depth_scale

        if not self.enable_depth:
            raise CameraError("当前未启用 depth")

        if not self._started or self._pipeline is None:
            raise CameraError("相机尚未启动，无法获取深度尺度")

        if self._latest_only:
            deadline = time.time() + 3.0
            while time.time() < deadline:
                if self._depth_scale is not None:
                    return self._depth_scale
                time.sleep(0.005)
            raise CameraError("深度尺度尚未就绪：请确认采集线程已输出至少一帧深度")

        try:
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                raise CameraError("无法获取深度尺度：取帧失败")

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                raise CameraError("无法获取深度尺度：无深度帧")

            scale = None

            try:
                scale = depth_frame.get_depth_scale()
            except Exception:
                pass

            if scale is None:
                try:
                    scale = depth_frame.get_value_scale()
                except Exception:
                    pass

            if scale is None:
                scale = 0.001

            self._depth_scale = float(scale)
            return self._depth_scale

        except Exception as e:
            raise CameraError(f"获取深度尺度失败: {e}") from e

    def get_depth_at_pixel(self, depth_image: np.ndarray, u: int, v: int) -> float:
        if depth_image is None:
            return 0.0

        h, w = depth_image.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return 0.0

        z = float(depth_image[v, u])

        if np.isnan(z) or np.isinf(z) or z < 0:
            return 0.0

        return z

    def get_valid_depth_near_pixel(
        self,
        depth_image: np.ndarray,
        u: int,
        v: int,
        kernel_size: int = 5,
    ) -> float:
        if depth_image is None:
            return 0.0

        h, w = depth_image.shape[:2]
        half = kernel_size // 2

        x1 = max(0, u - half)
        x2 = min(w, u + half + 1)
        y1 = max(0, v - half)
        y2 = min(h, v + half + 1)

        patch = depth_image[y1:y2, x1:x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]

        if valid.size == 0:
            return 0.0

        return float(np.median(valid))

    # =========================
    # 可视化
    # =========================
    def visualize_once(
        self,
        frame: Optional[FrameBundle] = None,
        show_depth_colormap: bool = True,
        window_name: str = "Camera Debug",
        wait_ms: int = 0,
    ) -> None:
        if frame is None:
            frame = self.get_frame()

        rgb = frame.rgb
        depth = frame.depth

        if rgb is None and depth is None:
            raise CameraError("当前没有可视化数据")

        vis = []

        if rgb is not None:
            vis.append(rgb)

        if depth is not None and show_depth_colormap:
            depth_vis = self.depth_to_colormap(depth)
            vis.append(depth_vis)

        if len(vis) == 1:
            canvas = vis[0]
        else:
            target_h = min(img.shape[0] for img in vis)
            resized = []
            for img in vis:
                scale = target_h / img.shape[0]
                target_w = int(img.shape[1] * scale)
                resized.append(cv2.resize(img, (target_w, target_h)))
            canvas = np.hstack(resized)

        cv2.imshow(window_name, canvas)
        cv2.waitKey(wait_ms)
        cv2.destroyWindow(window_name)

    @staticmethod
    def depth_to_colormap(depth_meters: np.ndarray, max_depth: float = 2.0) -> np.ndarray:
        depth = depth_meters.copy()
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth, 0, max_depth)
        depth_u8 = (depth / max_depth * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        return color

    # =========================
    # 坐标辅助
    # =========================
    @staticmethod
    def pixel_to_camera(
        u: int,
        v: int,
        depth_z: float,
        intrinsics: CameraIntrinsics,
    ) -> Tuple[float, float, float]:
        if depth_z <= 0:
            return 0.0, 0.0, 0.0

        x = (u - intrinsics.cx) * depth_z / intrinsics.fx
        y = (v - intrinsics.cy) * depth_z / intrinsics.fy
        z = depth_z
        return float(x), float(y), float(z)

    # =========================
    # 内部工具函数
    # =========================
    def _warmup_and_cache(self, warmup_frames: int = 5) -> None:
        if self._pipeline is None:
            return
        if self._latest_only:
            return

        for _ in range(warmup_frames):
            try:
                self._pipeline.wait_for_frames(500)
            except Exception:
                pass

        if self.enable_depth:
            try:
                _ = self.get_depth_scale()
            except Exception:
                pass

    def _get_color_format_preference(self) -> Any:
        # 先优先 RGB888，兼容性更好；拿不到再在 _select_video_profile 里回退
        return getattr(OBFormat, "RGB888", getattr(OBFormat, "RGB", None))

    def _get_depth_format_preference(self) -> Any:
        return getattr(OBFormat, "Y16", None)

    def _select_video_profile(
        self,
        pipeline: Any,
        sensor_type: Any,
        width: int,
        height: int,
        fmt: Any,
        fps: int,
    ) -> Any:
        """
        当前策略：
        1. 优先尝试用户指定 profile
        2. 再尝试常见候选格式
        3. 最后回退默认 profile
        """
        try:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
        except Exception as e:
            raise CameraError(f"获取 {sensor_type} 的 stream profile list 失败: {e}") from e

        # 1) 精确匹配
        if fmt is not None:
            try:
                return profile_list.get_video_stream_profile(width, height, fmt, fps)
            except Exception:
                pass

        # 2) 彩色流尝试多个格式
        if sensor_type == OBSensorType.COLOR_SENSOR:
            candidate_formats = [
                getattr(OBFormat, "RGB888", None),
                getattr(OBFormat, "RGB", None),
                getattr(OBFormat, "MJPG", None),
                getattr(OBFormat, "YUYV", None),
                getattr(OBFormat, "BGRA", None),
            ]
            for candidate_fmt in candidate_formats:
                if candidate_fmt is None:
                    continue
                try:
                    return profile_list.get_video_stream_profile(
                        width, height, candidate_fmt, fps
                    )
                except Exception:
                    pass

        # 3) depth 流尝试多个格式
        if sensor_type == OBSensorType.DEPTH_SENSOR:
            candidate_formats = [
                getattr(OBFormat, "Y16", None),
                getattr(OBFormat, "Y12", None),
                getattr(OBFormat, "Z16", None),
            ]
            for candidate_fmt in candidate_formats:
                if candidate_fmt is None:
                    continue
                try:
                    return profile_list.get_video_stream_profile(
                        width, height, candidate_fmt, fps
                    )
                except Exception:
                    pass

        # 4) 最后回退默认
        try:
            return profile_list.get_default_video_stream_profile()
        except Exception:
            pass

        raise CameraError(
            f"{sensor_type} 没有可用 profile: width={width}, height={height}, fps={fps}"
        )

    def _convert_color_frame_to_bgr(self, color_frame: Any) -> np.ndarray:
        width = int(color_frame.get_width())
        height = int(color_frame.get_height())
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        # RGB888
        expected_rgb = width * height * 3
        if data.size == expected_rgb:
            rgb = data.reshape((height, width, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr

        # BGRA
        expected_bgra = width * height * 4
        if data.size == expected_bgra:
            bgra = data.reshape((height, width, 4))
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            return bgr

        # YUYV
        expected_yuyv = width * height * 2
        if data.size == expected_yuyv:
            yuyv = data.reshape((height, width, 2))
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
            return bgr

        # MJPG / JPEG
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        # 最后再尝试一遍按 3 通道裸数据解释
        if data.size >= expected_rgb:
            try:
                rgb = data[:expected_rgb].reshape((height, width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                return bgr
            except Exception:
                pass

        raise CameraError(
            f"彩色帧解码失败，当前格式未适配: width={width}, height={height}, data_size={data.size}"
        )

    def _convert_depth_frame_to_meters(self, depth_frame: Any) -> np.ndarray:
        width = int(depth_frame.get_width())
        height = int(depth_frame.get_height())
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)

        expected = width * height
        if data.size != expected:
            raise CameraError(
                f"深度帧数据长度异常，expected={expected}, actual={data.size}"
            )

        depth_raw = data.reshape((height, width))
        scale = self._depth_scale_from_frame(depth_frame)
        depth_m = depth_raw.astype(np.float32) * scale
        return depth_m

    def _depth_scale_from_frame(self, depth_frame: Any) -> float:
        if self._depth_scale is not None:
            return self._depth_scale

        scale = None
        try:
            scale = depth_frame.get_depth_scale()
        except Exception:
            pass
        if scale is None:
            try:
                scale = depth_frame.get_value_scale()
            except Exception:
                pass
        if scale is None:
            scale = 0.001

        self._depth_scale = float(scale)
        return self._depth_scale

    @staticmethod
    def _safe_get_timestamp(frame: Any) -> Optional[float]:
        try:
            return float(frame.get_timestamp())
        except Exception:
            try:
                return float(frame.get_time_stamp())
            except Exception:
                return None


if __name__ == "__main__":
    cam = CameraManager(
        color_width=640,
        color_height=480,
        color_fps=30,
        depth_width=640,
        depth_height=480,
        depth_fps=30,
        enable_color=True,
        enable_depth=False,   # 先只跑 RGB
        align_to_color=False,
    )

    try:
        cam.start()
        _safe_print("相机启动成功")

        frame = cam.get_frame()
        _safe_print("rgb shape:", None if frame.rgb is None else frame.rgb.shape)
        _safe_print("depth shape:", None if frame.depth is None else frame.depth.shape)

        intr = cam.get_intrinsics()
        _safe_print("相机内参:", intr)

        # 如果你是本地桌面环境，可打开这个
        # cam.visualize_once(frame)

    except Exception as e:
        _safe_print("运行失败:", e)

    finally:
        cam.stop()
        _safe_print("相机已停止")