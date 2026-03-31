# -*- coding: utf-8 -*-
"""
camera_manager.py

当前版本目标：
1. 先稳定跑通 Orbbec DaBai 的 RGB
2. 默认不启用 depth，避免 USB2.0 + profile 不匹配导致启动失败
3. 后续再单独调通 depth profile

运行：
    python camera/camera_manager.py
"""

from __future__ import annotations

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
        enable_depth: bool = False,   # 先默认关掉 depth
        align_to_color: bool = False,
        device_index: int = 0,
        auto_start: bool = False,
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

        self._ctx = None
        self._pipeline = None
        self._config = None

        self._started = False
        self._lock = threading.Lock()

        self._cached_intrinsics: Optional[CameraIntrinsics] = None
        self._depth_scale: Optional[float] = None

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
                        fmt=OBFormat.RGB,
                        fps=self.color_fps,
                    )
                    self._config.enable_stream(color_profile)

                if self.enable_depth:
                    depth_profile = self._select_video_profile(
                        pipeline=self._pipeline,
                        sensor_type=OBSensorType.DEPTH_SENSOR,
                        width=self.depth_width,
                        height=self.depth_height,
                        fmt=OBFormat.Y16,
                        fps=self.depth_fps,
                    )
                    self._config.enable_stream(depth_profile)

                # 你的设备当前不支持 frame sync，默认不要开
                if self.align_to_color:
                    try:
                        self._pipeline.enable_frame_sync()
                    except Exception:
                        pass

                self._pipeline.start(self._config)
                self._started = True
                self._warmup_and_cache()

            except Exception as e:
                self._started = False
                raise CameraError(f"启动相机失败: {e}") from e

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return

            try:
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

    def is_started(self) -> bool:
        return self._started

    # =========================
    # 取帧
    # =========================
    def get_frame(self, timeout_ms: int = 1000) -> FrameBundle:
        if not self._started:
            raise CameraError("相机尚未启动，请先调用 start()")

        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
            if frames is None:
                raise CameraError("获取帧失败：wait_for_frames 返回空")

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

        except Exception as e:
            if isinstance(e, CameraError):
                raise
            raise CameraError(f"获取相机帧失败: {e}") from e

    # =========================
    # 内参与深度
    # =========================
    def get_intrinsics(self) -> CameraIntrinsics:
        if self._cached_intrinsics is not None:
            return self._cached_intrinsics

        if not self._started:
            raise CameraError("相机尚未启动，无法获取内参")

        try:
            if self.enable_color:
                profile = self._select_video_profile(
                    pipeline=self._pipeline,
                    sensor_type=OBSensorType.COLOR_SENSOR,
                    width=self.color_width,
                    height=self.color_height,
                    fmt=OBFormat.RGB,
                    fps=self.color_fps,
                )
            elif self.enable_depth:
                profile = self._select_video_profile(
                    pipeline=self._pipeline,
                    sensor_type=OBSensorType.DEPTH_SENSOR,
                    width=self.depth_width,
                    height=self.depth_height,
                    fmt=OBFormat.Y16,
                    fps=self.depth_fps,
                )
            else:
                raise CameraError("未启用任何流，无法获取内参")

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
            # 有些设备/模式下拿 profile intrinsic 会失败
            # 先给出一个友好的报错，不让程序崩得太难看
            raise CameraError("获取相机内参失败：当前 profile 下 SDK 没返回可用内参")

    def get_depth_scale(self) -> float:
        if self._depth_scale is not None:
            return self._depth_scale

        if not self.enable_depth:
            raise CameraError("当前未启用 depth")

        if not self._started:
            raise CameraError("相机尚未启动，无法获取深度尺度")

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
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)

    @staticmethod
    def depth_to_colormap(depth_meters: np.ndarray, max_depth: float = 2.0) -> np.ndarray:
        depth = depth_meters.copy()
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
        1. 优先取默认 profile（最稳）
        2. 再尝试指定 profile
        """
        try:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
        except Exception as e:
            raise CameraError(f"获取 {sensor_type} 的 stream profile list 失败: {e}") from e

        # 最稳：先取默认 profile
        try:
            return profile_list.get_default_video_stream_profile()
        except Exception:
            pass

        # 再尝试精确匹配
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            pass

        raise CameraError(f"{sensor_type} 没有可用 profile")

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

        raise CameraError("彩色帧解码失败，当前格式未适配")

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
        scale = self.get_depth_scale()
        depth_m = depth_raw.astype(np.float32) * scale
        return depth_m

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
        print("相机启动成功")

        frame = cam.get_frame()
        print("rgb shape:", None if frame.rgb is None else frame.rgb.shape)
        print("depth shape:", None if frame.depth is None else frame.depth.shape)

        # 先不强制取内参，避免当前 profile 下 SDK 不返回内参导致退出
        try:
            intr = cam.get_intrinsics()
            print("相机内参:", intr)
        except Exception as e:
            print("内参暂时获取失败:", e)

        # 远程 SSH 环境先别弹窗
        # cam.visualize_once(frame)

    except Exception as e:
        print("运行失败:", e)

    finally:
        cam.stop()
        print("相机已停止")