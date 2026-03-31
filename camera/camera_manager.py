# -*- coding: utf-8 -*-
"""
camera_manager.py

统一管理 RGB-D 相机输入，当前以 Orbbec 相机为主。
提供以下能力：
1. 初始化相机
2. 启动/停止数据流
3. 获取 RGB + Depth 帧
4. 获取相机内参
5. 获取指定像素点深度
6. 调试可视化

推荐目录：
robot_project/
└── camera/
    └── camera_manager.py
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2


# ==========
# 重要说明
# ==========
# 如果你本地安装的是:
#   from pyorbbecsdk import *
# 或者:
#   from pyorbbecsdk2 import *
# 这里改一下即可。
#
# 下面这套写法兼容“官方 Python SDK 风格”的常见接口命名。
# 如果你实际 SDK 返回字段略有差异，只需要在少数几个位置微调。
try:
    from pyorbbecsdk import (
        Context,
        Pipeline,
        Config,
        OBSensorType,
        OBFormat,
        VideoStreamProfile,
    )
except Exception:
    # 某些环境里类名不一定全都能直接 import 成功
    # 为了让文件先能被项目引用，这里保底。
    Context = None
    Pipeline = None
    Config = None
    OBSensorType = None
    OBFormat = None
    VideoStreamProfile = None


@dataclass
class CameraIntrinsics:
    """相机内参"""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FrameBundle:
    """单次取帧结果"""
    rgb: Optional[np.ndarray]
    depth: Optional[np.ndarray]
    timestamp: float
    rgb_timestamp: Optional[float] = None
    depth_timestamp: Optional[float] = None


class CameraError(Exception):
    """相机相关异常"""
    pass


class CameraManager:
    """
    RGB-D 相机管理器（优先面向 Orbbec）。

    用法示例：
        cam = CameraManager()
        cam.start()

        frame = cam.get_frame()
        rgb = frame.rgb
        depth = frame.depth

        intr = cam.get_intrinsics()
        z = cam.get_depth_at_pixel(depth, 320, 240)

        cam.stop()
    """

    def __init__(
        self,
        color_width: int = 640,
        color_height: int = 480,
        color_fps: int = 30,
        depth_width: int = 640,
        depth_height: int = 480,
        depth_fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
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
        """启动相机流"""
        with self._lock:
            if self._started:
                return

            if Pipeline is None or Config is None:
                raise CameraError(
                    "未检测到 pyorbbecsdk，请先正确安装 SDK，或修改 camera_manager.py 顶部 import。"
                )

            try:
                self._ctx = Context() if Context is not None else None
                self._pipeline = Pipeline()
                self._config = Config()

                profile_list = self._pipeline.get_stream_profile_list()

                if self.enable_color:
                    color_profile = self._select_video_profile(
                        profile_list=profile_list,
                        sensor_type=OBSensorType.COLOR_SENSOR,
                        width=self.color_width,
                        height=self.color_height,
                        fmt=OBFormat.RGB,
                        fps=self.color_fps,
                    )
                    self._config.enable_stream(color_profile)

                if self.enable_depth:
                    depth_profile = self._select_video_profile(
                        profile_list=profile_list,
                        sensor_type=OBSensorType.DEPTH_SENSOR,
                        width=self.depth_width,
                        height=self.depth_height,
                        fmt=OBFormat.Y16,
                        fps=self.depth_fps,
                    )
                    self._config.enable_stream(depth_profile)

                # 是否做深度对齐到彩色
                # 某些 SDK 支持 set_align_mode / enable_align
                # 不同版本接口不同，这里做兼容保护
                if self.align_to_color:
                    try:
                        # 某些版本是：
                        # self._config.set_align_mode(OBAlignMode.HW_MODE)
                        # 或 pipeline.enable_frame_sync()
                        self._pipeline.enable_frame_sync()
                    except Exception:
                        pass

                self._pipeline.start(self._config)
                self._started = True

                # 启动后先尝试缓存内参、深度尺度
                self._warmup_and_cache()

            except Exception as e:
                self._started = False
                raise CameraError(f"启动相机失败: {e}") from e

    def stop(self) -> None:
        """停止相机流"""
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
        """
        获取一帧 RGB + Depth。

        返回：
            FrameBundle(
                rgb=np.ndarray(H, W, 3) / None,
                depth=np.ndarray(H, W) / None,   # 建议统一为 float32，单位米
                timestamp=float
            )
        """
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

            # 彩色帧
            if self.enable_color:
                try:
                    color_frame = frames.get_color_frame()
                    if color_frame is not None:
                        color_image = self._convert_color_frame_to_bgr(color_frame)
                        rgb_ts = self._safe_get_timestamp(color_frame)
                except Exception as e:
                    raise CameraError(f"解析彩色帧失败: {e}") from e

            # 深度帧
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
        """
        获取彩色相机内参。
        如果拿不到彩色内参，可按需要改成深度内参。
        """
        if self._cached_intrinsics is not None:
            return self._cached_intrinsics

        if not self._started:
            raise CameraError("相机尚未启动，无法获取内参")

        try:
            profile_list = self._pipeline.get_stream_profile_list()

            if self.enable_color:
                profile = self._select_video_profile(
                    profile_list=profile_list,
                    sensor_type=OBSensorType.COLOR_SENSOR,
                    width=self.color_width,
                    height=self.color_height,
                    fmt=OBFormat.RGB,
                    fps=self.color_fps,
                )
            elif self.enable_depth:
                profile = self._select_video_profile(
                    profile_list=profile_list,
                    sensor_type=OBSensorType.DEPTH_SENSOR,
                    width=self.depth_width,
                    height=self.depth_height,
                    fmt=OBFormat.Y16,
                    fps=self.depth_fps,
                )
            else:
                raise CameraError("未启用任何流，无法获取内参")

            # 常见 SDK 接口：profile.get_intrinsic()
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

        except Exception as e:
            raise CameraError(f"获取相机内参失败: {e}") from e

    def get_depth_scale(self) -> float:
        """
        获取深度尺度，返回“原始深度值 * scale = 米”。
        有些 SDK 直接返回毫米图，有些返回原始 uint16 深度单位。
        这里做统一。
        """
        if self._depth_scale is not None:
            return self._depth_scale

        if not self._started:
            raise CameraError("相机尚未启动，无法获取深度尺度")

        try:
            # 常见做法：从 depth frame / profile / sensor 获取
            # 这里采用“取一帧后读 depth_scale”的兼容方式
            frames = self._pipeline.wait_for_frames(1000)
            if frames is None:
                raise CameraError("无法获取深度尺度：取帧失败")

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                raise CameraError("无法获取深度尺度：无深度帧")

            scale = None

            # 常见接口 1
            try:
                scale = depth_frame.get_depth_scale()
            except Exception:
                pass

            # 常见接口 2：有些版本在 frame 的 value scale 里
            if scale is None:
                try:
                    scale = depth_frame.get_value_scale()
                except Exception:
                    pass

            # 保底：很多设备默认毫米 -> 米
            if scale is None:
                scale = 0.001

            self._depth_scale = float(scale)
            return self._depth_scale

        except Exception as e:
            raise CameraError(f"获取深度尺度失败: {e}") from e

    def get_depth_at_pixel(self, depth_image: np.ndarray, u: int, v: int) -> float:
        """
        获取某像素点深度值（单位：米）。
        如果该点无效，返回 0.0
        """
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
        """
        当中心像素深度无效时，在邻域内找一个较稳健的深度值。
        返回中位数深度（米）。
        """
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
        """
        调试显示单帧。
        按任意键继续。
        """
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
            # 统一高度后横向拼接
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
        """
        将深度图（米）转成伪彩色图，便于调试显示。
        """
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
        """
        像素坐标 + 深度 -> 相机坐标系 3D 点（单位：米）
        """
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
        """
        启动后预热几帧，并缓存内参与深度尺度。
        """
        for _ in range(warmup_frames):
            try:
                self._pipeline.wait_for_frames(500)
            except Exception:
                pass

        try:
            _ = self.get_depth_scale() if self.enable_depth else None
        except Exception:
            pass

        try:
            _ = self.get_intrinsics()
        except Exception:
            pass

    def _select_video_profile(
        self,
        profile_list: Any,
        sensor_type: Any,
        width: int,
        height: int,
        fmt: Any,
        fps: int,
    ) -> Any:
        """
        选择视频流配置。
        不同 pyorbbecsdk 版本接口略有差异，这里做几层兼容。
        """
        try:
            # 常见接口
            return profile_list.get_video_stream_profile(
                sensor_type, width, height, fmt, fps
            )
        except Exception:
            pass

        try:
            # 某些版本先拿 sensor profile list
            sensor_profiles = profile_list.get_stream_profile_list(sensor_type)
            return sensor_profiles.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            pass

        try:
            # 再保底：取默认 profile
            sensor_profiles = profile_list.get_stream_profile_list(sensor_type)
            return sensor_profiles.get_default_video_stream_profile()
        except Exception as e:
            raise CameraError(
                f"无法选择流配置: sensor={sensor_type}, "
                f"size={width}x{height}, fps={fps}, fmt={fmt}, err={e}"
            ) from e

    def _convert_color_frame_to_bgr(self, color_frame: Any) -> np.ndarray:
        """
        将 SDK 彩色帧转为 OpenCV BGR 图像。
        """
        width = int(color_frame.get_width())
        height = int(color_frame.get_height())
        data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        # 常见情况：RGB888
        expected_rgb = width * height * 3
        if data.size == expected_rgb:
            rgb = data.reshape((height, width, 3))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr

        # 少数情况：MJPG / 其他压缩格式
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise CameraError("彩色帧解码失败，可能格式不是 RGB888/MJPG")
        return img

    def _convert_depth_frame_to_meters(self, depth_frame: Any) -> np.ndarray:
        """
        将深度帧统一转换为 float32 米单位深度图。
        """
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
        """
        兼容不同 SDK 的时间戳接口。
        """
        try:
            return float(frame.get_timestamp())
        except Exception:
            try:
                return float(frame.get_time_stamp())
            except Exception:
                return None


if __name__ == "__main__":
    """
    本文件直接运行可用于相机连通性测试：
        python camera_manager.py
    """
    cam = CameraManager(
        color_width=640,
        color_height=480,
        color_fps=30,
        depth_width=640,
        depth_height=480,
        depth_fps=30,
        enable_color=True,
        enable_depth=True,
        align_to_color=False,
    )

    try:
        cam.start()
        print("相机启动成功")

        intr = cam.get_intrinsics()
        print("相机内参:", intr)

        frame = cam.get_frame()
        print("rgb shape:", None if frame.rgb is None else frame.rgb.shape)
        print("depth shape:", None if frame.depth is None else frame.depth.shape)

        if frame.depth is not None:
            h, w = frame.depth.shape
            u, v = w // 2, h // 2
            z = cam.get_valid_depth_near_pixel(frame.depth, u, v)
            print(f"中心点深度: ({u}, {v}) -> {z:.4f} m")

            xyz = cam.pixel_to_camera(u, v, z, intr)
            print("中心点相机坐标:", xyz)

        cam.visualize_once(frame)

    except Exception as e:
        print("运行失败:", e)

    finally:
        cam.stop()
        print("相机已停止")