# -*- coding: utf-8 -*-
"""
yolo_grasp_prep_on_change_demo.py

在 yolo_grasp_prep_demo.py 基础上：
1. 仅当「当前最佳目标」相对上一次已发布结果发生有意义变化时，才打印 [TARGET]（在 grasp_prep 格式上增加深度与相机坐标）
2. 可选每隔 N 帧做一次 YOLO 推理（降低 GPU 占用；未推理帧沿用上次检测框叠画，快速运动时框可能略有偏差）
3. 默认开启 **RGB + 深度**；`--length-unit` 决定整条几何链单位。**mm** 时 SDK 深度在 `CameraManager` 内由原始值直接得到毫米（不经米）；手眼与 D2C 平移与之一致（标定文件用毫米填写）

说明：
- 延迟主要来自每帧 YOLO 推理与相机管线，而不是终端 print
- 深度与彩色分辨率不一致时，会将 (u,v) 映射到深度图再采样；几何对齐依赖设备/SDK，未做 D2C 时可能有偏差
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import configparser
import json
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera.camera_manager import CameraIntrinsics, CameraManager, CameraError

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError("请先安装 ultralytics: pip install ultralytics") from e


def safe_put_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> None:
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="本地 YOLO 模型路径，例如 ~/PiperClaw/models/yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu")
    parser.add_argument("--classes", type=str, default="", help="只检测指定类别，例如 cup,bottle")
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=10.0,
        help="中心点 (u,v) 移动超过该像素视为目标状态变化并刷新 [TARGET]",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=1,
        help="每 N 帧运行一次 YOLO（>=1）；>1 可降低 GPU 负载，未推理帧沿用上次检测结果叠画",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="关闭深度流，仅 RGB（与旧行为一致）",
    )
    parser.add_argument(
        "--align-to-color",
        action="store_true",
        help="尝试启用帧同步（设备不支持时会忽略）；部分机型上有利于 RGB/深度时间对齐",
    )
    parser.add_argument(
        "--show-depth-panel",
        action="store_true",
        help="并排显示深度伪彩色图（小窗 Depth）",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="不显示 OpenCV 窗口（可用于排查/规避 imshow 阻塞导致的高延迟）",
    )
    parser.add_argument(
        "--display-every",
        type=int,
        default=1,
        help="每 N 帧刷新一次窗口显示（>=1），降低 imshow/waitKey 开销",
    )
    parser.add_argument(
        "--waitkey-ms",
        type=int,
        default=1,
        help="cv2.waitKey 等待毫秒数（用于处理窗口事件；默认 1）",
    )
    parser.add_argument(
        "--min-depth",
        type=float,
        default=None,
        help="深度有效下限，单位与 --length-unit 一致；省略时默认 0.05(m) 或 50(mm)",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="深度有效上限，单位与 --length-unit 一致；省略时默认 3(m) 或 3000(mm)",
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        default="bbox",
        choices=["center", "bbox"],
        help="深度取样方式：center=中心邻域中值；bbox=bbox区域内有效深度中值（更稳，默认）",
    )
    parser.add_argument(
        "--depth-kernel",
        type=int,
        default=7,
        help="center 模式的邻域核大小（奇数），越大越稳但越可能混入背景",
    )
    parser.add_argument(
        "--depth-bbox-shrink",
        type=float,
        default=0.25,
        help="bbox 模式取样时对框做收缩比例（0~0.49），避免边缘背景干扰；0.25 表示四边各收缩 25%",
    )
    parser.add_argument(
        "--camera-param-ini",
        type=str,
        default="",
        help="Orbbec CameraParam ini（Color/Depth 内参 + D2CTransformParam）。D2C 的 trans0..2 单位须与 --length-unit 一致。",
    )
    parser.add_argument(
        "--cam-to-base-json",
        type=str,
        default="",
        help="手眼标定 JSON：相机→基座。rotation 无单位；translation/position 的单位与 --length-unit 一致（毫米模式请填毫米）。",
    )
    parser.add_argument(
        "--length-unit",
        type=str,
        default="m",
        choices=["m", "mm"],
        help="整条管线长度单位。mm：SDK 深度在 camera_manager 内直接输出毫米；D2C/手眼 JSON 的平移也用毫米。m：均为米。",
    )
    parser.add_argument(
        "--time-stat",
        action="store_true",
        help="开启耗时统计（取帧/YOLO/画图imshow）",
    )
    parser.add_argument(
        "--time-print-every",
        type=int,
        default=30,
        help="每 N 帧打印一次耗时统计（配合 --time-stat）",
    )
    return parser.parse_args()


def length_suffix(unit: str) -> str:
    return "mm" if unit == "mm" else "m"


def map_rgb_uv_to_depth_uv(
    u: int,
    v: int,
    rgb_shape: Tuple[int, ...],
    depth_shape: Tuple[int, ...],
) -> Tuple[int, int]:
    """彩色图像素映射到深度图坐标（分辨率不同时按比例）。"""
    rh, rw = int(rgb_shape[0]), int(rgb_shape[1])
    dh, dw = int(depth_shape[0]), int(depth_shape[1])
    if dh == rh and dw == rw:
        return u, v
    ud = int(round(u * dw / max(rw, 1)))
    vd = int(round(v * dh / max(rh, 1)))
    ud = max(0, min(dw - 1, ud))
    vd = max(0, min(dh - 1, vd))
    return ud, vd


def attach_depth_and_cam_xyz(
    det: Dict[str, Any],
    depth_image: Optional[np.ndarray],
    rgb_shape: Tuple[int, ...],
    intr_color: CameraIntrinsics,
    intr_depth: CameraIntrinsics,
    d2c_R: Optional[np.ndarray],
    d2c_t: Optional[np.ndarray],
    cam: CameraManager,
    min_depth: float,
    max_depth: float,
    depth_mode: str = "bbox",
    depth_kernel: int = 7,
    depth_bbox_shrink: float = 0.25,
) -> Dict[str, Any]:
    """附加 depth 与 cam_x/y/z；depth_image 单位与 CameraManager.depth_unit、min/max_depth 一致。"""
    out = det.copy()
    out["depth"] = 0.0
    out["cam_x"] = 0.0
    out["cam_y"] = 0.0
    out["cam_z"] = 0.0

    if depth_image is None:
        return out

    dep = depth_image.astype(np.float64, copy=False)

    u, v = int(det["u"]), int(det["v"])
    ud, vd = map_rgb_uv_to_depth_uv(u, v, rgb_shape, dep.shape)

    # 更鲁棒的深度取样：优先 bbox 区域有效深度中位数
    z = 0.0
    if depth_mode == "bbox":
        dh, dw = dep.shape[:2]
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        x1d, y1d = map_rgb_uv_to_depth_uv(x1, y1, rgb_shape, dep.shape)
        x2d, y2d = map_rgb_uv_to_depth_uv(x2, y2, rgb_shape, dep.shape)
        xa, xb = (x1d, x2d) if x1d <= x2d else (x2d, x1d)
        ya, yb = (y1d, y2d) if y1d <= y2d else (y2d, y1d)

        xa = max(0, min(dw - 1, xa))
        xb = max(0, min(dw - 1, xb))
        ya = max(0, min(dh - 1, ya))
        yb = max(0, min(dh - 1, yb))

        # shrink bbox to reduce background
        shrink = float(depth_bbox_shrink)
        shrink = max(0.0, min(0.49, shrink))
        w = max(0, xb - xa)
        h = max(0, yb - ya)
        xa2 = int(round(xa + w * shrink))
        xb2 = int(round(xb - w * shrink))
        ya2 = int(round(ya + h * shrink))
        yb2 = int(round(yb - h * shrink))
        if xb2 > xa2 and yb2 > ya2:
            patch = dep[ya2:yb2, xa2:xb2]
            valid = patch[(patch > min_depth) & (patch < max_depth) & np.isfinite(patch)]
            if valid.size > 0:
                z = float(np.median(valid))

    if z <= 0.0:
        k = int(depth_kernel)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        z = float(cam.get_valid_depth_near_pixel(dep, ud, vd, kernel_size=k))
    z = float(z)
    if (not np.isfinite(z)) or (z < min_depth) or (z > max_depth):
        z = 0.0
    out["depth"] = float(z)

    if z > 0 and np.isfinite(z):
        # Use depth intrinsics + depth pixel to compute 3D in depth camera coordinates.
        dx, dy, dz = CameraManager.pixel_to_camera(ud, vd, z, intr_depth)

        # Transform to color camera coordinates if D2C extrinsics provided.
        if d2c_R is not None and d2c_t is not None:
            p = np.array([dx, dy, dz], dtype=np.float64)
            pc = (d2c_R @ p) + d2c_t
            out["cam_x"] = float(pc[0])
            out["cam_y"] = float(pc[1])
            out["cam_z"] = float(pc[2])
        else:
            out["cam_x"] = float(dx)
            out["cam_y"] = float(dy)
            out["cam_z"] = float(dz)

    return out


def load_rigid_transform_json(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rigid transform from JSON. translation / position 的单位由调用方与 --length-unit 约定（米或毫米）。
    Supported formats:
      1) rotation/translation:
         - rotation: 3x3 nested list
         - translation: [tx, ty, tz]
      2) handeye_calibration_ros result:
         - position: [tx, ty, tz]
         - orientation: [qx, qy, qz, qw]
    """

    def _quat_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        n = x * x + y * y + z * z + w * w
        if n <= 1e-12:
            raise ValueError("orientation 四元数范数为 0，无法转换旋转矩阵")
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array(
            [
                [1.0 - (yy + zz), xy - wz, xz + wy],
                [xy + wz, 1.0 - (xx + zz), yz - wx],
                [xz - wy, yz + wx, 1.0 - (xx + yy)],
            ],
            dtype=np.float64,
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "rotation" in data and "translation" in data:
        R = np.array(data["rotation"], dtype=np.float64)
        t = np.array(data["translation"], dtype=np.float64).reshape(3)
        if R.shape != (3, 3):
            raise ValueError("rotation 必须是 3x3")
        return R, t

    if "position" in data and "orientation" in data:
        t = np.array(data["position"], dtype=np.float64).reshape(3)
        q = np.array(data["orientation"], dtype=np.float64).reshape(4)
        R = _quat_xyzw_to_rotmat(q)
        return R, t

    raise ValueError(
        "transform json 必须包含 rotation/translation 或 position/orientation 字段"
    )


def attach_robot_xyz(
    det: Dict[str, Any],
    cam_to_base_R: Optional[np.ndarray],
    cam_to_base_t: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    在 det 中补充 robot_x/y/z（基坐标系），单位与 cam_x/y/z 一致。
    """
    out = det.copy()
    out["robot_x"] = 0.0
    out["robot_y"] = 0.0
    out["robot_z"] = 0.0

    if cam_to_base_R is None or cam_to_base_t is None:
        return out

    x = float(out.get("cam_x", 0.0))
    y = float(out.get("cam_y", 0.0))
    z = float(out.get("cam_z", 0.0))
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
        return out
    if z <= 0.0:
        return out

    p_cam = np.array([x, y, z], dtype=np.float64)
    p_base = (cam_to_base_R @ p_cam) + cam_to_base_t
    out["robot_x"] = float(p_base[0])
    out["robot_y"] = float(p_base[1])
    out["robot_z"] = float(p_base[2])
    return out


def load_camera_param_ini(path: str) -> Tuple[CameraIntrinsics, CameraIntrinsics, np.ndarray, np.ndarray]:
    """
    Parse Orbbec Viewer exported CameraParam ini.
    D2CTransformParam trans0..2 的单位应与运行时 --length-unit 一致（米或毫米）。
    Returns: (color_intr, depth_intr, R(3,3), t(3,))
    """
    cp = configparser.ConfigParser()
    with open(path, "r", encoding="utf-8") as f:
        cp.read_file(f)

    def _intr(section: str) -> CameraIntrinsics:
        s = cp[section]
        return CameraIntrinsics(
            width=int(float(s.get("width"))),
            height=int(float(s.get("height"))),
            fx=float(s.get("fx")),
            fy=float(s.get("fy")),
            cx=float(s.get("cx")),
            cy=float(s.get("cy")),
        )

    color_intr = _intr("ColorIntrinsic")
    depth_intr = _intr("DepthIntrinsic")

    s = cp["D2CTransformParam"]
    R = np.array(
        [
            [float(s.get("rot0")), float(s.get("rot1")), float(s.get("rot2"))],
            [float(s.get("rot3")), float(s.get("rot4")), float(s.get("rot5"))],
            [float(s.get("rot6")), float(s.get("rot7")), float(s.get("rot8"))],
        ],
        dtype=np.float64,
    )
    t = np.array([float(s.get("trans0")), float(s.get("trans1")), float(s.get("trans2"))], dtype=np.float64)
    return color_intr, depth_intr, R, t


def get_detections(result) -> List[Dict[str, Any]]:
    detections = []

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
        x1, y1, x2, y2 = xyxy

        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        w = int(x2 - x1)
        h = int(y2 - y1)
        area = w * h

        detections.append(
            {
                "class_name": cls_name,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "u": u,
                "v": v,
                "w": w,
                "h": h,
                "area": area,
            }
        )

    return detections


def choose_best_detection(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not detections:
        return None

    detections = sorted(
        detections,
        key=lambda d: (d["conf"], d["area"]),
        reverse=True,
    )
    return detections[0]


def best_target_changed(
    prev: Optional[Dict[str, Any]],
    curr: Optional[Dict[str, Any]],
    pixel_threshold: float,
) -> bool:
    """判定是否应在控制台发布新的 [TARGET] 行。"""
    if prev is None and curr is None:
        return False
    if prev is None or curr is None:
        return True
    if prev["class_name"] != curr["class_name"]:
        return True
    du = float(prev["u"] - curr["u"])
    dv = float(prev["v"] - curr["v"])
    if (du * du + dv * dv) ** 0.5 > pixel_threshold:
        return True
    return False


def main():
    args = parse_args()
    if args.infer_every < 1:
        raise SystemExit("--infer-every 必须 >= 1")
    if args.display_every < 1:
        raise SystemExit("--display-every 必须 >= 1")
    if args.waitkey_ms < 0:
        raise SystemExit("--waitkey-ms 必须 >= 0")

    if args.min_depth is None:
        args.min_depth = 50.0 if args.length_unit == "mm" else 0.05
    if args.max_depth is None:
        args.max_depth = 3000.0 if args.length_unit == "mm" else 3.0

    len_suffix = length_suffix(args.length_unit)

    target_class_names = set()
    if args.classes.strip():
        target_class_names = {x.strip() for x in args.classes.split(",") if x.strip()}

    ini_color_intr: Optional[CameraIntrinsics] = None
    ini_depth_intr: Optional[CameraIntrinsics] = None
    d2c_R: Optional[np.ndarray] = None
    d2c_t: Optional[np.ndarray] = None
    cam_to_base_R: Optional[np.ndarray] = None
    cam_to_base_t: Optional[np.ndarray] = None
    if args.camera_param_ini.strip():
        ini_path = args.camera_param_ini.strip()
        if not os.path.isabs(ini_path):
            ini_path = os.path.join(PROJECT_ROOT, ini_path)
        ini_color_intr, ini_depth_intr, d2c_R, d2c_t = load_camera_param_ini(ini_path)
        if d2c_t is not None:
            d2c_t = np.asarray(d2c_t, dtype=np.float64)
        print("[INFO] loaded camera param ini:", ini_path)
        print("[INFO] ini color intr:", ini_color_intr)
        print("[INFO] ini depth intr:", ini_depth_intr)

    if args.cam_to_base_json.strip():
        tf_path = args.cam_to_base_json.strip()
        if not os.path.isabs(tf_path):
            tf_path = os.path.join(PROJECT_ROOT, tf_path)
        cam_to_base_R, cam_to_base_t = load_rigid_transform_json(tf_path)
        cam_to_base_t = np.asarray(cam_to_base_t, dtype=np.float64)
        print("[INFO] loaded hand-eye transform:", tf_path)
        print("[INFO] cam->base R:\n", cam_to_base_R)
        print(f"[INFO] cam->base t ({len_suffix}):", cam_to_base_t)

    print(f"[INFO] loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        raise RuntimeError(
            f"YOLO 模型加载失败，请检查模型文件是否存在且完整。\n"
            f"当前路径: {args.model}\n"
            f"原始错误: {e}"
        ) from e

    use_depth = not args.no_depth
    # If ini provided, use its resolution to match camera param/profile.
    cw = int(ini_color_intr.width) if ini_color_intr is not None else 640
    ch = int(ini_color_intr.height) if ini_color_intr is not None else 480
    dw = int(ini_depth_intr.width) if ini_depth_intr is not None else 640
    dh = int(ini_depth_intr.height) if ini_depth_intr is not None else 480
    cam = CameraManager(
        color_width=cw,
        color_height=ch,
        color_fps=30,
        depth_width=dw,
        depth_height=dh,
        depth_fps=30,
        enable_color=True,
        enable_depth=use_depth,
        align_to_color=args.align_to_color,
        latest_only=True,
        depth_unit=args.length_unit,
    )

    prev_time = time.time()
    fps = 0.0
    saved_target = None
    published_best: Optional[Dict[str, Any]] = None

    last_detections: List[Dict[str, Any]] = []
    last_best: Optional[Dict[str, Any]] = None
    frame_index = 0

    try:
        cam.start()
        print("[INFO] camera started")
        print("[INFO] frame grab: latest-only (后台线程持续取流，减轻 Pipeline 队列满/丢帧)")

        # Prefer ini intrinsics if provided; else use SDK intrinsics (color) as legacy default.
        intr_color = ini_color_intr if ini_color_intr is not None else cam.get_intrinsics()
        intr_depth = ini_depth_intr if ini_depth_intr is not None else (
            cam.get_depth_intrinsics() if hasattr(cam, "get_depth_intrinsics") else intr_color
        )
        print("[INFO] intrinsics(color):", intr_color)
        print("[INFO] intrinsics(depth):", intr_depth)
        print(f"[INFO] length unit: {len_suffix} (depth / cam_xyz / base_xyz / thresholds)")
        print(f"[INFO] depth valid range: [{args.min_depth:g}, {args.max_depth:g}] {len_suffix}")
        if use_depth:
            print(
                "[INFO] SDK depth stored as",
                "mm (uint16 × depth_scale × 1000, no meter buffer)"
                if args.length_unit == "mm"
                else "m (uint16 × depth_scale)",
            )
        print("[INFO] depth stream:", "on" if use_depth else "off (--no-depth)")

        window_name = "YOLO Grasp Prep (on change)"
        if not args.no_gui:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if args.show_depth_panel and use_depth:
                cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

        # windowed time stats
        stat_window_frames = 0
        stat_grab_s = 0.0
        stat_infer_s = 0.0
        stat_draw_det_s = 0.0
        stat_attach_depth_s = 0.0
        stat_draw_best_s = 0.0
        stat_print_s = 0.0
        stat_imshow_s = 0.0
        stat_infer_calls = 0
        stat_print_calls = 0

        while True:
            t_grab0 = time.perf_counter()
            frame_bundle = cam.get_frame(timeout_ms=1000)
            frame = frame_bundle.rgb
            depth_img = frame_bundle.depth if use_depth else None
            t_grab1 = time.perf_counter()

            if frame is None:
                print("[WARN] empty rgb frame")
                continue

            frame_index += 1
            run_infer = (frame_index % args.infer_every) == 0

            detections: List[Dict[str, Any]] = []
            best_target: Optional[Dict[str, Any]] = None

            t_infer0 = time.perf_counter()
            if run_infer:
                results = model.predict(
                    source=frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )

                detections = []
                best_target = None
                if len(results) > 0:
                    result = results[0]
                    detections = get_detections(result)
                    if target_class_names:
                        detections = [d for d in detections if d["class_name"] in target_class_names]
                    best_target = choose_best_detection(detections)

                last_detections = detections
                last_best = best_target
            else:
                detections = last_detections
                best_target = last_best
            t_infer1 = time.perf_counter()

            # 画图 + 深度采样 + imshow/waitKey（细分计时用于定位延迟）
            t_vis0 = time.perf_counter()
            vis = frame.copy()
            rgb_shape = frame.shape
            t_draw_det0 = time.perf_counter()
            t_attach0 = t_vis0
            t_attach1 = t_vis0
            t_draw_best0 = t_vis0
            t_draw_best1 = t_vis0
            t_print_dur = 0.0
            t_print_calls = 0

            if detections:
                for det in detections:
                    draw_box(vis, det["x1"], det["y1"], det["x2"], det["y2"], color=(0, 255, 0), thickness=2)
                    cv2.circle(vis, (det["u"], det["v"]), 4, (0, 0, 255), -1)

                    label = f'{det["class_name"]} {det["conf"]:.2f}'
                    safe_put_text(vis, label, (det["x1"], max(25, det["y1"] - 8)))
                    safe_put_text(vis, f'({det["u"]}, {det["v"]})', (det["x1"], min(vis.shape[0] - 10, det["y2"] + 20)), 0.5, 1)
            t_draw_det1 = time.perf_counter()

            if best_target is not None:
                t_attach0 = time.perf_counter()
                best_e = attach_depth_and_cam_xyz(
                    best_target,
                    depth_img,
                    rgb_shape,
                    intr_color,
                    intr_depth,
                    d2c_R,
                    d2c_t,
                    cam,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    depth_mode=args.depth_mode,
                    depth_kernel=args.depth_kernel,
                    depth_bbox_shrink=args.depth_bbox_shrink,
                )
                best_e = attach_robot_xyz(best_e, cam_to_base_R, cam_to_base_t)
                t_attach1 = time.perf_counter()

                t_draw_best0 = time.perf_counter()
                draw_box(
                    vis,
                    best_target["x1"],
                    best_target["y1"],
                    best_target["x2"],
                    best_target["y2"],
                    color=(255, 0, 0),
                    thickness=3,
                )
                cv2.circle(vis, (best_target["u"], best_target["v"]), 6, (255, 0, 0), -1)

                safe_put_text(
                    vis,
                    f'BEST: {best_target["class_name"]} ({best_target["u"]}, {best_target["v"]})',
                    (20, 95),
                    0.7,
                    2,
                    color=(255, 0, 0),
                )
                if use_depth:
                    safe_put_text(
                        vis,
                        f'Z={best_e["depth"]:.3f}{len_suffix}  '
                        f'cam=({best_e["cam_x"]:.3f},{best_e["cam_y"]:.3f},{best_e["cam_z"]:.3f})',
                        (20, 120),
                        0.55,
                        2,
                        color=(255, 0, 0),
                    )
                    if cam_to_base_R is not None and cam_to_base_t is not None:
                        safe_put_text(
                            vis,
                            f'base=({best_e["robot_x"]:.3f},{best_e["robot_y"]:.3f},{best_e["robot_z"]:.3f})',
                            (20, 145),
                            0.55,
                            2,
                            color=(255, 0, 0),
                        )
                t_draw_best1 = time.perf_counter()

                if best_target_changed(published_best, best_target, args.pixel_threshold):
                    published_best = best_e.copy()
                    if use_depth:
                        t_print0 = time.perf_counter()
                        print(
                            f'[TARGET] class={best_e["class_name"]}, '
                            f'conf={best_e["conf"]:.3f}, '
                            f'pixel=({best_e["u"]}, {best_e["v"]}), '
                            f'bbox=({best_e["x1"]}, {best_e["y1"]}, {best_e["x2"]}, {best_e["y2"]}), '
                            f'depth_{len_suffix}={best_e["depth"]:.3f}, '
                            f'cam_xyz_{len_suffix}=({best_e["cam_x"]:.4f}, {best_e["cam_y"]:.4f}, {best_e["cam_z"]:.4f})'
                            + (
                                f', base_xyz_{len_suffix}=({best_e["robot_x"]:.4f}, {best_e["robot_y"]:.4f}, {best_e["robot_z"]:.4f})'
                                if (cam_to_base_R is not None and cam_to_base_t is not None)
                                else ""
                            )
                        )
                        t_print_dur = time.perf_counter() - t_print0
                        t_print_calls = 1
                    else:
                        t_print0 = time.perf_counter()
                        print(
                            f'[TARGET] class={best_target["class_name"]}, '
                            f'conf={best_target["conf"]:.3f}, '
                            f'pixel=({best_target["u"]}, {best_target["v"]}), '
                            f'bbox=({best_target["x1"]}, {best_target["y1"]}, {best_target["x2"]}, {best_target["y2"]})'
                        )
                        t_print_dur = time.perf_counter() - t_print0
                        t_print_calls = 1
            else:
                if published_best is not None:
                    published_best = None
                    t_print0 = time.perf_counter()
                    print("[INFO] best target cleared (no detection)")
                    t_print_dur = time.perf_counter() - t_print0
                    t_print_calls = 1

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            safe_put_text(vis, f"FPS: {fps:.2f}", (20, 30), 0.8, 2)
            depth_tag = "depth:on" if use_depth else "depth:off"
            hint = f"q: quit | s: save | {depth_tag} | infer {args.infer_every}f | px>{args.pixel_threshold}"
            safe_put_text(vis, hint, (20, 60), 0.5, 1)

            saved_y = 155 if use_depth else 130
            if saved_target is not None:
                if use_depth and "depth" in saved_target:
                    safe_put_text(
                        vis,
                        f'SAVED: {saved_target["class_name"]} @ ({saved_target["u"]}, {saved_target["v"]}) '
                        f'Z={saved_target["depth"]:.3f}{len_suffix}',
                        (20, saved_y),
                        0.65,
                        2,
                        color=(0, 255, 255),
                    )
                else:
                    safe_put_text(
                        vis,
                        f'SAVED: {saved_target["class_name"]} @ ({saved_target["u"]}, {saved_target["v"]})',
                        (20, saved_y),
                        0.7,
                        2,
                        color=(0, 255, 255),
                    )

            key = 255
            t_imshow0 = time.perf_counter()
            if not args.no_gui:
                # 关键点：即使不刷新画面，也要每帧处理一次 GUI 事件，避免事件队列堆积导致后续 waitKey 阻塞很久
                if frame_index % args.display_every == 0:
                    cv2.imshow(window_name, vis)
                    if args.show_depth_panel and use_depth and depth_img is not None:
                        dvis = CameraManager.depth_to_colormap(
                            depth_img, max_depth=args.max_depth, depth_unit=args.length_unit
                        )
                        if dvis.shape[0] != vis.shape[0]:
                            scale = vis.shape[0] / max(dvis.shape[0], 1)
                            nw = max(1, int(dvis.shape[1] * scale))
                            dvis = cv2.resize(dvis, (nw, vis.shape[0]))
                        cv2.imshow("Depth", dvis)
                key = cv2.waitKey(int(args.waitkey_ms)) & 0xFF
            t_vis1 = time.perf_counter()
            t_imshow1 = t_vis1

            if key == ord("q"):
                break
            elif key == ord("s"):
                if best_target is not None:
                    best_e = attach_depth_and_cam_xyz(
                        best_target,
                        depth_img,
                        frame.shape,
                        intr_color,
                        intr_depth,
                        d2c_R,
                        d2c_t,
                        cam,
                        min_depth=args.min_depth,
                        max_depth=args.max_depth,
                        depth_mode=args.depth_mode,
                        depth_kernel=args.depth_kernel,
                        depth_bbox_shrink=args.depth_bbox_shrink,
                    )
                    best_e = attach_robot_xyz(best_e, cam_to_base_R, cam_to_base_t)
                    saved_target = best_e.copy()
                    if use_depth:
                        print(
                            f'[SAVE] selected target: '
                            f'class={saved_target["class_name"]}, '
                            f'pixel=({saved_target["u"]}, {saved_target["v"]}), '
                            f'bbox=({saved_target["x1"]}, {saved_target["y1"]}, {saved_target["x2"]}, {saved_target["y2"]}), '
                            f'depth_{len_suffix}={saved_target["depth"]:.3f}, '
                            f'cam_xyz_{len_suffix}=({saved_target["cam_x"]:.4f}, {saved_target["cam_y"]:.4f}, {saved_target["cam_z"]:.4f})'
                            + (
                                f', base_xyz_{len_suffix}=({saved_target["robot_x"]:.4f}, {saved_target["robot_y"]:.4f}, {saved_target["robot_z"]:.4f})'
                                if (cam_to_base_R is not None and cam_to_base_t is not None)
                                else ""
                            )
                        )
                    else:
                        print(
                            f'[SAVE] selected target: '
                            f'class={saved_target["class_name"]}, '
                            f'pixel=({saved_target["u"]}, {saved_target["v"]}), '
                            f'bbox=({saved_target["x1"]}, {saved_target["y1"]}, {saved_target["x2"]}, {saved_target["y2"]})'
                        )
                else:
                    print("[SAVE] no target to save")

            if args.time_stat:
                stat_window_frames += 1
                stat_grab_s += (t_grab1 - t_grab0)
                stat_infer_s += (t_infer1 - t_infer0) if run_infer else 0.0
                stat_draw_det_s += (t_draw_det1 - t_draw_det0)
                stat_attach_depth_s += (t_attach1 - t_attach0)
                stat_draw_best_s += (t_draw_best1 - t_draw_best0)
                stat_print_s += t_print_dur
                stat_print_calls += t_print_calls
                stat_imshow_s += (t_imshow1 - t_imshow0)
                if run_infer:
                    stat_infer_calls += 1

                if stat_window_frames >= max(1, int(args.time_print_every)):
                    avg_grab = stat_grab_s / stat_window_frames
                    avg_infer = stat_infer_s / stat_window_frames  # averaged per frame
                    avg_draw_det = stat_draw_det_s / stat_window_frames
                    avg_attach = stat_attach_depth_s / stat_window_frames
                    avg_draw_best = stat_draw_best_s / stat_window_frames
                    avg_imshow = stat_imshow_s / stat_window_frames
                    avg_print_total = stat_print_s / stat_window_frames
                    avg_total = (
                        stat_grab_s
                        + stat_infer_s
                        + stat_draw_det_s
                        + stat_attach_depth_s
                        + stat_draw_best_s
                        + stat_print_s
                        + stat_imshow_s
                    ) / stat_window_frames

                    infer_note = f"(infer calls={stat_infer_calls}, print calls={stat_print_calls})"
                    print(
                        f'[TIME] avg per frame: grab={avg_grab*1000:.1f}ms '
                        f'infer={avg_infer*1000:.1f}ms '
                        f'draw_det={avg_draw_det*1000:.1f}ms '
                        f'attach_depth={avg_attach*1000:.1f}ms '
                        f'draw_best={avg_draw_best*1000:.1f}ms '
                        f'imshow={avg_imshow*1000:.1f}ms '
                        f'print_total={avg_print_total*1000:.1f}ms '
                        f'total={avg_total*1000:.1f}ms {infer_note}'
                    )
                    stat_window_frames = 0
                    stat_grab_s = 0.0
                    stat_infer_s = 0.0
                    stat_draw_det_s = 0.0
                    stat_attach_depth_s = 0.0
                    stat_draw_best_s = 0.0
                    stat_print_s = 0.0
                    stat_imshow_s = 0.0
                    stat_infer_calls = 0
                    stat_print_calls = 0

        cv2.destroyAllWindows()

    except CameraError as e:
        print("[ERROR] camera error:", e)
    except KeyboardInterrupt:
        print("[INFO] interrupted by user")
    except Exception as e:
        print("[ERROR] runtime error:", e)
        raise
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("[INFO] camera stopped")


if __name__ == "__main__":
    main()
