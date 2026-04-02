# -*- coding: utf-8 -*-
"""
yolo_grasp_prep_on_change_demo.py

在 yolo_grasp_prep_demo.py 基础上：
1. 仅当「当前最佳目标」相对上一次已发布结果发生有意义变化时，才打印 [TARGET]（在 grasp_prep 格式上增加深度与相机坐标）
2. 可选每隔 N 帧做一次 YOLO 推理（降低 GPU 占用；未推理帧沿用上次检测框叠画，快速运动时框可能略有偏差）
3. 默认开启 **RGB + 深度**；在最佳目标中心邻域取深度（米），并换算为相机坐标 (X,Y,Z)m

说明：
- 延迟主要来自每帧 YOLO 推理与相机管线，而不是终端 print
- 深度与彩色分辨率不一致时，会将 (u,v) 映射到深度图再采样；几何对齐依赖设备/SDK，未做 D2C 时可能有偏差
"""

from __future__ import annotations

import os
import sys
import time
import argparse
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
    intr: CameraIntrinsics,
    cam: CameraManager,
) -> Dict[str, Any]:
    """为检测字典附加 depth_m 与 cam_x, cam_y, cam_z（米）。彩色与深度分辨率不同时对 (u,v) 做映射采样。"""
    out = det.copy()
    out["depth_m"] = 0.0
    out["cam_x"] = 0.0
    out["cam_y"] = 0.0
    out["cam_z"] = 0.0

    if depth_image is None:
        return out

    u, v = int(det["u"]), int(det["v"])
    ud, vd = map_rgb_uv_to_depth_uv(u, v, rgb_shape, depth_image.shape)
    z = cam.get_valid_depth_near_pixel(depth_image, ud, vd, kernel_size=5)
    out["depth_m"] = float(z)

    if z > 0 and np.isfinite(z):
        cx, cy, cz = CameraManager.pixel_to_camera(u, v, z, intr)
        out["cam_x"] = float(cx)
        out["cam_y"] = float(cy)
        out["cam_z"] = float(cz)

    return out


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

    target_class_names = set()
    if args.classes.strip():
        target_class_names = {x.strip() for x in args.classes.split(",") if x.strip()}

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
    cam = CameraManager(
        color_width=640,
        color_height=480,
        color_fps=30,
        depth_width=640,
        depth_height=480,
        depth_fps=30,
        enable_color=True,
        enable_depth=use_depth,
        align_to_color=args.align_to_color,
        latest_only=True,
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

        intr = cam.get_intrinsics()
        print("[INFO] intrinsics:", intr)
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
                    intr,
                    cam,
                )
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
                        f'Z={best_e["depth_m"]:.3f}m  cam=({best_e["cam_x"]:.3f},{best_e["cam_y"]:.3f},{best_e["cam_z"]:.3f})',
                        (20, 120),
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
                            f'depth_m={best_e["depth_m"]:.3f}, '
                            f'cam_xyz=({best_e["cam_x"]:.4f}, {best_e["cam_y"]:.4f}, {best_e["cam_z"]:.4f})'
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
                if use_depth and "depth_m" in saved_target:
                    safe_put_text(
                        vis,
                        f'SAVED: {saved_target["class_name"]} @ ({saved_target["u"]}, {saved_target["v"]}) '
                        f'Z={saved_target["depth_m"]:.3f}m',
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
            if (not args.no_gui) and (frame_index % args.display_every == 0):
                cv2.imshow(window_name, vis)
                if args.show_depth_panel and use_depth and depth_img is not None:
                    dvis = CameraManager.depth_to_colormap(depth_img)
                    if dvis.shape[0] != vis.shape[0]:
                        scale = vis.shape[0] / max(dvis.shape[0], 1)
                        nw = max(1, int(dvis.shape[1] * scale))
                        dvis = cv2.resize(dvis, (nw, vis.shape[0]))
                    cv2.imshow("Depth", dvis)
                key = cv2.waitKey(1) & 0xFF
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
                        intr,
                        cam,
                    )
                    saved_target = best_e.copy()
                    if use_depth:
                        print(
                            f'[SAVE] selected target: '
                            f'class={saved_target["class_name"]}, '
                            f'pixel=({saved_target["u"]}, {saved_target["v"]}), '
                            f'bbox=({saved_target["x1"]}, {saved_target["y1"]}, {saved_target["x2"]}, {saved_target["y2"]}), '
                            f'depth_m={saved_target["depth_m"]:.3f}, '
                            f'cam_xyz=({saved_target["cam_x"]:.4f}, {saved_target["cam_y"]:.4f}, {saved_target["cam_z"]:.4f})'
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
