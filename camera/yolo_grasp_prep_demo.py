# -*- coding: utf-8 -*-
"""
yolo_grasp_prep_demo.py

功能：
1. 打开 Orbbec RGB
2. YOLO 实时检测
3. 输出目标的图像坐标 (u, v)
4. 按 s 选中当前最佳目标，作为抓取候选
5. 按 q 退出

说明：
- 当前输出的是 2D 像素坐标，不是机械臂 3D 坐标
- 后续接 depth 后，可扩展为相机坐标
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

from camera.camera_manager import CameraManager, CameraError

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
    return parser.parse_args()


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
    """
    当前策略：
    1. 先按置信度排序
    2. 再偏向框面积更大的目标
    """
    if not detections:
        return None

    detections = sorted(
        detections,
        key=lambda d: (d["conf"], d["area"]),
        reverse=True,
    )
    return detections[0]


def main():
    args = parse_args()

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

    cam = CameraManager(
        color_width=640,
        color_height=480,
        color_fps=30,
        enable_color=True,
        enable_depth=False,   # 当前先只做 2D 坐标
        align_to_color=False,
    )

    prev_time = time.time()
    fps = 0.0
    saved_target = None

    try:
        cam.start()
        print("[INFO] camera started")

        intr = cam.get_intrinsics()
        print("[INFO] intrinsics:", intr)

        window_name = "YOLO Grasp Prep Demo"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            frame_bundle = cam.get_frame(timeout_ms=1000)
            frame = frame_bundle.rgb

            if frame is None:
                print("[WARN] empty rgb frame")
                continue

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            vis = frame.copy()
            best_target = None

            if len(results) > 0:
                result = results[0]
                detections = get_detections(result)

                if target_class_names:
                    detections = [d for d in detections if d["class_name"] in target_class_names]

                # 先把所有检测结果都画出来
                for det in detections:
                    draw_box(vis, det["x1"], det["y1"], det["x2"], det["y2"], color=(0, 255, 0), thickness=2)
                    cv2.circle(vis, (det["u"], det["v"]), 4, (0, 0, 255), -1)

                    label = f'{det["class_name"]} {det["conf"]:.2f}'
                    safe_put_text(vis, label, (det["x1"], max(25, det["y1"] - 8)))
                    safe_put_text(vis, f'({det["u"]}, {det["v"]})', (det["x1"], min(vis.shape[0] - 10, det["y2"] + 20)), 0.5, 1)

                # 选择一个最佳抓取候选
                best_target = choose_best_detection(detections)

                if best_target is not None:
                    # 高亮最佳目标
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

                    # 控制台持续输出当前候选坐标
                    print(
                        f'[TARGET] class={best_target["class_name"]}, '
                        f'conf={best_target["conf"]:.3f}, '
                        f'pixel=({best_target["u"]}, {best_target["v"]}), '
                        f'bbox=({best_target["x1"]}, {best_target["y1"]}, {best_target["x2"]}, {best_target["y2"]})'
                    )

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            safe_put_text(vis, f"FPS: {fps:.2f}", (20, 30), 0.8, 2)
            safe_put_text(vis, "q: quit | s: save current target", (20, 60), 0.6, 2)

            if saved_target is not None:
                safe_put_text(
                    vis,
                    f'SAVED: {saved_target["class_name"]} @ ({saved_target["u"]}, {saved_target["v"]})',
                    (20, 130),
                    0.7,
                    2,
                    color=(0, 255, 255),
                )

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                if best_target is not None:
                    saved_target = best_target.copy()
                    print(
                        f'[SAVE] selected target: '
                        f'class={saved_target["class_name"]}, '
                        f'pixel=({saved_target["u"]}, {saved_target["v"]}), '
                        f'bbox=({saved_target["x1"]}, {saved_target["y1"]}, {saved_target["x2"]}, {saved_target["y2"]})'
                    )
                else:
                    print("[SAVE] no target to save")

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