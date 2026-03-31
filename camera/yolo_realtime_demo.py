# -*- coding: utf-8 -*-
"""
yolo_realtime_demo.py

功能：
1. 调用 camera_manager.py 打开 Orbbec DaBai RGB
2. 使用 YOLO 做实时目标检测
3. 在画面上显示类别、置信度、中心点
4. 按 q 退出

运行：
    python camera/yolo_realtime_demo.py

可选：
    python camera/yolo_realtime_demo.py --model yolov8n.pt
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np

# ===== 兼容从项目根目录运行 =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera.camera_manager import CameraManager, CameraError

try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "未安装 ultralytics，请先执行: pip install ultralytics"
    ) from e


def safe_put_text(
    image: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """
    OpenCV 原生 putText 对中文支持一般，这里统一用英文标签更稳。
    """
    cv2.putText(
        image,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_detection(
    image: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    cls_name: str,
    conf: float,
) -> Tuple[int, int]:
    x1, y1, x2, y2 = box_xyxy
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)

    label = f"{cls_name} {conf:.2f}"
    text_y = max(25, y1 - 8)
    safe_put_text(image, label, (x1, text_y), font_scale=0.6, thickness=2)
    safe_put_text(image, f"({cx},{cy})", (x1, min(y2 + 20, image.shape[0] - 10)), font_scale=0.5, thickness=1)

    return cx, cy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path or model name")
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", type=str, default=None, help="cuda:0 / cpu / None")
    parser.add_argument("--classes", type=str, default="", help="只检测指定类别名，逗号分隔，例如: bottle,cup")
    return parser.parse_args()


def main():
    args = parse_args()

    target_class_names = set()
    if args.classes.strip():
        target_class_names = {x.strip() for x in args.classes.split(",") if x.strip()}

    print(f"[INFO] loading model: {args.model}")
    model = YOLO(args.model)

    cam = CameraManager(
        color_width=640,
        color_height=480,
        color_fps=30,
        enable_color=True,
        enable_depth=False,
        align_to_color=False,
    )

    prev_time = time.time()
    fps = 0.0

    try:
        cam.start()
        print("[INFO] camera started")

        intr = cam.get_intrinsics()
        print("[INFO] intrinsics:", intr)

        window_name = "Orbbec RGB + YOLO"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while True:
            frame_bundle = cam.get_frame(timeout_ms=1000)
            frame = frame_bundle.rgb

            if frame is None:
                print("[WARN] empty rgb frame")
                continue

            # YOLO 推理
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )

            vis = frame.copy()

            if len(results) > 0:
                result = results[0]
                names = result.names

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)

                        if target_class_names and cls_name not in target_class_names:
                            continue

                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        x1, y1, x2, y2 = xyxy

                        cx, cy = draw_detection(
                            vis,
                            (x1, y1, x2, y2),
                            cls_name,
                            conf,
                        )

                        # 这里只演示 2D 中心点；后续接 depth 后可转 3D
                        safe_put_text(
                            vis,
                            f"center: ({cx}, {cy})",
                            (x1, min(y2 + 40, vis.shape[0] - 10)),
                            font_scale=0.5,
                            thickness=1,
                        )

            # FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            safe_put_text(vis, f"FPS: {fps:.2f}", (20, 30), font_scale=0.8, thickness=2)
            safe_put_text(vis, "Press q to quit", (20, 60), font_scale=0.6, thickness=2)

            cv2.imshow(window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

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