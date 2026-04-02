# PiperClaw

## YOLO 实时检测演示（Orbbec DaBai RGB）

`camera/yolo_realtime_demo.py` 通过 `CameraManager` 打开 **Orbbec DaBai** 彩色相机，使用 **Ultralytics YOLO** 做实时目标检测，在画面上绘制框、类别、置信度与目标中心点；按 **q** 退出。

### 依赖

- Python 环境（示例中使用 conda 环境 `piperclaw`）
- OpenCV（`cv2`）、NumPy
- [Ultralytics](https://github.com/ultralytics/ultralytics)：`pip install ultralytics`
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)（Orbbec Python SDK），并已正确连接相机与驱动

### 启动方式

先激活 conda 环境，再进入项目根目录，最后运行脚本（将模型路径换成你本机的 `.pt` 文件）：

```bash
conda activate piperclaw
cd ~/PiperClaw
python camera/yolo_realtime_demo.py --model ~/PiperClaw/models/yolov8n.pt
```

若未指定 `--model`，默认使用当前工作目录下的 `yolov8n.pt`（首次运行可由 Ultralytics 自动下载）。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | YOLO 权重路径或模型名 | `yolov8n.pt` |
| `--conf` | 置信度阈值 | `0.35` |
| `--imgsz` | 推理输入边长 | `640` |
| `--device` | 设备，如 `cuda:0`、`cpu`；不传则由 YOLO 自动选择 | 自动 |
| `--classes` | 仅保留指定类别，逗号分隔，如 `bottle,cup` | 全部类别 |

### 说明

- 需从 **PiperClaw 项目根目录** 运行上述命令（脚本会依赖 `camera.camera_manager` 等模块）。
- 相机配置为彩色 **640×480@30**，默认仅彩色流、不启用深度（与 `camera_manager` 设计一致）。

---

## YOLO 抓取准备演示（`yolo_grasp_prep_demo.py`）

打开 Orbbec RGB，YOLO 实时检测并输出目标中心像素坐标 `(u, v)`；自动高亮当前“最佳”检测（置信度优先，其次框面积更大）。按 **s** 将当前最佳目标保存为抓取候选，按 **q** 退出。输出为 **2D 像素坐标**，非机械臂三维坐标（后续可接深度扩展）。

依赖与上文「YOLO 实时检测演示」相同。

### 启动方式

```bash
conda activate piperclaw
cd ~/PiperClaw
python camera/yolo_grasp_prep_demo.py --model ~/PiperClaw/models/yolov8n.pt
```

`--model` 为必填项，需指向本地 YOLO 权重文件。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | 本地 YOLO 模型路径（**必填**） | 无 |
| `--conf` | 置信度阈值 | `0.35` |
| `--imgsz` | 推理尺寸 | `640` |
| `--device` | 如 `cuda:0`、`cpu` | 自动 |
| `--classes` | 只检测指定类别，逗号分隔，如 `cup,bottle` | 全部类别 |

---

## YOLO 抓取准备（变化时刷新，`yolo_grasp_prep_on_change_demo.py`）

在 `yolo_grasp_prep_demo.py` 基础上：**仅当最佳目标相对上一次输出发生有意义变化时**才打印 `[TARGET]`（行格式与 grasp_prep 相同）；类别变化、检测从无到有/从有到无、或中心 `(u,v)` 移动超过阈值视为变化。失去最佳目标时会打印 `[INFO] best target cleared (no detection)`。可选 **`--infer-every N`** 每 N 帧做一次 YOLO，以降低 GPU 占用（未推理帧沿用上次框叠画，快速运动时可能略有偏差）。

依赖与上文相同。

### 启动方式

```bash
conda activate piperclaw
cd ~/PiperClaw
python camera/yolo_grasp_prep_on_change_demo.py --model ~/PiperClaw/models/yolov8n.pt
```

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--model` | 本地 YOLO 模型路径（**必填**） | 无 |
| `--conf` | 置信度阈值 | `0.35` |
| `--imgsz` | 推理尺寸 | `640` |
| `--device` | 如 `cuda:0`、`cpu` | 自动 |
| `--classes` | 只检测指定类别，逗号分隔 | 全部类别 |
| `--pixel-threshold` | 中心点移动超过该像素才再次输出 `[TARGET]` | `10` |
| `--infer-every` | 每 N 帧运行一次 YOLO（≥1）；大于 1 可降低推理频率 | `1` |

交互：**q** 退出，**s** 保存当前最佳目标，`[SAVE]` 行格式与 `yolo_grasp_prep_demo.py` 一致。
