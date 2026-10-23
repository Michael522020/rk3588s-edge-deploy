# RK3588S 边缘部署实验 · 代码与实测记录

《深度学习模型训练与边缘端部署实验手册》配套代码，覆盖 RK3588S（鲁班猫 LubanCat-4）
的 **PyTorch 训练 → ONNX 导出 → RKNN 量化 → NPU 板端推理** 全链路，
外加几个板上推理 demo。

**所有性能数字都是在这块板子上实测的**，不是从训练机抄的。
原始日志与 report JSON 一并放在 `3_edge_deploy/artifacts/`，
逐条可复现。

---

## 快速导航

| 想看什么 | 去哪 |
|---|---|
| 第五章性能基准表（实测延迟/帧率/精度） | [`3_edge_deploy/性能基准表.md`](3_edge_deploy/性能基准表.md) |
| RKNN 那条线的完整交付说明、坑、调参结论 | [`3_edge_deploy/README_RK3588S.md`](3_edge_deploy/README_RK3588S.md) |
| 交付权重（走 Release 下载） | [`3_edge_deploy/artifacts/models/README.md`](3_edge_deploy/artifacts/models/README.md) |

## 目录结构

```
1_cnn_vs_structbert/      TextCNN + StructBERT 文本分类推理（板上）
2_open_task/              Fashion-MNIST 图像分类 / 中文情感分类（板上）
3_edge_deploy/
├── autodl/               训练机（x86）侧：训练、导出 ONNX、转 RKNN
│   ├── 01_mlp_fraud_train_export.py      任务卡 1/2/3/4  信用卡 MLP
│   ├── 02_cnn_pneumonia_train_export.py  任务卡 9/10/11 肺炎 CNN (MobileNetV2)
│   ├── 02b_make_cnn_calib.py             仅补校准集/preprocess/清单（纯 CPU，不用重训）
│   ├── 03_yolov8n_export_onnx.py         任务卡 12     YOLOv8n 导 ONNX
│   ├── 04_convert_rknn.py                任务卡 5 + 实验二 ONNX→RKNN
│   ├── 05_convert_rknn_yolov8.py         任务卡 13     YOLOv8n→RKNN
│   ├── 06_yolo26_export_onnx.py          YOLO26n 变体
│   └── *.sh                              环境准备与流水线
├── board/                板端（aarch64）：NPU 推理
│   ├── 00_board_env_check.py             NPU/驱动/版本自检
│   ├── infer_mlp_rknn.py                 任务卡 6
│   ├── infer_cnn_rknn.py                 实验二 §五
│   ├── cam_person_detect.py              任务卡 14
│   └── cpu_load_harness.py               CPU 占用压测
├── artifacts/            实测日志、report JSON、交付权重（权重走 Release）
├── README_RK3588S.md     RKNN 链路完整说明
└── 性能基准表.md           第五章
4_minimind/               MiniMind 从预训练到 DPO、权重转换的脚本
win11_camera_stream.py    笔记本摄像头推 MJPEG 流，给无摄像头的板子当输入
```

---

## 当前状态（2026-10-23）

| 任务卡 | 内容 | 状态 |
|---|---|---|
| 任务卡 1/2/3/4 | 信用卡 MLP 训练 → 导出 ONNX → 校验 | ✅ ONNX 最大误差 1.19e-07 |
| 任务卡 5 | MLP → INT8 RKNN | ✅ |
| **任务卡 6** | **鲁班猫 MLP 推理 + 精度对比** | ✅ 一致率 99.956%、AUPRC 降幅 −0.01471、0.213 ms/条 |
| 任务卡 9/10/11 | 肺炎 CNN 训练 → ONNX | ✅ val_acc 0.9904、AUROC 0.9997 |
| 实验二 §五 | 肺炎 CNN → RKNN + 板端推理 | ⚠️ ONNX 正常（板端 AUROC 0.9628），**RKNN 模型失效** |
| **任务卡 12** | YOLOv8n 导 ONNX | ✅ 一致性校验 5/5 框匹配 |
| **任务卡 13** | YOLOv8n → RKNN | ✅ fp16（检出率 100%）／❌ INT8（类别头恒 0） |
| **任务卡 14** | 实时人员检测 | ✅ 用 fp16 版 |
| 第五章 | 性能基准表 | ✅ |

**已知未解决**：rknn-toolkit2 2.3.2 转出来的肺炎 CNN 与 YOLOv8n INT8 模型是坏的
（分类头输出恒为 0 / 全判同一类）。已逐项排除量化算法、校准图数量、`optimization_level`、
浮点精度、归一化烘焙、板端预处理等因素，同一转换器转 MLP 却完全正常。
排查过程与下一步方向见 [`README_RK3588S.md` 的「已知问题」](3_edge_deploy/README_RK3588S.md)。

---

## 环境

### 板端（LubanCat-4 / RK3588S）

| 项 | 值 |
|---|---|
| 系统 | Ubuntu 24.04.1 LTS，kernel `6.1.0-1025-rockchip` |
| 硬件 | RK3588S，8 核 aarch64，4 GB RAM |
| NPU | rknpu 驱动 0.9.7，设备节点 `/dev/dri/renderD129` |
| 运行时 | rknn-toolkit-lite2 **2.3.2** |
| 其他 | OpenCV 5.0.0、onnxruntime 1.31.0、torch 2.6.0 (CPU) |

```bash
# pip 走阿里云镜像（板子在国内，装包不挂）
mkdir -p ~/.pip && cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.aliyun.com/pypi/simple/
trusted-host = mirrors.aliyun.com
EOF

pip install --user --break-system-packages rknn-toolkit-lite2==2.3.2 scikit-learn
```

> ⚠️ 板上**没有** `/dev/rknpu`。RK3588 驱动走 DRM 路径，
> 判断 NPU 可用性要看 `/dev/dri/renderD129`，不是看 `/dev/rknpu` 存不存在。

### 训练机（x86 Linux / AutoDL）

```bash
python -c "import platform; print(platform.machine())"   # 必须 x86_64
pip install torch torchvision ultralytics onnx onnxruntime onnxsim \
            scikit-learn pandas rknn-toolkit2==2.3.2
```

> **版本纪律**：转换机 `rknn-toolkit2` 与板端 `rknn-toolkit-lite2` 必须同版本。
> AutoDL 无卡模式只给 **2 GiB 内存 / 0.5 核 CPU**，转视觉模型会 OOM（退出码 137），
> 降级组合见 [`README_RK3588S.md`](3_edge_deploy/README_RK3588S.md#量化转换怎么调实测结论)。

---

## 板子上没有摄像头怎么办

`cam_person_detect.py --source` 支持三种输入，本地文件即可验证整条检测链路：

```bash
# 本地图片（最省事）
python3 ~/edge/cam_person_detect.py --rknn models/yolov8n_fp.rknn \
        --source models/bus.jpg --frames 1 --headless --save-shot out/shot.jpg

# 笔记本推 MJPEG 流，板子拉流
python win11_camera_stream.py                    # Win11 上跑，默认 :8080
python3 ~/edge/cam_person_detect.py --rknn models/yolov8n_fp.rknn \
        --source http://<笔记本IP>:8080/stream
```

---

## 数据集

均**不随本仓库分发**，请按原仓库条款自行获取：

| 数据集 | 来源 | 许可 |
|---|---|---|
| Credit Card Fraud Detection | Kaggle `mlg-ulb/creditcardfraud` | CC BY 4.0 |
| Chest X-Ray Images (Pneumonia) | Kaggle `paultimothymooney/chest-xray-pneumonia` | CC BY 4.0 |
| COCO val2017 | `ultralytics` / `cocodataset.org` | 各方自有条款 |
| YOLOv8n / YOLO26n COCO 权重 | Ultralytics 自动下载 | AGPL-3.0 |

> **授权提醒**：YOLOv8n 的 COCO 权重为 Ultralytics 发布，
> 仓库里的 `.rknn` 由其转换而来，**闭源商业用途需自行确认授权**。

## 第三方组件

`rknn-toolkit2`（Rockchip）、`ultralytics`、`torchvision`、
`transformers` / `MiniMind`。本仓库只包含实验脚本与实测记录。