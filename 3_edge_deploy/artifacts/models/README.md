# 交付权重说明

这些文件不进 git（体积大），以 **GitHub Release** 的形式分发：
<https://github.com/<你的账号>/rk3588s-edge-deploy/releases>

## rknn/ —— 板端可直接跑

| 文件 | 模型 | 精度 | 板端实测 | 备注 |
|---|---|---|---|---|
| `mlp_fraud.rknn` | 信用卡欺诈 MLP | INT8 | 一致率 99.956%、AUPRC 降幅 −0.01471、0.213 ms/条 | **交付版**，用 `kl_divergence` + 校准集裁剪 ±16 转出 |
| `mlp_fraud_fp.rknn` | 同上 | fp16 | 与 ONNX 判定一致率 100%、logit 最大差 0.0027 | 量化前的链路验证基线 |
| `pneumonia_fp.rknn` | 肺炎 CNN | fp16 | **精度失效**，见 `../README_RK3588S.md`「已知问题」 | 仅留作问题复现，**不要用于交付** |
| `yolov8n_fp.rknn` | YOLOv8n | fp16 | **检出率 100%**、50.88 ms/帧 | **交付版**（任务卡 13/14 请用这个） |
| `yolo26n_fp.rknn` | YOLO26n | fp16 | 未做板端验证 | 转换链路与 YOLOv8n 同源 |

> **不要用 INT8 版的 YOLOv8n**：分类头输出恒为 0，一个目标都检不出。
> 仓库里没有 INT8 版，因为它是坏的——需要的话按 README 的命令可复现出同样的坏模型。

## onnx/ —— 转换输入与浮点基线

| 文件 | 用途 |
|---|---|
| `mlp_fraud.onnx` | MLP 的 ONNX（静态 batch=1） |
| `mlp_splits.npz` | 训练时已标准化的分层切分数据，板端推理直接喂，**不要重新标准化** |
| `metrics_mlp.json` | 训练侧指标，含**判定阈值 0.77**——板端必须用这个阈值，用默认的 0.5 结果全错 |
| `pneumonia.onnx` | 肺炎 CNN 的 ONNX。**板端用它跑得对**（AUROC 0.9628，与训练侧逐项一致），失效的只是 RKNN |
| `preprocess.json` | CNN 板端预处理口径（mean/std/resize/输入输出名），从 `preprocess.json` 读，不要写死 |
| `test_manifest.txt` | 640 条测试集清单（`路径<TAB>标签`），配合 `--image-root` 使用 |
| `yolov8n.onnx` / `yolo26n.onnx` | YOLO 的 ONNX（转换输入） |

## 许可与出处

- 模型权重由 COCO 预训练的 YOLOv8n / YOLO26n 与 ImageNet 预训练的 MobileNetV2 转换而来，
  遵循各自上游许可证。
- `yolov8n.onnx` / `yolov8n_fp.rknn` 对应的 COCO 权重来自 Ultralytics，
  AGPL-3.0；**若用于闭源商业场景需注意授权**。
- 数据集（creditcard.csv、Chest X-Ray Images、COCO val2017）**不在本仓库**，
  均来自 Kaggle / ultralytics，各自带 CC BY 4.0 或类似条款，请自行按原仓库条款获取。