# -*- coding: utf-8 -*-
"""
文档2-①：Fashion-MNIST 分类推理（板上）
─────────────────────────────────────────────────────────────────

数据集：Fashion-MNIST（10 类服装图像，60k 训练 + 10k 测试）
预训练模型：HuggingFace 上 sayakpaul/fashion-mnist-resnet 等已微调模型

板上推理策略：
  * 自动从 HuggingFace 下载数据集 + 预训练权重
  * 推理 10 张测试图，打印 top-3 类别与置信度
  * 保存带预测标签的网格图到 ./output.png

依赖（阿里源）：
  pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      transformers torch torchvision datasets pillow

运行：
  python3 fashion_mnist_infer.py
"""

import os, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# HuggingFace 阿里云镜像加速（可选）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("=" * 70)
print("Fashion-MNIST 分类推理（板上）")
print("=" * 70)

try:
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModelForImageClassification
except ImportError:
    print("[ERR] 缺包，请先：")
    print("      pip install -i https://mirrors.aliyun.com/pypi/simple/ \\")
    print("          transformers datasets pillow torchvision")
    sys.exit(1)

# 1) 加载 F-MNIST（HF 上的 zalando-datasets/fashion_mnist）
print("\n[1/3] 下载 Fashion-MNIST 测试集 ...", flush=True)
t0 = time.time()
ds = load_dataset("zalando-datasets/fashion_mnist", split="test[:64]")
print(f"      加载 {len(ds)} 条测试样本  (用时 {time.time()-t0:.1f}s)")

# 2) 加载预训练 F-MNIST ResNet
MODEL_NAME = os.environ.get("FM_MODEL", "sayakpaul/resnet50-fashion-mnist")
print(f"\n[2/3] 下载预训练模型 {MODEL_NAME} ...", flush=True)
t0 = time.time()
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"[WARN] {MODEL_NAME} 不可用 ({e.__class__.__name__})")
    print("       退到通用 ResNet18 + ImageNet 预训练（10 类输出不匹配，仅展示流程）")
    MODEL_NAME = "microsoft/resnet-18"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=10, ignore_mismatched_sizes=True
    )
print(f"      加载完成  (用时 {time.time()-t0:.1f}s)")
print(f"      类别标签: {model.config.id2label}")

model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"      参数量: {n_params:,} ({n_params/1e6:.2f}M)")

# 3) 推理
print("\n[3/3] 推理前 64 条 ...", flush=True)
t0 = time.time()
labels_all, preds_all = [], []
with torch.no_grad():
    for i in range(len(ds)):
        img = ds[i]["image"].convert("RGB")
        enc = processor(img, return_tensors="pt")
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
        top3 = torch.topk(probs, 3)
        idx = ds[i]["label"]
        gt = model.config.id2label.get(idx, str(idx))
        pred = model.config.id2label[top3.indices[0].item()]
        labels_all.append(gt); preds_all.append(pred)
        if i < 5 or (gt != pred):
            print(f"  [{i:3d}] 真值={gt:12s}  预测={pred:12s}  "
                  f"top3={[model.config.id2label[j.item()] for j in top3.indices]}  "
                  f"conf={top3.values[0].item():.3f}")
print(f"      推理完成  (用时 {time.time()-t0:.1f}s, 平均 {(time.time()-t0)/len(ds)*1000:.1f} ms/张)")

# 评估
correct = sum(1 for a, b in zip(labels_all, preds_all) if a == b)
print(f"\n[result] {correct}/{len(labels_all)} = {correct/len(labels_all):.2%}")

# 4) 保存标注网格图（可选，需 matplotlib）
try:
    import matplotlib.pyplot as plt
    n_show = min(16, len(ds))
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for k, ax in enumerate(axes.flat):
        if k >= n_show: break
        img = ds[k]["image"].convert("RGB")
        ax.imshow(img); ax.axis("off")
        ok = "✓" if labels_all[k] == preds_all[k] else "✗"
        ax.set_title(f"{ok} {preds_all[k]}", color=("green" if ok == "✓" else "red"), fontsize=9)
    plt.suptitle("Fashion-MNIST Predictions")
    plt.tight_layout()
    out_png = Path("./output.png").resolve()
    plt.savefig(out_png, dpi=80)
    print(f"[save ] {out_png}")
except Exception as e:
    print(f"[skip] matplotlib 出图失败: {e}")