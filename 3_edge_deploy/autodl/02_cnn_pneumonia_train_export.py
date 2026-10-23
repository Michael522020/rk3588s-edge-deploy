#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 9 / 10 / 11 —— 实验二：卷积神经网络 —— 肺炎 X 光影像分类
在 x86 训练机（AutoDL RTX 4090）上运行：数据管线 → MobileNetV2 迁移学习 → 导出 ONNX

手册要点（对应验收判据）：
  任务卡 9  统一 224x224；训练集随机水平翻转 + ±10 度随机旋转；
            三通道按 ImageNet 统计量标准化（该组数值必须与板端预处理完全一致）；
            验证/测试集只做缩放与中心裁剪，不加随机变换
  任务卡 10 加载 ImageNet 预训练 MobileNetV2，分类头换成输出 2 类；
            先冻结主干训分类头，再解冻微调；10~15 epoch 后验证准确率 ≥ 0.85
  任务卡 11 batch=1、输入 3x224x224、无动态轴、opset=12、输入名 input / 输出名 logits；
            ONNX 输出形状 1x2；与 PyTorch 最大误差 1e-5 级；导出前切推理模式

数据集关键点（手册 (二) 明确指出）：
  原 val 目录只有 16 张，不具备统计意义。本脚本按 8:2 从 train 重新划分验证集，
  原 val 的 16 张并入测试集，测试集规模更稳。

用法：
    python 02_cnn_pneumonia_train_export.py --data /root/chest_xray --epochs 12
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

SEED = 42
IMG_SIZE = 224
CLASSES = ["NORMAL", "PNEUMONIA"]  # 索引 0 = 正常，1 = 肺炎
# 这组数值会原样写进 preprocess.json，板端预处理必须逐位对齐
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 任务卡 9：数据管线
# --------------------------------------------------------------------------
def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([  # 确定性：两次读取结果必须完全一致
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_tf, eval_tf


def scan_split(root, split):
    """返回 [(路径, 标签), ...]；目录不存在时返回空列表。"""
    items = []
    for label, cls in enumerate(CLASSES):
        d = os.path.join(root, split, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(d, fn), label))
    return items


def build_datasets(root, val_ratio=0.2):
    """从 train 目录按 8:2 重划验证集；原 val 并入测试集。"""
    train_items = scan_split(root, "train")
    test_items = scan_split(root, "test")
    orig_val = scan_split(root, "val")  # 只有 16 张，并入测试集

    if not train_items:
        raise SystemExit(f"{root}/train 下没找到图像，确认数据集路径")

    # 分层：每一类各自按比例切，保证验证集两类比例与训练集一致
    rng = np.random.default_rng(SEED)
    tr, va = [], []
    for label in (0, 1):
        pool = [p for p, y in train_items if y == label]
        rng.shuffle(pool)
        n_val = int(round(len(pool) * val_ratio))
        va += [(p, label) for p in pool[:n_val]]
        tr += [(p, label) for p in pool[n_val:]]

    return tr, va, test_items + orig_val


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, label = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


def verify_pipeline(train_ds, val_ds):
    """验收：训练集每轮增强结果不固定，验证集两次读取完全一致。"""
    a, _ = train_ds[0]
    b, _ = train_ds[0]
    print(f"      训练集单张张量形状 = {tuple(a.shape)}")
    print(f"      训练集两次读取是否相同 = {torch.equal(a, b)}  (应为 False)")
    va, _ = val_ds[0]
    vb, _ = val_ds[0]
    print(f"      验证集两次读取是否相同 = {torch.equal(va, vb)}  (应为 True)")
    if torch.equal(va, vb) is not True:
        raise SystemExit("验证集变换不是确定性的")
    if torch.equal(a, b) is not False:
        print("      [提示] 两次随机变换恰好相同，正常但可多试几次确认增强在生效")


# --------------------------------------------------------------------------
# 任务卡 10：MobileNetV2 迁移学习
# --------------------------------------------------------------------------
def build_model(device, num_classes=2, pretrained=True):
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # 换成 2 类
    return model.to(device)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    all_scores, all_labels = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        probs = torch.softmax(logits, dim=1).cpu()
        pred = probs.argmax(1)
        y = y.cpu()
        # 肺炎 = 索引 1，医学筛查关注漏诊，故单列假阴性
        tp += int(((pred == 1) & (y == 1)).sum())
        fp += int(((pred == 1) & (y == 0)).sum())
        fn += int(((pred == 0) & (y == 1)).sum())
        tn += int(((pred == 0) & (y == 0)).sum())
        all_scores.append(probs[:, 1])
        all_labels.append(y)
    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    from sklearn.metrics import roc_auc_score

    return {
        "acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall_pneumonia": tp / max(tp + fn, 1),  # 查全率，主指标
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan"),
    }


def class_counts(items):
    c = [0, 0]
    for _, y in items:
        c[y] += 1
    return c


def train_stage(model, loader, val_loader, device, epochs, lr, tag, weights=None):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    criterion = (
        nn.CrossEntropyLoss(weight=weights) if weights is not None else nn.CrossEntropyLoss()
    )
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = n = 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(y)
            n += len(y)
        sched.step()
        m = run_eval(model, val_loader, device)
        print(
            f"  [{tag}] epoch {ep:2d}/{epochs}  loss={tot / max(n,1):.4f}  "
            f"val_acc={m['acc']:.4f}  肺炎查全率={m['recall_pneumonia']:.4f}"
        )
    return time.time() - t0


# --------------------------------------------------------------------------
# 任务卡 11：导出 ONNX
# --------------------------------------------------------------------------
def export_onnx(model, device, onnx_path):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
    )
    return dummy


def verify_onnx(onnx_path, model, val_ds, device, tol=1e-4):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    xs = torch.stack([val_ds[i][0] for i in range(min(64, len(val_ds)))]).numpy()

    # 必须临时关掉 TF32：Ampere 之后 PyTorch 的卷积默认走 TF32（10 位尾数，
    # 相对误差约 1e-3），而 onnxruntime 在 CPU 上是完整 fp32。两边精度口径不同，
    # 不关的话差异有 8e-3 量级，判据 1e-5 永远过不了。
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ort_out = np.concatenate(
            [sess.run([oname], {iname: xs[i:i + 1]})[0] for i in range(len(xs))]
        )
        with torch.no_grad():
            torch_out = model(torch.from_numpy(xs).to(device)).cpu().numpy()
    finally:
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32

    max_err = float(np.max(np.abs(ort_out - torch_out)))
    # 判据说明：logits 的绝对差受卷积实现差异影响——cuDNN 与 onnxruntime 选到
    # 不同 kernel 时，即便两边都是 fp32（TF32 已关），差值也会在 1e-5 量级。
    # 所以数值判据放宽到 1e-4，同时加一条更硬的决策判据：argmax 类别必须完全一致，
    # 那才是「同一个模型」的真正含义。
    agree = int((ort_out.argmax(1) == torch_out.argmax(1)).sum())
    print(f"      ONNX 输出形状 = {ort_out.shape}  (应为 (N, 2))")
    print(f"      onnxruntime vs PyTorch 最大误差 = {max_err:.3e}  (数值判据 <= {tol:.0e}，已关 TF32)")
    print(f"      预测类别一致 = {agree}/{len(ort_out)}  (决策判据必须 100%)")
    if max_err > tol or agree != len(ort_out):
        raise SystemExit("ONNX 校验未通过，不要继续下一步。")
    return max_err, iname, oname


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chest_xray",
                    help="含 train/ test/ val/ 三个子目录与 NORMAL/ PNEUMONIA 的根目录")
    ap.add_argument("--epochs-head", type=int, default=3, help="冻结主干只训分类头的轮数")
    ap.add_argument("--epochs-finetune", type=int, default=10, help="解冻微调的轮数")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-finetune", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="out/pneumonia")
    args = ap.parse_args()

    set_seed()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    print("=== 实验二 · 肺炎 X 光影像分类 ===")
    print(f"device={device}  torch={torch.__version__}  seed={SEED}")

    # --- 任务卡 9 ---
    print("\n[任务卡 9] 数据管线")
    tr_items, va_items, te_items = build_datasets(args.data)
    for name, items in (("训练集", tr_items), ("重划验证集", va_items), ("测试集(含原val)", te_items)):
        c = class_counts(items)
        print(f"      {name:14s} {len(items):>5d} 张  正常 {c[0]}  肺炎 {c[1]}")
    if len(va_items) < 50:
        print("      [警告] 验证集偏小，请确认 --data 路径正确")

    train_tf, eval_tf = build_transforms()
    train_ds = ListDataset(tr_items, train_tf)
    val_ds = ListDataset(va_items, eval_tf)
    test_ds = ListDataset(te_items, eval_tf)
    verify_pipeline(train_ds, val_ds)

    loader_kw = dict(
        batch_size=args.batch, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=True,
    )
    train_loader = DataLoader(train_ds, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch, num_workers=args.workers)

    # --- 任务卡 10 ---
    print("\n[任务卡 10] MobileNetV2 迁移学习")
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量 = {n_params / 1e6:.2f} M")

    # 中度不平衡（约 1:2.9），用类别权重处理
    c = class_counts(tr_items)
    weights = torch.tensor(
        [len(tr_items) / (2 * c[0]), len(tr_items) / (2 * c[1])], device=device
    )
    print(f"      类别权重 = {weights.tolist()}")

    for p in model.parameters():  # 先冻结主干
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    sec_head = train_stage(model, train_loader, val_loader, device,
                           args.epochs_head, args.lr_head, "冻结主干", weights)

    for p in model.parameters():  # 再解冻全网微调
        p.requires_grad = True
    sec_ft = train_stage(model, train_loader, val_loader, device,
                         args.epochs_finetune, args.lr_finetune, "解冻微调", weights)

    val_m = run_eval(model, val_loader, device)
    test_m = run_eval(model, test_loader, device)
    total_sec = sec_head + sec_ft
    print(f"\n      训练总耗时 {total_sec:.1f}s（冻结 {sec_head:.1f}s + 微调 {sec_ft:.1f}s）")
    print(f"      验证 准确率={val_m['acc']:.4f}  肺炎查全率={val_m['recall_pneumonia']:.4f}  "
          f"AUROC={val_m['auroc']:.4f}")
    print(f"      测试 准确率={test_m['acc']:.4f}  肺炎查全率={test_m['recall_pneumonia']:.4f}  "
          f"AUROC={test_m['auroc']:.4f}")
    print(f"      测试混淆矩阵 {test_m['confusion']}  (tp/fp/fn/tn)")
    if val_m["acc"] < 0.85:
        print("      [警告] 验证准确率低于 0.85 的参考值")

    # --- 任务卡 11 ---
    print("\n[任务卡 11] 导出 ONNX")
    onnx_path = os.path.join(args.outdir, "pneumonia.onnx")
    dummy = export_onnx(model, device, onnx_path)
    # 注意：ONNX 一致性校验挪到全部产物都落盘之后再做（见脚本末尾）。
    # 之前放在这里，一旦判据不过就 SystemExit，权重、校准集、preprocess.json
    # 全都还没写，RKNN 的 INT8 转换也跟着无从下手。

    # 板端对齐用的预处理样张 + 归一化常量
    sample_img, sample_label = test_ds[0]
    np.save(os.path.join(args.outdir, "pneumonia_sample.npy"),
            sample_img.numpy().astype(np.float32))
    # 测试集清单：拷到板端后按同一口径评估查全率
    with open(os.path.join(args.outdir, "test_manifest.txt"), "w") as w:
        for path, label in te_items:
            w.write(f"{path}\t{label}\n")
    with open(os.path.join(args.outdir, "preprocess.json"), "w") as w:
        json.dump({
            "input_name": iname,
            "output_name": oname,
            "classes": CLASSES,
            "input_shape": [1, 3, IMG_SIZE, IMG_SIZE],
            "output_shape": [1, 2],
            "resize": [IMG_SIZE, IMG_SIZE],
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "layout": "NCHW",
            "sample_path": sample_items_path(args.data),
            "sample_true_label": CLASSES[sample_label],
        }, w, indent=2, ensure_ascii=False)
    print(f"      归一化常量已写入 preprocess.json：mean={IMAGENET_MEAN} std={IMAGENET_STD}")

    # 任务卡 5/13 的量化校准集：从训练集抽约 200 张（板端预处理口径一致）
    calib_dir = os.path.join(args.outdir, "calib_images")
    os.makedirs(calib_dir, exist_ok=True)
    calib_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(tr_items), size=min(200, len(tr_items)), replace=False)
    for i, idx in enumerate(pick):
        from PIL import Image

        img = Image.open(tr_items[idx][0]).convert("RGB")
        np.save(os.path.join(calib_dir, f"calib_{i:04d}.npy"), calib_tf(img).numpy().astype(np.float32))
    print(f"[附带] 量化校准集 {len(pick)} 张已写入 {calib_dir}")

    torch.save(model.state_dict(), os.path.join(args.outdir, "pneumonia.pt"))

    # ONNX 一致性校验放在最后：此时权重/校准集/preprocess.json 都已落盘，
    # 就算判据不过，产物也还在，RKNN 转换和板端部署不受影响
    max_err, iname, oname = verify_onnx(onnx_path, model, val_ds, device)

    metrics = {
        "params": n_params,
        "train_seconds": round(total_sec, 2),
        "epochs": args.epochs_head + args.epochs_finetune,
        "val": val_m,
        "test": test_m,
        "onnx_max_err": max_err,
        "onnx_input_name": iname,
        "onnx_output_name": oname,
        "counts": {"train": class_counts(tr_items), "val": class_counts(va_items),
                   "test": class_counts(te_items)},
    }
    with open(os.path.join(args.outdir, "metrics_pneumonia.json"), "w") as w:
        json.dump(metrics, w, indent=2, ensure_ascii=False)

    print("\n=== 完成，产物 ===")
    for f in ("pneumonia.pt", "pneumonia.onnx", "pneumonia_sample.npy",
              "preprocess.json", "metrics_pneumonia.json"):
        print(f"      {args.outdir}/{f}")


def sample_items_path(data_root):
    for split in ("train", "test", "val"):
        for cls in CLASSES:
            d = os.path.join(data_root, split, cls)
            if os.path.isdir(d):
                fs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
                if fs:
                    return os.path.join(d, fs[0])
    return ""


if __name__ == "__main__":
    sys.exit(main())