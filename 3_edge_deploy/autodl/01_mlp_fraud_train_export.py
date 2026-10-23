#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 1 / 2 / 3 / 4 —— 实验一：全连接神经网络 —— 信用卡欺诈检测
在 x86 训练机（AutoDL RTX 4090）上运行：训练 → 指标 → 导出 ONNX → onnxruntime 校验

手册要点（对应验收判据）：
  任务卡 1  丢弃 Time，29 维特征；先分层切 20% 测试，再从剩余切 20% 验证；
            标准化器只在训练集拟合；固定种子；切分落盘可复现
  任务卡 2  29 -> 64 -> 32 -> 1，ReLU，第一层后 Dropout(0.3)，输出单个 logit，
            网络内不加 sigmoid；导出前切推理模式使 Dropout 失效
  任务卡 3  正样本权重 = 负样本数 / 正样本数（按实际切分计算，不写死）；
            用 BCEWithLogitsLoss（带 logits，数值稳定）
  任务卡 4  batch=1、输入 29 维、无动态轴、opset=12、输入名 input / 输出名 logits；
            导出验证集首条样本为 npy 供板端对齐；onnxruntime 与 PyTorch 误差 1e-5 级

用法：
    python 01_mlp_fraud_train_export.py --data data/creditcard.csv
    python 01_mlp_fraud_train_export.py --epochs 15 --device cuda
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
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]  # 29 维，丢弃 Time
TEST_SIZE = 0.20
VAL_SIZE = 0.20  # 相对剩余部分，即占全集 16%
EPOCHS_DEFAULT = 15


# --------------------------------------------------------------------------
# 任务卡 2：全连接网络定义
# --------------------------------------------------------------------------
class FraudMLP(nn.Module):
    """29 -> 64 -> 32 -> 1，约 4033 个参数。"""

    def __init__(self, in_dim: int = 29, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        # 网络内不做 sigmoid：输出是 logit，激活交给损失函数

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(x)  # 训练时生效，eval() 下失效
        x = self.act(self.fc2(x))
        return self.fc3(x)  # (N, 1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 任务卡 1：数据准备
# --------------------------------------------------------------------------
def load_raw(csv_path: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_COLS + ["Class"] if c not in df.columns]
    if missing:
        raise SystemExit(f"creditcard.csv 缺少列: {missing}")
    x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int64)
    return x, y


def make_splits(x, y, seed=SEED):
    """先分层切 20% 测试，再从剩余 80% 里切 20% 作验证。三份正样本占比一致。"""
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=VAL_SIZE, random_state=seed, stratify=y_tr
    )
    return (x_tr, y_tr), (x_va, y_va), (x_te, y_te)


def standardize(splits, out_npz: str):
    """标准化器只在训练集上拟合，再变换验证/测试，避免数据泄漏。"""
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = splits
    scaler = StandardScaler().fit(x_tr)  # 只用训练集拟合
    out = {
        "x_train": scaler.transform(x_tr).astype(np.float32),
        "y_train": y_tr,
        "x_val": scaler.transform(x_va).astype(np.float32),
        "y_val": y_va,
        "x_test": scaler.transform(x_te).astype(np.float32),
        "y_test": y_te,
    }
    np.savez(out_npz, **out)
    return out, scaler


# --------------------------------------------------------------------------
# 任务卡 3：训练与评估
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_logits(model, x, device, batch=8192):
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        chunk = torch.from_numpy(x[i : i + batch]).to(device)
        outs.append(model(chunk).cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(-1)


def pick_threshold(y_true, logits):
    """在验证集上按 F1 选阈值，再拿去报测试集的查全率/查准率。"""
    scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid 只在评估时施加
    best_f1, best_thr = -1.0, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (scores >= thr).astype(int)
        p, r = precision_score(y_true, pred, zero_division=0), recall_score(
            y_true, pred, zero_division=0
        )
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


def evaluate(model, data, device):
    res = {}
    for split in ("val", "test"):
        x, y = data[f"x_{split}"], data[f"y_{split}"]
        logits = predict_logits(model, x, device)
        res[f"{split}_auprc"] = float(average_precision_score(y, logits))
        res[f"{split}_roc_auc"] = float(roc_auc_score(y, logits))
    thr = pick_threshold(data["y_val"], predict_logits(model, data["x_val"], device))
    res["threshold"] = thr
    test_logits = predict_logits(model, data["x_test"], device)
    pred = (test_logits >= thr).astype(int)
    y_test = data["y_test"]
    res["test_precision"] = float(precision_score(y_test, pred, zero_division=0))
    res["test_recall"] = float(recall_score(y_test, pred, zero_division=0))
    # 退化检测：不能全判为负类
    res["test_pred_positive_rate"] = float(pred.mean())
    res["test_positive_rate"] = float(y_test.mean())
    return res


def train(model, data, device, epochs, lr=1e-3):
    (x_tr, y_tr) = data["x_train"], data["y_train"]
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # 带 logits，数值稳定

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt = torch.from_numpy(x_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device).reshape(-1, 1)

    model.train()
    losses = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        out = model(xt)  # 全量 batch，本数据集只有 22 万行，4090 上放得下
        loss = criterion(out, yt)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        print(f"  epoch {ep:2d}/{epochs}  loss={losses[-1]:.5f}")
    return losses, time.time() - t0


# --------------------------------------------------------------------------
# 任务卡 4：导出 ONNX + onnxruntime 校验
# --------------------------------------------------------------------------
def export_onnx(model, device, onnx_path, sample_npy):
    model.eval()  # 使 Dropout 失效，保证两次前向完全一致
    dummy = torch.randn(1, len(FEATURE_COLS), device=device)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,  # 手册统一取 12：RKNN 支持 12~19，ATC 普遍 11~12
        do_constant_folding=True,
        export_params=True,
    )
    # 导出验证集首条样本，供板端精度对齐
    np.save(sample_npy, dummy.detach().cpu().numpy().astype(np.float32))
    print(f"[ok] {onnx_path}  +  {sample_npy}")


def verify_onnx(onnx_path, model, data, device, tol=1e-5):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    oname = sess.get_outputs()[0].name
    print(f"      Netron 口径 -> 输入 '{iname}' 输出 '{oname}'")

    # 导出时固定了 batch=1（RKNN/NPU 推理就是这个口径），校验必须按同样口径喂，
    # 否则 onnxruntime 会直接报 INVALID_ARGUMENT。
    shape = sess.get_inputs()[0].shape
    fixed_batch = shape[0] if shape and isinstance(shape[0], int) else 0
    bs = fixed_batch if fixed_batch > 0 else 256
    print(f"      输入形状 {shape} -> 校验按 batch={bs} 分片喂入")

    x_te = data["x_test"][:2048]
    ort_out = np.concatenate(
        [sess.run([oname], {iname: x_te[i : i + bs]})[0] for i in range(0, len(x_te), bs)],
        axis=0,
    ).reshape(-1)
    torch_out = predict_logits(model, x_te, device)
    max_err = float(np.max(np.abs(ort_out - torch_out)))
    print(f"      onnxruntime vs PyTorch 最大误差 = {max_err:.3e}  (判据 <= {tol:.0e})")
    if max_err > tol:
        raise SystemExit("ONNX 校验未通过，不要继续下一步。")
    return max_err, iname, oname


def check_consistency(model, data, device, tol=0.0):
    """推理模式下两次前向结果完全一致（任务卡 2 验收）。"""
    x = data["x_val"][:1024]
    a = predict_logits(model, x, device)
    b = predict_logits(model, x, device)
    d = float(np.max(np.abs(a - b)))
    print(f"      推理模式两次前向最大差 = {d:.3e}")
    if d > tol:
        raise SystemExit("推理模式下前向结果不一致，Dropout 可能未失效。")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="out/mlp_fraud")
    args = ap.parse_args()

    set_seed()
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device(args.device)
    print(f"=== 实验一 · 信用卡欺诈检测 ===")
    print(f"device={device}  torch={torch.__version__}  seed={SEED}")

    # --- 任务卡 1：数据 ---
    print("\n[任务卡 1] 分层切分与标准化")
    x, y = load_raw(args.data)
    print(f"      全集 {x.shape[0]} 笔，正样本 {int(y.sum())} 笔，占比 {y.mean():.5%}")
    splits = make_splits(x, y)
    data, scaler = standardize(splits, os.path.join(args.outdir, "mlp_splits.npz"))
    for name in ("train", "val", "test"):
        print(
            f"      {name:5s}: {data[f'x_{name}'].shape[0]:>7d} 条  "
            f"正样本占比 {data[f'y_{name}'].mean():.5%}"
        )
    # 重新加载验证：切分结果确实落盘且形状一致
    chk = np.load(os.path.join(args.outdir, "mlp_splits.npz"))
    for k in ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"):
        assert np.array_equal(chk[k], data[k]), f"{k} 落盘后不一致"
    print("      切分结果已落盘并重新加载校验一致")

    # --- 任务卡 2：模型 ---
    print("\n[任务卡 2] 全连接网络定义")
    model = FraudMLP(in_dim=len(FEATURE_COLS)).to(device)
    n_params = count_params(model)
    print(f"      结构 29 -> 64 -> 32 -> 1，Dropout(0.3)，参数量 = {n_params}")
    if not (3500 <= n_params <= 4500):
        print("      [警告] 参数量偏离约 4 千的预期")
    with torch.no_grad():
        print(f"      单样本输出形状 = {tuple(model(torch.randn(1,29,device=device)).shape)}")
    check_consistency(model, data, device)

    # --- 任务卡 3：训练 ---
    print("\n[任务卡 3] 带权重的二分类损失")
    losses, train_sec = train(model, data, device, args.epochs, args.lr)
    metrics = evaluate(model, data, device)
    metrics.update(
        train_seconds=round(train_sec, 2),
        params=n_params,
        epochs=args.epochs,
        final_loss=round(losses[-1], 6),
    )
    print(f"\n      训练耗时 {train_sec:.1f}s")
    print(f"      验证 AUPRC   = {metrics['val_auprc']:.5f}  "
          f"(随机基线 {data['y_train'].mean():.5f})")
    print(f"      验证 ROC-AUC = {metrics['val_roc_auc']:.5f}")
    print(f"      测试 AUPRC   = {metrics['test_auprc']:.5f}")
    print(f"      阈值 {metrics['threshold']:.3f} -> 查准率 {metrics['test_precision']:.4f} "
          f"查全率 {metrics['test_recall']:.4f}")
    print(f"      预测正类占比 {metrics['test_pred_positive_rate']:.5f} vs "
          f"真实 {metrics['test_positive_rate']:.5f}")
    if metrics["val_auprc"] <= data["y_train"].mean() * 2:
        print("      [警告] 验证 AUPRC 未显著高于正样本占比，模型可能退化")

    # --- 任务卡 4：ONNX ---
    print("\n[任务卡 4] 导出 ONNX 并校验")
    onnx_path = os.path.join(args.outdir, "mlp_fraud.onnx")
    sample_npy = os.path.join(args.outdir, "mlp_fraud_sample.npy")
    export_onnx(model, device, onnx_path, sample_npy)
    metrics["onnx_max_err"], iname, oname = verify_onnx(onnx_path, model, data, device)
    metrics["onnx_input_name"] = iname
    metrics["onnx_output_name"] = oname

    # 校准集：任务卡 5 要用，随机抽 100~200 条训练样本（已按训练口径标准化）
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(data["x_train"]), size=200, replace=False)
    calib = data["x_train"][idx]
    calib_dir = os.path.join(args.outdir, "calib_npy")
    os.makedirs(calib_dir, exist_ok=True)
    for i, row in enumerate(calib):
        np.save(os.path.join(calib_dir, f"calib_{i:04d}.npy"), row.astype(np.float32))
    print(f"\n[附带] 校准集 {len(calib)} 条已写入 {calib_dir}")

    torch.save(model.state_dict(), os.path.join(args.outdir, "mlp_fraud.pt"))
    with open(os.path.join(args.outdir, "metrics_mlp.json"), "w") as w:
        json.dump(metrics, w, indent=2, ensure_ascii=False)

    print("\n=== 完成，产物 ===")
    for f in ("mlp_fraud.pt", "mlp_fraud.onnx", "mlp_fraud_sample.npy",
              "mlp_splits.npz", "metrics_mlp.json"):
        print(f"      {args.outdir}/{f}")


if __name__ == "__main__":
    sys.exit(main())