#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二第五节 —— 鲁班猫板端推理（肺炎 CNN 分类）

在 RK3588S 鲁班猫上运行：加载 .rknn，对测试集输出二分类结果。

手册要点（对应验收判据）：
  板端推理流程与实验一相同，区别在于预处理要复现训练时的归一化参数，
    否则精度会明显下降 —— 这是图像部署最常见的坑。
    本脚本从 preprocess.json 读 mean/std，绝不写死。
  输入 1x3x224x224，输出 1x2
  医学筛查漏诊代价高：主指标是肺炎类的查全率，其次准确率与 AUROC
  同时输出混淆矩阵，说明阈值调整对查全率与查准率的取舍
  记录单帧耗时

用法：
    python3 infer_cnn_rknn.py \
        --rknn models/pneumonia.rknn \
        --preprocess models/preprocess.json \
        --manifest models/test_manifest.txt \
        --image-root /home/ubuntu/chest_xray
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def load_rknn_runtime(rknn_path, verbose=False):
    from rknnlite.api import RKNNLite

    rknn = RKNNLite(verbose=verbose)
    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise SystemExit(f"load_rknn 失败，返回码 {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise SystemExit(
            f"init_runtime 失败，返回码 {ret}。\n"
            f"先跑 00_board_env_check.py 看 NPU 节点诊断。"
        )
    return rknn


def preprocess(path, cfg):
    """复现训练时的预处理；mean/std 一律取自 preprocess.json，不在此处写死。

    缩放必须走 PIL，不能用 cv2.resize：训练侧是 torchvision 的
    transforms.Resize，底层是 PIL 的 resize，降采样自带抗锯齿；
    cv2.resize 的 INTER_LINEAR 降采样**不抗锯齿**，把 1800x1300 直接压到
    224 会混叠出严重的假高频。实测同样这 640 张图：
        cv2  预处理 -> 板端 ONNX  acc=0.7672  AUROC=0.9588  tn=94
        PIL  预处理 -> 板端 ONNX  acc=0.8500  AUROC=0.9628  tn=147
    后者与训练侧逐项完全一致（tp397 fp95 fn1 tn147），差的这几十个假阳性
    全部来自缩放混叠。
    """
    from PIL import Image

    h, w = cfg["resize"]
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    mean = np.array(cfg["mean"], dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)
    x = (x - mean) / std                                 # 归一化，必须与训练一致
    x = np.transpose(x, (2, 0, 1))[None]                 # HWC -> NCHW
    return np.ascontiguousarray(x.astype(np.float32))


def rknn_infer(rknn, paths, cfg, warmup=5, progress=True):
    outs = np.zeros((len(paths), 2), dtype=np.float32)
    lat = np.zeros(len(paths), dtype=np.float64)

    for i in range(min(warmup, len(paths))):
        rknn.inference(inputs=[preprocess(paths[i], cfg)])

    for i, p in enumerate(paths):
        x = preprocess(p, cfg)
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[x])
        lat[i] = (time.perf_counter() - t0) * 1000.0
        arr = np.asarray(out).reshape(-1)[:2]
        outs[i] = arr
        if progress and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(paths)}  累计 {lat[:i+1].mean():.2f} ms/张", flush=True)
    return outs, lat


DATA_SPLITS = ("train", "test", "val")


def rebase_to_image_root(p, image_root):
    """清单里存的是训练机上的绝对路径，拷到板子上一个都不存在。

    从路径里认出 train/test/val 这一层，把它及其之后的相对部分接到
    --image-root 下面。这样同一份 manifest 在训练机和板子上都能用，
    不用在板子上重建数据集目录结构去迁就路径前缀。
    认不出来就原样返回，交给后面的存在性检查报错。
    """
    parts = p.replace("\\", "/").split("/")
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in DATA_SPLITS:
            return os.path.join(image_root, *parts[i:])
    return p


def read_manifest(manifest, image_root=None):
    """清单每行 `路径<TAB>标签`；路径按原样或相对 image_root 拼接。"""
    paths, labels = [], []
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.rsplit(None, 1)
            p, y = parts[0], int(parts[1])
            if image_root:
                if not os.path.isabs(p):
                    joined = os.path.join(image_root, p)
                    # 清单里的前缀是训练机上的相对布局（data/chest_xray/test/...），
                    # 板上的根目录未必一样，拼出来不存在就再按 train/test/val 重映射
                    p = joined if os.path.exists(joined) else rebase_to_image_root(p, image_root)
                elif not os.path.exists(p):
                    p = rebase_to_image_root(p, image_root)
            paths.append(p)
            labels.append(y)
    return paths, np.array(labels, dtype=np.int64)


def metrics_from(probs, labels, classes):
    pred = probs.argmax(1)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    res = {
        "acc": (tp + tn) / max(len(labels), 1),
        "precision": tp / max(tp + fp, 1),
        "recall_pneumonia": tp / max(tp + fn, 1),   # 主指标
        "recall_normal": tn / max(tn + fp, 1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    try:
        from sklearn.metrics import roc_auc_score

        res["auroc"] = float(roc_auc_score(labels, probs[:, 1]))
    except Exception:
        res["auroc"] = float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default="models/pneumonia.rknn")
    ap.add_argument("--preprocess", default="models/preprocess.json")
    ap.add_argument("--manifest", default="models/test_manifest.txt")
    ap.add_argument("--image-root", default=None, help="清单里是相对路径时用它补全")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", default="out/board_cnn_report.json")
    args = ap.parse_args()

    print("=== 实验二 · 鲁班猫板端推理（肺炎 CNN）===")
    for f in (args.rknn, args.preprocess, args.manifest):
        if not os.path.exists(f):
            raise SystemExit(f"找不到 {f}")

    with open(args.preprocess) as f:
        cfg = json.load(f)
    print(f"  预处理口径（来自 preprocess.json，务必与训练一致）:")
    print(f"    输入名 {cfg['input_name']}  形状 {cfg['input_shape']}  布局 {cfg['layout']}")
    print(f"    缩放   {cfg['resize']}")
    print(f"    均值   {cfg['mean']}")
    print(f"    标准差 {cfg['std']}")
    print(f"    类别   {cfg['classes']}")

    paths, labels = read_manifest(args.manifest, args.image_root)
    if args.limit:
        paths, labels = paths[: args.limit], labels[: args.limit]
    missing = [p for p in paths[:20] if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"清单里的图片找不到，例如 {missing[0]}\n"
                         f"请把测试集图片拷到板端后用 --image-root 指定根目录")
    print(f"\n  测试集 {len(paths)} 张  正常 {int((labels==0).sum())}  "
          f"肺炎 {int((labels==1).sum())}")

    rknn = load_rknn_runtime(args.rknn)
    try:
        print(f"\n[推理] NPU 逐张推理 ...")
        logits, lat = rknn_infer(rknn, paths, cfg)
    finally:
        rknn.release()
        print("  rknn.release() 已释放")

    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    pred = probs.argmax(1)

    print(f"\n[耗时] 均值 {lat.mean():.2f} ms/张  中位 {np.median(lat):.2f} ms  "
          f"P99 {np.percentile(lat, 99):.2f} ms")
    print(f"[前 8 张]")
    for i in range(min(8, len(paths))):
        print(f"  {os.path.basename(paths[i]):<28s} 真实={cfg['classes'][labels[i]]:<9s} "
              f"预测={cfg['classes'][pred[i]]:<9s} "
              f"概率={probs[i,1]:.4f}")

    m = metrics_from(probs, labels, cfg["classes"])
    print(f"\n  准确率        = {m['acc']:.4f}")
    print(f"  肺炎查全率 ★  = {m['recall_pneumonia']:.4f}   （主指标，漏诊代价高）")
    print(f"  肺炎查准率    = {m['precision']:.4f}")
    print(f"  正常查全率    = {m['recall_normal']:.4f}")
    print(f"  AUROC         = {m['auroc']:.4f}")
    c = m["confusion"]
    print(f"\n  混淆矩阵（行=真实，列=预测）  类别: {cfg['classes']}")
    print(f"              预测正常  预测肺炎")
    print(f"    真实正常   {c['tn']:>6d}   {c['fp']:>6d}")
    print(f"    真实肺炎   {c['fn']:>6d}   {c['tp']:>6d}")
    print(f"  漏诊 {c['fn']} 例 —— 医学筛查里这一项最该压低")
    print(f"  阈值下调可换查全率、代价是查准率下降；本脚本默认取 argmax(阈值 0.5)")

    report = {
        "n_test": len(paths),
        "latency_ms_mean": round(float(lat.mean()), 3),
        "latency_ms_median": round(float(np.median(lat)), 3),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
        "preprocess": {k: cfg[k] for k in ("input_shape", "mean", "std", "resize", "layout")},
        **{k: v for k, v in m.items() if k != "confusion"},
        "confusion": m["confusion"],
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)
    print(f"\n[完成] 报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())