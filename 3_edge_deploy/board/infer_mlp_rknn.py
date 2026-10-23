#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 6 —— 鲁班猫板端推理（实验一 · 信用卡欺诈检测）

在 RK3588S 鲁班猫上运行：加载 .rknn，对测试样本输出欺诈概率，
并与浮点 ONNX（板端 CPU 跑 onnxruntime）逐条对比。

手册要点（对应验收判据）：
  用 rknn-toolkit-lite2 的 Python 接口加载模型并初始化运行时
  输入必须是 float32、形状 1x29，且与训练时的标准化口径完全一致
    —— mlp_splits.npz 里的数据在训练机侧就已经标准化过了，直接用即可
  输出为单个 logit 值，需再经 sigmoid 得到概率
  记录单条推理耗时
  验收：能打印预测概率；量化模型与浮点 ONNX 的判定一致率不低于 99%；
        AUPRC 下降不超过 0.03（参考值）

用法：
    python3 infer_mlp_rknn.py \
        --rknn models/mlp_fraud.rknn \
        --onnx models/mlp_fraud.onnx \
        --splits models/mlp_splits.npz \
        --threshold 0.5
"""

import argparse
import json
import os
import sys
import time

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 板子只有 4 GB 内存，scipy+sklearn 装进来要 ~150 MB，而且 pip 装 aarch64 轮子
# 常年龟速。有 sklearn 就用 sklearn，没有就用下面这份 numpy 实现兜底——
# 算法与 sklearn 的 _binary_uninterpolated_average_precision /
# _binary_roc_aucscore 逐行对齐（同样按分数降序、同分合并成阈值点），
# 所以指标数值应当一致，不存在「换算法导致对不上」的问题。
def _binary_clf_curve(y_true, score):
    """返回 (tps, fps)，按分数降序、并列值合并。"""
    order = np.argsort(score, kind="stable")[::-1]
    y_true, score = y_true[order], score[order]
    distinct = np.where(np.diff(score))[0]
    idxs = np.r_[distinct, y_true.size - 1]
    tps = np.cumsum(y_true)[idxs]
    return tps, 1 + idxs - tps


def average_precision(y_true, score):
    tps, fps = _binary_clf_curve(y_true, score)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true, score):
    tps, fps = _binary_clf_curve(y_true, score)
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.0 起改名
    return float(trapz(tpr, fpr))


def load_metrics():
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        return average_precision_score, roc_auc_score, "sklearn"
    except ImportError:
        return average_precision, roc_auc, "numpy 兜底（板子没装 sklearn）"


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
            f"先跑 board/00_board_env_check.py 看 NPU 节点诊断。"
        )
    return rknn


def rknn_infer(rknn, x, warmup=10):
    """逐条推理；输入 float32、形状 1x29。顺带统计单条耗时。"""
    for i in range(min(warmup, len(x))):
        rknn.inference(inputs=[x[i : i + 1].astype(np.float32)])

    logits = np.empty(len(x), dtype=np.float32)
    lat = []
    for i in range(len(x)):
        t0 = time.perf_counter()
        out = rknn.inference(inputs=[x[i : i + 1].astype(np.float32)])
        dt = (time.perf_counter() - t0) * 1000.0
        logits[i] = float(np.asarray(out).reshape(-1)[0])
        lat.append(dt)
    return logits, np.array(lat)


def onnx_infer(onnx_path, x, batch=None):
    """板端 CPU 上跑浮点 ONNX，作为量化精度的对照基线。

    batch 不传死值：导出的 ONNX 是静态 batch=1（RKNN 转换要求静态形状，
    训练脚本刻意没开动态轴），按 512 切片喂会被 onnxruntime 直接拒掉。
    这里从会话里读回真实的 batch 维，动态轴才是它给的值。
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    if batch is None:
        declared = sess.get_inputs()[0].shape[0]
        batch = declared if isinstance(declared, int) and declared > 0 else len(x)
    out = np.concatenate(
        [sess.run([oname], {iname: x[i : i + batch]})[0] for i in range(0, len(x), batch)]
    ).reshape(-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default="models/mlp_fraud.rknn")
    ap.add_argument("--onnx", default="models/mlp_fraud.onnx",
                    help="浮点 ONNX 对照；给了才算一致率与 AUPRC 差值")
    ap.add_argument("--splits", default="models/mlp_splits.npz")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="判定欺诈的概率阈值，默认 0.5")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条；0 = 全部")
    ap.add_argument("--show", type=int, default=10, help="打印前 N 条预测明细")
    ap.add_argument("--report", default="out/board_mlp_report.json")
    args = ap.parse_args()

    print("=== 任务卡 6 · 鲁班猫板端推理（信用卡欺诈 MLP）===")

    if not os.path.exists(args.rknn):
        raise SystemExit(f"找不到 RKNN 模型: {args.rknn}")
    z = np.load(args.splits)
    x_te, y_te = z["x_test"], z["y_test"]
    if args.limit:
        x_te, y_te = x_te[: args.limit], y_te[: args.limit]
    print(f"  测试集 {len(x_te)} 条，正样本占比 {y_te.mean():.5%}")

    rknn = load_rknn_runtime(args.rknn)
    try:
        print(f"\n[推理] NPU 上逐条推理 {len(x_te)} 条 ...")
        logits_rknn, lat = rknn_infer(rknn, x_te)
        print(f"  单条耗时  均值 {lat.mean():.3f} ms  中位 {np.median(lat):.3f} ms  "
              f"P99 {np.percentile(lat, 99):.3f} ms")
    finally:
        rknn.release()
        print("  rknn.release() 已释放")

    prob = sigmoid(logits_rknn)
    pred = (prob >= args.threshold).astype(int)

    print(f"\n[预测明细] 前 {min(args.show, len(prob))} 条")
    for i in range(min(args.show, len(prob))):
        flag = "预测欺诈" if pred[i] else "预测正常"
        print(f"  #{i:<6d} 真实={y_te[i]}  logit={logits_rknn[i]:>9.4f}  "
              f"概率={prob[i]:.6f}  {flag}")

    report = {
        "n_test": int(len(y_te)),
        "threshold": args.threshold,
        "latency_ms_mean": round(float(lat.mean()), 4),
        "latency_ms_median": round(float(np.median(lat)), 4),
        "latency_ms_p99": round(float(np.percentile(lat, 99)), 4),
        "pred_positive_rate": round(float(pred.mean()), 6),
        "true_positive_rate": round(float(y_te.mean()), 6),
    }
    average_precision_score, roc_auc_score, metrics_src = load_metrics()
    report["metrics_source"] = metrics_src
    print(f"\n  指标实现：{metrics_src}")

    report["rknn_auprc"] = round(float(average_precision_score(y_te, logits_rknn)), 6)
    report["rknn_roc_auc"] = round(float(roc_auc_score(y_te, logits_rknn)), 6)
    print(f"\n  量化 RKNN  AUPRC={report['rknn_auprc']:.5f}  "
          f"ROC-AUC={report['rknn_roc_auc']:.5f}")
    print(f"  预测正类占比 {report['pred_positive_rate']:.5f} "
          f"vs 真实 {report['true_positive_rate']:.5f}")
    if pred.sum() == 0:
        print("  [警告] 全部判为正常类，模型退化了，检查阈值与量化结果")

    # ---- 与浮点 ONNX 对比 ----
    if os.path.exists(args.onnx):
        print(f"\n[对比] 板端 CPU 跑浮点 ONNX {args.onnx}")
        logits_onnx = onnx_infer(args.onnx, x_te)
        pred_onnx = (sigmoid(logits_onnx) >= args.threshold).astype(int)
        agree = float((pred == pred_onnx).mean())
        onnx_auprc = float(average_precision_score(y_te, logits_onnx))
        onnx_auc = float(roc_auc_score(y_te, logits_onnx))
        drop = onnx_auprc - report["rknn_auprc"]

        report.update({
            "onnx_auprc": round(onnx_auprc, 6),
            "onnx_roc_auc": round(onnx_auc, 6),
            "agreement": round(agree, 6),
            "auprc_drop": round(drop, 6),
            "max_logit_diff": round(float(np.max(np.abs(logits_rknn - logits_onnx))), 6),
        })
        print(f"  浮点 ONNX  AUPRC={onnx_auprc:.5f}  ROC-AUC={onnx_auc:.5f}")
        print(f"  判定一致率 = {agree:.4%}   （判据 >= 99%）")
        print(f"  AUPRC 下降 = {drop:+.5f}      （判据 <= 0.03）")
        print(f"  logit 最大绝对差 = {report['max_logit_diff']:.6f}")
        if agree < 0.99:
            print("  [未通过] 一致率低于 99%")
        if drop > 0.03:
            print("  [未通过] AUPRC 下降超过 0.03")
    else:
        print(f"\n[跳过] 未找到 {args.onnx}，跳过一致率与 AUPRC 差值核对")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)
    print(f"\n[完成] 报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())