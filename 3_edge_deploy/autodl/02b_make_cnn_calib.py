#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补齐肺炎 CNN 的「模型无关」产物：量化校准集 / preprocess.json / 测试清单。

背景（2026-10-23）：10-21 那次 AutoDL 上跑 02 脚本时，用的是修复前的版本——
ONNX 一致性校验放在导出之后立刻做，判据 1e-5 被 3.3e-5 顶掉就 SystemExit，
于是 calib_images / preprocess.json / test_manifest.txt / pneumonia.pt 全都没落盘，
只有 pneumonia.onnx 侥幸写出来了。结果 04 脚本开 INT8 时找不到校准图直接报错。

本地这份 02 已经修好（校验挪到产物落盘之后、判据放宽并加 argmax 决策判据），
但训练要 GPU，补这些产物不要 GPU——它们只依赖数据集和已导出的 ONNX，
所以单独写这个脚本在 CPU 上补，训练权重原样沿用 AutoDL 上已导出的那个 ONNX。

校准集的口径必须和 02 脚本逐位一致（同样的 rng、同样的 transform），
否则量化区间对不上，板端精度会莫名其妙地掉。这里直接从 02 里 import，
不复制常量，避免两边漂移。

用法：
    python 02b_make_cnn_calib.py --data data/chest_xray --onnx out/pneumonia/pneumonia.onnx
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("cnn02", os.path.join(_HERE, "02_cnn_pneumonia_train_export.py"))
cnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cnn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/chest_xray")
    ap.add_argument("--onnx", default="out/pneumonia/pneumonia.onnx")
    ap.add_argument("--outdir", default="out/pneumonia")
    ap.add_argument("--calib-n", type=int, default=200,
                    help="量化校准张量个数，与 02 脚本一致时保持 200")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        raise SystemExit(f"找不到 ONNX：{args.onnx}")
    os.makedirs(args.outdir, exist_ok=True)

    # 输入/输出名直接问 ONNX，不依赖训练时的 model 对象
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    print(f"ONNX  {args.onnx}")
    print(f"      输入 {iname} {sess.get_inputs()[0].shape}  输出 {oname} {sess.get_outputs()[0].shape}")

    # 数据切分与 02 脚本完全一致（同样的 SEED、同样的 8:2 分层切分）
    tr_items, va_items, te_items = cnn.build_datasets(args.data)
    print(f"      训练集 {len(tr_items)}  验证集 {len(va_items)}  测试集 {len(te_items)}")

    from PIL import Image
    from torchvision import transforms

    # ---- 1. 量化校准集：从训练集抽 calib_n 张，transform 与 02 脚本逐字一致
    calib_dir = os.path.join(args.outdir, "calib_images")
    os.makedirs(calib_dir, exist_ok=True)
    calib_tf = transforms.Compose([
        transforms.Resize((cnn.IMG_SIZE, cnn.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cnn.IMAGENET_MEAN, std=cnn.IMAGENET_STD),
    ])
    rng = np.random.default_rng(cnn.SEED)
    pick = rng.choice(len(tr_items), size=min(args.calib_n, len(tr_items)), replace=False)
    for i, idx in enumerate(pick):
        img = Image.open(tr_items[idx][0]).convert("RGB")
        np.save(os.path.join(calib_dir, f"calib_{i:04d}.npy"),
                calib_tf(img).numpy().astype(np.float32))
    sample = np.load(os.path.join(calib_dir, "calib_0000.npy"))
    print(f"[1/3] 校准集 {len(pick)} 个张量 -> {calib_dir}")
    print(f"      形状 {sample.shape} {sample.dtype}  范围 [{sample.min():.3f}, {sample.max():.3f}]")

    # ---- 2. 板端预处理样张 + 归一化常量
    eval_tf = cnn.build_transforms()[1]
    test_ds = cnn.ListDataset(te_items, eval_tf)
    sample_img, sample_label = test_ds[0]
    np.save(os.path.join(args.outdir, "pneumonia_sample.npy"),
            sample_img.numpy().astype(np.float32))
    with open(os.path.join(args.outdir, "preprocess.json"), "w") as w:
        json.dump({
            "input_name": iname,
            "output_name": oname,
            "classes": cnn.CLASSES,
            "input_shape": [1, 3, cnn.IMG_SIZE, cnn.IMG_SIZE],
            "output_shape": [1, 2],
            "resize": [cnn.IMG_SIZE, cnn.IMG_SIZE],
            "mean": list(cnn.IMAGENET_MEAN),
            "std": list(cnn.IMAGENET_STD),
            "layout": "NCHW",
            "sample_path": cnn.sample_items_path(args.data),
            "sample_true_label": cnn.CLASSES[sample_label],
        }, w, indent=2, ensure_ascii=False)
    print(f"[2/3] preprocess.json + pneumonia_sample.npy  "
          f"mean={cnn.IMAGENET_MEAN} std={cnn.IMAGENET_STD}")

    # ---- 3. 测试集清单：板端按同一口径算查全率
    with open(os.path.join(args.outdir, "test_manifest.txt"), "w") as w:
        for path, label in te_items:
            w.write(f"{path}\t{label}\n")
    c = cnn.class_counts(te_items)
    print(f"[3/3] test_manifest.txt {len(te_items)} 行  正常 {c[0]} 肺炎 {c[1]}")

    print("\n完成，可以开 INT8 了：")
    print(f"  python 04_convert_rknn.py --model cnn --onnx {args.onnx} "
          f"--calib-dir {calib_dir} --calib-limit {args.calib_n} --outdir out/rknn --quant")
    return 0


if __name__ == "__main__":
    sys.exit(main())