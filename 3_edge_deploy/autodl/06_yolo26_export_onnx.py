#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO26n 导出 ONNX 并校验（实验三 YOLO 分支的 YOLO26 版本）

与 03_yolov8n_export_onnx.py 的关键差异——**输出契约完全不同**：
    YOLOv8n : output0 [1, 84, 8400]   = 4 个框坐标 + 80 类分数，检测前需要手工 NMS
    YOLO26n : output0 [1, 300, 6]     = 端到端 NMS-free，300 行已按置信度降序，
                                        每行是 xyxy(4) + conf(1) + cls(1)
所以这里不再手写解码+NMS，改为直接比对「.pt 官方 predict」与「ONNX 官方 predict」
在同一张图上框出来的目标，验收口径仍是手册的「位置、类别、置信度一致」。

导出参数沿用手册口径：输入 640、输入名 images、opset 12、关闭动态轴、批固定 1、
开启简化去除冗余算子。

用法：
    python 06_yolo26_export_onnx.py --weights yolo26n.pt --outdir out/yolo26n
"""

import argparse
import os
import sys

import numpy as np

INPUT_NAME = "images"
INPUT_SIZE = 640
OPSET = 12
TEST_IMAGE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def fetch_test_images(n, outdir):
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, url in enumerate(TEST_IMAGE_URLS):
        dst = os.path.join(outdir, f"test_{i}.jpg")
        if os.path.exists(dst):
            paths.append(dst)
            continue
        try:
            from ultralytics.utils import downloads

            downloads.safe_download(url, dst)
            paths.append(dst)
        except Exception as e:
            print(f"      [跳过] {url} 下载失败: {e}")
    return paths[:n]


def collect_calib_images(outdir, coco_dir, need=200):
    """量化校准素材：优先用 COCO val2017（贴近真实检测分布），否则退回测试图。"""
    os.makedirs(outdir, exist_ok=True)
    picked = []
    for sub in ("val2017", "train2017"):
        d = os.path.join(coco_dir, sub)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        # 均匀取样，避免只取到文件名前缀连续的同一批
        step = max(1, len(files) // need)
        for f in files[::step][:need]:
            picked.append(os.path.join(d, f))
        if len(picked) >= need:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26n.pt")
    ap.add_argument("--outdir", default="out/yolo26n")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--calib-dir", default="/root/coco/images")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 72)
    print("实验三 · YOLO26n 导出 ONNX")
    print("=" * 72)

    from ultralytics import YOLO

    model = YOLO(args.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"  权重      : {args.weights}")
    print(f"  参数量    : {n_params / 1e6:.2f}M")
    print(f"  类别数    : {len(model.names)}")

    # ---------------------------------------------------------------- 导出
    print("\n[1] 导出 ONNX")
    produced = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        opset=OPSET,
        simplify=True,
        dynamic=False,
        batch=1,
        nms=False,
        verbose=False,
    )
    target = os.path.join(args.outdir, "yolov8n.onnx" if "v8" in args.weights else "yolo26n.onnx")
    if os.path.abspath(produced) != os.path.abspath(target):
        os.replace(produced, target)
    onnx_path = target
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  [ok] {onnx_path}  ({size_mb:.2f} MB)")

    # ---------------------------------------------------------------- 形状核对
    print("\n[2] Netron 口径核对")
    import onnx as onnx_mod

    mo = onnx_mod.load(onnx_path)

    def dims(ti):
        return [d.dim_value or d.dim_param for d in ti.type.tensor_type.shape.dim]

    for i in mo.graph.input:
        print(f"  输入 {i.name} {dims(i)}")
    for o in mo.graph.output:
        print(f"  输出 {o.name} {dims(o)}")

    outs = [o.name for o in mo.graph.output]
    inps = [i.name for i in mo.graph.input]
    iname, oname = inps[0], outs[0]
    want_in = [1, 3, INPUT_SIZE, INPUT_SIZE]
    if "images" not in iname:
        print(f"  [警告] 输入名是 '{iname}'，手册要求 'images'")
    shp = dims(mo.graph.output[0])
    if len(shp) == 3 and shp[2] == 6:
        print("  [ok] 端到端 NMS-free 输出 [1,300,6]，板端解码不需要 NMS")
    else:
        print(f"  [注意] 输出不是 [1,300,6] 形态，实际 {shp}，解码方式需按此形状适配")

    # ---------------------------------------------------------------- 一致性校验
    print("\n[3] 一致性校验：.pt 官方 predict  vs  ONNX 官方 predict")
    imgs = fetch_test_images(2, os.path.join(args.outdir, "test_images"))
    if not imgs:
        print("  [跳过] 没有测试图")
        return 0

    onnx_model = YOLO(onnx_path)   # 让 ultralytics 走 ONNX Runtime 推理路径
    worst_box = 0.0
    worst_conf = 0.0
    for p in imgs:
        # rect=False：与 ONNX 固定 640×640 的输入口径一致，否则框数都会对不上
        a = model.predict(source=p, conf=0.25, imgsz=INPUT_SIZE, rect=False, verbose=False)[0]
        b = onnx_model.predict(source=p, conf=0.25, imgsz=INPUT_SIZE, rect=False, verbose=False)[0]
        ba, bb = a.boxes, b.boxes
        ca = ba.cls.cpu().numpy().astype(int)
        cb = bb.cls.cpu().numpy().astype(int)
        xa = ba.xyxy.cpu().numpy()
        xb = bb.xyxy.cpu().numpy()
        print(f"  {os.path.basename(p):<14s} pytorch {len(ba)} 框 / onnx {len(bb)} 框")
        if len(ca) == len(cb):
            if len(ca):
                matched = 0
                used = set()
                for i, x in enumerate(xa):
                    d = np.linalg.norm(xb - x[None], axis=1)
                    d[list(used)] = np.inf
                    j = int(d.argmin())
                    if d[j] < 20.0 and cb[j] == ca[i]:
                        matched += 1
                        used.add(j)
                        worst_box = max(worst_box, float(d[j]))
                conf_a = ba.conf.cpu().numpy()
                conf_b = bb.conf.cpu().numpy()
                worst_conf = max(worst_conf, float(np.abs(conf_a - conf_b).max()))
                print(f"      类别+位置匹配 {matched}/{len(ca)}，"
                      f"最大框偏差 {worst_box:.2f} px，最大置信度差 {worst_conf:.4f}")
        else:
            print(f"      [警告] 框数不一致：{len(ca)} vs {len(cb)}")

    print(f"\n  全部图片最大框偏差 = {worst_box:.2f} px，最大置信度差 = {worst_conf:.4f}")

    # ---------------------------------------------------------------- 校准清单
    print("\n[4] 量化校准清单（任务卡 13 用）")
    calib = collect_calib_images(args.outdir, args.calib_dir, args.calib_n)
    if len(calib) < args.calib_n:
        extra = fetch_test_images(8, os.path.join(args.outdir, "test_images"))
        calib += [e for e in extra if e not in calib]
    list_path = os.path.join(args.outdir, "calib_list.txt")
    with open(list_path, "w", encoding="utf-8") as w:
        for p in calib:
            w.write(os.path.abspath(p) + "\n")
    print(f"  {len(calib)} 张 -> {list_path}")
    if len(calib) < 100:
        print(f"  [提示] 手册建议 100~200 张，当前 {len(calib)} 张；"
              f"把 COCO 图放到 {args.calib_dir}/val2017 后重跑本步骤")

    print("\n" + "=" * 72)
    print(f"[完成] {onnx_path}")
    print("下一步：05_convert_rknn_yolov8.py --onnx " + onnx_path +
          f" --dataset {list_path} --outdir out/rknn [--quant]")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())