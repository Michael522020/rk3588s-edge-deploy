#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 13 —— 实验三：YOLOv8n 的 ONNX 转 RKNN

手册要点（对应验收判据）：
  优先使用 rknn_model_zoo 的 yolov8 转换脚本，避免输出切分不一致导致
    "无报错但检不出目标"
  量化校准清单每行一个图片绝对路径，取 COCO 图片或摄像头实拍图 100~200 张
  归一化配置与官方 yolov8 示例保持一致 —— 均值全为 0、标准差全为 255，
    即仅做除以 255 的缩放
  目标平台设为 rk3588
  先关闭量化跑通浮点版本，再开启 INT8 量化
  转换器以详细模式运行；结束释放资源

两种模式：
  --use-model-zoo  克隆并调用 rknn_model_zoo 官方 examples/yolov8/python/convert.py
                   （手册推荐，与官方示例逐字对齐）
  默认（自包含）    用与官方示例完全相同的配置直接转换，不依赖克隆

用法：
    python 05_convert_rknn_yolov8.py --onnx out/yolov8n/yolov8n.onnx \
        --dataset out/yolov8n/calib_list.txt --outdir out/rknn
    python 05_convert_rknn_yolov8.py ... --quant          # INT8
    python 05_convert_rknn_yolov8.py ... --use-model-zoo # 走官方脚本
"""

import argparse
import os
import shutil
import subprocess
import sys

TARGET = "rk3588"
INPUT_SIZE = 640
# 官方 yolov8 示例的归一化口径：均值全 0、标准差全 255，等价于只做 /255
IMG_MEAN = [0, 0, 0]
IMG_STD = [255, 255, 255]


def check_dataset(path):
    if not os.path.exists(path):
        raise SystemExit(f"找不到校准清单 {path}")
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    missing = [l for l in lines if not os.path.exists(l)]
    if missing:
        print(f"  [警告] 清单中有 {len(missing)} 个文件不存在，例如 {missing[0]}")
    print(f"  校准清单 {len(lines)} 行，可用 {len(lines) - len(missing)} 个")
    if len(lines) < 100:
        print("  [提示] 手册建议 100~200 张，少于 100 张量化可能不稳")
    return lines


def dataset_for_rknn(path, workdir):
    """rknn-toolkit2 2.3.2 独立转换用的校准清单：每行一个图片绝对路径。

    别和上面的 dataset_for_model_zoo 混用——那个会往每行后面追加
    `宽 高 通道 mean...std...`，那是官方 convert.py 的清单格式；
    2.3.2 自己解析时只认纯路径，会报 `Unsupport file 640!`。
    均值/标准差已经通过 config 的 mean_values/std_values 传进去了。
    """
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    out = os.path.join(workdir, "dataset_rknn.txt")
    with open(out, "w", encoding="utf-8") as w:
        for line in lines:
            w.write(os.path.abspath(line) + "\n")
    print(f"  校准清单(2.3.2 格式): {out}，{len(lines)} 张")
    return out


def dataset_for_model_zoo(path, workdir):
    """官方 convert.py 用的清单每行需要形如 `path 192 192 3 0 0 0 255 255 255`"""
    lines = [l.strip() for l in open(path) if l.strip()]
    out = os.path.join(workdir, "dataset_zoo.txt")
    with open(out, "w") as w:
        for line in lines:
            w.write(f"{line} {INPUT_SIZE} {INPUT_SIZE} 3 "
                    f"{IMG_MEAN[0]} {IMG_MEAN[1]} {IMG_MEAN[2]} "
                    f"{IMG_STD[0]} {IMG_STD[1]} {IMG_STD[2]}\n")
    return out


def run_model_zoo(onnx_path, dataset, outdir, quant):
    """克隆并调用 rknn_model_zoo 的官方 yolov8 转换脚本（手册推荐路径）。"""
    root = os.path.abspath("rknn_model_zoo")
    if not os.path.isdir(root):
        print("  克隆 rknn_model_zoo ...")
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://github.com/airockchip/rknn_model_zoo.git", root,
        ])
    ex_dir = os.path.join(root, "examples", "yolov8", "python")
    convert_py = os.path.join(ex_dir, "convert.py")
    if not os.path.exists(convert_py):
        raise SystemExit(
            f"未找到官方脚本 {convert_py}。\n"
            f"rknn_model_zoo 的目录结构可能已变化，请手动查看 {ex_dir}"
        )

    tag = "" if quant else "_fp"
    want = f"yolov8n_{TARGET}_{1 if quant else 0}{tag}.rknn"
    dst = os.path.abspath(os.path.join(outdir, f"yolov8n{tag}.rknn"))
    local_onnx = os.path.join(ex_dir, "yolov8n.onnx")
    shutil.copy(onnx_path, local_onnx)

    cmd = [sys.executable, "convert.py", onnx_path, TARGET, dataset, want]
    print("  执行:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=ex_dir)
    produced = os.path.join(ex_dir, want)
    if rc != 0 or not os.path.exists(produced):
        raise SystemExit(f"官方脚本失败，返回码 {rc}")
    shutil.copy(produced, dst)
    return dst


def convert_standalone(onnx_path, dataset, out_path, quant,
                        quant_algorithm="kl_divergence", optimization_level=3):
    """自包含转换：配置与官方 yolov8 示例逐项对齐。"""
    # rknn-toolkit2 2.3.2 的入口是 rknn.api.RKNN；包里没有 rknn.config / RKNNModel
    from rknn.api import RKNN

    rknn = RKNN(verbose=True)  # 详细模式
    try:
        # rknn-toolkit2 2.3.2 的配置口径（与 1.x/2.0 不同）：
        #   target_device 已移除 / float_bit -> float_dtype / quantized_alg -> quantized_algorithm
        #   quantized_type -> quantized_dtype / input_mean_values -> mean_values
        #   config 不再接受 input_size，输入尺寸由模型本身决定
        rknn.config(
            target_platform=TARGET,
            float_dtype="float16",
            quantized_algorithm=quant_algorithm,
            quantized_dtype="w8a8",
            quantized_method="layer",
            optimization_level=optimization_level,
            mean_values=IMG_MEAN,      # 均值全 0
            std_values=IMG_STD,        # 标准差全 255（等价于 /255 归一化）
        )
        rknn.load_onnx(model=onnx_path)
        if quant:
            rknn.build(do_quantization=True, dataset=dataset)
        else:
            rknn.build(do_quantization=False)
        rknn.export_rknn(out_path)   # 2.3.2 里是位置参数 export_path，不是 out_path
    finally:
        rknn.release()  # 释放会话资源
        print("---- rknn.release() 已释放 ----")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--dataset", required=True, help="每行一个图片绝对路径的校准清单")
    ap.add_argument("--outdir", default="out/rknn")
    ap.add_argument("--quant", action="store_true", help="开启 INT8 量化；不加以产出浮点版本")
    ap.add_argument("--use-model-zoo", action="store_true", help="调用 rknn_model_zoo 官方脚本")
    ap.add_argument("--quant-algorithm", default="kl_divergence",
                    choices=["normal", "mmse", "kl_divergence", "gdq"],
                    help="逐层激活截断点的搜索方式，默认 kl_divergence（见 04 的说明）")
    ap.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="图优化强度。3 最激进但最吃内存——AutoDL 无卡模式 "
                         "cgroup 只给 2 GiB，转 YOLO/CNN 会被 OOM 杀掉（退出码 137），"
                         "内存不够时降到 1 或 0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=== 任务卡 13 · YOLOv8n ONNX 转 RKNN ===")
    print(f"  ONNX    : {args.onnx}")
    print(f"  目标平台: {TARGET}")
    print(f"  归一化  : mean={IMG_MEAN} std={IMG_STD}  (等价于只做 /255)")
    print(f"  量化    : {'INT8' if args.quant else '关闭（浮点版本，用于先验证链路）'}")

    if not os.path.exists(args.onnx):
        raise SystemExit(f"找不到 ONNX: {args.onnx}")
    check_dataset(args.dataset)

    tag = "" if args.quant else "_fp"
    # 产物名跟 ONNX 源文件走：yolov8n.onnx -> yolov8n.rknn，yolo26n.onnx -> yolo26n.rknn，
    # 否则两种模型会产出同名文件互相覆盖
    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    out_path = os.path.join(args.outdir, f"{stem}{tag}.rknn")

    if args.use_model_zoo:
        dst = run_model_zoo(args.onnx, args.dataset, args.outdir, args.quant)
        out_path = dst
    else:
        dataset = dataset_for_rknn(args.dataset, args.outdir)
        convert_standalone(args.onnx, dataset, out_path, args.quant,
                           quant_algorithm=args.quant_algorithm,
                           optimization_level=args.optimization_level)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[完成] {out_path}  ({size_kb:.1f} KB)")
    print("下一步：拷到鲁班猫，用 board/cam_person_detect.py 做 USB 摄像头实时检测（任务卡 14）")
    print("      注意板端解码要匹配本脚本采用的输出切分，否则会出现'无报错但检不出目标'")
    if not args.quant:
        print("提示：先用浮点版本确认链路无误，再加 --quant 跑 INT8，并在同一张图上比对框")
    return 0


if __name__ == "__main__":
    sys.exit(main())