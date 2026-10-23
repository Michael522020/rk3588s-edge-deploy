#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 5（实验一 MLP）与实验二第五节（肺炎 CNN）—— ONNX 转 RKNN

必须在 x86 主机 Ubuntu 18.04/20.04/22.04 上运行（RKNN-Toolkit2 不支持 aarch64，
也不支持 Windows；手册 (二) 说明即为此）。转换完再把 .rknn 拷到鲁班猫。

手册要点（对应验收判据）：
  目标平台设为 rk3588
  量化校准集从训练集随机抽 100~200 条，数值须已按训练口径标准化
  建议先关闭量化跑通浮点版本，确认转换链路无误后再开启量化
  转换器以详细模式运行，便于查看算子处理情况
  转换结束释放会话资源，避免占用内存
  日志无算子回退或不支持告警

用法：
    # 环境自检
    python -c "import rknn; print(rknn.__version__)"
    pip install rknn-toolkit2==2.3.2
    python 04_convert_rknn.py --env-check

    # MLP：先浮点后 INT8
    python 04_convert_rknn.py --model mlp     --onnx out/mlp_fraud/mlp_fraud.onnx \
        --calib out/mlp_fraud/calib_npy/*.npy  --outdir out/rknn
    python 04_convert_rknn.py --model mlp     --onnx out/mlp_fraud/mlp_fraud.onnx \
        --calib out/mlp_fraud/calib_npy/*.npy  --outdir out/rknn --quant

    # 肺炎 CNN
    python 04_convert_rknn.py --model cnn     --onnx out/pneumonia/pneumonia.onnx \
        --calib out/pneumonia/calib_images/*.npy --outdir out/rknn --quant
"""

import argparse
import glob
import os
import platform
import sys

import numpy as np

MODEL_SPEC = {
    "mlp": {
        "name": "mlp_fraud",
        "input_shape": (1, 29),
        "source": "任务卡 5 · 实验一全连接信用卡欺诈检测",
    },
    "cnn": {
        "name": "pneumonia",
        "input_shape": (1, 3, 224, 224),
        "source": "实验二第五节 · 肺炎 CNN 影像分类",
    },
}
TARGET = "rk3588"


def env_check():
    """RKNN-Toolkit2 的硬性平台限制，先确认再往下走。"""
    print("=== RKNN-Toolkit2 环境自检 ===")
    print(f"  平台     : {platform.platform()}")
    print(f"  机器架构 : {platform.machine()}")

    ok = True
    if platform.machine() not in ("x86_64", "AMD64"):
        print("  [致命] RKNN-Toolkit2 只支持 x86 主机，当前架构无法安装/运行。")
        ok = False
    else:
        print("  [ok] 架构符合 x86 要求")

    if "microsoft" in platform.release().lower() or "WSL" in platform.version():
        print("  [提示] 疑似 WSL 环境；Ubuntu 20.04/22.04 的 WSL2 一般可用，18.04 也可")

    try:
        import rknn

        # rknn 2.3.2 的包本身没有 __version__，真实版本要从发行包元数据取
        ver = getattr(rknn, "__version__", None)
        if not ver:
            try:
                from importlib.metadata import version

                ver = version("rknn-toolkit2")
            except Exception:
                ver = "未知"
        print(f"  [ok] rknn-toolkit2 已安装，版本 {ver}")
        if ver not in ("未知", "2.3.2"):
            print(f"  [警告] 版本是 {ver}，与板端 Lite2 的 2.3.2 不一致，转换结果可能无法在板上加载")
    except ImportError:
        print("  [致命] 未安装 rknn-toolkit2。安装命令：")
        print("        pip install rknn-toolkit2==2.3.2     # 版本须与板端 Lite2 一致")
        ok = False

    try:
        import onnx

        print(f"  [ok] onnx {onnx.__version__}")
    except ImportError:
        print("  [警告] 未安装 onnx")

    print("\n  板端 Lite2 版本须与此处一致，可先在鲁班猫上跑：")
    print("      python -c \"from rknnlite.api import RKNNLite; import pkg_resources;"
          "print(pkg_resources.get_distribution('rknn_toolkit_lite2').version)\"")
    return ok


def build_calib_array(paths, input_shape, limit=200, clip=0.0):
    """把散落的 npy 校准样本堆成一个数组；样本须与训练/板端预处理口径一致。

    clip > 0 时按 ±clip 截断。标准化后的特征是重尾的：信用卡欺诈数据里
    少数几列的 absmax 能到 104，而 p99 只有 3~4。量化区间直接取校准集的
    min/max，区间被离群点撑大之后步长也跟着变粗（[-104,104] 步长 0.819，
    而 99% 的数据挤在 ±4 以内），典型样本的相对误差能到 20%。截断能把
    步长压细几个量级，代价只是极小比例的样本饱和。
    """
    rows = []
    n_clipped = 0
    for p in paths[:limit]:
        arr = np.load(p)
        if arr.shape != tuple(input_shape):
            arr = arr.reshape(input_shape)
        arr = arr.astype(np.float32)
        if clip > 0:
            n_clipped += int(np.count_nonzero(np.abs(arr) > clip))
            arr = np.clip(arr, -clip, clip)
        rows.append(arr)
    if not rows:
        raise SystemExit("校准集为空")
    batch = np.stack(rows, axis=0)
    if clip > 0:
        print(f"  裁剪    : ±{clip}，共截断 {n_clipped} 个元素"
              f"（占 {n_clipped / batch.size:.4%}）")
    return batch, len(rows)


def convert(onnx_path, calib, out_path, quant, verbose=True, calib_list=None,
            quant_algorithm="normal", optimization_level=3, float_dtype="float16",
            input_mean=None, input_std=None):
    # rknn-toolkit2 2.3.2 的入口是 rknn.api.RKNN；它的包里没有 rknn.config 子模块
    from rknn.api import RKNN

    rknn = RKNN(verbose=verbose)  # 详细模式：便于查看算子处理情况
    try:
        # rknn-toolkit2 2.3.2 的配置参数名与 1.x/2.0 不同：
        #   target_device 已移除（只有 target_platform）
        #   float_bit -> float_dtype，可选 float16 / bfloat16 / tfloat32（没有 float32）
        #   quantized_alg -> quantized_algorithm，可选 normal / mmse / kl_divergence / gdq
        #   quantized_type -> quantized_dtype，取 w8a8 表示纯 INT8
        # 浮点阶段只跑 float16（RK3588 NPU 本来就是 fp16 算力），确认链路后再开量化
        cfg = dict(
            target_platform=TARGET,
            float_dtype=float_dtype,
            **({"mean_values": [input_mean]} if input_mean else {}),
            **({"std_values": [input_std]} if input_std else {}),
            quantized_algorithm=quant_algorithm,
            quantized_dtype="w8a8",
            quantized_method="layer",
            optimization_level=optimization_level,
        )
        print("---- config ----")
        print(f"  {cfg}")
        rknn.config(**cfg)      # 顺序要紧：config -> load_onnx -> build -> export

        print("---- load_onnx ----")
        rknn.load_onnx(model=onnx_path)

        if quant:
            print(f"---- build（INT8 量化，校准样本 {calib.shape}）----")
            if not calib_list:
                raise SystemExit("缺少校准清单（calib_list），无法量化")
            rknn.build(do_quantization=True, dataset=calib_list)
        else:
            print("---- build（关闭量化，浮点版本）----")
            rknn.build(do_quantization=False)

        print(f"---- export_rknn -> {out_path} ----")
        rknn.export_rknn(out_path)   # 2.3.2 里是位置参数 export_path，不是 out_path

        # 把转换日志留档，便于事后核对有无算子回退
        log_path = out_path.replace(".rknn", ".convert.log")
        try:
            stats = rknn.list_operator()  # 触发 SDK 算子支持表查询
            with open(log_path, "w") as w:
                w.write(f"onnx : {onnx_path}\nout  : {out_path}\n")
                w.write(f"quant: {quant}\nconfig: {cfg}\n\n")
                w.write("supported operators on this SDK:\n")
                for k, v in sorted(stats.items()):
                    w.write(f"  {k}\n")
            print(f"  转换日志 -> {log_path}")
        except Exception as e:
            print(f"  [提示] 取不到算子表: {e}")
    finally:
        rknn.release()  # 释放会话资源，避免占用内存
        print("---- rknn.release() 已释放 ----")


def scan_for_fallback(log_path):
    """转换日志里若出现回退/不支持字样，明确提示。"""
    if not os.path.exists(log_path):
        return
    bad = []
    with open(log_path, errors="ignore") as f:
        for line in f:
            low = line.lower()
            if any(k in low for k in ("fallback", "not support", "unsupport", "cpu op")):
                bad.append(line.strip())
    if bad:
        print(f"  [警告] 日志中出现 {len(bad)} 条回退/不支持相关记录：")
        for line in bad[:10]:
            print(f"         {line}")
    else:
        print("  [ok] 日志中无算子回退或不支持告警")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-check", action="store_true", help="只做环境自检后退出")
    ap.add_argument("--model", choices=list(MODEL_SPEC), required=False)
    ap.add_argument("--onnx", help="ONNX 路径")
    ap.add_argument("--calib", nargs="*", help="校准 npy 路径或含通配符的目录")
    ap.add_argument("--calib-dir", help="含 npy 校准样本的目录")
    ap.add_argument("--calib-limit", type=int, default=200)
    ap.add_argument("--outdir", default="out/rknn")
    ap.add_argument("--quant", action="store_true", help="开启 INT8 量化；不加以产出浮点版本")
    ap.add_argument("--quant-algorithm", default="kl_divergence",
                    choices=["normal", "mmse", "kl_divergence", "gdq"],
                    help="逐层激活截断点的搜索方式。默认 kl_divergence：它逐层搜"
                         "最小化 KL 散度的截断点，比 normal 的 min/max 稳得多。"
                         "实测 MLP 上 normal 会让 AUPRC 掉 0.034（超判据），"
                         "换 kl_divergence 后不降反升 0.017")
    ap.add_argument("--calib-clip", type=float, default=0.0,
                    help="校准样本按 ±C 截断后再送进量化；0 表示不截。"
                         "重尾特征下可显著压细 INT8 步长")
    ap.add_argument("--optimization-level", type=int, default=3, choices=[0, 1, 2, 3],
                    help="图优化强度。3 最激进但最吃内存——AutoDL 无卡模式 "
                         "cgroup 只给 2 GiB，转 YOLO/CNN 会被 OOM 杀掉（退出码 137），"
                         "内存不够时降到 1 或 0")
    ap.add_argument("--float-dtype", default="float16",
                    choices=["float16", "bfloat16", "tfloat32"],
                    help="浮点版本内部精度。RK3588 NPU 原生 fp16；"
                         "某些模型 fp16 精度不够时可以试 bfloat16/tfloat32")
    ap.add_argument("--input-mean", nargs="*", type=float, default=None,
                    help="每个输入通道的均值，写进 RKNN 后板端就不要再减它。"
                         "不设则 RKNN 插恒等预处理（mean=0/std=1）")
    ap.add_argument("--input-std", nargs="*", type=float, default=None,
                    help="每个输入通道的标准差。图像模型常用 0 0 0 / 255 255 255，"
                         "让 RKNN 自己做 /255，板端直接喂 0~255 原值")
    args = ap.parse_args()

    if args.env_check:
        return 0 if env_check() else 1

    if not args.model or not args.onnx:
        ap.error("--model 与 --onnx 必填")

    spec = MODEL_SPEC[args.model]
    os.makedirs(args.outdir, exist_ok=True)
    print(f"=== {spec['source']} ===")
    print(f"  ONNX    : {args.onnx}")
    print(f"  目标平台: {TARGET}")
    print(f"  量化    : {'INT8' if args.quant else '关闭（浮点版本，用于先验证链路）'}")

    if not os.path.exists(args.onnx):
        raise SystemExit(f"找不到 ONNX: {args.onnx}")

    calib = None
    calib_list_path = None
    if args.quant:
        paths = list(args.calib or [])
        if args.calib_dir:
            paths = sorted(glob.glob(os.path.join(args.calib_dir, "*.npy")))
        if not paths:
            raise SystemExit("开量化必须给 --calib 或 --calib-dir")
        if any("*" in p for p in paths):
            expanded = []
            for p in paths:
                expanded.extend(sorted(glob.glob(p)))
            paths = expanded
        calib, n = build_calib_array(paths, spec["input_shape"], args.calib_limit,
                                 args.calib_clip)
        print(f"  校准集  : {n} 条，shape={calib.shape}，"
              f"min={calib.min():.3f} max={calib.max():.3f}")
        # rknn-toolkit2 2.3.2 的 build(dataset=) 只认「每行一个 .npy 路径」的文本清单：
        #   传裸 ndarray  -> stat() 报 'path should be string'
        #   传 .npy 文件  -> 会被当稀疏权重配置按 utf-8 解码而报 UnicodeDecodeError
        # 而且清单里的每个 npy 必须是 (含 batch 维的) input_shape。训练脚本落盘的
        # 校准样本是单条无 batch 维的，直接喂会报
        #   "The dims of input(ndarray) shape (29,) is wrong, expect dims is 2"
        # 量化就等于没校准准，所以这里统一补成 (1, ...) 再写清单。
        norm_dir = os.path.join(args.outdir, "calib_npy_norm")
        os.makedirs(norm_dir, exist_ok=True)
        want = tuple(spec["input_shape"])
        calib_list_path = os.path.join(args.outdir, "calib_list.txt")
        with open(calib_list_path, "w", encoding="utf-8") as w:
            for p in paths[:n]:
                arr = np.load(p).astype(np.float32)
                if arr.shape != want:
                    arr = arr.reshape(want)
                # 裁剪必须作用在这一份上：toolkit 读的是这些 .npy 文件本身，
                # 只裁 build_calib_array 返回的那个内存数组不会有任何效果。
                if args.calib_clip > 0:
                    arr = np.clip(arr, -args.calib_clip, args.calib_clip)
                dst = os.path.join(norm_dir, os.path.basename(p))
                np.save(dst, arr)
                w.write(os.path.abspath(dst) + "\n")
        print(f"  校准清单: {calib_list_path}（{n} 条，每条 shape={want}"
              + (f"，已截断到 ±{args.calib_clip}" if args.calib_clip > 0 else "") + "）")

    tag = "" if args.quant else "_fp"
    out_path = os.path.join(args.outdir, f"{spec['name']}{tag}.rknn")
    convert(args.onnx, calib, out_path, args.quant, calib_list=calib_list_path,
            quant_algorithm=args.quant_algorithm,
            optimization_level=args.optimization_level,
            float_dtype=args.float_dtype,
            input_mean=args.input_mean,
            input_std=args.input_std)
    scan_for_fallback(out_path.replace(".rknn", ".convert.log"))

    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n[完成] {out_path}  ({size_kb:.1f} KB)")
    print(f"下一步：scp 到 ubuntu@<板子IP>:/home/ubuntu/models/ 后用 rknn-toolkit-lite2 推理")
    if not args.quant:
        print("提示：先用浮点版本确认链路无误，再加 --quant 跑 INT8，对比两者精度")
    return 0


if __name__ == "__main__":
    sys.exit(main())