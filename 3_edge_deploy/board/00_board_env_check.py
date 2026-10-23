#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
鲁班猫 RK3588S 板端环境自检 —— 跑 RKNN 推理前必须先过这一关。

逐项检查：
  1. NPU 驱动与设备节点
  2. rknn-toolkit-lite2 是否安装、版本
  3. OpenCV 版本与视频采集能力（USB 摄像头）
  4. 磁盘、内存余量
  5. 模型文件是否就位

关于 NPU 节点的重要说明：
  rknn-toolkit-lite2 2.3.x 的运行时（librknnrt）走 DRM 路径 /dev/dri/renderD129，
  而旧版 Rockchip 驱动走 /dev/rknpu 这个 misc 节点。本板（LubanCat-4，
  内核 6.1.0-1025-rockchip，rknpu 驱动 0.9.7）是新版 DRM 路径：
  没有 /dev/rknpu 属正常现象，不是故障。本脚本据此判可用性。

用法：
    python3 00_board_env_check.py
    python3 00_board_env_check.py --models ~/models
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

OK, BAD, WARN = "[ ok ]", "[FAIL]", "[WARN]"


def run(cmd):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception as e:
        return f"<执行失败: {e}>"


def check_npu():
    print("\n=== 1. NPU 驱动 ===")
    devs = sorted(glob.glob("/dev/rknpu*"))
    npu_drm = [
        d for d in sorted(glob.glob("/dev/dri/renderD*"))
        if "npu" in os.path.realpath(d + "/device")
    ]

    log = run("dmesg 2>/dev/null | grep -i rknpu | tail -10")
    if log:
        print("  dmesg 里的 RKNPU 记录：")
        for line in log.splitlines():
            print(f"        {line}")
    else:
        print(f"  {WARN} dmesg 里没有 RKNPU 记录（可能需要 sudo 才能读）")

    drv = "/sys/devices/platform/fdab0000.npu/driver"
    if os.path.islink(drv):
        print(f"  {OK} RKNPU 驱动已绑定 -> "
              f"{os.path.basename(os.path.realpath(drv))}")

    # 运行时库到底会打开哪个路径，直接从 .so 里读出来，不靠猜
    lib_paths = []
    try:
        import rknnlite.api

        base = os.path.dirname(rknnlite.api.__file__)
        for so in glob.glob(os.path.join(base, "*.so")):
            out = subprocess.run(["strings", so], capture_output=True,
                                 text=True).stdout
            lib_paths += [l for l in out.splitlines()
                          if l.startswith("/dev/dri/") or l == "/dev/rknpu"]
    except Exception:
        pass
    lib_paths = sorted(set(lib_paths))

    print(f"\n  传统节点 /dev/rknpu* -> {devs or '不存在（旧驱动才需要，本板可忽略）'}")
    print(f"  {OK if npu_drm else WARN} DRM 节点 NPU 路径  -> {npu_drm or '未找到'}")
    for d in npu_drm:
        print(f"        {d} -> {os.path.realpath(d + '/device')}")
    if lib_paths:
        print(f"  运行时库内置路径    -> {lib_paths}")
    if shutil.which("librga2-config") or os.path.exists("/usr/lib/librga.so"):
        print(f"  {OK} librga2 已在（RGA 硬件缩放可用）")

    usable = bool(devs) or bool(npu_drm)
    if usable:
        print(f"\n  {OK} 结论：NPU 可用，rknn-toolkit-lite2 能初始化运行时。")
        if not devs:
            print("        本板走 DRM 路径，没有 /dev/rknpu 属正常现象，不是故障。")
    else:
        print(f"""
  {BAD} 结论：NPU 不可用，init_runtime() 一定失败。

  按顺序排查：
    1) sudo dmesg | grep -iE "rknpu|fdab0000" —— 看探测在哪一步停下
    2) 若停在 devfreq 初始化，多半是缺 rknpu-supply 电源 regulator
    3) 镜像可能没装驱动：sudo apt install rknpu-rk3588
    4) 或换官方最新镜像
""")
    return usable


def check_lite2():
    print("\n=== 2. rknn-toolkit-lite2 ===")
    try:
        import rknnlite.api as api

        print(f"  {OK} rknnlite 可导入，路径 {api.__file__}")
    except ImportError:
        print(f"  {BAD} 未安装 rknn-toolkit-lite2")
        print("      安装（走阿里源）：")
        print("        pip install rknn-toolkit-lite2==2.3.2")
        return False

    ver = run(
        "python3 -c \"import importlib.metadata as m;"
        "print(m.version('rknn-toolkit-lite2'))\" 2>/dev/null"
    )
    print(f"  版本: {ver or '未知'}")
    print("  [对照] 转换机侧 rknn-toolkit2 必须与此处同版本")
    return True


def check_opencv():
    print("\n=== 3. OpenCV 与视频采集 ===")
    try:
        import cv2

        print(f"  cv2 {cv2.__version__}")
        if not cv2.__version__.startswith("4"):
            print(f"  {WARN} 手册要求 OpenCV 4.x，当前 5.x，部分 API 行为有差异")
    except ImportError:
        print(f"  {BAD} 未安装 OpenCV：pip install opencv-python")
        return False

    vids = sorted(glob.glob("/dev/video[0-9]*"))
    real = [v for v in vids if os.access(v, os.R_OK | os.W_OK)]
    print(f"  /dev/video[0-9]* -> {vids or '不存在'}")
    print(f"  当前用户可访问   -> {real or '无'}")

    if not real:
        print(f"  {WARN} 没有本地 UVC 采集节点")
        print("      /dev/video-enc0 / -dec0 是硬件编解码节点，不是摄像头采集")
        print("      替代方案：用 Win11 笔记本摄像头推 MJPEG 流，板端用 IP 拉流")
        print("        （见 board/cam_person_detect.py --source http://<笔记本IP>:8080/stream）")
        if shutil.which("v4l2-ctl"):
            print(run("v4l2-ctl --list-devices 2>&1 | head -20"))
    else:
        if shutil.which("v4l2-ctl"):
            print("  v4l2-ctl --list-devices:")
            for line in run("v4l2-ctl --list-devices 2>&1 | head -12").splitlines():
                print(f"        {line}")
    return bool(real)


def check_resources():
    print("\n=== 4. 资源 ===")
    print(f"  磁盘: {run('df -h / | tail -1')}")
    print(f"  内存: {run('free -h | head -2 | tail -1')}")
    print(f"  CPU : {run('nproc')} 核")


def check_models(models_dir):
    print(f"\n=== 5. 模型文件（{models_dir}）===")
    if not os.path.isdir(models_dir):
        print(f"  {BAD} 目录不存在")
        return
    files = sorted(glob.glob(os.path.join(models_dir, "*")))
    if not files:
        print(f"  {WARN} 目录为空")
    for f in files:
        kb = os.path.getsize(f) / 1024
        mark = "<- RKNN" if f.endswith(".rknn") else ""
        print(f"    {os.path.basename(f):<40s} {kb:>10.1f} KB  {mark}")
    if not any(f.endswith(".rknn") for f in files):
        print(f"  {WARN} 没有 .rknn 文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.expanduser("~/models"))
    args = ap.parse_args()

    print("=== 鲁班猫 RK3588S 板端环境自检 ===")
    print(f"  主机: {run('hostname')}    内核: {run('uname -r')}")
    print(f"  架构: {run('uname -m')}    Python: {sys.version.split()[0]}")

    npu = check_npu()
    lite = check_lite2()
    cam = check_opencv()
    check_resources()
    check_models(args.models)

    print("\n=== 汇总 ===")
    print(f"  NPU 驱动           : {'可用' if npu else '不可用 —— RKNN 推理无法进行'}")
    print(f"  rknn-toolkit-lite2 : {'已安装' if lite else '未安装'}")
    print(f"  本地 USB 摄像头    : {'可用' if cam else '不可用（改用笔记本推流）'}")
    return 0 if npu else 1


if __name__ == "__main__":
    sys.exit(main())