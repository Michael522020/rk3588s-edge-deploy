#!/usr/bin/env bash
# 重建 RKNN 转换环境：跳过 torch 依赖栈
#
# 为什么不按 pip 默认来：rknn-toolkit2 2.3.2 声明 torch<=2.4.0，与镜像自带的
# torch 2.8.0 冲突，干净 venv 里 pip 会去下整套 torch + nvidia CUDA 库（数 GB，
# 实测跑了 50 分钟还在下 cudnn）。而 04/05 两个转换脚本根本不 import torch
# （只用 numpy / onnx / rknn），所以直接 --no-deps 装 rknn，按需补它其余依赖。
set -euo pipefail
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
E=/root/autodl-tmp/edge
V=$E/venv-rknn
BASE=/root/miniconda3/bin/python

say() { echo -e "\n===== [$(date +%T)] $* ====="; }

say "停掉还在下 CUDA 栈的 pip"
for p in $(pgrep -x pip 2>/dev/null); do kill "$p" 2>/dev/null || true; done
sleep 2

say "重建 venv $V（继承 base 的 torch，装进 venv 的 numpy 1.26.4 会覆盖 base 的 2.3.2）"
rm -rf "$V"
"$BASE" -m venv --system-site-packages "$V"
"$V/bin/pip" install -q -U pip wheel -i "$MIRROR"

say "补齐 rknn 真正需要的依赖（不含 torch）"
"$V/bin/pip" install -q -i "$MIRROR" \
    "numpy==1.26.4" "onnx>=1.16.1" "onnxruntime>=1.17.0" \
    "protobuf>=4.21.6,<=4.25.4" "psutil>=5.9.0" "ruamel.yaml>=0.17.21" \
    "scipy>=1.9.3" "tqdm>=4.64.1" "opencv-python>=4.5.5.64" "fast-histogram>=0.11"

say "安装 rknn-toolkit2==2.3.2（--no-deps，跳过 torch 栈）"
"$V/bin/pip" install -q --no-deps -i "$MIRROR" "rknn-toolkit2==2.3.2"

say "自检：能否 import 并实例化 RKNN"
"$V/bin/python" - <<'PY'
import sys, numpy, onnx, onnxruntime, rknn
from importlib.metadata import version
print(f"  python         {sys.version.split()[0]}")
print(f"  numpy          {numpy.__version__}   (rknn 要求 <=1.26.4；来自 venv，覆盖了 base 的 2.x)")
print(f"  onnx           {onnx.__version__}")
print(f"  onnxruntime    {onnxruntime.__version__}")
print(f"  rknn-toolkit2  {version('rknn-toolkit2')}   <-- 必须 2.3.2")
import torch
print(f"  torch          {torch.__version__}  (继承自 base；rknn.api 在 import 阶段就要它)")
# rknn 2.3.2 的入口是 rknn.api.RKNN；rknn.config / RKNNModel 是别的版本的布局，不存在
from rknn.api import RKNN
r = RKNN(verbose=False)
print("  from rknn.api import RKNN -> [OK]，实例化成功")
r.release()
print("  release() -> [OK]")
PY

say "完成"
