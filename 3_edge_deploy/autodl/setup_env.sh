#!/usr/bin/env bash
# AutoDL 环境准备：base conda 完全不动，训练和转换各用一个独立 venv
set -euo pipefail
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
BASE=/root/miniconda3/bin/python
E=/root/autodl-tmp/edge

say() { echo -e "\n===== [$(date +%T)] $* ====="; }

# ---- 训练环境：继承 base 的 torch 2.8.0+cu128 / numpy 2.3.2，只补缺的
say "建 venv-train (--system-site-packages)"
[ -d "$E/venv-train" ] || "$BASE" -m venv --system-site-packages "$E/venv-train"
"$E/venv-train/bin/pip" install -q -U pip wheel -i "$MIRROR"

say "装训练/导出依赖"
"$E/venv-train/bin/pip" install -q -i "$MIRROR" \
    onnx onnxruntime onnxsim ultralytics pandas scikit-learn matplotlib pillow tqdm

# ---- 转换环境：干净 venv，numpy 压到 1.x 以配 rknn-toolkit2 2.3.2
say "建 venv-rknn (干净)"
[ -d "$E/venv-rknn" ] || "$BASE" -m venv "$E/venv-rknn"
"$E/venv-rknn/bin/pip" install -q -U pip wheel -i "$MIRROR"

say "装 rknn-toolkit2==2.3.2（必须与板端 lite2 对齐）+ numpy 1.26.4 + onnx"
"$E/venv-rknn/bin/pip" install -q -i "$MIRROR" numpy==1.26.4 onnx
"$E/venv-rknn/bin/pip" install -q -i "$MIRROR" "rknn-toolkit2==2.3.2"

# ---- 自检
say "自检：训练环境"
"$E/venv-train/bin/python" - <<'PY'
import sys, torch, onnx, onnxruntime, numpy
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}   cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda        {torch.version.cuda}   gpu={torch.cuda.get_device_name(0)}")
    x = torch.randn(512, 512, device="cuda"); print(f"  算子自检     matmul on cuda -> {float((x@x).abs().mean()):.4f}")
print(f"  numpy       {numpy.__version__}")
print(f"  onnx        {onnx.__version__}")
print(f"  onnxruntime {onnxruntime.__version__}")
import ultralytics; print(f"  ultralytics {ultralytics.__version__}")
PY

say "自检：转换环境"
"$E/venv-rknn/bin/python" - <<'PY'
import sys, numpy, onnx, rknn
print(f"  python      {sys.version.split()[0]}")
print(f"  numpy       {numpy.__version__}")
print(f"  onnx        {onnx.__version__}")
print(f"  rknn-toolkit2 {rknn.__version__}   <-- 必须 2.3.2")
PY

say "环境就绪"