#!/usr/bin/env bash
# 在 AutoDL (x86_64 Ubuntu) 上一次性跑完：
#   任务卡 1/2/3/4  信用卡 MLP 训练 + 导出 ONNX + RKNN(浮点→INT8)
#   任务卡 9/10/11 肺炎 CNN 训练 + 导出 ONNX + RKNN(浮点→INT8)
#   任务卡 12/13   YOLOv8n 导 ONNX + 转 RKNN
#
# 不动机器配置：所有 Python 依赖装在 /root/autodl-tmp/edge/venv 里，
# 系统 / base 环境的 python、CUDA、驱动一律不碰。
#
# 用法： bash run_on_autodl.sh            # 全流程
#       bash run_on_autodl.sh env        # 只装依赖 + 自检

set -euo pipefail

ROOT=/root/autodl-tmp/edge
WORK=$ROOT/autodl
DATA=$WORK/data
VENV=$ROOT/venv
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
ZIP=$ROOT/Chest\ X-Ray\ Images.zip     # 2.4G 上传后放在 $ROOT 下
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

say() { echo -e "\n\033[1;36m==> $*\033[0m"; }

# ---------------------------------------------------------------- 0. 目录
mkdir -p "$WORK" "$DATA" "$ROOT/x" "$ROOT/autodl-tmp-check"
cd "$WORK"

# ---------------------------------------------------------------- 1. 依赖
say "准备 venv: $VENV"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -q -U pip wheel -i "$MIRROR"

say "装 PyTorch (CUDA 12.1) + 训练/导出依赖"
pip install -q torch torchvision --index-url "$MIRROR"
pip install -q -i "$MIRROR" \
    numpy pandas scikit-learn matplotlib pillow tqdm \
    onnx onnxruntime onnxsim ultralytics

say "装 rknn-toolkit2 —— 必须锁 2.3.2，与板端 rknn-toolkit-lite2 2.3.2 对齐"
pip install -q -i "$MIRROR" "rknn-toolkit2==2.3.2"

say "环境自检"
python - <<'PY'
import torch, onnx, onnxruntime, rknn, sys
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda        {torch.version.cuda}  gpu={torch.cuda.get_device_name(0)}")
print(f"  onnx        {onnx.__version__}")
print(f"  onnxruntime {onnxruntime.__version__}  providers={onnxruntime.get_available_providers()}")
print(f"  rknn        {rknn.__version__}   <-- 必须是 2.3.2")
PY
python 04_convert_rknn.py --env-check || true

[ "${1:-}" = "env" ] && { say "只做环境，结束"; exit 0; }

# ---------------------------------------------------------------- 2. 数据
if [ ! -f "$DATA/creditcard.csv" ]; then
    say "缺少 $DATA/creditcard.csv —— 先把本地的 creditcard.csv 传上来"
    exit 1
fi
if [ ! -d "$DATA/chest_xray/train" ]; then
    if [ -f "$ZIP" ]; then
        say "解压胸片数据集（2.4G，约 1~2 分钟）"
        unzip -q -o "$ZIP" -d "$DATA"
        # 解压后可能是 $DATA/chest_xray 或 $DATA/Chest_xray，统一成小写名
        [ -d "$DATA/Chest_xray" ] && mv "$DATA/Chest_xray" "$DATA/chest_xray.tmp" \
            && rm -rf "$DATA/chest_xray" && mv "$DATA/chest_xray.tmp" "$DATA/chest_xray"
    else
        say "缺少数据集：请上传 Chest X-Ray Images.zip 到 $ROOT"
        exit 1
    fi
fi
say "数据就位："
echo "  欺诈样本: $(wc -l < "$DATA/creditcard.csv") 行"
echo "  胸片: $(find "$DATA/chest_xray" -name '*.jpeg' -o -name '*.jpg' | wc -l) 张"

# ---------------------------------------------------------------- 3. 训练 + 导出 ONNX
say "任务卡 1/2/3/4 —— 信用卡 MLP"
python 01_mlp_fraud_train_export.py --data "$DATA/creditcard.csv" \
    --device cuda --outdir out/mlp_fraud

say "任务卡 9/10/11 —— 肺炎 CNN（MobileNetV2 预训练，3 轮训头 + 10 轮微调）"
python 02_cnn_pneumonia_train_export.py --data "$DATA/chest_xray" \
    --device cuda --batch 32 --outdir out/pneumonia

say "任务卡 12 —— YOLOv8n 导 ONNX（640 / opset 12 / 简化 / 静态 batch=1）"
python 03_yolov8n_export_onnx.py --weights yolov8n.pt --outdir out/yolov8n

# ---------------------------------------------------------------- 4. ONNX -> RKNN
say "实验一 MLP -> RKNN：先浮点确认链路，再上 INT8"
python 04_convert_rknn.py --model mlp --onnx out/mlp_fraud/mlp_fraud.onnx \
    --calib-dir out/mlp_fraud/calib_npy --calib-limit 200 --outdir out/rknn
python 04_convert_rknn.py --model mlp --onnx out/mlp_fraud/mlp_fraud.onnx \
    --calib-dir out/mlp_fraud/calib_npy --calib-limit 200 --outdir out/rknn --quant

say "实验二 肺炎 CNN -> RKNN"
python 04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images --calib-limit 200 --outdir out/rknn
python 04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images --calib-limit 200 --outdir out/rknn --quant

say "任务卡 13 —— YOLOv8n -> RKNN（先浮点，确认无算子回退再量化）"
python 05_convert_rknn_yolov8.py --onnx out/yolov8n/yolov8n.onnx \
    --dataset out/yolov8n/calib_list.txt --outdir out/rknn
python 05_convert_rknn_yolov8.py --onnx out/yolov8n/yolov8n.onnx \
    --dataset out/yolov8n/calib_list.txt --outdir out/rknn --quant

# ---------------------------------------------------------------- 5. 收尾
say "产物清单"
find out -maxdepth 3 \( -name '*.onnx' -o -name '*.rknn' -o -name '*.pt' \) \
    -printf '%-52p %10s bytes\n' 2>/dev/null || ls -lR out

say "下一步：把 *.rknn 拷回 RK3588S 板子验证"
echo "  scp -P 22880 -r root@connect.<region>.seetacloud.com:/root/autodl-tmp/edge/autodl/out/rknn ./"
echo "  然后 scp 到 ubuntu@192.168.1.100:/home/ubuntu/models/，用板端 rknn-toolkit-lite2 跑推理"