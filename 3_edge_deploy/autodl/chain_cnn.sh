#!/usr/bin/env bash
# 等 MiniMind SFT 结束，立刻接上肺炎 CNN 训练+导出（别让 GPU 空转）
set -uo pipefail
MM=/root/autodl-tmp/minimind
E=/root/autodl-tmp/edge

echo "[$(date +%T)] 等 SFT 进程退出..."
while true; do
    running=0
    for p in $(pgrep -x python 2>/dev/null); do
        tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_full_sft && running=1
    done
    [ "$running" -eq 0 ] && break
    sleep 20
done
echo "[$(date +%T)] SFT 已结束，产物："
ls -l "$MM/minimind/out/"*.pth

# 84G 卡上 MobileNetV2@224 远用不满，batch 给到 64、dataloader worker 给到 32
echo "[$(date +%T)] 启动肺炎 CNN 训练（batch 64, workers 32）"
cd "$E/autodl"
"$E/venv-train/bin/python" -u 02_cnn_pneumonia_train_export.py \
    --data "$E/autodl/data/chest_xray" \
    --device cuda \
    --batch 64 \
    --workers 32 \
    --epochs-head 3 \
    --epochs-finetune 10 \
    --outdir out/pneumonia \
    > "$E/logs/cnn.log" 2>&1
echo "[$(date +%T)] CNN 训练结束，开始转 RKNN（浮点 + INT8）"
"$E/venv-rknn/bin/python" -u 04_convert_rknn.py --model cnn \
    --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images \
    --calib-limit 200 --outdir out/rknn > "$E/logs/cnn_rknn_fp.log" 2>&1
"$E/venv-rknn/bin/python" -u 04_convert_rknn.py --model cnn \
    --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images \
    --calib-limit 200 --outdir out/rknn --quant > "$E/logs/cnn_rknn_int8.log" 2>&1
echo "[$(date +%T)] 全部完成"
ls -l "$E/autodl/out/rknn/"*.rknn