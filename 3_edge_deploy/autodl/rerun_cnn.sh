#!/usr/bin/env bash
# 重跑肺炎 CNN（TF32 修复后）+ 两个 RKNN 版本
set -uo pipefail
E=/root/autodl-tmp/edge
cd "$E/autodl"

echo "[$(date +%T)] CNN 训练 + 导出 ONNX"
"$E/venv-train/bin/python" -u 02_cnn_pneumonia_train_export.py \
    --data "$E/autodl/data/chest_xray" \
    --device cuda --batch 64 --workers 32 \
    --epochs-head 3 --epochs-finetune 10 \
    --outdir out/pneumonia > "$E/logs/cnn.log" 2>&1
tail -22 "$E/logs/cnn.log"

if [ ! -f out/pneumonia/pneumonia.onnx ]; then
    echo "ONNX 没产出，RKNN 转换跳过"
    exit 1
fi

echo "[$(date +%T)] 转 RKNN 浮点"
"$E/venv-rknn/bin/python" -u 04_convert_rknn.py --model cnn \
    --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images --calib-limit 200 \
    --outdir out/rknn > "$E/logs/cnn_rknn_fp.log" 2>&1
grep -E '完成|校准' "$E/logs/cnn_rknn_fp.log" | tail -3

echo "[$(date +%T)] 转 RKNN INT8"
"$E/venv-rknn/bin/python" -u 04_convert_rknn.py --model cnn \
    --onnx out/pneumonia/pneumonia.onnx \
    --calib-dir out/pneumonia/calib_images --calib-limit 200 \
    --outdir out/rknn --quant > "$E/logs/cnn_rknn_int8.log" 2>&1
grep -E '完成|校准' "$E/logs/cnn_rknn_int8.log" | tail -3

echo "[$(date +%T)] 全部完成"
ls -l out/rknn/ out/pneumonia/*.onnx