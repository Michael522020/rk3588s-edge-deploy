#!/usr/bin/env bash
# YOLO26n 全链路：导 ONNX -> RKNN 浮点 -> RKNN INT8
# 与 MiniMind 的 SFT 训练并行跑（RKNN 转换是 CPU 密集，不抢 GPU）
set -uo pipefail
source /etc/network_turbo >/dev/null 2>&1
E=/root/autodl-tmp/edge
cd "$E/autodl"

PY_TRAIN=$E/venv-train/bin/python
PY_RKNN=$E/venv-rknn/bin/python

say() { echo -e "\n===== [$(date +%T)] $* ====="; }

say "1/3 YOLO26n 导 ONNX"
"$PY_TRAIN" -u 06_yolo26_export_onnx.py --weights yolo26n.pt --outdir out/yolo26n \
    --calib-dir /root/coco \
    2>&1 | grep -vE 'Warning|Requirement|^Downloading|libcublasLt|onnxruntime'
[ -f out/yolo26n/yolo26n.onnx ] || { echo "ONNX 没产出，后续跳过"; exit 1; }

say "2/3 YOLO26n -> RKNN 浮点"
"$PY_RKNN" -u 05_convert_rknn_yolov8.py \
    --onnx out/yolo26n/yolo26n.onnx \
    --dataset out/yolo26n/calib_list.txt \
    --outdir out/rknn_yolo26 2>&1 | grep -vE '^D RKNN|^I RKNN|^W RKNN|^D |^E RKNN' | tail -20

say "3/3 YOLO26n -> RKNN INT8"
"$PY_RKNN" -u 05_convert_rknn_yolov8.py \
    --onnx out/yolo26n/yolo26n.onnx \
    --dataset out/yolo26n/calib_list.txt \
    --outdir out/rknn_yolo26 --quant 2>&1 | grep -vE '^D RKNN|^I RKNN|^W RKNN|^D |^E RKNN' | tail -20

say "产物"
ls -l out/rknn_yolo26/ 2>/dev/null
say "完成"