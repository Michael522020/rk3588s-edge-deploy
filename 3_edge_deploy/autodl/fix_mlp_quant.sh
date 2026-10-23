#!/usr/bin/env bash
# MLP 的 INT8 精度补救：把「量化算法」和「校准集截断」两个杠杆分开试，
# 好知道 AUPRC 那 0.0037 的缺口到底是谁造成的。
#
# 背景：板端任务卡 6 实测 判定一致率 99.998%（判据 >=99% 通过），
# 但 AUPRC 从浮点的 0.51740 掉到 0.48372，降幅 0.03368，超过 <=0.03 的判据。
# 用浮点 RKNN 复跑同一批数据：一致率 100%、logit 最大差仅 0.0027，
# 说明 NPU 后端和 ONNX->RKNN 链路都没问题，损失全部来自 INT8 量化本身。
#
# 根因：标准化后的特征是重尾的。训练集全局 absmax=104.4，而 p99 只有 3~4。
# 量化区间取校准集 min/max，区间被离群点撑大之后步长变粗
# （[-104,104] -> 0.819），99% 的数据挤在 ±4 以内，典型样本相对误差 ~20%。
set -uo pipefail
E=/root/autodl-tmp/edge
cd "$E/autodl"
PY=$E/venv-rknn/bin/python
ONNX=out/mlp_fraud/mlp_fraud.onnx
CALIB=out/mlp_fraud/calib_npy

say() { echo -e "\n########## [$(date +%T)] $* ##########"; }

run() {   # run <名字> <算法> <裁剪>
    say "转换 $1  算法=$2  裁剪=$3"
    "$PY" -u 04_convert_rknn.py --model mlp --onnx "$ONNX" \
        --calib-dir "$CALIB" --calib-limit 200 \
        --outdir "out/$1" --quant \
        --quant-algorithm "$2" --calib-clip "$3" \
        > "$E/logs/mlp_$1.log" 2>&1
    grep -hE '裁剪|校准清单|config|完成' "$E/logs/mlp_$1.log" | tail -4
}

run rknn_kl          kl_divergence 0
run rknn_clip16      normal        16
run rknn_clip16_kl   kl_divergence 16

say "全部产物"
ls -l out/rknn_kl/*.rknn out/rknn_clip16/*.rknn out/rknn_clip16_kl/*.rknn 2>/dev/null