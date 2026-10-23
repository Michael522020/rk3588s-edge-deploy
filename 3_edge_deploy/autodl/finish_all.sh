#!/usr/bin/env bash
# 收尾链：CNN 重跑 -> MiniMind 05 推理验证 -> 06 LoRA -> 07 DPO -> 08 权重转换
set -uo pipefail
E=/root/autodl-tmp/edge
MM=/root/autodl-tmp/minimind
REPO=$MM/minimind
PY=$MM/venv/bin/python
PY_T=$E/venv-train/bin/python
PY_R=$E/venv-rknn/bin/python

say() { echo -e "\n########## [$(date +%T)] $* ##########"; }

# ============================================================ 1. 肺炎 CNN 重跑
say "1/6 肺炎 CNN 重跑（TF32 已关）"
cd "$E/autodl"
"$PY_T" -u 02_cnn_pneumonia_train_export.py --data "$E/autodl/data/chest_xray" \
    --device cuda --batch 64 --workers 32 \
    --epochs-head 3 --epochs-finetune 10 --outdir out/pneumonia \
    > "$E/logs/cnn.log" 2>&1
grep -E '准确率|查全率|最大误差|ok\]|完成' "$E/logs/cnn.log" | tail -8
if [ -f out/pneumonia/pneumonia.onnx ]; then
    "$PY_R" -u 04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
        --calib-dir out/pneumonia/calib_images --calib-limit 200 \
        --outdir out/rknn > "$E/logs/cnn_rknn_fp.log" 2>&1
    "$PY_R" -u 04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
        --calib-dir out/pneumonia/calib_images --calib-limit 200 \
        --outdir out/rknn --quant > "$E/logs/cnn_rknn_int8.log" 2>&1
    grep -h '完成' "$E/logs/cnn_rknn_fp.log" "$E/logs/cnn_rknn_int8.log" | tail -2
fi

# ============================================================ 2. 实验 05 推理验证
say "2/6 实验 05 · SFT 推理验证（与预训练的'答非所问'对比）"
cd "$REPO"
echo 0 | "$PY" -u eval_llm.py --weight full_sft > "$MM/exp05_eval_sft.log" 2>&1
grep -E '💬|🧠|Speed|Model Params' "$MM/exp05_eval_sft.log" | head -12

# ============================================================ 3. 实验 06 LoRA
say "3/6 实验 06 · LoRA 垂域微调（lora_identity）"
cd "$REPO/trainer"
"$PY" -u train_lora.py --lora_name lora_identity --from_weight full_sft \
    --data_path ../dataset/lora_identity.jsonl --save_dir ../out \
    > "$MM/exp06_lora.log" 2>&1
grep -iE 'lora|占比|trainable|Model Params' "$MM/exp06_lora.log" | head -8
ls -l "$REPO/out/"lora* 2>/dev/null

# ============================================================ 4. LoRA 后推理对比
say "4/6 LoRA 后再问一遍「你叫什么名字」"
cd "$REPO"
echo 0 | "$PY" -u eval_llm.py --weight full_sft --lora_weight lora_identity \
    > "$MM/exp06_eval_lora.log" 2>&1
grep -E '💬|🧠' "$MM/exp06_eval_lora.log" | head -4

# ============================================================ 5. 实验 07 DPO
say "5/6 实验 07 · DPO 偏好对齐（--log_interval 1 才能看到 0.6931 锚点）"
cd "$REPO/trainer"
"$PY" -u train_dpo.py --data_path ../dataset/dpo.jsonl --from_weight full_sft \
    --log_interval 1 --save_dir ../out > "$MM/exp07_dpo.log" 2>&1
head -5 "$MM/exp07_dpo.log"
grep -E 'dpo_loss' "$MM/exp07_dpo.log" | head -4
ls -l "$REPO/out/"dpo* 2>/dev/null

# ============================================================ 6. 实验 08 权重转换
say "6/6 实验 08 · 权重转换（transformers 格式）"
cd "$REPO/scripts"
"$PY" -u convert_model.py ../out/full_sft_768.pth ../out/minimind-3-sft \
    > "$MM/exp08_convert.log" 2>&1
tail -5 "$MM/exp08_convert.log"
ls -l "$REPO/out/minimind-3-sft" 2>/dev/null | head

say "收尾链完成"
ls -l "$E/autodl/out/rknn/"*.rknn "$REPO/out/"*.pth 2>/dev/null
df -h /root/autodl-tmp | tail -1