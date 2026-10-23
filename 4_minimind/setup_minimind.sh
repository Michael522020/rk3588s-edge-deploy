#!/usr/bin/env bash
# MiniMind 实验 01/02 的准备：独立 venv + 依赖 + 数据下载
# base conda 不动；torch 用镜像自带的（requirements.txt 里本就注释掉了 torch/peft）
set -euo pipefail
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
MM=/root/autodl-tmp/minimind
REPO=$MM/minimind
FILES="pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl dpo.jsonl lora_identity.jsonl lora_medical.jsonl"

say() { echo -e "\n===== [$(date +%T)] $* ====="; }

say "建 venv $MM/venv (继承镜像自带 torch)"
[ -d "$MM/venv" ] || /root/miniconda3/bin/python -m venv --system-site-packages "$MM/venv"
"$MM/venv/bin/pip" install -q -U pip wheel -i "$MIRROR"

say "装 requirements.txt（torch/torchvision/peft/matplotlib 已注释，保持不装）"
cd "$REPO"
"$MM/venv/bin/pip" install -q -r requirements.txt -i "$MIRROR"

say "关键版本确认"
"$MM/venv/bin/python" - <<'PY'
import torch, transformers, numpy, datasets
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}  gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"  transformers {transformers.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  datasets     {datasets.__version__}")
for m in ("modelscope", "swanlab", "flask", "streamlit", "fastapi", "trl"):
    try:
        mod = __import__(m); print(f"  {m:<12} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:<12} [FAIL] {e.__class__.__name__}: {e}")
PY

say "下载数据集到 $REPO/dataset（3.07 GB）"
mkdir -p "$REPO/dataset"
cd "$REPO"
"$MM/venv/bin/python" -m modelscope download --dataset gongjy/minimind_dataset \
    --include $FILES --local_dir "$REPO/dataset"

say "数据到位情况"
ls -l "$REPO/dataset"
du -sh "$REPO/dataset"
df -h /root/autodl-tmp | tail -1

say "完成"