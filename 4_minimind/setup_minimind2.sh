#!/usr/bin/env bash
# 接续 setup_minimind.sh：修 pyarrow/numpy 冲突，然后下载数据
# 冲突原因：requirements.txt 钉 numpy==1.26.4，但 pip 解析出的 pyarrow 26.0.0
#           硬性要求 numpy>=2.0，导致 `import datasets` 直接 ImportError。
# 处置：保留官方钉的 numpy==1.26.4，把 pyarrow 降到仍支持 numpy 1.x 的 18.1.0。
#       （已确认仓库代码没有 np.float/np.object 这类 numpy 1.x 废弃别名）
set -euo pipefail
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
MM=/root/autodl-tmp/minimind
REPO=$MM/minimind
PY=$MM/venv/bin/python
FILES="pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl dpo.jsonl lora_identity.jsonl lora_medical.jsonl"

say() { echo -e "\n===== [$(date +%T)] $* ====="; }

say "降 pyarrow 到 18.1.0（兼容 numpy 1.26.4）"
"$MM/venv/bin/pip" install -q "pyarrow==18.1.0" -i "$MIRROR"

say "依赖复查（这次要真能 import 成功）"
"$PY" - <<'PY'
import torch, transformers, numpy, datasets, pyarrow
print(f"  torch        {torch.__version__}  cuda={torch.cuda.is_available()}  "
      f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"  transformers {transformers.__version__}")
print(f"  numpy        {numpy.__version__}")
print(f"  pyarrow      {pyarrow.__version__}")
print(f"  datasets     {datasets.__version__}   <- 能 import 才算修好")
for m in ("modelscope", "swanlab", "flask", "streamlit", "fastapi", "trl", "sentence_transformers"):
    try:
        mod = __import__(m)
        print(f"  {m:<22} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:<22} [FAIL] {e.__class__.__name__}: {e}")
PY

say "下载数据集（3.07 GB，走 ModelScope）"
mkdir -p "$REPO/dataset"
cd "$REPO"
"$PY" -m modelscope download --dataset gongjy/minimind_dataset \
    --include $FILES --local_dir "$REPO/dataset"

say "数据到位"
ls -l "$REPO/dataset"
du -sh "$REPO/dataset"
df -h /root/autodl-tmp | tail -1
say "完成"