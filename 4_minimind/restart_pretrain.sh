#!/usr/bin/env bash
# 重启实验 03 预训练（加 -u 关闭缓冲，否则长跑看不见进度）
# 注意：不要用 `pkill -f train_pretrain.py`——那条命令自身的命令行里也含这个字符串，
# 会把执行它的 shell 一起杀掉（连接直接断开，exit 128）。只按进程名筛 python。
set -uo pipefail
MM=/root/autodl-tmp/minimind
PY=$MM/venv/bin/python
REPO=$MM/minimind

echo "--- 当前训练进程 ---"
for p in $(pgrep -x python 2>/dev/null); do
    if tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_pretrain; then
        echo "  kill $p"
        kill "$p" 2>/dev/null
    fi
done
sleep 5

cd "$REPO/trainer"
nohup "$PY" -u train_pretrain.py \
    --data_path ../dataset/pretrain_t2t_mini.jsonl \
    --max_seq_len 768 \
    --save_dir ../out \
    > "$MM/exp03_pretrain.log" 2>&1 &
echo "已重启，日志 $MM/exp03_pretrain.log"
sleep 3
pgrep -x python | while read -r p; do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q train_pretrain && echo "  运行中 PID $p"
done
exit 0