#!/usr/bin/env bash
# 无卡模式只有 2 GiB cgroup 内存，RKNN 转换器转 YOLO/CNN 会被 OOM 杀掉（退出码 137）。
# 这里三管齐下：先把 page cache 丢掉腾出 ~700 MB，再把图优化强度降下来，
# 校准图也减到 50 张（手册建议 100~200，但类别头只要覆盖到常见目标就够，
# YOLOv8n 那次栽的跟就是校准图只有 2 张，50 张远高于那个下限）。
set -uo pipefail
E=/root/autodl-tmp/edge
cd "$E/autodl"
PY_R=$E/venv-rknn/bin/python

say() { echo -e "\n########## [$(date +%T)] $* ##########"; }

free_mem() {
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    echo -n "  当前 cgroup 用量: "
    awk '{printf "%.0f MiB / 2048 MiB\n", $1/1048576}' /sys/fs/cgroup/memory.current
}

for OL in 1 0; do
    say "肺炎 CNN -> RKNN INT8  (optimization_level=$OL)"
    free_mem
    "$PY_R" -u 04_convert_rknn.py --model cnn \
        --onnx out/pneumonia/pneumonia.onnx \
        --calib-dir out/pneumonia/calib_images --calib-limit 50 \
        --outdir out/rknn --quant --optimization-level "$OL" \
        > "$E/logs/cnn_rknn_int8.log" 2>&1
    echo "  退出码 $?"
    tail -c 400 "$E/logs/cnn_rknn_int8.log" | tr '\r' '\n' | grep -vE '^\s*$' | tail -2
    if [ -f out/rknn/pneumonia.rknn ]; then
        echo "  >>> 成了：$(ls -lh out/rknn/pneumonia.rknn | awk '{print $5}')"
        break
    fi
done

for OL in 1 0; do
    say "YOLOv8n -> RKNN INT8  (optimization_level=$OL, 50 张 COCO 校准)"
    free_mem
    head -50 out/yolov8n/calib_list_200.txt > out/yolov8n/calib_list_50.txt
    "$PY_R" -u 05_convert_rknn_yolov8.py \
        --onnx out/yolov8n/yolov8n.onnx \
        --dataset out/yolov8n/calib_list_50.txt \
        --outdir out/rknn_yolo_fix --quant --optimization-level "$OL" \
        > "$E/logs/yolov8n_int8_fix.log" 2>&1
    echo "  退出码 $?"
    tail -c 400 "$E/logs/yolov8n_int8_fix.log" | tr '\r' '\n' | grep -vE '^\s*$' | tail -2
    if [ -f out/rknn_yolo_fix/yolov8n.rknn ]; then
        echo "  >>> 成了：$(ls -lh out/rknn_yolo_fix/yolov8n.rknn | awk '{print $5}')"
        break
    fi
done

say "产物"
ls -l out/rknn/*.rknn out/rknn_yolo_fix/*.rknn 2>/dev/null