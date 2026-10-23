#!/usr/bin/env bash
# 等补传的分片到齐 -> 合并 -> 校验 sha256 -> 解压胸片数据集
set -uo pipefail
E=/root/autodl-tmp/edge
SRC="D:/down/数据集/Chest X-Ray Images.zip"
EXPECT_SHA=f569fe885b0f921e836f3d6bcc8d7b3442f5e0ca4db4533d06b8cf25d2114ea1
NEED=460668898
TARGET=$(( 620691456 + 460668463 + 460668887 + 460667731 + NEED ))

echo "[$(date +%T)] 等待补传分片到齐（需要 $NEED 字节）"
for i in $(seq 1 240); do
    have=$(stat -c%s "$E/fix.bin" 2>/dev/null || echo 0)
    if [ "$have" -ge "$NEED" ]; then
        echo "[$(date +%T)] 分片到齐 $have 字节"
        break
    fi
    echo "[$(date +%T)] fix.bin 进度 $have / $NEED"
    sleep 60
done

have=$(stat -c%s "$E/fix.bin" 2>/dev/null || echo 0)
if [ "$have" -lt "$NEED" ]; then
    echo "等待超时，fix.bin 仍只有 $have 字节，未做合并"
    exit 1
fi

echo "[$(date +%T)] 合并 p0 p1 p2 p3 fix.bin -> Chest X-Ray Images.zip"
cd "$E"
# 注意：必须逐个列出。写成 p[0-9] 会把 p0 也匹配进去，前缀就会被拼两遍
cat p0 p1 p2 p3 fix.bin > merged.zip
SIZE=$(stat -c%s merged.zip)
echo "合并后 $SIZE 字节，期望 $TARGET"
if [ "$SIZE" -ne "$TARGET" ]; then
    echo "大小不符，保留 merged.zip 供排查"
    exit 1
fi

echo "[$(date +%T)] 校验 sha256"
SHA=$(sha256sum merged.zip | cut -d' ' -f1)
echo "远端 $SHA"
echo "本地 $EXPECT_SHA"
if [ "$SHA" != "$EXPECT_SHA" ]; then
    echo "sha256 不一致！文件损坏，不要用于训练"
    exit 1
fi
mv -f merged.zip "Chest X-Ray Images.zip"
rm -f p0 p1 p2 p3 fix.bin
echo "[$(date +%T)] 校验通过，清理分片完成"

echo "[$(date +%T)] 解压到 $E/autodl/data/"
mkdir -p "$E/autodl/data"
cd "$E/autodl/data"
unzip -q -o "$E/Chest X-Ray Images.zip"
# 官方包解压后根目录是 chest_xray（Windows 上可能是 Chest_xray），统一成小写
if [ -d Chest_xray ] && [ ! -d chest_xray ]; then
    mv Chest_xray chest_xray
fi
rm -f "$E/Chest X-Ray Images.zip"

echo "[$(date +%T)] 数据集结构"
ls "$E/autodl/data"
for split in train val test; do
    d="$E/autodl/data/chest_xray/$split"
    [ -d "$d" ] || continue
    n=$(find "$d" -type f \( -name '*.jpeg' -o -name '*.jpg' \) | wc -l)
    echo "  $split: $n 张"
    for cls in "$d"/*/; do
        [ -d "$cls" ] || continue
        c=$(find "$cls" -type f \( -name '*.jpeg' -o -name '*.jpg' \) | wc -l)
        echo "      $(basename "$cls"): $c"
    done
done
df -h /root/autodl-tmp | tail -1
echo "[$(date +%T)] 完成"