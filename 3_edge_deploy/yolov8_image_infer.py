# -*- coding: utf-8 -*-
"""
文档3-①：YOLOv8 图片人员检测推理（板上）
─────────────────────────────────────────────────────────────────

板上已有 /home/ubuntu/yolov8n.pt（6.5 MB），无需下载权重。
自动从网上拉一张含人的测试图（COCO val2017），跑完输出 output.jpg

依赖（板上已装）：
  ultralytics 8.2.x、opencv-python、torch

运行：
  python3 yolov8_image_infer.py [--weights /home/ubuntu/yolov8n.pt] [--source URL_OR_PATH]
"""

import argparse, time, urllib.request, sys
from pathlib import Path

print("=" * 70)
print("YOLOv8 图片人员检测推理（板上）")
print("=" * 70)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERR] 缺 ultralytics: pip install -i https://mirrors.aliyun.com/pypi/simple/ ultralytics")
    sys.exit(1)


def fetch_image(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 1024:
        return True
    print(f"[get ] {url}")
    try:
        urllib.request.urlretrieve(url, dst)
        return dst.stat().st_size > 1024
    except Exception as e:
        print(f"[ERR ] 下载失败: {e}")
        return False


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/home/ubuntu/yolov8n.pt")
ap.add_argument(
    "--source",
    default="https://ultralytics-docs-images.s3.amazonaws.com/coco_datasets/coco_val2017_000000001.jpg",
)
ap.add_argument("--out", default="output.jpg")
args = ap.parse_args()

w = Path(args.weights)
if not w.exists():
    print(f"[ERR] 权重文件不存在: {w}")
    print("      请先确认 /home/ubuntu/yolov8n.pt，或用 --weights 指定别的路径")
    sys.exit(1)

src = Path(args.source)
if not src.exists() and args.source.startswith("http"):
    cache = Path("./_test.jpg")
    if not fetch_image(args.source, cache):
        print("[ERR] 无法获取 source, 用 --source 指定本地图片")
        sys.exit(1)
    src = cache

print(f"[info] weights = {w}  size = {w.stat().st_size/1024:.0f} KB")
print(f"[info] source  = {src}")

print("\n[load] YOLO ...")
t0 = time.time()
model = YOLO(str(w))
print(f"      加载完成  (用时 {time.time()-t0:.1f}s)")
print(f"      类别 (person=0): {model.names}")

print("\n[infer] ...")
t0 = time.time()
results = model(str(src), conf=0.25, classes=[0])  # 仅保留 person
dt = time.time() - t0
r = results[0]
n = len(r.boxes)
print(f"      检测到 {n} 个 person  (用时 {dt:.2f}s)")

for i, b in enumerate(r.boxes):
    cls = int(b.cls.item()); conf = float(b.conf.item()); xyxy = b.xyxy[0].tolist()
    print(f"      [{i}] class={model.names[cls]}  conf={conf:.3f}  xyxy=({int(xyxy[0])},{int(xyxy[1])},{int(xyxy[2])},{int(xyxy[3])})")

out_path = Path(args.out).resolve()
r.save(filename=str(out_path))
print(f"\n[save] {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")

if n == 0:
    print("\n[hint] 这张图里没有 person，可以换一张有人的图试试。试试：")
    print("       --source https://ultralytics-docs-images.s3.amazonaws.com/0_jpg.jpg")