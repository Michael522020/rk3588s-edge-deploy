# -*- coding: utf-8 -*-
"""
文档3-②：YOLOv8 视频人员检测推理（板上）
─────────────────────────────────────────────────────────────────

板上无 USB 摄像头 → 用本地视频文件代替（自动下载示例 mp4）。
输入：--source URL 或本地 mp4
输出：--out output.mp4（带检测框）
边缘效果：仅检测 person (class 0)

依赖（板上已装）：
  ultralytics、opencv-python、torch

运行：
  python3 yolov8_video_infer.py
"""

import argparse, urllib.request, time
from pathlib import Path

print("=" * 70)
print("YOLOv8 视频人员检测推理（板上，无摄像头用本地视频替代）")
print("=" * 70)

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    print(f"[ERR] 缺包: {e}")
    print("      pip install -i https://mirrors.aliyun.com/pypi/simple/ ultralytics opencv-python")
    raise SystemExit(1)


def fetch_video(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 1024:
        return True
    print(f"[get ] {url} -> {dst}")
    try:
        urllib.request.urlretrieve(url, dst)
        return dst.stat().st_size > 1024
    except Exception as e:
        print(f"[ERR ] {e}")
        return False


ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="/home/ubuntu/yolov8n.pt")
ap.add_argument(
    "--source",
    default="https://ultralytics-docs-images.s3.amazonaws.com/coco_datasets/coco_1280.mp4",
    help="视频源 (URL 或本地路径)；默认 COCO 样例 30s 行人片段",
)
ap.add_argument("--out", default="output.mp4")
ap.add_argument("--max-frames", type=int, default=120, help="最多处理多少帧（默认 120 ≈ 4 秒）")
args = ap.parse_args()

w = Path(args.weights)
src = Path(args.source)
if not src.exists() and args.source.startswith("http"):
    cache = Path("./_sample.mp4")
    if not fetch_video(args.source, cache):
        raise SystemExit("[ERR] 无法获取 source")
    src = cache

if not w.exists():
    raise SystemExit(f"[ERR] 权重不存在: {w}")

print(f"[info] weights = {w}")
print(f"[info] source  = {src}  size = {src.stat().st_size/1024:.0f} KB")

cap = cv2.VideoCapture(str(src))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[info] video {W}x{H} {fps:.1f}fps  {total} frames")
print(f"[info] 将处理前 {args.max_frames} 帧")

print("\n[load] YOLO ...")
model = YOLO(str(w))

print("[run ] 推理 ...")
t0 = time.time()
frames = 0
person_count_total = 0
for res in model.predict(source=str(src), stream=True, conf=0.25, classes=[0],
                          max_det=20, verbose=False):
    frames += 1
    n = len(res.boxes)
    person_count_total += n
    if frames % 20 == 0:
        print(f"      帧 {frames}: 检测到 {n} person")
    if frames >= args.max_frames:
        break
dt = time.time() - t0
fps_real = frames / dt
print(f"\n[done] 处理 {frames} 帧  用时 {dt:.1f}s  实测 {fps_real:.1f} fps")
print(f"[stat] 总检出 person 框 {person_count_total} 个, 平均 {person_count_total/frames:.1f} 个/帧")

# 用 cv2 写一份带标注的 mp4（慢，但确认渲染）
print("\n[save] 写带标注的视频 ...")
cap.release()
cap = cv2.VideoCapture(str(src))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_path = Path(args.out).resolve()
writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
t0 = time.time()
i = 0
for res in model.predict(source=str(src), stream=True, conf=0.25, classes=[0],
                          max_det=20, verbose=False):
    i += 1
    frame = res.plot()  # 带框的 ndarray
    writer.write(frame)
    if i >= args.max_frames:
        break
writer.release()
cap.release()
print(f"[done] 写出 {i} 帧, 用时 {time.time()-t0:.1f}s, file {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
print("\n[hint] 把 output.mp4 拷回 Win11 用 scp / WinSCP 即可查看：")
print(f"       scp ubuntu@192.168.1.100:{out_path} .")