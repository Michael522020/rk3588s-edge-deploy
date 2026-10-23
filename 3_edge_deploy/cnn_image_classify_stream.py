# -*- coding: utf-8 -*-
"""
板子端：从 Win11 摄像头拉 MJPEG 流 + MobileNetV2 (ImageNet) 图像分类
========================================================================

完整对应文档3·实验二 (CNN 图像分类)：用 MobileNetV2 迁移学习的卷积网络，
对实时视频流做 1000 类 ImageNet 分类（摄像头实际拍的是 RGB 场景，不是 X 光，
用预训练权重做 1000 类物体识别作为 demo）。

板上可行性：
  ✅ MobileNetV2 ImageNet 预训练（torchvision 自带，板上不需训练）
  ✅ cv2.VideoCapture 拉 HTTP MJPEG 流
  ⚠️  CPU 推理 ~200-400 ms/帧
  ⚠️  无桌面 → 结果写到 classify.mp4 文件，scp 回 Win11 看

用法：
  # 1) 先在 Win11 上跑 win11_camera_stream.py（推流）
  # 2) 板子端（默认 source 指向 Win11 笔记本的 IP 192.168.1.2）
  python3 cnn_image_classify_stream.py
  python3 cnn_image_classify_stream.py --source http://192.168.1.2:8080/stream
  python3 cnn_image_classify_stream.py --max-frames 100 --no-record
"""

import argparse
import time
import urllib.request

import cv2
import torch
import torchvision.models as models
from PIL import Image


def probe_source(url: str) -> bool:
    """先 HEAD 一次流，避免 cap.read() 死等。"""
    print(f"[probe] GET {url}")
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as r:
            ct = r.headers.get("Content-Type", "")
            print(f"        HTTP {r.status}  Content-Type={ct}")
            if r.status == 200 and "multipart" in ct:
                # 读一帧验证流
                boundary = ct.split("boundary=")[-1].split(";")[0].strip()
                buf = b""
                while True:
                    chunk = r.read(1024)
                    if not chunk:
                        break
                    buf += chunk
                    if b"--" + boundary.encode() in buf and buf.endswith(b"\r\n\r\n"):
                        break
                print(f"        拉取首段 {len(buf)} bytes OK")
                return True
            return False
    except Exception as e:
        print(f"        ERR: {e.__class__.__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        default="http://192.168.1.2:8080/stream",
        help="Win11 MJPEG 流 URL",
    )
    ap.add_argument("--out", default="classify.mp4", help="输出 mp4（板子上无桌面看不了实时窗口）")
    ap.add_argument("--max-frames", type=int, default=600)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--no-record", action="store_true", help="不写视频文件")
    ap.add_argument("--write-fps", type=float, default=10.0)
    args = ap.parse_args()

    print("=" * 70)
    print("板子端：摄像头流 + CNN 图像分类（MobileNetV2 ImageNet 1000 类）")
    print("=" * 70)
    print(f"[src ] {args.source}")
    print(f"[out ] {args.out}  (no-record={args.no_record})")

    if not probe_source(args.source):
        print("[!] source 暂时拉不到——继续尝试 cv2.VideoCapture（有时 Windows 防火墙会放过这个）")

    print("\n[load] MobileNetV2 (ImageNet) ...", flush=True)
    t0 = time.time()
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model = models.mobilenet_v2(weights=weights).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量: {n_params/1e6:.2f}M  (用时 {time.time()-t0:.1f}s)")
    print(f"      类别数: {len(categories)}")

    print(f"\n[open] cv2.VideoCapture({args.source})")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERR] 打不开流 —— 检查 Win11 防火墙是否放行 python.exe")
        return

    W_out = 640
    H_out = 480
    fps_out = args.write_fps
    writer = None
    if not args.no_record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, fps_out, (W_out, H_out))
        print(f"[rec ] {args.out}  fps={fps_out}  {W_out}x{H_out}")

    print("\n[run ] 推流 + 推理（按 Ctrl+C 停止）\n")
    t_start = time.time()
    n = 0
    last_label = ""
    try:
        while n < args.max_frames:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                print(f"[{n:4d}] 拉流失败（网络问题或 Win11 推流停止），重试 ...")
                time.sleep(0.5)
                continue

            if frame.shape[1] != W_out or frame.shape[0] != H_out:
                frame = cv2.resize(frame, (W_out, H_out))

            # 推理（torchvision transforms 需要 PIL Image，所以先把 ndarray 转 PIL）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = preprocess(pil_img).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
            probs = logits.softmax(dim=1)[0]
            top = torch.topk(probs, args.topk)

            lines = []
            for i in range(args.topk):
                cls = categories[top.indices[i].item()]
                conf = top.values[i].item()
                lines.append(f"{i+1}. {cls:25s} {conf:.2%}")
            last_label = lines[0]

            # 顶部黑底画 top-k
            overlay_h = 28 * (args.topk + 1) + 6
            cv2.rectangle(frame, (0, 0), (W_out, overlay_h), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line, (10, 24 * (i + 1) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA,
                )

            # 底部统计
            fps_now = 1.0 / max(time.time() - t0, 1e-6)
            cv2.putText(
                frame, f"frame {n+1}  fps~{fps_now:.1f}", (10, H_out - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            n += 1
            dt = time.time() - t0
            if n <= 5 or n % 10 == 0:
                print(f"[{n:4d}] {last_label}  ({dt*1000:.0f} ms, ~{fps_now:.1f} fps)")

            # 节流到 ~10 fps
            if dt < 0.10:
                time.sleep(0.10 - dt)

    except KeyboardInterrupt:
        print("\n[stop] 用户中断")

    elapsed = time.time() - t_start
    print(f"\n[done] {n} 帧, 用时 {elapsed:.1f}s, 平均 {n/elapsed:.2f} fps")
    if writer is not None:
        writer.release()
    cap.release()
    print(f"[save] {args.out}")
    print(f"\n[hint] 把视频拷回 Win11：")
    print(f"       scp ubuntu@192.168.1.100:~/work/20261013/3_edge_deploy/{args.out} .")


if __name__ == "__main__":
    main()