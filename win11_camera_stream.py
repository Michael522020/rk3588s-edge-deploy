# -*- coding: utf-8 -*-
"""
Win11 端摄像头推流服务（MJPEG over HTTP）
=============================================================

把笔记本内置摄像头以 MJPEG 流形式推到局域网。板子用 cv2.VideoCapture 拉流。

用法（Win11 上跑）：
  python3 win11_camera_stream.py                   # 默认 cam 0, 640x480, port 8080
  python3 win11_camera_stream.py --list-cams      # 查看可用摄像头编号
  python3 win11_camera_stream.py --cam 1          # 用外接 USB 摄像头
  python3 win11_camera_stream.py --port 9000      # 改端口

依赖：
  opencv-python  (Win11 已确认 cv2 5.0.0 OK)

防火墙：
  第一次运行 Windows 会问"是否允许 Python 通过防火墙"——必须选 **是**
  (否则板子拉流会一直被 Windows Defender 拦截)

输出示例（在 Win11 终端）：

  ========== MJPEG 推流服务 ==========
  Win11 IP: 192.168.1.2
  URL:      http://192.168.1.2:8080/stream
  板子端用 cv2.VideoCapture(<URL>) 拉流
  ====================================
"""

import argparse
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


class State:
    cap = None
    width = 640
    height = 480
    quality = 70
    fps_limit = 0.05   # 50ms 一帧 ≈ 20fps 上限


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/stream":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("Use /stream\n".encode())
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        cap = State.cap
        if cap is None or not cap.isOpened():
            return

        try:
            while True:
                t0 = time.time()
                ok, buf = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                ok, jpg = cv2.imencode(".jpg", buf, [cv2.IMWRITE_JPEG_QUALITY, State.quality])
                if not ok:
                    continue
                self.wfile.write(b"--frame\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpg)))
                self.end_headers()
                self.wfile_write_safe(jpg.tobytes())
                self.wfile_write_safe(b"\r\n")
                # 限速到 ~20 fps
                dt = time.time() - t0
                if dt < State.fps_limit:
                    time.sleep(State.fps_limit - dt)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            return

    def wfile_write_safe(self, b):
        try:
            self.wfile.write(b)
        except (BrokenPipeError, ConnectionResetError):
            raise

    def log_message(self, fmt, *args):
        # 静默默认日志（按需可改成 print 调试）
        pass


def list_local_ips():
    ips = []
    try:
        hostname = socket.gethostname()
        all_ips = socket.gethostbyname_ex(hostname)[2]
        ips = [ip for ip in all_ips if not ip.startswith("127.")]
    except Exception:
        pass
    # 再补一个直连 socket 试探
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.insert(0, s.getsockname()[0])
        s.close()
    except Exception:
        pass
    # 去重保序
    seen, out = set(), []
    for ip in ips:
        if ip not in seen:
            out.append(ip); seen.add(ip)
    return out


def open_cam(idx, w, h):
    # 用 DirectShow 后端更稳（Win 上）
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--quality", type=int, default=70)
    ap.add_argument("--fps-limit", type=float, default=20.0)
    ap.add_argument("--list-cams", action="store_true")
    args = ap.parse_args()

    if args.list_cams:
        print("[scan] 检测 0~3 号摄像头 ...")
        for i in range(4):
            c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not c.isOpened():
                c = cv2.VideoCapture(i)
            ok = c.isOpened()
            if ok:
                w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"  cam[{i}] OK ({w}x{h})")
            else:
                print(f"  cam[{i}] 不可用")
            c.release()
        return

    cap = open_cam(args.cam, args.width, args.height)
    if cap is None:
        print(f"[ERR] 摄像头 {args.cam} 打不开")
        print("       试试 --list-cams 看哪些摄像头可用，或 --cam 1 用外接 USB 摄像头")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[cam] opened cam[{args.cam}]: {actual_w}x{actual_h}")

    State.cap = cap
    State.width = actual_w
    State.height = actual_h
    State.quality = args.quality
    State.fps_limit = 1.0 / max(args.fps_limit, 0.5)

    server = ThreadingHTTPServer(("0.0.0.0", args.port), MJPEGHandler)

    ips = list_local_ips()
    print()
    print("=" * 60)
    print(" MJPEG 推流服务 (Win11)")
    print("=" * 60)
    for ip in ips:
        print(f"   http://{ip}:{args.port}/stream")
    print("=" * 60)
    print(" 板子端拉流命令 (在 192.168.1.100 上跑):")
    for ip in ips:
        print(f"   python3 ~/work/20261013/3_edge_deploy/cnn_image_classify_stream.py \\")
        print(f"         --source http://{ip}:{args.port}/stream")
        break
    print(" 按 Ctrl+C 停止")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[stream] stopping")
    finally:
        cap.release()
        server.server_close()


if __name__ == "__main__":
    main()