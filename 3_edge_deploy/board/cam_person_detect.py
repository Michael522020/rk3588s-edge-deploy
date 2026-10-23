#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 14 —— 鲁班猫 USB 摄像头实时人员检测（只保留 person 类）

在 RK3588S 鲁班猫上运行：打开 USB 摄像头，逐帧送入 RKNN 模型，画面里只框出人。

手册要点（对应验收判据）：
  先用 ls /dev/video* 与 v4l2-ctl --list-devices 确认设备编号与支持格式
  用 OpenCV 打开摄像头时显式指定 V4L2 后端，先设 MJPG 编码再设分辨率；
    部分摄像头不支持 MJPG 会回退到 YUYV，打印返回的四字符编码确认即可，不必强求
  循环内依次做：读帧 -> 保持宽高比缩放到 640x640 并填充 -> 转 RGB -> NPU 推理
    -> 解码与 NMS -> 只保留类别号 0 的框 -> 按缩放与填充把坐标映射回原图
    -> 画框并标注置信度
  循环保持轻量，不要每帧写文件或打印日志
  退出时释放摄像头并关闭窗口，打印平均帧率与单帧耗时
  验收：画面能实时显示并稳定框出人员，杯子、椅子等非人目标不出现框；
        退出后设备正常释放、可再次打开；平均帧率与单帧耗时记入基准表

两个必须注意的坑（手册提示）：
  1) 无桌面环境时窗口弹不出来 —— 改用 --save-video 把结果存成文件再拷回本地看
  2) RKNN 转换时归一化已按官方示例烘焙进模型（mean=0, std=255），
     所以板端输入是 RGB、float32、0~255 原值，**不要再除以 255**

用法：
    python3 cam_person_detect.py --rknn models/yolov8n.rknn --device 0
    python3 cam_person_detect.py --rknn models/yolov8n.rknn --device 0 \
        --save-video out/person.mp4          # 无桌面环境
    python3 cam_person_detect.py --rknn models/yolov8n.rknn --device 0 --headless
"""

import argparse
import os
import sys
import time

import numpy as np

PERSON_CLASS = 0  # COCO 八十类中 person 是第 0 类
INPUT_SIZE = 640  # 640 已是 16 的倍数，不会触发 RGA 宽高步长错误
BOXES, CLASSES = 4, 80


# --------------------------------------------------------------------------
# 摄像头
# --------------------------------------------------------------------------
def open_camera(source, width, height, fourcc="MJPG"):
    """打开视频源。

    source 以 http/https 开头 -> 拉网络流（Win11 笔记本摄像头推的 MJPEG）。
        MJPEG 流自带分辨率与编码，fourcc/宽高设置对其无效，直接读。
    否则视为本地设备编号 / 显式 V4L2 后端；
        先设 MJPG 再设分辨率，MJPG 不支持时回退 YUYV 也能用。
    """
    import cv2

    if source.startswith(("http://", "https://", "rtsp://")):
        print(f"  [网络流] {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(
                f"拉流失败: {source}\n"
                f"  1) 笔记本上是否已经跑起推流脚本？\n"
                f"  2) Windows 防火墙是否放行了那个端口（第一次运行会弹窗询问）\n"
                f"  3) 两者是否在同一网段（板子 {__import__('socket').gethostbyname(__import__('socket').gethostname())}）"
            )
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise SystemExit(f"连上了但读不出帧，检查推流端是否在正常采集")
        h, w = frame.shape[:2]
        print(f"  [网络流] 首帧 {w}x{h}，编解码 {int(cap.get(cv2.CAP_PROP_FOURCC)):08x}")
        return cap

    if os.path.isfile(source):
        # 这块板子本身没有 UVC 摄像头，但模型链路（RKNN 加载、NPU 推理、
        # 后处理解码）仍然要能验证——否则只能等有人插摄像头才跑得起来。
        print(f"  [本地文件] {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"打不开视频/图片文件: {source}")
        # 尺寸只能从属性取，不能 cap.read() 探测：图片只有一帧，探测读完就没了；
        # 视频则是读完再 set(CAP_PROP_POS_FRAMES, 0) 也回不来（实测 read -> set
        # -> read 第二次直接 False），主循环会一帧都拿不到。
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise SystemExit(f"读不出 {source} 的画面尺寸，文件可能已损坏")
        print(f"  [本地文件] {w}x{h}，总帧数 {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
        return cap

    device = int(source)
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"打不开 /dev/video{device}。先用 ls /dev/video* 确认编号，"
                         f"或检查是否插了 UVC 免驱摄像头、是否在 video 组。"
                         f"若板子没接摄像头，可用 --source http://<笔记本IP>:8080/stream 拉流。")

    want = cv2.VideoWriter_fourcc(*fourcc)
    cap.set(cv2.CAP_PROP_FOURCC, want)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    got = int(cap.get(cv2.CAP_PROP_FOURCC))
    got_name = "".join([chr((got >> (8 * i)) & 0xFF) for i in range(4)])
    if got_name != fourcc:
        print(f"  [提示] 摄像头不支持 {fourcc}，实际回退为 {got_name}；不影响使用")
    else:
        print(f"  编码格式 {got_name}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  分辨率 {w}x{h}")
    if w == 0 or h == 0:
        print("  [警告] 分辨率读取为 0，摄像头可能未正常出流")
    return cap


# --------------------------------------------------------------------------
# 预处理
# --------------------------------------------------------------------------
def letterbox(img, size=INPUT_SIZE, pad=114):
    """保持宽高比缩放并填充到 size x size，返回图与缩放/填充参数。"""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), pad, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def to_input(canvas):
    """BGR -> RGB，HWC -> NCHW，float32，0~255 原值（归一化已烘焙进 RKNN 模型）。"""
    rgb = canvas[:, :, ::-1]
    chw = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))
    return chw[None].astype(np.float32)


def scale_boxes(boxes, r, left, top, w0, h0):
    """把 640x640 画布上的坐标映射回原始帧尺寸。"""
    out = boxes.copy()
    out[:, [0, 2]] -= left
    out[:, [1, 3]] -= top
    out[:, [0, 2]] /= r
    out[:, [1, 3]] /= r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out


# --------------------------------------------------------------------------
# 解码
# --------------------------------------------------------------------------
def nms(boxes, scores, classes, thres):
    """按类别分别做 NMS。"""
    out = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        b, s = boxes[idx], scores[idx]
        order = s.argsort()[::-1]
        while len(order):
            i = order[0]
            out.append((b[i], s[i], c))
            if len(order) == 1:
                break
            xx1 = np.maximum(b[i, 0], b[order[1:], 0])
            yy1 = np.maximum(b[i, 1], b[order[1:], 1])
            xx2 = np.minimum(b[i, 2], b[order[1:], 2])
            yy2 = np.minimum(b[i, 3], b[order[1:], 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            area_i = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1])
            area_o = ((b[order[1:], 2] - b[order[1:], 0]) *
                      (b[order[1:], 3] - b[order[1:], 1]))
            order = order[1:][inter / np.maximum(area_i + area_o - inter, 1e-9) <= thres]
    if not out:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,)), np.zeros((0,))
    return (np.array([o[0] for o in out], dtype=np.float32),
            np.array([o[1] for o in out], dtype=np.float32),
            np.array([o[2] for o in out], dtype=np.int32))


def postprocess(raw_outputs, conf_thres, nms_thres):
    """解码 + NMS + 只保留 person 类，返回可直接绘制的 (N,5) xyxy+conf。

    兼容两种输出切分：
      * 单输出 (1, 84, 8400)      —— 直接可用
      * 官方 rknn_model_zoo 切分后的多输出，按通道数拼回 84 通道
    拼接方式是把每个输出 reshape 成 (C, -1) 后按序相接，因此
    boxes 与 scores 谁在前由实际张量形状决定，不靠猜。
    若通道数对不上，会明确提示转换脚本与解码器不匹配，
    而不是默默输出错误结果。
    """
    parts = []
    for out in raw_outputs:
        a = np.asarray(out)
        if a.ndim == 3:
            a = a[0]
        parts.append(a.reshape(a.shape[0], -1))
    pred = np.concatenate(parts, axis=0)          # (84, anchors)
    pred = pred.T                                  # (anchors, 84)

    if pred.shape[1] != BOXES + CLASSES:
        raise SystemExit(
            f"输出通道数 {pred.shape[1]} 不符合预期（应为 {BOXES + CLASSES}）。"
            f"这通常意味着转换脚本与本解码器不匹配 —— "
            f"请用 rknn_model_zoo 的 yolov8 转换脚本（任务卡 13）。"
        )

    xywh, scores = pred[:, :BOXES], pred[:, BOXES:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = (conf >= conf_thres) & (cls == PERSON_CLASS)   # 只保留类别号 0
    xywh, conf = xywh[keep], conf[keep]
    if len(xywh) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    boxes = np.empty((len(xywh), 4), dtype=np.float32)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    b, s, _ = nms(boxes, conf, np.zeros(len(conf), dtype=np.int32), nms_thres)
    if not len(b):
        return np.zeros((0, 5), dtype=np.float32)
    return np.column_stack([b, s])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", default="models/yolov8n.rknn")
    ap.add_argument("--source", default="0",
                    help="视频源：本地设备编号 0，本地视频/图片文件路径，"
                         "或网络流地址 http://<笔记本IP>:8080/stream")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--headless", action="store_true", help="无桌面环境，不弹窗")
    ap.add_argument("--save-video", help="把带框画面存成 mp4（无桌面环境用）")
    ap.add_argument("--save-shot", default=None, help="检测到人时抓一帧存成 jpg")
    ap.add_argument("--frames", type=int, default=0, help="跑多少帧后退出；0 = 一直跑")
    ap.add_argument("--report", default="out/board_yolov8_report.json")
    args = ap.parse_args()

    import cv2
    from rknnlite.api import RKNNLite

    print("=== 任务卡 14 · 鲁班猫 USB 摄像头实时人员检测 ===")
    if not os.path.exists(args.rknn):
        raise SystemExit(f"找不到 RKNN 模型: {args.rknn}")

    print("\n[1/4] 检查视频源")
    is_net = args.source.startswith(("http://", "https://", "rtsp://"))
    if is_net:
        print(f"  模式: 网络拉流  <-  {args.source}")
    else:
        vids = sorted(int(v[6:]) for v in os.listdir("/dev")
                      if v.startswith("video") and v[6:].isdigit())
        print(f"  模式: 本地摄像头  可用 /dev/video* -> "
              f"{[f'/dev/video{v}' for v in vids] or '无'}")
        if vids and int(args.source) not in vids:
            raise SystemExit(f"--source {args.source} 不在可用列表里，请改成 {vids[0]}；"
                             f"若板子没接摄像头，可用 --source http://<笔记本IP>:8080/stream")

    print("\n[2/4] 加载 RKNN 模型")
    rknn = RKNNLite(verbose=False)
    if rknn.load_rknn(args.rknn) != 0:
        raise SystemExit("load_rknn 失败")
    if rknn.init_runtime() != 0:
        raise SystemExit(
            "init_runtime 失败。先跑 00_board_env_check.py 看 NPU 节点诊断。"
        )
    print("  init_runtime 完成")

    print("\n[3/4] 打开视频源")
    cap = open_camera(args.source, args.width, args.height, args.fourcc)

    writer = None
    if args.save_video:
        os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)
        writer = cv2.VideoWriter(args.save_video,
                                 cv2.VideoWriter_fourcc(*"mp4v"), 15, (args.width, args.height))
        print(f"  结果写入 {args.save_video}")

    print(f"\n[4/4] 主循环  conf={args.conf}  nms={args.nms}  "
          f"只保留类别号 {PERSON_CLASS}(person)")
    print("  Ctrl-C 退出\n")

    n_frames = 0
    n_hit = 0
    lat_inf, t_start = [], time.perf_counter()
    shot_saved = False
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if os.path.isfile(args.source):
                    # 文件读到尾是正常结束，不是故障——图片只有 1 帧，视频播完也没了
                    print("  已读到文件末尾，正常结束")
                else:
                    print("  [警告] 读帧失败，摄像头可能断开")
                break
            h0, w0 = frame.shape[:2]

            canvas, r, left, top = letterbox(frame)
            t0 = time.perf_counter()
            outs = rknn.inference(inputs=[to_input(canvas)])
            lat_inf.append((time.perf_counter() - t0) * 1000.0)

            det = postprocess(outs, args.conf, args.nms)
            if len(det):
                n_hit += 1
                boxes = scale_boxes(det[:, :4], r, left, top, w0, h0)
                for b, c in zip(boxes, det[:, 4]):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"person {c:.2f}", (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                if args.save_shot and not shot_saved:
                    cv2.imwrite(args.save_shot, frame)
                    shot_saved = True
                    print(f"  已抓帧 -> {args.save_shot}")

            if writer:
                writer.write(frame)

            cv2.putText(frame, f"person: {len(det)}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if not args.headless:
                cv2.imshow("RK3588S person detect", frame)
                if cv2.waitKey(1) & 0xFF == 27:   # ESC 退出
                    break
            elif n_frames % 30 == 0:
                # 无桌面时也要能确认还活着，但不要每帧刷屏
                fps = n_frames / max(time.perf_counter() - t_start, 1e-9)
                print(f"  已处理 {n_frames} 帧  {fps:.1f} fps  本帧检出 {len(det)}", flush=True)

            n_frames += 1
            if args.frames and n_frames >= args.frames:
                break
    except KeyboardInterrupt:
        print("\n  收到 Ctrl-C，准备退出")
    finally:
        cap.release()                       # 退出时释放摄像头，可再次打开
        if writer:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        rknn.release()
        print("\n  摄像头、窗口、NPU 资源均已释放")

    elapsed = time.perf_counter() - t_start
    fps = n_frames / max(elapsed, 1e-9)
    lat = np.array(lat_inf) if lat_inf else np.zeros(1)

    print("\n=== 性能基准（填入第五章表格「YOLOv8n / 鲁班猫 NPU」行）===")
    print(f"  处理帧数   : {n_frames}")
    print(f"  平均帧率   : {fps:.2f} fps")
    print(f"  单帧 NPU 耗时: 均值 {lat.mean():.2f} ms  中位 {np.median(lat):.2f} ms  "
          f"P99 {np.percentile(lat, 99):.2f} ms")
    print(f"  检出率     : {n_hit}/{n_frames} 帧检出人员 "
          f"({n_hit / max(n_frames, 1):.1%})")
    print(f"  备注       : USB 摄像头实时人员检测，仅 person 类")

    import json

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as w:
        json.dump({
            "frames": n_frames,
            "fps": round(fps, 3),
            "latency_ms_mean": round(float(lat.mean()), 3),
            "latency_ms_median": round(float(np.median(lat)), 3),
            "latency_ms_p99": round(float(np.percentile(lat, 99)), 3),
            "frames_with_person": n_hit,
            "resolution": [args.width, args.height],
            "fourcc": args.fourcc,
            "note": "USB 摄像头实时人员检测，仅 person 类",
        }, w, indent=2, ensure_ascii=False)
    print(f"\n[完成] 报告 -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())