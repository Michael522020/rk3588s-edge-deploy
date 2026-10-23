#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务卡 12 —— 实验三：YOLOv8 —— 导出 ONNX（不训练，直接用官方 COCO 预训练权重）

手册要点（对应验收判据）：
  加载官方 COCO 预训练权重，不做任何训练
  输入尺寸 640，输入名 images（后续 RKNN / OM 转换脚本按此名引用）
  opset 取 12；关闭动态轴、批固定为 1；开启简化去除冗余算子，减轻 NPU 转换负担
  输出形状 (1, 84, 8400)
  同一张测试图上 ONNX 推理与原始权重推理框出的目标位置、类别、置信度一致

另附：按手册 (三) 的说明导出 200 张校准图清单，供任务卡 13 的 RKNN 量化使用。

用法：
    python 03_yolov8n_export_onnx.py --outdir out/yolov8n
"""

import argparse
import os
import sys

import numpy as np
import torch

INPUT_NAME = "images"   # 手册指定，转换脚本与板端推理都按这个名字引用
INPUT_SIZE = 640
OPSET = 12

TEST_IMAGE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def download_test_images(n, outdir):
    """抓几张公开测试图（含行人）作为一致性校验与量化校准素材。"""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, url in enumerate(TEST_IMAGE_URLS):
        dst = os.path.join(outdir, f"test_{i}.jpg")
        try:
            from ultralytics.utils import downloads

            downloads.safe_download(url, dst)
            paths.append(dst)
        except Exception as e:  # 网络不通时不致命，后面会提示
            print(f"      [跳过] {url} 下载失败: {e}")
    # 不够就复制 COCO 训练图里的样本补齐
    if len(paths) < n:
        extra = os.path.join(os.path.expanduser("~"), "coco", "images", "train2017")
        if os.path.isdir(extra):
            for fn in sorted(os.listdir(extra))[: n * 3]:
                if fn.lower().endswith((".jpg", ".png")):
                    paths.append(os.path.join(extra, fn))
                if len(paths) >= n:
                    break
    return paths[:n]


def export_onnx(weights, outdir):
    from ultralytics import YOLO

    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=INPUT_SIZE,
        opset=OPSET,
        simplify=True,       # 去冗余算子，减轻 NPU 转换负担
        dynamic=False,       # 关闭动态轴
        batch=1,             # 批固定为 1
        nms=False,
        verbose=False,
    )
    # ultralytics 会在权重同目录生成，改名并搬到 outdir
    produced = onnx_path
    target = os.path.join(outdir, "yolov8n.onnx")
    if os.path.abspath(produced) != os.path.abspath(target):
        os.replace(produced, target)
    return YOLO(weights), target


def inspect_onnx(onnx_path):
    import onnx

    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    g = m.graph
    ins = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in g.input]
    outs = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in g.output]
    print(f"      输入 {ins}")
    print(f"      输出 {outs}")
    if ins[0][0] != INPUT_NAME:
        raise SystemExit(f"输入名应为 {INPUT_NAME}，实际为 {ins[0][0]}")
    if list(outs[0][1]) != [1, 84, 8400]:
        raise SystemExit(f"输出形状应为 [1, 84, 8400]，实际为 {outs[0][1]}")
    print(f"      [ok] 输入名与输出形状均符合手册")
    return ins[0][0], outs[0][0]


@torch.no_grad()
def torch_raw_output(yolo, img_t):
    """取原始检测头输出 (1, 84, 8400)，绕过后处理以便和 ONNX 逐元素比。

    新版 ultralytics 在 eval 模式下 forward 返回的是 tuple/dict
    （_inference 里 `y if self.training else (y, x)`），要按目标张量解包。
    """
    out = yolo.model(img_t)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = next(v for v in out.values() if torch.is_tensor(v))
    if not torch.is_tensor(out):
        raise TypeError(f"无法从 yolo.model() 的输出里取到张量，实际类型 {type(out)}")
    return out


def letterbox(img, size=INPUT_SIZE):
    """与 ultralytics 推理口径一致的等比缩放 + 居中填充。

    返回 (1, 3, size, size) float32、0~1 的 numpy 数组。
    注意：ultralytics 的 LetterBox 只接受 ndarray（PIL 会报
    'Image' object has no attribute 'shape'），且默认返回 HWC uint8，
    不做 HWC->CHW 和归一化，这两步在这里补齐。
    """
    from ultralytics.data.augment import LetterBox

    if not isinstance(img, np.ndarray):
        img = np.asarray(img.convert("RGB") if hasattr(img, "convert") else img)
    padded = LetterBox(new_shape=(size, size), auto=False, stride=32)(image=img)
    if not isinstance(padded, np.ndarray):
        padded = padded.numpy()
    arr = padded.astype(np.float32) / 255.0
    return np.ascontiguousarray(arr.transpose(2, 0, 1))[None]


def compare_raw(yolo, onnx_path, img_paths, tol=1e-4):
    """逐元素比对：ONNX 原始输出 vs PyTorch 原始输出。"""
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    worst = 0.0
    for p in img_paths[:3]:
        img = Image.open(p).convert("RGB")
        arr = letterbox(img)  # (1,3,640,640) float 0~1
        t = torch.from_numpy(arr).float()
        o_ort = sess.run([oname], {iname: arr})[0]
        o_torch = torch_raw_output(yolo, t).cpu().numpy()
        if o_torch.shape != o_ort.shape:
            print(f"      [警告] 形状不一致 pytorch={o_torch.shape} onnx={o_ort.shape}")
        d = float(np.max(np.abs(o_torch - o_ort)))
        worst = max(worst, d)
        print(f"      {os.path.basename(p):<14s} 原始输出最大误差 {d:.3e}")
    print(f"      全部图片最大误差 = {worst:.3e}")
    return worst


def decode_and_compare(yolo, onnx_path, img_paths, conf=0.25):
    """检测级比对：框的位置、类别、置信度是否一致。"""
    import onnxruntime as ort
    from PIL import Image

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    iname, oname = sess.get_inputs()[0].name, sess.get_outputs()[0].name

    for p in img_paths[:2]:
        img = Image.open(p).convert("RGB")
        h0, w0 = img.height, img.width

        # --- PyTorch 侧：走 ultralytics 官方后处理 ---
        # rect=False 必须显式给：官方 predict 默认 rect=True，会用「矩形最小填充」
        # 而不是 640×640 正方形 letterbox，和 ONNX 固定输入的口径不同，
        # 框数都会不一样（实测同一张图 6 框 vs 5 框）。
        res = yolo.predict(source=p, conf=conf, imgsz=INPUT_SIZE,
                           rect=False, verbose=False)[0]
        pt_boxes = res.boxes.xyxy.cpu().numpy()
        pt_cls = res.boxes.cls.cpu().numpy().astype(int)
        pt_conf = res.boxes.conf.cpu().numpy()

        # --- ONNX 侧：手写解码 ---
        arr = letterbox(img)  # (1,3,640,640) float 0~1
        out = sess.run([oname], {iname: arr})[0][0]      # (84, 8400)
        out = torch.from_numpy(out.T).float()   # (8400, 84)，转张量后下面全是张量运算
        xywh, scores = out[:, :4], out[:, 4:]
        cls = scores.argmax(1)
        confs = scores.max(1).values
        keep = confs >= conf
        xywh, cls, confs = xywh[keep], cls[keep], confs[keep]

        # --- ONNX 侧 NMS ---
        # ultralytics 8.4 已移除 non_max_suppression。这里改用 torchvision 的实现，
        # 目的是「独立解码」再和官方后处理对拍，本来就不该跟着 ultralytics 的
        # 版本变动跑偏；坐标换算口径照搬官方：xywh 在 640 尺度上做 NMS，
        # 再 scale_boxes 映射回原图尺寸。
        from torchvision.ops import nms as tv_nms
        from ultralytics.utils.ops import scale_boxes

        xyxy = torch.empty_like(xywh)
        xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        keep = tv_nms(xyxy, confs, 0.7)
        det = torch.cat(
            (xywh[keep], confs[keep, None], cls[keep, None].float()), dim=1
        )                                                    # (N, 6) xywh+conf+cls
        if len(det):
            # det[:, :4] 是 xywh，必须显式告诉 scale_boxes；默认按 xyxy 处理会得到
            # x2 < x1 的废框（ultralytics 的 non_max_suppression 返回 xyxy，所以
            # 老代码没这个参数是对的，换成 torchvision NMS 后语义就变了）
            det[:, :4] = scale_boxes(arr.shape[2:], det[:, :4], (h0, w0), xywh=True)
        on_conf = det[:, 4].numpy()
        # det[:, :4] 是 xywh，官方 boxes.xyxy 是 xyxy——必须统一成 xyxy 再比，
        # 否则两种格式混着算欧氏距离，永远匹配不上
        _c, _cw, _ch = det[:, 0], det[:, 2], det[:, 3]
        on_boxes = torch.stack(
            [_c - _cw / 2, det[:, 1] - _ch / 2, _c + _cw / 2, det[:, 1] + _ch / 2], dim=1
        ).numpy()
        on_cls = det[:, 5].numpy().astype(int)

        print(f"      {os.path.basename(p)}:  pytorch {len(pt_boxes)} 框 / onnx {len(on_boxes)} 框")
        if len(pt_boxes) and len(on_boxes):
            # 贪心匹配：两侧都是按置信度降序，逐个找最近的同类框。
            # used_pt 与 used_on 必须分开记——之前用一个 set 混着记
            # pt 下标和 onnx 下标，逻辑上是错的。
            best = 0.0
            used_on = set()
            matched = 0
            for i, b in enumerate(pt_boxes):
                d = np.linalg.norm(on_boxes - b[None], axis=1)
                d[list(used_on)] = np.inf          # 已被配上的 onnx 框不再重复使用
                j = int(d.argmin())
                if d[j] < 20.0 and on_cls[j] == pt_cls[i]:
                    matched += 1
                    used_on.add(j)
                    best = max(best, float(d[j]))
            print(f"      类别+位置匹配 {matched}/{len(pt_boxes)}，"
                  f"最大中心偏差 {best:.2f} px，最大置信度差 "
                  f"{np.abs(on_conf[:len(pt_cls)] - pt_conf[:len(on_conf)]).max():.4f}")
        persons_pt = int((pt_cls == 0).sum())
        print(f"      其中 person 类 {persons_pt} 个（实验三只保留类别号 0）")


def write_calib_list(paths, out_path):
    """任务卡 13 用：校准清单每行一个图片绝对路径。"""
    with open(out_path, "w") as w:
        for p in paths:
            w.write(os.path.abspath(p) + "\n")
    print(f"      校准清单 {len(paths)} 行 -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--outdir", default="out/yolov8n")
    ap.add_argument("--calib-n", type=int, default=200,
                    help="任务卡 13 的量化校准图数量")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=== 实验三 · YOLOv8n 导出 ONNX（不训练）===")
    print(f"torch={torch.__version__}")

    yolo, onnx_path = export_onnx(args.weights, args.outdir)
    print(f"[ok] {onnx_path}")

    print("\n[任务卡 12] Netron 口径核对")
    iname, oname = inspect_onnx(onnx_path)

    print("\n[任务卡 12] 一致性校验")
    imgs = download_test_images(8, os.path.join(args.outdir, "test_images"))
    if not imgs:
        raise SystemExit("没有可用的测试图片，请手工放几张含行人的图到 test_images/")
    compare_raw(yolo, onnx_path, imgs)
    decode_and_compare(yolo, onnx_path, imgs)

    print("\n[附带] 任务卡 13 的量化校准素材")
    calib = download_test_images(args.calib_n, os.path.join(args.outdir, "calib_images"))
    if len(calib) < 100:
        print(f"      [提示] 只拿到 {len(calib)} 张；手册建议 100~200 张，"
              f"可把 COCO 图拷进 calib_images 后重跑本步骤")
    write_calib_list(calib, os.path.join(args.outdir, "calib_list.txt"))

    print("\n=== 完成，产物 ===")
    print(f"      {onnx_path}")
    print(f"      {args.outdir}/calib_list.txt")
    print(f"\n下一步：把 {onnx_path} 传到 x86 转换机，用 RKNN-Toolkit2 转 RKNN（任务卡 13）")


if __name__ == "__main__":
    sys.exit(main())