# 实验手册 3 · RK3588S（鲁班猫 LubanCat-4）部分

手册：`3.深度学习模型训练与边缘端部署实验手册.docx`

本目录只覆盖 RK358S 链路。昇腾（OM / MindSpore）部分不在此列。

> **当前状态（2026-10-23）**：板端实测已完成，性能数据见
> [`性能基准表.md`](性能基准表.md)。任务卡 6、12、13(fp16)、14 已通过；
> **肺炎 CNN 与 YOLOv8n 的 RKNN 模型转换失效**，见文末「已知问题」。

## 交付物对照表

| 任务卡 | 内容 | 脚本 | 跑在哪 | 状态 |
|---|---|---|---|---|
| 任务卡 1/2/3/4 | 信用卡 MLP：分层切分+标准化 → 训练 → 导出 ONNX → 校验 | `autodl/01_mlp_fraud_train_export.py` | 4090 | ✅ |
| 任务卡 5 | `mlp_fraud.onnx` → INT8 RKNN | `autodl/04_convert_rknn.py --model mlp` | 4090（x86） | ✅ |
| 任务卡 6 | 鲁班猫 MLP 推理 + 精度对比 | `board/infer_mlp_rknn.py` | 鲁班猫 | ✅ |
| 任务卡 9/10/11 | 肺炎 CNN：数据管线 → MobileNetV2 → 导出 ONNX | `autodl/02_cnn_pneumonia_train_export.py` | 4090 | ✅ |
| 实验二 §(五) | 肺炎 CNN → RKNN + 板端推理 | `autodl/04_convert_rknn.py --model cnn`<br>`board/infer_cnn_rknn.py` | 4090 / 鲁班猫 | ⚠️ ONNX 正常，RKNN 失效 |
| 任务卡 12 | YOLOv8n 导出 ONNX（不训练） | `autodl/03_yolov8n_export_onnx.py` | 4090 | ✅ |
| 任务卡 13 | `yolov8n.onnx` → RKNN | `autodl/05_convert_rknn_yolov8.py` | 4090（x86） | ⚠️ fp16 ✅ / INT8 失效 |
| 任务卡 14 | 实时人员检测 | `board/cam_person_detect.py` | 鲁班猫 | ✅（用 fp16 版） |
| 第五章 | 性能基准表 | [`性能基准表.md`](性能基准表.md) | — | ✅ |

辅助脚本（本次新增，为修问题而加）：

| 脚本 | 用途 |
|---|---|
| `autodl/02b_make_cnn_calib.py` | 只补 CNN 的量化校准集 / preprocess.json / 测试清单，不训练——训练要 GPU，这三样不要 |
| `autodl/fix_mlp_quant.sh` | 复现 MLP 量化的三组对照（normal / 裁剪 / kl_divergence） |
| `autodl/finish_quant.sh` | 在 2 GiB 内存的无卡容器里把 CNN + YOLO 的 INT8 转出来 |
| `artifacts/` | 从 AutoDL 抢救回来的全部模型、日志、MiniMind 权重 |

## 关键约定（三个模型统一）

| 项 | 值 | 原因 |
|---|---|---|
| ONNX opset | **12** | RKNN-Toolkit2 1.6.0+ 支持 12~19；ATC 普遍 11~12。取交集 |
| batch | **1**，无动态轴 | RKNN 转换要求静态形状 |
| MLP 输入/输出名 | `input` / `logits` | 任务卡 4 指定 |
| CNN 输入/输出名 | `input` / `logits` | 任务卡 11 指定 |
| YOLOv8 输入名 | `images` | 任务卡 12 指定，转换脚本按此名引用 |
| 目标平台 | `rk3588` | RK3588S 与 RK3588 共用 RKNN 目标 |
| 版本对齐 | RKNN-Toolkit2 与板端 rknn-toolkit-lite2 **必须同版本** | 手册 §(三) 版本纪律 |
| 量化算法 | **`kl_divergence`** | `normal`（逐层 min/max）实测会让 MLP 的 AUPRC 掉 0.034，超判据 |
| MLP 判定阈值 | **0.77** | 训练时调出来的；脚本默认 0.5 是错的，会把大量正常样本误判为欺诈 |

## 执行顺序

### A. 4090 上（x86 Linux）

```bash
# 0. 依赖（版本纪律见下方表格；rknn-toolkit2 必须 2.3.2，与板端 lite2 对齐）
pip install torch torchvision ultralytics onnx onnxruntime onnxsim sklearn pandas \
            rknn-toolkit2==2.3.2
python -c "import platform; print(platform.machine())"   # 必须输出 x86_64

# 1. 数据
#    creditcard.csv      Kaggle mlg-ulb/creditcardfraud
#    chest_xray/         Kaggle paultimothymooney/chest-xray-pneumonia
#                        结构需为 chest_xray/{train,test,val}/{NORMAL,PNEUMONIA}

# 2. 实验一：MLP（顺手记下 metrics 里的 threshold，板端要用）
python autodl/01_mlp_fraud_train_export.py --data data/creditcard.csv --epochs 15

# 3. 实验二：肺炎 CNN
python autodl/02_cnn_pneumonia_train_export.py --data data/chest_xray --epochs-finetune 10

# 4. 实验三：YOLOv8n 导出（会自动下载官方 COCO 权重）
python autodl/03_yolov8n_export_onnx.py

# 5. 转 RKNN —— 每个模型先浮点、后 INT8，浮点跑通再开量化
python autodl/04_convert_rknn.py --env-check
python autodl/04_convert_rknn.py --model mlp --onnx out/mlp_fraud/mlp_fraud.onnx
python autodl/04_convert_rknn.py --model mlp --onnx out/mlp_fraud/mlp_fraud.onnx \
        --calib-dir out/mlp_fraud/calib_npy --quant --calib-clip 16
python autodl/04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx
python autodl/04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
        --calib-dir out/pneumonia/calib_images --quant
python autodl/05_convert_rknn_yolov8.py --onnx out/yolov8n/yolov8n.onnx \
        --dataset out/yolov8n/calib_list.txt
python autodl/05_convert_rknn_yolov8.py --onnx out/yolov8n/yolov8n.onnx \
        --dataset out/yolov8n/calib_list.txt --quant
```

> **转换完必须上板实测，别只看转换日志。** 转换器全程无报错、`.rknn` 正常产出，
> 但分类头输出恒为 0 的情况真的发生过（见文末「已知问题」）。
>
> `05` 加 `--use-model-zoo` 可改走 rknn_model_zoo 官方 `examples/yolov8/python/convert.py`
> （手册推荐路径，会自动 clone）。
>
> 训练脚本若在 ONNX 校验处退出，`calib_images/`、`preprocess.json`、
> `test_manifest.txt` 都不会落盘。此时不要重训，用
> `python autodl/02b_make_cnn_calib.py --data data/chest_xray --onnx out/pneumonia/pneumonia.onnx`
> 在 CPU 上补齐这三样——它们只依赖数据集和已导出的 ONNX，跟模型权重无关。

### B. 拷到鲁班猫

```bash
scp out/rknn/*.rknn        ubuntu@<板子IP>:~/models/
scp out/mlp_fraud/mlp_splits.npz out/mlp_fraud/mlp_fraud.onnx  ubuntu@<板子IP>:~/models/
scp out/pneumonia/preprocess.json out/pneumonia/test_manifest.txt ubuntu@<板子IP>:~/models/
scp board/*.py             ubuntu@<板子IP>:~/edge/
# 肺炎测试集图片也要拷过去：640 张，test/ 和 val/ 两个目录
#   manifest 里是训练机的相对路径 data/chest_xray/test/...，
#   脚本会自动按 train/test/val 重映射到 --image-root，所以拷成
#   ~/chest_xray/test/... ~/chest_xray/val/... 即可
```

### C. 鲁班猫上

```bash
# 0. 先自检
pip install --user --break-system-packages rknn-toolkit-lite2==2.3.2
python3 edge/00_board_env_check.py

# 1. 任务卡 6：MLP（阈值 0.77！用默认的 0.5 结果全错）
python3 edge/infer_mlp_rknn.py --rknn models/mlp_fraud.rknn \
        --onnx models/mlp_fraud.onnx --splits models/mlp_splits.npz \
        --threshold 0.77

# 2. 实验二：肺炎 CNN
python3 edge/infer_cnn_rknn.py --rknn models/pneumonia.rknn \
        --preprocess models/preprocess.json \
        --manifest models/test_manifest.txt \
        --image-root ~/chest_xray

# 3. 任务卡 14：人员检测
#    板子没有 UVC 摄像头，--source 支持三种：设备编号 / 网络流 / 本地文件
ls /dev/video*                                        # 无卡时这里是空的
python3 edge/cam_person_detect.py --rknn models/yolov8n_fp.rknn \
        --source models/bus.jpg --frames 1 --headless \
        --save-shot out/bus_shot.jpg                  # 本地图片
python3 edge/cam_person_detect.py --rknn models/yolov8n_fp.rknn \
        --source http://<笔记本IP>:8080/stream         # 笔记本推 MJPEG 流
```

## 量化转换怎么调（实测结论）

### ① 量化算法是决定性因素

`--quant-algorithm` 默认值已从 `normal` 改成 `kl_divergence`。MLP 上实测：

| 变体 | AUPRC | AUPRC 降幅 | 一致率 |
|---|---|---|---|
| `normal` + 不裁剪（原始配置） | 0.48372 | +0.03368 ❌ 超判据 | 99.998% |
| `normal` + `--calib-clip 16` | 0.49637 | +0.02103 ✅ | 99.993% |
| `kl_divergence` | 0.53426 | −0.01686 ✅ | 99.967% |
| **`kl_divergence` + `--calib-clip 16`（交付版）** | **0.53211** | **−0.01471** ✅ | 99.956% |

### ② 校准集截断：重尾特征必须截

标准化后的信用卡欺诈特征是重尾的——训练集全局 `absmax=104.4`，但 p99 只有 3~4。
量化区间直接取校准集 min/max，区间被离群点撑大后 INT8 步长变粗，典型样本相对误差能到 20%。

```bash
python autodl/04_convert_rknn.py --model mlp --onnx out/mlp_fraud/mlp_fraud.onnx \
        --calib-dir out/mlp_fraud/calib_npy --quant --calib-clip 16
```

> ⚠️ 裁剪必须作用在**写盘给 toolkit 的那份 npy**（`out/rknn/calib_npy_norm/`）。
> 脚本里两处都加了裁剪，只裁内存里那个数组是没有效果的。

### ③ AutoDL 无卡模式的内存限制

无卡模式 cgroup 只给 **2 GiB**、CPU 只给 **0.5 核**（`nproc` 会报 208，别被骗）。
RKNN 转换器转 YOLO/CNN 会 OOM，退出码 **137**。能塞进去的组合：

```bash
# 视觉模型：normal + optimization_level=0 + 20~50 张校准图
python autodl/04_convert_rknn.py --model cnn --onnx out/pneumonia/pneumonia.onnx \
        --calib-dir out/pneumonia/calib_images --calib-limit 50 \
        --outdir out/rknn --quant --optimization-level 0 --quant-algorithm normal
```

`kl_divergence` 要逐层存直方图，更吃内存，2 GiB 下用不了——这也是为什么
MLP 能用 kl_divergence（模型小）而 YOLO 不行。

### ④ 校准集不是可有可无

YOLOv8n 最早只用 **2 张**图做校准（导出脚本自己都打印了「手册建议 100~200 张」的提示，
当时没当回事），结果板端一个目标都检不出。同一个模型浮点版检出率 100%。

## 容易踩的坑

1. **`/dev/rknpu` 不存在** → `init_runtime()` 必失败。RK3588 驱动走 DRM 路径，
   设备节点是 `/dev/dri/renderD129`。判断 NPU 可用性要查这个。
2. **板端预处理必须用 PIL 缩放，不能用 cv2.resize**。训练侧是 torchvision
   （底层 PIL，降采样自带抗锯齿），cv2 的 INTER_LINEAR 降采样**不抗锯齿**，
   1800×1300 直接压到 224 会混叠出严重假高频。实测同一批 640 张图：
   cv2 版准确率 0.7672 / AUROC 0.9588 / tn=94，PIL 版 0.8500 / 0.9628 / tn=147，
   与训练侧逐项完全一致。
3. **ONNX 是静态 batch=1**。板端脚本按 512 批量喂会被 onnxruntime 直接拒
   （`Got: 512 Expected: 1`）。`infer_mlp_rknn.py` 的 `onnx_infer()` 现在会从
   会话里读回真实 batch 维。
4. **RKNN 版本不一致** → 转换机 toolkit2 与板端 lite2 必须同版本，否则加载失败。
5. **"无报错但检不出目标"** → 输出切分与解码器不匹配，或（本次的情况）
   量化后分类头输出恒为 0。见文末「已知问题」。
6. **摄像头不支持 MJPG** → 回退 YUYV 也能跑，脚本会打印实际编码，不必强求。
7. **板子没有摄像头** → `--source` 传本地图片/视频文件就能验证整条检测链路；
   注意别用 `cap.read()` 探测首帧，图片只有一帧，探测读完就没了
   （且这个后端 read → set(POS_FRAMES,0) → read 第二次直接 False），
   尺寸要从 `cap.get(CAP_PROP_FRAME_WIDTH/HEIGHT)` 取。
8. **`00_board_env_check.py` 报 `[FAIL] /dev/rknpu` 是误报**，看汇总行的
   「NPU 驱动: 可用」为准。

## 已知问题（未解决）

### 肺炎 CNN 与 YOLOv8n 转出来的 RKNN 是坏的

| 现象 | 肺炎 CNN | YOLOv8n |
|---|---|---|
| 框坐标 / 中间层 | 正常 | 正常（通道 0~3 max≈643） |
| 类别分通道 | 640 张全判肺炎 | **精确为 0.0000** |
| 板端 AUROC | 0.5465(INT8) / 0.6343(fp16) | — |
| 同模型 ONNX 板端 | **0.9628 ✅** | **检出率 100% ✅** |

已逐项排除：量化算法（normal/mmse/kl_divergence/gdq）、校准图数量（2/20/50/200）、
`optimization_level`（0/1/3）、浮点精度（fp16/bf16；bf16 运行时直接报
`input dtype is undefine!`）、归一化是否烘焙进 RKNN、板端预处理。

ONNX 图本身很干净：52 Conv / 35 Clip / 10 Add / 1 GlobalAveragePool / 1 Gemm，
opset 12，无 RKNN 不支持的算子。同一个转换器转 MLP 完全正常（一致率 99.956%）。
**结论是 rknn-toolkit2 2.3.2 对这两个图的转换有问题，不是参数没调对。**

下一步方向（都需要重新开 AutoDL）：换 rknn-toolkit2 版本；走 rknn_model_zoo
官方转换脚本；或把 CNN 的 `Gemm` 改写成 `MatMul`+`Add` 再转。

在修好之前，**交付时用 `yolov8n_fp.rknn`**（fp16 版检出率 100%，50.88 ms/帧）。
CNN 那条暂时只能用 ONNX 在板端 CPU 上跑（AUROC 0.9628，与训练侧一致）。

## 数据集说明（手册 §附录A）

| 数据集 | 来源 |
|---|---|
| Credit Card Fraud Detection | Kaggle `mlg-ulb/creditcardfraud`，CC BY 4.0 |
| Chest X-Ray Images (Pneumonia) | Kaggle `paultimothymooney/chest-xray-pneumonia`，CC BY 4.0 |
| YOLOv8n COCO 预训练权重 | `github.com/ultralytics/ultralytics`，自动下载 |
| 量化校准图（YOLO） | COCO val2017，从 `ultralytics/assets` 拿测试图也可 |