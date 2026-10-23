# -*- coding: utf-8 -*-
"""
文档2-②：中文情感分类推理（板上）—— ChnSentiCorp 任务域
─────────────────────────────────────────────────────────────────

数据集：ChnSentiCorp（中文情感二分类，~12k 条）
预训练模型：HuggingFace 上 uer/roberta-base-finetuned-dianping-chinese
            —— 在大众点评评论上微调，与 ChnSentiCorp 分布相近。

依赖（阿里源）：
  pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      transformers modelscope safetensors

运行：
  python3 chnsenticorp_infer.py
"""

import os, sys, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("=" * 70)
print("中文情感分类推理（板上，对齐 ChnSentiCorp 任务）")
print("=" * 70)

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("[ERR] 缺包，请先：")
    print("      pip install -i https://mirrors.aliyun.com/pypi/simple/ \\")
    print("          transformers safetensors")
    sys.exit(1)

MODEL_NAME = "uer/roberta-base-finetuned-dianping-chinese"
print(f"\n[1/3] 加载模型 {MODEL_NAME} ...", flush=True)
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"[WARN] 主模型失败 ({e.__class__.__name__})，退到 ernie-3.0")
    MODEL_NAME = "nghuyong/ernie-3.0-base-zh-finetuned-dianping-chinese"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"      加载完成  (用时 {time.time()-t0:.1f}s)")
print(f"      类别映射: {model.config.id2label}")
print(f"      参数量: {n_params:,} ({n_params/1e6:.1f}M)")

texts = [
    "这家餐厅口味非常不错，服务态度也好，强烈推荐！",
    "上菜太慢，等了一个小时，菜还咸得要死，不会再来了。",
    "环境一般，价格略贵，但味道还行，整体不算太失望。",
    "性价比超高，味道好分量足，下次还会再来！",
    "停车不方便，位置难找，菜的味道倒是还可以。",
    "完整体验非常差，态度恶劣，强烈差评！",
]

print("\n[2/3] 单条推理示例：")
for text in texts:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        out = model(**enc).logits[0]
    probs = out.softmax(dim=0)
    pred_id = probs.argmax().item()
    label = model.config.id2label[pred_id]
    conf = probs[pred_id].item()
    print(f"  [{label:5s} {conf:.3f}] {text[:30]}...")

print("\n[3/3] 推理耗时（10 条平均）：")
import random
sample = random.choices(texts, k=10)
t0 = time.time()
with torch.no_grad():
    for t in sample:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
        _ = model(**enc).logits
dt = (time.time() - t0) * 100
print(f"      平均 {dt:.1f} ms/条")

# 加载 ChnSentiCorp 测试集做更全面的评估（自动下载）
print("\n[bonus] 加载 ChnSentiCorp 测试集评估 ...")
try:
    from datasets import load_dataset
    ds = load_dataset("lansinuote/ChnSentiCorp", split="validation[:64]")
    correct = 0
    t0 = time.time()
    for ex in ds:
        enc = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            pred = model(**enc).logits.argmax(-1).item()
        if pred == ex["label"]: correct += 1
    n = len(ds)
    print(f"      ChnSentiCorp acc on {n} samples = {correct/n:.2%}  (用时 {time.time()-t0:.1f}s)")
except Exception as e:
    print(f"      [skip] {e.__class__.__name__}: {e}")