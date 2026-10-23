# -*- coding: utf-8 -*-
"""
文档1：CNN vs StructBERT —— 板上推理演示（不训练）
─────────────────────────────────────────────────────────────────

任务背景：TextCNN 从零训练 + StructBERT 微调（训练部分已跳过）。

板上推理策略：
  * TextCNN   —— 因文档要求"必须从零训练"且无公开预训练权重，此处仅演示架构：
                随机初始化 → 跑一次 forward → 打印参数量和推理耗时。
  * StructBERT —— 加载 damo/nlp_structbert_backbone_base_std 预训练权重，
                做 masked-token-prediction 与文本分类前向演示。

依赖（阿里源）：
  pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      modelscope transformers safetensors

运行：
  python3 infer.py
"""

import os, sys, time, tempfile

# ───────────────────────────── 1. TextCNN 推理演示（随机权重） ─────────────────────────────
print("=" * 70)
print("[1/2] TextCNN 推理演示（随机初始化，未训练）")
print("=" * 70)

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """Kim 2014 TextCNN (字符级)."""

    def __init__(self, vocab_size=30000, embed_dim=64, n_filters=128,
                 kernel_sizes=(2, 3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, n_filters, k, padding=k // 2) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        e = self.embed(x).transpose(1, 2)              # (B, E, L)
        h = [F.relu(c(e)).max(dim=2)[0] for c in self.convs]
        return self.fc(self.dropout(torch.cat(h, dim=1)))


model_cnn = TextCNN().eval()
n_params_cnn = sum(p.numel() for p in model_cnn.parameters())
print(f"[TextCNN] 参数量 = {n_params_cnn:,}")

# 模拟字符级输入（batch=4, seq_len=300）
sample_text_chars = 300
x_cnn = torch.randint(0, 30000, (4, sample_text_chars))
with torch.no_grad():
    t0 = time.time()
    out = model_cnn(x_cnn)
    dt = (time.time() - t0) * 1000
print(f"[TextCNN] input shape = {tuple(x_cnn.shape)}  output shape = {tuple(out.shape)}")
print(f"[TextCNN] 单 batch 推理耗时 = {dt:.1f} ms  (CPU, 未优化)")
print("[TextCNN] 输出是随机 logits —— 因为权重未训练，仅演示前向传播")


# ───────────────────────────── 2. StructBERT 推理（加载预训练） ─────────────────────────────
print()
print("=" * 70)
print("[2/2] StructBERT 推理（加载 damo/nlp_structbert_backbone_base_std）")
print("=" * 70)

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
except ImportError:
    print("[ERR] 缺包，请先：")
    print("      pip install -i https://mirrors.aliyun.com/pypi/simple/ \\")
    print("          modelscope transformers safetensors")
    sys.exit(1)

SB_NAME = "damo/nlp_structbert_backbone_base_std"
print(f"[StructBERT] 模型: {SB_NAME}")
print(f"[StructBERT] 缓存目录: ~/.cache/modelscope")

# 2a. 下载（已下过会自动跳过）
t0 = time.time()
cache_dir = snapshot_download(SB_NAME, cache_dir=os.path.expanduser("~/.cache/modelscope"))
print(f"[StructBERT] 路径: {cache_dir}  (用时 {time.time()-t0:.1f}s)")

# 2b. 加载 tokenizer + backbone
tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True)
backbone = AutoModel.from_pretrained(cache_dir, trust_remote_code=True)
backbone.eval()
n_params_sb = sum(p.numel() for p in backbone.parameters())
print(f"[StructBERT] 参数量 = {n_params_sb:,}  ({n_params_sb/1e6:.1f}M)")

# 2c. masked LM 演示
mlm = AutoModelForMaskedLM.from_pretrained(cache_dir, trust_remote_code=True)
mlm.eval()

# 美团点评风格示例
texts = [
    "这家店口味很[MASK]，服务员态度也不错",
    "排队[MASK]太长，等了一个小时",
    "[MASK]干净卫生，装修很有格调",
]
print("\n[StructBERT] Masked LM 演示：")
for text in texts:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = mlm(**enc)
    mask_pos = (enc.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)
    if mask_pos[1].numel():
        idx = mask_pos[1][0].item()
        top5 = out.logits[0, idx].topk(5)
        toks = tokenizer.convert_ids_to_tokens(top5.indices.tolist())
        print(f"  文本: {text}")
        print(f"  Top5 预测: {[toks[i] + f'({top5.values[i].item():.2f})' for i in range(5)]}")
        print()

# 2d. backbone 推理耗时
text = "这家餐厅的口味相当不错，服务也很周到，强烈建议来尝尝。"
enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    t0 = time.time()
    out = backbone(**enc)
    dt = (time.time() - t0) * 1000
print(f"[StructBERT] 单条文本({len(text)}字) backbone 推理耗时 = {dt:.1f} ms")
print(f"[StructBERT] hidden shape = {tuple(out.last_hidden_state.shape)}")
print()
print("=" * 70)
print("结论:")
print(f"  TextCNN    参数量 = {n_params_cnn/1e3:.1f} K (随机初始化)")
print(f"  StructBERT 参数量 = {n_params_sb/1e6:.1f} M (已加载官方预训练)")
print(f"  架构差异: CNN 局部感受野, StructBERT 全局自注意力")
print("=" * 70)