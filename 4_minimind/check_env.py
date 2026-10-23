#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 01｜环境搭建与验收

验收判据（手册 1.2 / 六、总验收清单）：
    · 依赖逐项 [OK]
    · 实例化 MiniMindForCausalLM 后参数量精确等于 63,912,192
    · 显存总量按 1024³ 换算打印（4090 应显示约 24.0 GB）

这个数字同时验证三件事：依赖装对了、模型代码能跑通、手册 2.1 节的核算是对的。
对不上就别往下走。

运行：
    cd /root/autodl-tmp/minimind/minimind
    ../venv/bin/python check_env.py
"""

import importlib
import os
import platform
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)))))

GIB = 1024 ** 3
EXPECTED_PARAMS = 63_912_192

print("=" * 72)
print("实验 01 ｜ MiniMind 环境自检")
print("=" * 72)

# ------------------------------------------------------------ 1. 解释器与框架
print("\n[1] 解释器 / 框架")
print(f"  Python        {platform.python_version()}  ({sys.executable})")
print(f"  平台          {platform.platform()}")

import torch

print(f"  PyTorch       {torch.__version__}")
print(f"  CUDA 编译版本  {torch.version.cuda or 'N/A (CPU 版)'}")
cuda_ok = torch.cuda.is_available()
print(f"  CUDA 可用      {cuda_ok}")
if not cuda_ok:
    print("      [FAIL] torch.cuda.is_available() 为 False。")
    print("             最常见原因：手动 pip install torch 覆盖成了 CPU 版。")
    print("             处置：卸载 torch，恢复镜像自带版本 —— requirements.txt 里")
    print("                   torch/torchvision 是被注释掉的，不要装。")
else:
    print(f"  GPU 型号       {torch.cuda.get_device_name(0)}")
    print(f"  GPU 数量       {torch.cuda.device_count()}")
    props = torch.cuda.get_device_properties(0)
    print(f"  显存总量       {props.total_memory / GIB:.1f} GB  ({props.total_memory} 字节)")
    print(f"  计算能力       sm_{props.major}{props.minor}")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  BF16 支持      {bf16}   （训练脚本 --dtype bfloat16 需要它）")
    if not bf16:
        print("      [WARN] BF16 不可用，训练请改 --dtype float16 或 float32")

# 显存除法口径演示：手册要求用 1024³
print(f"  口径对照       {props.total_memory} / 1024³ = {props.total_memory / GIB:.1f} GB")
print(f"                 {props.total_memory} / 1000³ = {props.total_memory / 1000 ** 3:.1f} GB (十进制)")

# ------------------------------------------------------------ 2. 依赖逐项 try-import
print("\n[2] 依赖逐项检查")
# (模块名, 说明)
MODULES = [
    ("transformers", "权重/配置加载"),
    ("datasets", "JSONL 流式数据集"),
    ("numpy", "数值计算"),
    ("modelscope", "数据集与模型下载"),
    ("swanlab", "训练曲线"),
    ("flask", "API / Web 服务"),
]
failed = []
for name, desc in MODULES:
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, "__version__", "无版本号")
        print(f"  [OK]   {name:<14} {ver:<12} {desc}")
    except Exception as e:
        failed.append(name)
        print(f"  [FAIL] {name:<14} {'':<12} {e.__class__.__name__}: {e}")

# 附加项：装了就用得上，没装不算失败
for name, desc in [("streamlit", "实验 08 WebUI"), ("fastapi", "实验 08 API 服务")]:
    try:
        mod = importlib.import_module(name)
        print(f"  [OK]   {name:<14} {getattr(mod, '__version__', '无版本号'):<12} {desc}")
    except Exception:
        print(f"  [SKIP] {name:<14} {'-':<12} {desc}（实验 08 前再装）")

# ------------------------------------------------------------ 3. 模型参数对账
print("\n[3] 模型参数对账")
try:
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
except Exception as e:
    print(f"  [FAIL] 无法导入模型代码：{e.__class__.__name__}: {e}")
    print("         确认在仓库根目录下运行（sys.path 需要能 import model 包）")
    sys.exit(1)

cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, vocab_size=6400)
print(f"  hidden_size              {cfg.hidden_size}")
print(f"  num_hidden_layers        {cfg.num_hidden_layers}")
print(f"  vocab_size               {cfg.vocab_size}")
print(f"  num_attention_heads      {cfg.num_attention_heads}")
print(f"  num_key_value_heads      {cfg.num_key_value_heads}   (GQA)")
print(f"  head_dim                 {cfg.head_dim}")
print(f"  intermediate_size        {cfg.intermediate_size}   (FFN)")
print(f"  max_position_embeddings  {cfg.max_position_embeddings}")
print(f"  rope_theta               {cfg.rope_theta}")
print(f"  rms_norm_eps             {cfg.rms_norm_eps}")
print(f"  tie_word_embeddings      {cfg.tie_word_embeddings}")

model = MiniMindForCausalLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model Params: {n_params:,}")
print(f"  Model Params: {n_params / 1e6:.2f}M")

if n_params == EXPECTED_PARAMS:
    print(f"  [OK] 参数量精确等于 {EXPECTED_PARAMS:,}，与手册 2.1 表一致")
else:
    diff = n_params - EXPECTED_PARAMS
    print(f"  [FAIL] 参数量与手册差 {diff:+,}，先别往下走")
    print("         常见原因：改过 hidden_size / 层数 / 词表，或 checkout 到了不同版本")

# 逐项拆解，让「这个数字怎么来的」能讲清楚
print("\n  参数量拆解：")
embed = cfg.vocab_size * cfg.hidden_size
head_dim, n_kv = cfg.head_dim, cfg.num_key_value_heads
q = cfg.hidden_size * cfg.num_attention_heads * head_dim
k = cfg.hidden_size * n_kv * head_dim
v = k
o = cfg.num_attention_heads * head_dim * cfg.hidden_size
qkv_o = q + k + v + o
ffn = 3 * cfg.hidden_size * cfg.intermediate_size      # SwiGLU: gate/up/down
qkv_o += 2 * head_dim                                  # q_norm / k_norm 的 RMSNorm
attn_all = cfg.num_hidden_layers * qkv_o
ffn_all = cfg.num_hidden_layers * ffn
norms = 2 * cfg.hidden_size                             # 每层 input_norm + post_attention_norm
norm_all = cfg.num_hidden_layers * norms
print(f"    词嵌入(与输出共享)   {embed:>12,}")
print(f"    每层注意力 + q/k_norm{attn_all // cfg.num_hidden_layers:>12,}  × {cfg.num_hidden_layers} 层")
print(f"    每层 SwiGLU FFN      {ffn:>12,}  × {cfg.num_hidden_layers} 层")
print(f"    每层两个 RMSNorm     {norms:>12,}  × {cfg.num_hidden_layers} 层")
print(f"    末尾 norm            {cfg.hidden_size:>12,}")
print(f"    ----------------------------------")
print(f"    合计                 {embed + attn_all + ffn_all + norm_all + cfg.hidden_size:>12,}")

# ------------------------------------------------------------ 4. 前向冒烟
print("\n[4] 前向冒烟测试")
try:
    m = model.to("cuda" if cuda_ok else "cpu").eval()
    ids = torch.tensor([[1, 42, 7, 9, 2]], device=m.device)
    with torch.no_grad():
        out = m(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    print(f"  [OK] forward 输出形状 {tuple(logits.shape)}  (batch, seq, vocab)")
    print(f"       末位 logits 前 5 个值：{[round(float(x), 4) for x in logits[0, -1, :5]]}")
    loss_ref = torch.nn.functional.cross_entropy(
        logits[0, :-1].float(), ids[0, 1:].long()
    )
    print(f"       随机初始化下的 loss ≈ {float(loss_ref):.4f}")
    print(f"       理论下界 ln(vocab_size) = ln({cfg.vocab_size}) = {torch.log(torch.tensor(float(cfg.vocab_size))):.4f}")
    print("       —— 实验 03 的起始 loss 应该落在这个数附近（手册说的 8.7）")
except Exception as e:
    print(f"  [FAIL] 前向失败：{e.__class__.__name__}: {e}")

# ------------------------------------------------------------ 5. 结论
print("\n" + "=" * 72)
ok = (not failed) and n_params == EXPECTED_PARAMS and cuda_ok
if ok:
    print("结论：[OK] 环境验收通过，可以进入实验 02（数据准备）")
else:
    if failed:
        print(f"结论：[FAIL] 以下依赖未就绪：{', '.join(failed)}")
    if n_params != EXPECTED_PARAMS:
        print(f"结论：[FAIL] 参数量不符，差 {n_params - EXPECTED_PARAMS:+,}")
    if not cuda_ok:
        print("结论：[FAIL] GPU 不可用，先修 torch")
print("=" * 72)
sys.exit(0 if ok else 1)