# -*- coding: utf-8 -*-
"""
文档4：MiniMind 大模型推理（板上）
─────────────────────────────────────────────────────────────────

从 ModelScope 下载 minimind-3（~64M 参数，~250MB），跑单轮对话。
板上内存 3.8 GiB 紧张，ML 限制 max_new_tokens≤256 避免 OOM。

依赖（阿里源）：
  pip install -i https://mirrors.aliyun.com/pypi/simple/ \
      modelscope transformers safetensors

运行：
  python3 minimind_inference.py

交互命令：
  /quit  /q    退出
  /clear /c   清屏
"""

import os, sys, time

os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))

print("=" * 70)
print("MiniMind-3 板上对话推理")
print("=" * 70)

try:
    from modelscope import snapshot_download
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    print("[ERR] 缺包：")
    print("      pip install -i https://mirrors.aliyun.com/pypi/simple/ \\")
    print("          modelscope transformers safetensors")
    sys.exit(1)

MODEL_ID = "gongjy/minimind-3"

print(f"\n[1/4] 下载 {MODEL_ID} (~250MB) ...", flush=True)
t0 = time.time()
cache_dir = snapshot_download(MODEL_ID, cache_dir=os.environ["MODELSCOPE_CACHE"])
print(f"      路径: {cache_dir}  (用时 {time.time()-t0:.1f}s)")

print("\n[2/4] 加载 tokenizer ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(cache_dir, trust_remote_code=True)
print(f"      done  (用时 {time.time()-t0:.1f}s)")
print(f"      vocab_size = {tokenizer.vocab_size}")

print("\n[3/4] 加载模型（CPU，fp32，约 250MB 内存）...", flush=True)
print("      ⚠  3.8G 内存偏紧；如果 OOM，加 --int4 参数（需 bitsandbytes）")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    cache_dir,
    trust_remote_code=True,
    torch_dtype="auto",
).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"      加载完成  (用时 {time.time()-t0:.1f}s)  参数量 = {n_params/1e6:.1f}M")


def chat_once(prompt: str, max_new_tokens: int = 256):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0
    reply = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    n_new = out.shape[1] - inputs.input_ids.shape[1]
    return reply.strip(), n_new, dt


import torch  # noqa: E402  (放在这里避免提前导入影响启动信息)

print("\n[4/4] 对话 (输入 /quit 退出)")
print("-" * 70)

prompts = [
    "你好，你叫什么名字？",
    "介绍一下你自己",
    "用 Python 写一个快速排序",
    "解释一下 LoRA 微调的原理",
]
print("[demo] 自动跑 4 条样本：\n")
for p in prompts:
    print(f"  user: {p}")
    reply, n_new, dt = chat_once(p, max_new_tokens=200)
    print(f"  MiniMind: {reply[:300]}")
    print(f"  [gen {n_new} tokens, {dt:.1f}s, {n_new/dt:.1f} tok/s]")
    print()

print("=" * 70)
print("进入交互模式（直接回车跳过，输入 /quit 退出）")
print("=" * 70)
while True:
    try:
        q = input("\nuser> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q: continue
    if q in ("/quit", "/q", "exit"):
        break
    if q in ("/clear", "/c"):
        os.system("clear")
        continue
    try:
        reply, n_new, dt = chat_once(q)
        print(f"\nMiniMind> {reply}\n  [{n_new} tok, {dt:.1f}s, {n_new/dt:.1f} tok/s]")
    except Exception as e:
        print(f"[ERR] {e.__class__.__name__}: {e}")