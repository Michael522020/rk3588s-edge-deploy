#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验 02｜数据准备与校验

验收判据（手册 3.2 / 六、总验收清单）：
    pretrain_t2t_mini.jsonl = 1,241,043,656 字节（约 1.24 GB）
    sft_t2t_mini.jsonl      = 1,739,201,170 字节（约 1.74 GB）
    · 逐行统计行数，扫描过程不出现 JSONDecodeError
    · 抽前 3 行打印字段名，并断言目标字段存在（预训练查 text、SFT 查 conversations）

自查（手册 3.2）：若体积只有几十 MB，说明下载被中断或只下到了元数据。

运行：
    cd /root/autodl-tmp/minimind/minimind
    ../venv/bin/python check_data.py
"""

import json
import os
import sys
import time

BASE = 1000 ** 3   # 十进制口径：106 = MB、109 = GB，与手册 3.2 表对齐
TIB = 1024 ** 4

# 手册 3.2 表的实测值（2026-10-10，官方文件清单）
EXPECTED = {
    "pretrain_t2t_mini.jsonl": dict(bytes=1_241_043_656, field="text"),
    "sft_t2t_mini.jsonl":      dict(bytes=1_739_201_170, field="conversations"),
    "dpo.jsonl":               dict(bytes=53_653_322,   field="chosen"),
    "lora_identity.jsonl":     dict(bytes=22_698,       field="conversations"),
    "lora_medical.jsonl":      dict(bytes=34_002_385,   field="conversations"),
}


def human(n):
    """十进制口径的可读体积"""
    if n >= BASE:
        return f"{n / BASE:.2f} GB ({n:,} 字节)"
    if n >= 1000 ** 2:
        return f"{n / 1000 ** 2:.2f} MB ({n:,} 字节)"
    if n >= 1000:
        return f"{n / 1000:.2f} KB ({n:,} 字节)"
    return f"{n} 字节"


def scan(path, field, progress_every=100_000):
    """逐行读，统计行数 + 校验 JSON 可解析 + 记录字段名"""
    lines = 0
    bad_json = 0
    first_keys = []
    field_missing = 0
    t0 = time.time()

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                bad_json += 1
                if bad_json <= 3:
                    print(f"      [JSONDecodeError] 第 {i} 行: {e}")
                continue
            lines += 1
            if field not in obj:
                field_missing += 1
                if field_missing <= 3:
                    print(f"      [字段缺失] 第 {i} 行没有 '{field}'，实际字段: {list(obj)}")
            if len(first_keys) < 3:
                first_keys.append(list(obj.keys()))
            if lines % progress_every == 0:
                rate = lines / max(time.time() - t0, 1e-9)
                print(f"      ... 已扫 {lines:,} 行（{rate:,.0f} 行/秒）", flush=True)

    return dict(
        lines=lines,
        bad_json=bad_json,
        field_missing=field_missing,
        first_keys=first_keys,
    )


def main():
    ds_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset"
    print("=" * 72)
    print(f"实验 02 ｜ 数据校验   目录：{os.path.abspath(ds_dir)}")
    print("=" * 72)

    if not os.path.isdir(ds_dir):
        print(f"[FAIL] 目录不存在：{ds_dir}")
        print("       下载命令：")
        print("       ../venv/bin/python -m modelscope download --dataset gongjy/minimind_dataset \\")
        print("           --include pretrain_t2t_mini.jsonl sft_t2t_mini.jsonl dpo.jsonl \\")
        print("                     lora_identity.jsonl lora_medical.jsonl --local_dir ./dataset")
        sys.exit(1)

    results = {}
    all_ok = True

    for name, spec in EXPECTED.items():
        path = os.path.join(ds_dir, name)
        print(f"\n--- {name} " + "-" * (60 - len(name)))

        if not os.path.exists(path):
            print(f"  [FAIL] 文件不存在")
            all_ok = False
            continue

        size = os.path.getsize(path)
        exp = spec["bytes"]
        ratio = size / exp
        print(f"  体积    {human(size)}")
        print(f"  手册值  {human(exp)}   本次/手册 = {ratio:.4f}")
        if size == exp:
            print(f"  [OK] 体积与手册完全一致")
        elif 0.99 <= ratio <= 1.01:
            print(f"  [OK] 体积在手册值的 ±1% 内（数据集更新属正常）")
        else:
            print(f"  [FAIL] 体积与手册差 {ratio - 1:+.2%} —— 手册自查：体积只有几十 MB "
                  f"说明下载被中断或只下到了元数据，需要重下")
            all_ok = False

        print(f"  扫描中（逐行 JSON 校验，每 10 万行报进度）...", flush=True)
        r = scan(path, spec["field"])
        results[name] = r

        print(f"  行数    {r['lines']:,}")
        json_msg = "[OK] 全部可解析" if r["bad_json"] == 0 else f"[FAIL] {r['bad_json']} 行解析失败"
        field_msg = "[OK] 全部命中" if r["field_missing"] == 0 else f"[FAIL] {r['field_missing']} 行缺失"
        print(f"  JSON    {json_msg}")
        print(f"  字段    目标字段 '{spec['field']}'  {field_msg}")
        for i, keys in enumerate(r["first_keys"]):
            print(f"          第 {i + 1} 行字段: {keys}")
        if r["bad_json"] or r["field_missing"]:
            all_ok = False

    # ------------------------------------------------------------ 汇总
    print("\n" + "=" * 72)
    print("汇总（体积 MB + 行数，十进制口径）")
    print("=" * 72)
    print(f"  {'文件':<26}{'体积(MB)':>12}{'行数':>14}  备注")
    total = 0
    for name in EXPECTED:
        p = os.path.join(ds_dir, name)
        if not os.path.exists(p):
            print(f"  {name:<26}{'—':>12}{'—':>14}  缺失")
            continue
        size = os.path.getsize(p)
        total += size
        lines = results[name]["lines"]
        note = ""
        if name in ("pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl"):
            note = "主数据（验收锚点）"
        print(f"  {name:<26}{size / 1000 ** 2:>12,.1f}{lines:>14,}  {note}")
    print(f"  {'合计':<26}{total / 1000 ** 3:>11.2f} GB")

    # 行数记进实验记录表用的格式
    print("\n行数（抄进实验记录表）：")
    for name in EXPECTED:
        if name in results:
            print(f"  {name:<26} {results[name]['lines']:>12,} 行")

    print("\n磁盘余量：")
    st = os.statvfs(ds_dir)
    free = st.f_bavail * st.f_frsize
    print(f"  可用 {human(free)}   {'[OK] 够用' if free > 10 * BASE else '[WARN] 不足 10 GB'}")

    print("\n" + "=" * 72)
    print("结论：[OK] 数据验收通过，可以进入实验 03（预训练）" if all_ok
          else "结论：[FAIL] 有项目未通过，见上面标 [FAIL] 的行")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()