#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板端 CPU 负载控制台 —— 把开发板稳定压到指定占用率（默认 80%）

实现方式：闭环控制，不是估算。
  1) 每 0.5 秒从 /proc/stat 采样两次，算出真实 CPU 占用（含每核）
  2) 起 W 个工作进程，各绑一个核（os.sched_setaffinity），用占空比调节负荷
  3) 测量值与目标差多少，就按比例调占空比，直到稳定在目标附近
  4) 退出时汇总：实测持续均值、峰值、每核占用、达标时间占比

模式：
  --mode synthetic  纯合成负荷。不需要模型，现在就能验证板子能否稳定压到 80%。
  --mode hold       只测量、不加负荷。用来核对「别的任务占到了多少」。

用法：
  python3 cpu_load_harness.py --mode synthetic --target 80 --duration 60
  python3 cpu_load_harness.py --mode synthetic --target 80 --duration 0    # 一直跑到 Ctrl-C
  python3 cpu_load_harness.py --mode hold --duration 10
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

SAMPLE_INTERVAL = 0.5


# --------------------------------------------------------------------------
# 采样
# --------------------------------------------------------------------------
def read_cpu_times():
    """返回 (所有核的 busy/idle 累计 jiffies, 每核 busy 列表)。"""
    with open("/proc/stat") as f:
        lines = [l for l in f if l.startswith("cpu")]
    agg_busy = agg_total = 0
    per_core = []
    for line in lines:
        parts = line.split()
        name, vals = parts[0], [int(v) for v in parts[1:]]
        total = sum(vals)
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        busy = total - idle
        agg_busy += busy
        agg_total += total
        if name != "cpu":
            per_core.append((busy, total))
    return agg_busy, agg_total, per_core


class CpuSampler:
    """基于 /proc/stat 差分的 CPU 占用测量。

    本板内核的 /proc/stat 记账较粗（实测约每秒前进 800 jiffies，8 核合计，
    即每核 ~100 jiffies/s）。如果两次读数落在同一个记账周期内，差分会是 0~2，
    算出 0% 或 100% 的假值。所以这里强制最小测量窗口，并丢弃退化样本。
    """

    MIN_WINDOW = 0.50     # 秒；小于此窗口的差分不可信。sample() 自己按这个节奏走
    MIN_JIFFIES = 40      # 差分过小说明没跨过记账边界

    def __init__(self):
        self.prev = read_cpu_times()
        self.prev_t = time.monotonic()
        self.dropped = 0

    def sample(self):
        # 保证距上次读数至少 MIN_WINDOW
        wait = self.MIN_WINDOW - (time.monotonic() - self.prev_t)
        if wait > 0:
            time.sleep(wait)

        prev_busy, prev_total, prev_cores = self.prev
        cur = read_cpu_times()
        now = time.monotonic()
        dt = now - self.prev_t
        self.prev, self.prev_t = cur, now

        d_busy = cur[0] - prev_busy
        d_total = cur[1] - prev_total
        if d_total < self.MIN_JIFFIES:
            self.dropped += 1
            return None  # 退化样本：丢掉，不要喂给控制器

        util = 100.0 * d_busy / d_total

        cores = []
        for (b1, t1), (b2, t2) in zip(prev_cores, cur[2]):
            d_t, d_b = t2 - t1, b2 - b1
            cores.append(100.0 * d_b / d_t if d_t else 0.0)
        return util, cores, dt, (d_busy, d_total)


# --------------------------------------------------------------------------
# 工作进程
# --------------------------------------------------------------------------
def worker(core_id, duty_val, stop_flag, _unused):
    """在指定核上按占空比做纯计算负荷。

    占空比 = 忙的时间 / 周期。忙时做真实计算（不是 sleep），
    闲时 sleep 让出 CPU，这样 W 个进程可以平滑地逼近任意总占用率。

    duty_val 是跨进程共享变量，必须每个周期重新读取 ——
    在这里取一次就固定下来，是之前压不满 80% 的原因。
    """
    try:
        os.sched_setaffinity(0, {core_id})
    except (AttributeError, OSError):
        pass

    period = 0.05          # 50ms 一个周期
    x = 12345
    while not stop_flag.value:
        duty = max(0.0, min(1.0, float(duty_val.value)))   # 每周期重读
        busy_t = period * duty
        idle_t = period - busy_t

        if busy_t > 0:
            end = time.monotonic() + busy_t
            while time.monotonic() < end:
                for _ in range(2000):
                    x = (x * 1103515245 + 12345) & 0x7FFFFFFF
                    x ^= x >> 7
        if idle_t > 0:
            time.sleep(idle_t)
    return x


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synthetic", "hold"], default="synthetic")
    ap.add_argument("--target", type=float, default=80.0, help="目标 CPU 占用 %%")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="持续秒数；0 = 一直跑到 Ctrl-C")
    ap.add_argument("--workers", type=int, default=0, help="0 = 自动按核数取满")
    ap.add_argument("--settle", type=float, default=8.0,
                    help="调节阶段时长，之后进入稳定观测期")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="判定达标的允许偏差（百分点）")
    ap.add_argument("--report", default="out/cpu_load_report.json")
    args = ap.parse_args()

    nproc = os.cpu_count() or 8
    workers = args.workers or nproc
    workers = max(1, min(workers, nproc))
    print("=== 板端 CPU 负载控制台 ===")
    print(f"  核数 {nproc}  工作进程 {workers}  目标 {args.target:.1f}%  模式 {args.mode}")

    sampler = CpuSampler()

    if args.mode == "hold":
        print("\n[只测量，不加负荷]")
        samples, cores_acc = [], []
        t_end = time.time() + (args.duration or 10)
        while time.time() < t_end:
            r = sampler.sample()
            if r is None:
                continue          # 退化样本，跳过
            u, cores, _, raw = r
            samples.append(u)
            cores_acc.append(cores)
            print(f"  总占用 {u:5.1f}%   每核 " +
                  " ".join(f"{c:4.0f}" for c in cores), flush=True)
        print(f"\n  均值 {sum(samples)/len(samples):.1f}%  峰值 {max(samples):.1f}%")
        return 0

    # ---- 闭环控制 ----
    duty = 0.10
    ctx = mp.get_context("fork")
    duty_val = ctx.Value("d", duty)
    stop_flag = ctx.Value("b", False)
    procs = [
        ctx.Process(target=worker, args=(i, duty_val, stop_flag, None), daemon=True)
        for i in range(workers)
    ]
    for p in procs:
        p.start()
    time.sleep(1.0)
    alive = sum(1 for p in procs if p.is_alive())
    if alive == 0:
        raise SystemExit("所有工作进程都没起来，检查 multiprocessing 是否被限制")
    print(f"  {alive}/{workers} 个工作进程已启动")
    print(f"\n[调节阶段] 逐步逼近 {args.target:.1f}% ...")
    print("  按 Ctrl-C 停止\n")

    all_samples, all_cores = [], []
    t0 = time.time()
    settle_end = t0 + args.settle
    end = t0 + args.duration if args.duration else float("inf")
    steady = []
    saturated = 0

    try:
        while time.time() < end:
            r = sampler.sample()
            if r is None:
                continue          # 退化样本，跳过，不参与控制与统计
            u, cores, dt, raw = r
            all_samples.append(u)
            all_cores.append(cores)

            if time.time() >= settle_end:
                steady.append(u)

            phase = "调" if time.time() < settle_end else "稳"
            print(f"  [{phase}] 总占用 {u:5.1f}%  目标 {args.target:5.1f}%  "
                  f"占空比 {duty_val.value * 100:5.1f}%  |  " +
                  " ".join(f"{c:3.0f}" for c in cores) +
                  f"   [raw busy={raw[0]} total={raw[1]}]", flush=True)

            # 比例调节：观测到的偏差越大，占空比调得越多
            err = args.target - u
            if duty_val.value >= 0.999 and u < args.target - args.tolerance:
                saturated += 1
            else:
                saturated = 0
            new_duty = duty_val.value * (1.0 + 0.25 * err / 100.0)
            new_duty = max(0.005, min(1.0, new_duty))
            if abs(u - args.target) <= args.tolerance:
                new_duty = min(max(new_duty, duty_val.value * 0.98), 1.0)
            duty_val.value = new_duty

    except KeyboardInterrupt:
        print("\n  收到 Ctrl-C")
    finally:
        stop_flag.value = True
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()

    # ---- 汇总 ----
    if not all_samples:
        print("没有采到样本")
        return 1

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "mean": sum(xs) / n,
            "min": xs[0],
            "max": xs[-1],
            "p50": xs[n // 2],
            "p95": xs[min(n - 1, int(n * 0.95))],
        }

    overall = stats(all_samples)
    steady_s = stats(steady) if steady else overall
    core_means = [sum(c[i] for c in all_cores) / len(all_cores)
                  for i in range(len(all_cores[0]))]
    in_band = sum(1 for s in steady if abs(s - args.target) <= args.tolerance)
    band_ratio = in_band / len(steady) if steady else 0.0

    print("\n=== 结果 ===")
    print(f"  全程     均值 {overall['mean']:.1f}%  最低 {overall['min']:.1f}%  "
          f"最高 {overall['max']:.1f}%")
    if steady:
        print(f"  稳态     均值 {steady_s['mean']:.1f}%  P50 {steady_s['p50']:.1f}%  "
              f"P95 {steady_s['p95']:.1f}%")
    print(f"  每核均值 " + " ".join(f"{c:.0f}" for c in core_means))
    print(f"  有效样本 {len(all_samples)}  丢弃退化样本 {sampler.dropped}")
    print(f"  达标率   {band_ratio:.1%}（|实测-目标| <= {args.tolerance} 个百分点）")
    if args.duration:
        print(f"  持续     {args.duration:.0f} 秒 + {args.settle:.0f} 秒调节期")
    if saturated > 5:
        print(f"  [警告] 占空比一度打满仍远低于目标 —— 说明工作进程提供的负荷上限不够，"
              f"加大 --workers 或缩短采样间隔后再试")

    report = {
        "mode": args.mode,
        "nproc": nproc,
        "workers": workers,
        "target_pct": args.target,
        "tolerance_pp": args.tolerance,
        "duration_s": args.duration,
        "overall": {k: round(v, 2) for k, v in overall.items()},
        "steady": {k: round(v, 2) for k, v in steady_s.items()},
        "per_core_mean": [round(c, 2) for c in core_means],
        "in_band_ratio": round(band_ratio, 4),
        "samples": len(all_samples),
        "dropped_degenerate": sampler.dropped,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as w:
        json.dump(report, w, indent=2, ensure_ascii=False)
    print(f"\n[完成] 报告 -> {args.report}")

    if band_ratio < 0.9:
        print(f"  [提示] 达标率偏低，可能是温度降频或工作进程不足，可加大 --duration 重测")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())