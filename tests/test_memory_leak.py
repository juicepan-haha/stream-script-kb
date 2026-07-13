#!/usr/bin/env python3
"""
Smoke Test: 内存泄漏检测（模拟 5 路并发 × 循环压测）

验证目标:
  1. asyncio.Queue 在长时间高频 put/get 后不发生内存泄漏
  2. 4 级 Worker 管线在持续背压下内存占用稳定
  3. 第 55 分钟内存 ≈ 第 1 分钟内存

用法:
  python tests/test_memory_leak.py [--duration 3600] [--interval 60]
"""
import asyncio
import gc
import os
import sys
import time
import argparse
import random
import tracemalloc

import numpy as np

# =========================================================================
# Mock 管线（模拟真实 Worker 行为，但无外部依赖）
# =========================================================================

QUEUE_SIZES = {"download": 20, "text": 50, "chunk": 100, "enriched": 100}

# 模拟 30 秒的音频数据 (16kHz mono s16le → float32)
_MOCK_AUDIO = np.zeros(30 * 16000, dtype=np.float32)


async def mock_download_worker(dl_q, text_q, task_count: int):
    """[Mock Worker 1] 模拟 ffmpeg 下载 + 推 numpy 块到 text_queue。"""
    for _ in range(task_count):
        task_id = f"mock_{random.randint(1000, 9999)}"
        blocks = random.randint(10, 60)  # 5-30 分钟音频
        for blk in range(blocks):
            await text_q.put({
                "task_id": task_id,
                "audio": _MOCK_AUDIO.copy(),  # 模拟真实 numpy 数组
                "chunk_idx": blk,
                "offset_sec": blk * 30.0,
            })
        await text_q.put({"task_id": task_id, "done": True, "total_chunks": blocks})
        dl_q.task_done()


async def mock_transcribe_worker(text_q, chunk_q):
    """[Mock Worker 2] 模拟 Whisper 转录，输出带时间戳的段列表。"""
    while True:
        chunk = await text_q.get()
        try:
            task_id = chunk["task_id"]
            if chunk.get("done"):
                await chunk_q.put({"task_id": task_id, "done": True})
                text_q.task_done()
                continue

            # 模拟转录：每块生成 1-5 个段
            segs = []
            offset = chunk["offset_sec"]
            for _ in range(random.randint(1, 5)):
                dur = random.uniform(1.0, 15.0)
                segs.append({
                    "start": offset,
                    "end": offset + dur,
                    "text": "测试话术文本 " * random.randint(5, 30),
                })
                offset += dur

            await chunk_q.put({"task_id": task_id, "segments": segs})
        finally:
            text_q.task_done()


async def mock_chunking_worker(chunk_q, enriched_q):
    """[Mock Worker 3] 模拟滑动窗口切块。"""
    buffers: dict[str, list] = {}
    while True:
        item = await chunk_q.get()
        try:
            task_id = item["task_id"]
            if item.get("done"):
                if task_id in buffers and buffers[task_id]:
                    combined = "".join(s["text"] for s in buffers[task_id])
                    await enriched_q.put({
                        "task_id": task_id,
                        "chunk": {"text": combined, "char_count": len(combined)},
                    })
                    buffers.pop(task_id, None)
                await enriched_q.put({"task_id": task_id, "done": True})
                chunk_q.task_done()
                continue

            if task_id not in buffers:
                buffers[task_id] = []
            buffers[task_id].extend(item["segments"])

            # 模拟切块条件
            if len("".join(s["text"] for s in buffers[task_id])) >= 500:
                combined = "".join(s["text"] for s in buffers[task_id])
                await enriched_q.put({
                    "task_id": task_id,
                    "chunk": {"text": combined, "char_count": len(combined)},
                })
                buffers[task_id] = []
        finally:
            chunk_q.task_done()


async def mock_enrich_worker(enriched_q, results: list):
    """[Mock Worker 4] 模拟 DeepSeek 富化。"""
    while True:
        item = await enriched_q.get()
        try:
            task_id = item["task_id"]
            if item.get("done"):
                results.append({"task_id": task_id, "done": True})
                enriched_q.task_done()
                continue
            # 模拟 API 调用延迟
            await asyncio.sleep(random.uniform(0.05, 0.2))
            results.append({
                "task_id": task_id,
                "chunk_id": f"{task_id}_chunk_{random.randint(1,9999)}",
                "icebreaker": "测试破冰话术",
                "painpoint": "测试痛点话术",
                "mechanism": "测试卖点话术",
                "close_order": "测试促单话术",
            })
        finally:
            enriched_q.task_done()


# =========================================================================
# 内存监控
# =========================================================================

def _get_memory_mb() -> float:
    """获取当前进程 RSS 内存 (MB)。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return -1.0


def _get_python_heap() -> float:
    """通过 tracemalloc 获取 Python 堆内存 (MB)。"""
    current, peak = tracemalloc.get_traced_memory()
    return current / 1024 / 1024


# =========================================================================
# 主测试
# =========================================================================

async def run_stress_test(duration_sec: int = 600, report_interval: int = 60):
    """运行内存压力测试。"""
    # 创建队列
    dl_q = asyncio.Queue(maxsize=QUEUE_SIZES["download"])
    text_q = asyncio.Queue(maxsize=QUEUE_SIZES["text"])
    chunk_q = asyncio.Queue(maxsize=QUEUE_SIZES["chunk"])
    enriched_q = asyncio.Queue(maxsize=QUEUE_SIZES["enriched"])
    results = []

    # 启动 tracemalloc
    tracemalloc.start()

    # 启动 4 个 Mock Worker
    tasks_per_round = 5  # 模拟 5 个直播间
    workers = [
        asyncio.create_task(
            mock_download_worker(dl_q, text_q, tasks_per_round)
        ),
        asyncio.create_task(mock_transcribe_worker(text_q, chunk_q)),
        asyncio.create_task(mock_chunking_worker(chunk_q, enriched_q)),
        asyncio.create_task(mock_enrich_worker(enriched_q, results)),
    ]

    print(f"{'='*70}")
    print(f"内存泄漏 Smoke Test 开始")
    print(f"测试时长: {duration_sec}s ({duration_sec/60:.0f} 分钟)")
    print(f"每轮注入: {tasks_per_round} 个任务")
    print(f"队列上限: {QUEUE_SIZES}")
    print(f"{'='*70}")

    start_time = time.time()
    snapshots = []
    round_num = 0

    while time.time() - start_time < duration_sec:
        round_num += 1
        elapsed = time.time() - start_time

        # 注入一轮任务
        for _ in range(tasks_per_round):
            await dl_q.put({"url": f"mock://live{random.randint(1,5)}",
                            "task_id": f"task_{int(time.time()*1000)}_{round_num}"})

        # 等待本轮任务全部流过管道
        await dl_q.join()

        # 强制 GC
        gc.collect()

        # 采样内存
        elapsed_min = elapsed / 60
        rss_mb = _get_memory_mb()
        heap_mb = _get_python_heap()

        snapshots.append((elapsed_min, rss_mb, heap_mb))
        print(f"[{elapsed_min:6.1f}min] RSS={rss_mb:8.1f}MB  Heap={heap_mb:8.1f}MB  "
              f"Queues: dl={dl_q.qsize()} txt={text_q.qsize()} "
              f"chk={chunk_q.qsize()} enr={enriched_q.qsize()}  "
              f"Results: {len(results)}")

        if round_num % 5 == 0:
            # 定期清理结果列表（模拟真实场景中查询后释放）
            results.clear()
            gc.collect()

        # 等待到下一个采样点
        await asyncio.sleep(max(0, report_interval - (time.time() - start_time) % report_interval))

    # --- 分析 ---
    tracemalloc.stop()

    if len(snapshots) < 2:
        print("\n❌ 采样点不足，无法分析")
        return

    first_rss = snapshots[0][1]
    # 取后半段中位数 vs 前半段中位数
    mid = len(snapshots) // 2
    first_half_avg = sum(s[1] for s in snapshots[:mid]) / mid
    second_half_avg = sum(s[1] for s in snapshots[mid:]) / (len(snapshots) - mid)

    growth_pct = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    print(f"\n{'='*70}")
    print(f"内存泄漏分析报告")
    print(f"{'='*70}")
    print(f"前半段平均 RSS:     {first_half_avg:8.1f} MB")
    print(f"后半段平均 RSS:     {second_half_avg:8.1f} MB")
    print(f"增长:               {growth_pct:+.1f}%")
    print(f"")

    if growth_pct > 15:
        print(f"❌ FAIL: 内存增长 {growth_pct:.1f}% > 15%，可能存在内存泄漏！")
        print(f"   请检查: asyncio.Queue 是否有堆积、结果列表是否无界增长、")
        print(f"   numpy 数组是否有循环引用")
    elif growth_pct > 5:
        print(f"⚠️  WARN: 内存增长 {growth_pct:.1f}%，轻微上涨（可能在正常范围）")
    else:
        print(f"✅ PASS: 内存增长 {growth_pct:.1f}% ≤ 5%，无内存泄漏")

    # 详细时间线
    print(f"\n详细时间线 (RSS MB):")
    for elapsed_min, rss, heap in snapshots:
        bar = "█" * int(rss / max(s[1] for s in snapshots) * 40)
        print(f"  {elapsed_min:5.0f}min  {rss:7.1f}MB  {bar}")

    # 关 Worker
    for w in workers:
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description="Smoke Test: 内存泄漏检测")
    parser.add_argument("--duration", type=int, default=600,
                        help="测试时长（秒），默认 600 (10 分钟)")
    parser.add_argument("--interval", type=int, default=60,
                        help="采样间隔（秒），默认 60")
    args = parser.parse_args()

    asyncio.run(run_stress_test(args.duration, args.interval))


if __name__ == "__main__":
    main()
