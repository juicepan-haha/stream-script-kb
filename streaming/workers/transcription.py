"""Whisper 常驻转录 Worker。"""

import time

from shared import config
from streaming.state import (
    chunk_queue,
    task_progress,
    text_queue,
    transcript_results,
)


async def whisper_transcribe_worker():
    """消费内存音频块，生成带时间戳的字幕段。"""
    import torch
    from faster_whisper import WhisperModel

    if torch.cuda.is_available():
        gpu_idx = config.WHISPER_DEVICE_INDEX
        if gpu_idx == -1:
            gpu_idx = 0
        total_vram = (
            torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
        )
        usable = total_vram * config.WHISPER_GPU_MEMORY_FRACTION * 0.6
        model_map = {
            "large-v3": 4.5,
            "medium": 2.5,
            "small": 1.5,
            "base": 0.7,
            "tiny": 0.6,
        }
        best = "tiny"
        for model_name, required_vram in model_map.items():
            if required_vram <= usable:
                best = model_name
                break
        device, compute = "cuda", "float16"
        batch = max(4, min(32, int((usable - model_map[best]) / 0.1)))
        torch.cuda.set_per_process_memory_fraction(
            config.WHISPER_GPU_MEMORY_FRACTION, gpu_idx
        )
        print(
            f"[Worker 2][转录] 🖥️ GPU: {total_vram:.1f}GB → "
            f"model={best}, batch_size={batch}"
        )
    else:
        best, device, compute = "base", "cpu", "int8"
        print("[Worker 2][转录] ⚠️ 无 GPU，降级 CPU base")

    print(f"[Worker 2][转录] 加载模型: {best} ...")
    started_at = time.time()
    model = WhisperModel(best, device=device, compute_type=compute)
    print(
        f"[Worker 2][转录] 模型就绪 ({time.time() - started_at:.1f}s)，"
        "等待音频..."
    )

    while True:
        chunk = await text_queue.get()
        try:
            task_id = chunk["task_id"]
            task_progress.setdefault(task_id, {})["stage"] = "transcribing"

            if chunk.get("done"):
                print(
                    f"[Worker 2][转录] [{task_id}] ✅ 转录完成 "
                    f"({chunk.get('total_chunks', 0)} 块)"
                )
                await chunk_queue.put({"task_id": task_id, "done": True})
                continue

            if chunk.get("error"):
                print(
                    f"[Worker 2][转录] [{task_id}] ❌ "
                    f"上游错误: {chunk['error']}"
                )
                continue

            audio = chunk["audio"]
            offset = chunk["offset_sec"]
            chunk_idx = chunk["chunk_idx"]

            print(
                f"[Worker 2][转录] [{task_id}] 块 {chunk_idx} "
                f"(偏移 {offset:.0f}s)...",
                end=" ",
                flush=True,
            )
            transcribe_started_at = time.time()
            segments, _ = model.transcribe(
                audio,
                beam_size=config.WHISPER_BEAM_SIZE,
                language=config.WHISPER_LANGUAGE,
                vad_filter=config.WHISPER_VAD_FILTER,
            )

            segment_list = [
                {
                    "start": segment.start + offset,
                    "end": segment.end + offset,
                    "text": segment.text.strip(),
                }
                for segment in segments
            ]

            elapsed = time.time() - transcribe_started_at
            ratio = len(audio) / 16000 / elapsed if elapsed > 0 else 0
            print(f"OK ({elapsed:.1f}s, {ratio:.1f}x, {len(segment_list)} 段)")

            transcript_results.setdefault(task_id, []).extend(segment_list)
            await chunk_queue.put({
                "task_id": task_id,
                "segments": segment_list,
            })
        except Exception as exc:
            print(
                f"[Worker 2][转录] [{chunk.get('task_id', '?')}] "
                f"❌ 错误: {exc}"
            )
        finally:
            text_queue.task_done()
