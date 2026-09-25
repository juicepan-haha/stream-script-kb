"""转录文本的流式语义切块 Worker。"""

import re

from shared import config
from streaming.state import chunk_queue, enriched_queue, task_progress


SENTENCE_END = re.compile(r"[。！？.!?]$")


async def chunking_worker():
    """按字数、句末标点和停顿把字幕段增量组合成话术块。"""
    buffers: dict[str, list[dict]] = {}
    buffer_chars: dict[str, int] = {}
    chunk_indices: dict[str, int] = {}

    min_chars = config.MIN_CHARS
    max_chars = config.MAX_CHARS
    silence_gap = config.SILENCE_GAP_SEC

    def emit_chunk(task_id: str, entries: list[dict]) -> dict:
        index = chunk_indices.get(task_id, 0)
        combined = "".join(entry["text"] for entry in entries)
        chunk_data = {
            "chunk_id": f"{task_id}_chunk_{index + 1:04d}",
            "source_file": task_id,
            "start_time": entries[0]["start"],
            "end_time": entries[-1]["end"],
            "char_count": len(combined),
            "text": combined,
        }
        chunk_indices[task_id] = index + 1
        return chunk_data

    print("[Worker 3][切块] 就绪，等待转录段...")

    while True:
        item = await chunk_queue.get()
        try:
            task_id = item["task_id"]

            if item.get("done"):
                if task_id in buffers and buffers[task_id]:
                    chunk = emit_chunk(task_id, buffers[task_id])
                    await enriched_queue.put({
                        "task_id": task_id,
                        "chunk": chunk,
                    })
                    print(
                        f"[Worker 3][切块] [{task_id}] 尾部残片 → "
                        f"{chunk['chunk_id']} ({chunk['char_count']} 字)"
                    )
                await enriched_queue.put({"task_id": task_id, "done": True})
                task_progress.setdefault(task_id, {})["stage"] = "chunked"
                buffers.pop(task_id, None)
                buffer_chars.pop(task_id, None)
                chunk_indices.pop(task_id, None)
                print(f"[Worker 3][切块] [{task_id}] ✅ 切块完成")
                continue

            segments = item["segments"]
            if not segments:
                continue

            if task_id not in buffers:
                buffers[task_id] = []
                buffer_chars[task_id] = 0
                chunk_indices[task_id] = 0

            buffer = buffers[task_id]

            for index, segment in enumerate(segments):
                text = segment["text"]
                char_count = len(text)

                if char_count > max_chars:
                    if buffer:
                        chunk = emit_chunk(task_id, buffer)
                        await enriched_queue.put({
                            "task_id": task_id,
                            "chunk": chunk,
                        })
                        print(
                            f"[Worker 3][切块] [{task_id}] 清缓冲区 → "
                            f"{chunk['chunk_id']} ({chunk['char_count']} 字)"
                        )
                        buffer.clear()
                        buffer_chars[task_id] = 0

                    position = 0
                    while position < char_count:
                        segment_text = text[position:position + max_chars]
                        hard_chunk = {
                            "chunk_id": (
                                f"{task_id}_chunk_"
                                f"{chunk_indices[task_id] + 1:04d}"
                            ),
                            "source_file": task_id,
                            "start_time": segment["start"],
                            "end_time": segment["end"],
                            "char_count": len(segment_text),
                            "text": segment_text,
                        }
                        chunk_indices[task_id] += 1
                        await enriched_queue.put({
                            "task_id": task_id,
                            "chunk": hard_chunk,
                        })
                        position += max_chars
                    continue

                buffer.append(segment)
                buffer_chars[task_id] += char_count

                should_split = buffer_chars[task_id] >= max_chars
                if not should_split and buffer_chars[task_id] >= min_chars:
                    combined = "".join(entry["text"] for entry in buffer)
                    if SENTENCE_END.search(combined):
                        should_split = True
                    elif index + 1 < len(segments):
                        next_segment = segments[index + 1]
                        should_split = (
                            next_segment["start"] - segment["end"] > silence_gap
                        )

                if should_split:
                    chunk = emit_chunk(task_id, buffer)
                    await enriched_queue.put({
                        "task_id": task_id,
                        "chunk": chunk,
                    })
                    buffer.clear()
                    buffer_chars[task_id] = 0
        except Exception as exc:
            print(
                f"[Worker 3][切块] [{item.get('task_id', '?')}] "
                f"❌ 错误: {exc}"
            )
        finally:
            chunk_queue.task_done()
