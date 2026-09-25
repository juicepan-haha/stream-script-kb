"""直播音频下载和内存解码 Worker。"""

import asyncio
import time

import numpy as np

from shared import config
from streaming.state import download_queue, task_progress, text_queue


CHUNK_SECONDS = 30
BYTES_PER_CHUNK = CHUNK_SECONDS * 16000 * 2  # s16le mono 16kHz


def parse_cookies(netscape_path: str) -> str:
    """将 Netscape 格式 cookie 转为 HTTP Cookie 请求头。"""
    cookies = []
    try:
        with open(netscape_path) as cookie_file:
            for line in cookie_file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    value = parts[6] if len(parts) >= 7 else ""
                    cookies.append(f"{parts[5]}={value}")
    except FileNotFoundError:
        pass
    return "; ".join(cookies)


async def audio_download_and_decode_worker():
    """通过 ffmpeg 下载音频，按 30 秒切成 NumPy 块送入转录队列。"""
    cookie_str = parse_cookies("cookies.txt")
    header_line = f"Cookie: {cookie_str}\r\n" if cookie_str else ""

    print("[Worker 1][下载] 就绪，等待任务...")

    while True:
        task_data = await download_queue.get()
        url = task_data["url"]
        task_id = task_data["task_id"]
        print(f"[Worker 1][下载] [{task_id}] 开始流式捕获: {url}")

        task_progress.setdefault(task_id, {})["stage"] = "downloading"

        cmd = [
            config.FFMPEG_PATH,
            "-headers", header_line,
            "-i", url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            chunk_idx = 0
            offset_sec = 0.0
            started_at = time.time()

            while True:
                data = await proc.stdout.read(BYTES_PER_CHUNK)
                if not data:
                    break

                samples = (
                    np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
                await text_queue.put({
                    "task_id": task_id,
                    "audio": samples,
                    "chunk_idx": chunk_idx,
                    "offset_sec": offset_sec,
                })

                chunk_idx += 1
                offset_sec += CHUNK_SECONDS

                if chunk_idx % 10 == 0:
                    elapsed = time.time() - started_at
                    print(
                        f"[Worker 1][下载] [{task_id}] 已推送 {chunk_idx} 块 "
                        f"({offset_sec / 60:.0f} 分钟), "
                        f"{offset_sec / elapsed:.1f}x 实时率"
                    )

            await text_queue.put({
                "task_id": task_id,
                "done": True,
                "total_chunks": chunk_idx,
            })

            stderr_data = await proc.stderr.read()
            await proc.wait()
            if proc.returncode != 0:
                err_msg = stderr_data.decode(errors="replace")[-500:]
                print(
                    f"[Worker 1][下载] [{task_id}] ⚠️ "
                    f"ffmpeg rc={proc.returncode}: {err_msg}"
                )

            elapsed = time.time() - started_at
            realtime_ratio = offset_sec / elapsed if elapsed > 0 else 0
            print(
                f"[Worker 1][下载] [{task_id}] ✅ 完成: {chunk_idx} 块, "
                f"{elapsed:.0f}s, {realtime_ratio:.1f}x 实时率"
            )
        except Exception as exc:
            print(f"[Worker 1][下载] [{task_id}] ❌ 错误: {exc}")
            await text_queue.put({"task_id": task_id, "error": str(exc)})
        finally:
            download_queue.task_done()
