#!/usr/bin/env python3
"""Step 0: 用 Python requests 并发下载 m3u8 的所有 TS 片段，本地 ffmpeg 合并。

特征:
  - ThreadPoolExecutor 16 并发下载，5000+ 片段数分钟完成
  - 单片段 3 次重试 + 异常隔离
  - 线程安全进度上报
"""
import argparse
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

import config

# CDN 并发上限，避免触发反爬
MAX_WORKERS = 16
MAX_RETRIES = 3


def _download_one(seg_url: str, fpath: Path,
                  headers: dict, retries: int = MAX_RETRIES) -> Path | None:
    """下载单个 TS 片段（线程安全）。成功返回 Path，失败返回 None。"""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(seg_url, headers=headers, timeout=60)
            r.raise_for_status()
            fpath.write_bytes(r.content)
            return fpath
        except Exception:
            if attempt < retries:
                continue
    return None


def download_m3u8_segments(m3u8_url: str, cookie: str = "") -> list[Path]:
    """并发下载 m3u8 中的所有 TS 片段到临时目录，返回按序号排列的路径列表。"""
    headers = {"User-Agent": "Mozilla/5.0"}
    if cookie:
        headers["Cookie"] = cookie

    # 下载 m3u8 播放列表
    print(f"[STEP0] Fetching playlist: {m3u8_url}")
    resp = requests.get(m3u8_url, headers=headers, timeout=30)
    resp.raise_for_status()
    playlist = resp.text

    # 解析 segment URLs
    base_url = m3u8_url.rsplit("/", 1)[0]
    segments = []
    for line in playlist.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            if line.startswith("http"):
                segments.append(line)
            else:
                segments.append(f"{base_url}/{line}")

    total = len(segments)
    print(f"[STEP0] Found {total} segments, downloading with {MAX_WORKERS} workers")

    tmpdir = Path(tempfile.mkdtemp(prefix="stream_segments_"))

    # 构建 (idx, url, path) 任务列表
    tasks = []
    for i, seg_url in enumerate(segments, 1):
        fpath = tmpdir / f"{i:05d}.ts"
        tasks.append((i, seg_url, fpath))

    # 并发下载
    results: dict[int, Path] = {}   # idx → Path，保证顺序
    done_lock = threading.Lock()
    done_count = 0
    failed_count = 0

    def _task(args):
        idx, url, fpath = args
        nonlocal done_count, failed_count
        result = _download_one(url, fpath, headers)
        with done_lock:
            nonlocal done_count, failed_count
            done_count += 1
            if result is None:
                failed_count += 1
            if done_count % 200 == 0 or done_count == total:
                print(f"[STEP0] [{done_count}/{total}] "
                      f"({failed_count} failed)")
        return idx, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_task, t): t[0] for t in tasks}
        for future in as_completed(futures):
            idx, path = future.result()
            if path is not None:
                results[idx] = path

    # 按序号排序返回
    segment_files = [results[i] for i in sorted(results)]

    if failed_count > 0:
        print(f"[STEP0] ⚠️  {failed_count}/{total} segments failed")

    print(f"[STEP0] Downloaded {len(segment_files)}/{total} segments")
    return segment_files


def merge_to_m4a(segment_files: list[Path], output_path: Path) -> bool:
    """用 ffmpeg concat 将所有 TS 片段合并为 m4a。"""
    if output_path.exists():
        print(f"[STEP0] ⏭️  Already exists, skipping: {output_path.name}")
        return True

    if not segment_files:
        print("[STEP0] No segments to merge", file=sys.stderr)
        return False

    # 创建 concat 文件列表
    concat_list = segment_files[0].parent / "concat.txt"
    with open(concat_list, "w") as f:
        for sf in segment_files:
            f.write(f"file '{sf}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        config.FFMPEG_PATH,
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-vn",
        "-acodec", "copy",      # TS 已经是 AAC，直接复制，秒级完成
        "-y",
        str(output_path),
    ]

    print(f"[STEP0] Merging {len(segment_files)} segments → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[STEP0] ffmpeg error:\n{result.stderr[:2000]}", file=sys.stderr)
        return False

    size = output_path.stat().st_size
    print(f"[STEP0] Done: {output_path.name} ({size / 1024 / 1024:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Step 0: 下载 m3u8 直播回放 → m4a 音频")
    parser.add_argument("--url", required=True, help="m3u8 URL")
    parser.add_argument("--name", required=True, help="Output filename (without extension)")
    parser.add_argument("--cookie", type=str, default="", help="HTTP Cookie string")
    args = parser.parse_args()

    segment_files = download_m3u8_segments(args.url, args.cookie)

    output_path = config.CHUNK_DIR / f"{args.name}.m4a"
    ok = merge_to_m4a(segment_files, output_path)

    # 清理临时文件
    if segment_files:
        tmpdir = segment_files[0].parent
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
