#!/usr/bin/env python3
"""Step 0: 用 ffmpeg 下载 m3u8 直播流/回放为音频文件。

输入: m3u8 URL
输出: config.CHUNK_DIR/*.m4a

用法:
  python step0_download.py --url "https://..." --name "主播名_20260712"
  python step0_download.py --url "https://..." --name "test" --duration 30
  python step0_download.py --batch data/stream_urls.json --duration 3600

前置:
  ffmpeg 需已安装（config.FFMPEG_PATH）
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config


def download_m3u8(url: str, output_path: Path, duration: int = 0) -> bool:
    """用 ffmpeg 下载 m3u8 并转码为 m4a。

    duration=0 表示不限时长（适用于有结束的回放），
    >0 表示录制 N 秒（适用于直播流）。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[STEP0] ⏭️  已存在，跳过: {output_path.name}")
        return True

    cmd = [
        config.FFMPEG_PATH,
        # --- 稳定连接 ---
        "-reconnect", "1",
        "-reconnect_at_eof", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        "-timeout", "30000000",     # 30s 超时
        # --- 可靠解析 ---
        "-probesize", "50M",
        "-analyzeduration", "100M",
        "-fflags", "+genpts",       # 修复丢失的时间戳
        # --- 输入输出 ---
        "-i", url,
        "-vn",                    # 不要视频
        "-acodec", "aac",         # AAC 音频编码
        "-b:a", config.FFMPEG_BITRATE,
        "-f", "mp4",              # MP4 容器
        "-movflags", "+faststart",  # 下载完成后可流式播放
        "-y",                     # 覆盖已有文件
    ]
    if duration > 0:
        cmd += ["-t", str(duration)]
    cmd.append(str(output_path))

    duration_str = f" (限时 {duration} 秒)" if duration > 0 else ""
    print(f"[STEP0] 📥 下载中 → {output_path.name}{duration_str}")
    print(f"[STEP0] 命令: ffmpeg -i <m3u8> -vn -acodec aac -b:a {config.FFMPEG_BITRATE}{duration_str}")

    # 显示 ffmpeg 的实时进度
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    ) as proc:
        last_line = ""
        for line in proc.stdout or []:
            if "time=" in line:
                # ffmpeg 进度行，只显示最后一行
                last_line = line.strip()
            elif "error" in line.lower() or "Error" in line:
                print(f"  {line.strip()}")

        if last_line:
            # 从 time=00:01:23.45 格式提取进度
            import re
            m = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", last_line)
            if m:
                h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                total_sec = h * 3600 + mnt * 60 + s
                print(f"[STEP0] ⏱️  录制时长: {total_sec // 60} 分 {total_sec % 60:.0f} 秒")

    if output_path.exists():
        size = output_path.stat().st_size
        print(f"[STEP0] ✅ 完成: {output_path.name} ({size / 1024 / 1024:.1f} MB)")
        return True
    else:
        print(f"[STEP0] ❌ 未生成文件", file=sys.stderr)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Step 0: ffmpeg 下载 m3u8 → m4a",
        epilog="""
示例:
  %(prog)s --url "https://...m3u8" --name "主播_20260712"
  %(prog)s --url "https://...m3u8" --name "test" --duration 30
  %(prog)s --batch data/stream_urls.json --duration 3600
        """,
    )
    parser.add_argument("--url", help="m3u8 地址")
    parser.add_argument("--name", help="输出文件名（不含后缀）")
    parser.add_argument(
        "--duration", type=int, default=0,
        help="录制时长（秒），0=不限（默认: %(default)s）",
    )
    parser.add_argument(
        "--batch", type=Path,
        help="批量：JSON 文件，格式 [{'url': '...', 'name': '...', 'duration': 3600}, ...]",
    )
    args = parser.parse_args()

    if args.url:
        name = args.name or f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output = config.CHUNK_DIR / f"{name}.m4a"
        ok = download_m3u8(args.url, output, args.duration)
        if not ok:
            sys.exit(1)

    elif args.batch:
        if not args.batch.exists():
            print(f"[STEP0] ❌ 文件不存在: {args.batch}", file=sys.stderr)
            sys.exit(1)
        with open(args.batch) as f:
            items = json.load(f)
        ok = fail = 0
        for i, item in enumerate(items, 1):
            name = item.get("name") or f"download_{i:03d}"
            dur = item.get("duration", args.duration)
            output = config.CHUNK_DIR / f"{name}.m4a"
            print(f"\n[STEP0] [{i}/{len(items)}] {name}")
            if download_m3u8(item["url"], output, dur):
                ok += 1
            else:
                fail += 1
        print(f"\n[STEP0] 完成: {ok}/{len(items)} 成功, {fail} 失败")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
