#!/usr/bin/env python3
"""试验脚本：用 yt-dlp 下载直播回放音频。

用法:
  # 查看 URL 信息和可用格式（不下载）
  python experiments/test_ytdlp.py --url "https://..." --dry-run

  # 下载音频
  python experiments/test_ytdlp.py --url "https://..."

  # 指定输出目录和文件名
  python experiments/test_ytdlp.py --url "https://..." --output-dir /tmp --name test_001

前置:
  pip install yt-dlp
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# 尝试引入项目配置（可选，不强制）
try:
    import config
    DEFAULT_OUTPUT_DIR = config.CHUNK_DIR
except ImportError:
    DEFAULT_OUTPUT_DIR = Path("./downloads")


def probe_info(url: str, probe_timeout: int = 120) -> dict | None:
    """用 yt-dlp 探测 URL 的元信息（标题、时长、格式等），不下载。

    返回解析后的 JSON 字典，失败返回 None。
    """
    print(f"[INFO] 探测: {url}")

    # 先检查网络连通性
    domain = url.split("/")[2] if "//" in url else url
    ping = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--connect-timeout", "5", f"https://{domain}"],
        capture_output=True, text=True, timeout=10,
    )
    if ping.returncode != 0:
        print(f"[WARN] 无法连接到 {domain}（网络不通或被屏蔽）")
        print(f"       提示: 某些平台（如 YouTube）可能需要代理才能访问")
        print(f"       你可以通过 HTTP_PROXY 环境变量设置代理:")
        print(f"         HTTP_PROXY=http://127.0.0.1:7890 {sys.argv[0]} ...")

    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True,
            text=True,
            timeout=probe_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[ERROR] 探测超时（{probe_timeout}秒）")
        print(f"       可能原因: 网络连接慢、视频太长、或被屏蔽")
        return None

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        print(f"[ERROR] yt-dlp 探测失败:\n{error_msg}")
        if "HTTP Error 403" in error_msg:
            print(f"       提示: 该平台可能需要 cookies 或登录态")
            print(f"       尝试添加 --cookies cookies.txt 参数")
        return None

    info = json.loads(result.stdout.strip())
    return info


def download_audio(url: str, output_path: Path, dl_timeout: int = 600) -> bool:
    """下载最佳音频流并转为 m4a。

    已存在的文件自动跳过（幂等）。
    """
    if output_path.exists():
        print(f"[SKIP] 已存在: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",          # 最佳音频
        "--extract-audio",               # 提取音频
        "--audio-format", "m4a",         # 转为 m4a
        "--audio-quality", "0",          # 最佳音质
        "-o", str(output_path),
        "--print", "after_move:filepath",  # 下载完成后打印实际路径
        url,
    ]

    print(f"[DOWNLOAD] 目标: {output_path}")
    print(f"[DOWNLOAD] 命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=dl_timeout)
    except subprocess.TimeoutExpired:
        print(f"[ERROR] 下载超时（{dl_timeout}秒）")
        return False

    if result.returncode != 0:
        error_msg = result.stderr.strip()
        print(f"[ERROR] 下载失败:\n{error_msg}")
        return False

    print(f"[DONE] ✅ 下载完成: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="试验：用 yt-dlp 下载直播回放音频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --dry-run
  %(prog)s --url "https://www.bilibili.com/video/BV1xx411c7mD"
  %(prog)s --url "https://live.taobao.com/..." --name "taobao_live_001"
        """,
    )
    parser.add_argument("--url", required=True, help="直播回放或视频 URL")
    parser.add_argument("--name", help="输出文件名（不含后缀，默认自动）")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="输出目录（默认: %(default)s）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只探测信息，不下载",
    )
    parser.add_argument(
        "--probe-timeout",
        type=int,
        default=120,
        help="探测超时秒数（默认: %(default)s）",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=600,
        help="下载超时秒数（默认: %(default)s）",
    )
    args = parser.parse_args()

    # ---------- Step 1: 探测 ----------
    print("=" * 60)
    print(f"URL: {args.url}")
    print("=" * 60)

    info = probe_info(args.url, args.probe_timeout)
    if info is None:
        sys.exit(1)

    # 展示关键信息
    title = info.get("title", "未知标题")
    duration = info.get("duration", 0)
    ext = info.get("ext", "?")
    webpage_url = info.get("webpage_url", args.url)

    print(f"\n📺 标题: {title}")
    print(f"⏱️  时长: {duration // 60} 分 {duration % 60} 秒")
    print(f"📦 格式: {ext}")
    print(f"🔗 来源: {webpage_url}")

    # 如果有缩略图
    thumbnail = info.get("thumbnail")
    if thumbnail:
        print(f"🖼️  封面: {thumbnail}")

    # 如果有上传者
    uploader = info.get("uploader") or info.get("channel") or ""
    if uploader:
        print(f"👤 主播/UP主: {uploader}")

    # 如果有描述，截取前 200 字
    description = info.get("description", "")
    if description:
        desc_short = description[:200].replace("\n", " ")
        print(f"📝 简介: {desc_short}...")

    # 打印格式列表（简略）
    formats = info.get("formats", [])
    if formats:
        audio_formats = [f for f in formats if f.get("vcodec") == "none" and f.get("acodec") != "none"]
        print(f"\n🎵 可用音频格式: {len(audio_formats)} 种")
        for af in audio_formats[:5]:
            abr = af.get("abr", "?")
            acodec = af.get("acodec", "?")
            fmt_id = af.get("format_id", "?")
            print(f"   [{fmt_id}] {acodec} @ {abr}kbps")
        if len(audio_formats) > 5:
            print(f"   ... 还有 {len(audio_formats) - 5} 种（用 --dry-run 查看完整列表）")

    print()

    # ---------- Step 2: 下载（除非 --dry-run） ----------
    if args.dry_run:
        print("[DRY-RUN] 不下载，仅打印信息")
        return

    # 确定输出文件名
    if args.name:
        filename = f"{args.name}.m4a"
    else:
        # 用视频标题做文件名（清理特殊字符）
        safe_title = "".join(c if c.isalnum() or c in " _-." else "_" for c in title)
        safe_title = safe_title.strip()[:80]
        filename = f"{safe_title}.m4a"

    output_path = args.output_dir / filename
    success = download_audio(args.url, output_path, args.download_timeout)

    if success:
        file_size = output_path.stat().st_size
        print(f"\n📊 文件大小: {file_size / 1024 / 1024:.1f} MB")
        print(f"📂 保存位置: {output_path.resolve()}")
        print(f"\n💡 下一步: 运行管道处理此文件")
        print(f"   cp '{output_path}' ~/stream-script-kb/audio_chunks/")
        print(f"   cd ~/stream-script-kb && python step1_transcribe.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
