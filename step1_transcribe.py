#!/usr/bin/env python3
"""Step 1: 使用 faster-whisper 将 .m4a 音频转录为 VTT 字幕文件。

输出: data/transcripts/<同名>.vtt
断点续跑: 已有 .vtt 的文件自动跳过。
"""
import os
import sys
from pathlib import Path

import config


def _format_vtt(segments, info) -> str:
    """将 faster-whisper 的 segments 转为 VTT 格式字符串。"""
    lines = ["WEBVTT\n"]
    if info and info.language:
        lines.append(f"Language: {info.language}\n")
    for seg in segments:
        start = seg.start
        end = seg.end
        start_ts = (
            f"{int(start // 3600):02d}:{int(start % 3600 // 60):02d}:{start % 60:06.3f}"
        )
        end_ts = (
            f"{int(end // 3600):02d}:{int(end % 3600 // 60):02d}:{end % 60:06.3f}"
        )
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def main():
    from faster_whisper import WhisperModel

    print(f"[STEP1] Loading model: {config.WHISPER_MODEL}")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )

    chunks = sorted(
        [f for f in os.listdir(config.CHUNK_DIR) if f.endswith(".m4a")]
    )
    total = len(chunks)
    print(f"[STEP1] Found {total} .m4a files")

    for i, chunk in enumerate(chunks, 1):
        vtt_name = Path(chunk).stem + ".vtt"
        vtt_path = config.TRANSCRIPT_DIR / vtt_name

        if vtt_path.exists():
            print(f"[STEP1] [{i}/{total}] {chunk} → SKIP (already exists)")
            continue

        chunk_path = os.path.join(config.CHUNK_DIR, chunk)
        print(f"[STEP1] [{i}/{total}] {chunk} → {vtt_name}")

        try:
            segments, info = model.transcribe(
                chunk_path,
                beam_size=config.WHISPER_BEAM_SIZE,
                language=config.WHISPER_LANGUAGE,
            )
            vtt_content = _format_vtt(segments, info)
            vtt_path.write_text(vtt_content, encoding="utf-8")
            print(f"[STEP1] [{i}/{total}] ✅ {vtt_name} done")
        except Exception as e:
            print(
                f"[STEP1] [{i}/{total}] ❌ {chunk} failed: {e}",
                file=sys.stderr,
            )

        # 显存清理
        import torch
        torch.cuda.empty_cache()

    print(f"[STEP1] All done. VTT files in {config.TRANSCRIPT_DIR}")


if __name__ == "__main__":
    main()
