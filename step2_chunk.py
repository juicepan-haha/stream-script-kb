#!/usr/bin/env python3
"""Step 2: 将 VTT 字幕文件按字数切块 (500-1000 字/块)。

输入: data/transcripts/*.vtt
输出: data/chunks.json
"""
import json
import re
from pathlib import Path

import config


_SENTENCE_END = re.compile(r"[。！？.!?]$")


def _parse_timestamp(ts: str) -> float:
    """将 VTT 时间戳 'HH:MM:SS.mmm' 转为秒数。"""
    parts = ts.strip().split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def parse_vtt(content: str) -> list[dict]:
    """解析 VTT 文本, 返回字幕条目列表。

    返回: [{"start": float, "end": float, "text": str}, ...]
    """
    entries = []
    lines = content.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        # 跳过头部和空行
        if line in ("", "WEBVTT") or line.startswith("Language:") or line.startswith("Kind:"):
            i += 1
            continue
        # 匹配时间戳行
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start = _parse_timestamp(parts[0])
                end = _parse_timestamp(parts[1])
                # 收集所有文本行直到空行
                text_parts = []
                i += 1
                while i < len(lines) and lines[i].strip() != "":
                    text_parts.append(lines[i].strip())
                    i += 1
                entries.append({
                    "start": start,
                    "end": end,
                    "text": "".join(text_parts),
                })
                continue
        i += 1

    return entries


def build_chunks(
    entries: list[dict],
    source_file: str,
    min_chars: int = 500,
    max_chars: int = 1000,
) -> list[dict]:
    """将字幕条目按字数拼接为 chunk。"""
    if not entries:
        return []

    chunks = []
    buffer_entries = []
    buffer_chars = 0
    chunk_idx = 0
    stem = Path(source_file).stem

    for entry in entries:
        text = entry["text"]
        char_count = len(text)

        # 单条超过 max_chars: 先保存 buffer, 然后硬切
        if char_count > max_chars:
            if buffer_entries:
                chunks.append(_make_chunk(buffer_entries, stem, chunk_idx))
                chunk_idx += 1
                buffer_entries = []
                buffer_chars = 0
            # 硬切
            pos = 0
            while pos < char_count:
                seg = text[pos:pos + max_chars]
                chunks.append({
                    "chunk_id": f"{stem}_{chunk_idx + 1:03d}",
                    "source_file": source_file,
                    "start_time": entry["start"],
                    "end_time": entry["end"],
                    "char_count": len(seg),
                    "text": seg,
                })
                chunk_idx += 1
                pos += max_chars
            continue

        buffer_entries.append(entry)
        buffer_chars += char_count

        combined = "".join(e["text"] for e in buffer_entries)

        # 达到 max_chars: 强制切分
        if buffer_chars >= max_chars:
            chunks.append(_make_chunk(buffer_entries, stem, chunk_idx))
            chunk_idx += 1
            buffer_entries = []
            buffer_chars = 0
        # 达到 min_chars 且有句末标点: 自然切分
        elif buffer_chars >= min_chars and _SENTENCE_END.search(combined):
            chunks.append(_make_chunk(buffer_entries, stem, chunk_idx))
            chunk_idx += 1
            buffer_entries = []
            buffer_chars = 0

    # 尾部残片
    if buffer_entries:
        if buffer_chars < min_chars and chunks:
            chunks[-1]["text"] += "".join(e["text"] for e in buffer_entries)
            chunks[-1]["char_count"] = len(chunks[-1]["text"])
            chunks[-1]["end_time"] = buffer_entries[-1]["end"]
        else:
            chunks.append(_make_chunk(buffer_entries, stem, chunk_idx))

    return chunks


def _make_chunk(entries: list[dict], stem: str, idx: int) -> dict:
    return {
        "chunk_id": f"{stem}_{idx + 1:03d}",
        "source_file": stem + ".vtt",
        "start_time": entries[0]["start"],
        "end_time": entries[-1]["end"],
        "char_count": sum(len(e["text"]) for e in entries),
        "text": "".join(e["text"] for e in entries),
    }


def main():
    vtt_files = sorted(config.TRANSCRIPT_DIR.glob("*.vtt"))
    total = len(vtt_files)
    print(f"[STEP2] Found {total} VTT files")

    if total == 0:
        print("[STEP2] No VTT files found. Run step1_transcribe.py first.")
        return

    all_chunks = []
    for i, vtt_path in enumerate(vtt_files, 1):
        source = vtt_path.name
        print(f"[STEP2] [{i}/{total}] Processing {source}")
        content = vtt_path.read_text(encoding="utf-8")
        entries = parse_vtt(content)
        chunks = build_chunks(entries, source, config.MIN_CHARS, config.MAX_CHARS)
        all_chunks.extend(chunks)
        print(
            f"[STEP2] [{i}/{total}] {source} → {len(chunks)} chunks "
            f"({len(entries)} subtitle entries)"
        )

    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[STEP2] Done. {len(all_chunks)} chunks → {config.CHUNKS_FILE}")

    if all_chunks:
        lengths = [c["char_count"] for c in all_chunks]
        print(
            f"[STEP2] Chunk stats: min={min(lengths)}, max={max(lengths)}, "
            f"avg={sum(lengths)/len(lengths):.0f}"
        )


if __name__ == "__main__":
    main()
