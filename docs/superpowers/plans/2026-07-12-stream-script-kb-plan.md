# Stream Script KB 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建直播话术知识库 — 从音频转录到语义搜索的完整管道

**Architecture:** 5 个独立脚本串行执行，中间结果落 JSON/文件，最后 Streamlit 提供 Web 检索界面。每步输入输出独立，失败可重跑。

**Tech Stack:** faster-whisper (large-v3, CUDA), DeepSeek API (deepseek-chat), BAAI/bge-small-zh-v1.5, PostgreSQL + pgvector, Streamlit

## Global Constraints

- Python >= 3.10
- faster-whisper 使用 CUDA, device="cuda", compute_type="float16"
- DeepSeek API key 从环境变量 `DEEPSEEK_API_KEY` 读取
- Embedding 模型: BAAI/bge-small-zh-v1.5, normalize_embeddings=True
- PostgreSQL 数据库名: `stream_scripts`, 表名: `scripts`
- 切块字数: MIN_CHARS=500, MAX_CHARS=1000
- DeepSeek 并发数: 5, 重试: 3 次
- Streamlit 每页 20 条结果

---

### Task 1: 项目脚手架 (config.py + requirements.txt)

**Files:**
- Create: `config.py`
- Create: `requirements.txt`

**Interfaces:**
- Produces: `config.CHUNK_DIR`, `config.TRANSCRIPT_DIR`, `config.DATA_DIR`, `config.MIN_CHARS`, `config.MAX_CHARS`, `config.DEEPSEEK_MODEL`, `config.DEEPSEEK_CONCURRENCY`, `config.DEEPSEEK_RETRIES`, `config.EMBEDDING_MODEL`, `config.PG_HOST`, `config.PG_PORT`, `config.PG_USER`, `config.PG_PASSWORD`, `config.PG_DB`, `config.CHUNKS_FILE`, `config.ENRICHED_FILE`, `config.ERRORS_LOG`

- [ ] **Step 1: 编写 config.py**

```python
"""stream-script-kb 全局配置"""
import os
from pathlib import Path

# --- 路径 ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNK_DIR = Path("/home/justin/audio_chunks")
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 1: 转录 ---
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_BEAM_SIZE = 5
WHISPER_LANGUAGE = "zh"

# --- Step 2: 切块 ---
MIN_CHARS = 500
MAX_CHARS = 1000
CHUNKS_FILE = DATA_DIR / "chunks.json"

# --- Step 3: DeepSeek ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CONCURRENCY = 5
DEEPSEEK_RETRIES = 3
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_MAX_TOKENS = 2048
ENRICHED_FILE = DATA_DIR / "enriched.json"
ERRORS_LOG = DATA_DIR / "errors.log"

# --- Step 4: 向量化 ---
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_DB = os.environ.get("PG_DB", "stream_scripts")

# --- Step 5: Streamlit ---
STREAMLIT_PAGE_SIZE = 20
STREAMLIT_TITLE = "直播话术知识库"
```

- [ ] **Step 2: 编写 requirements.txt**

```
faster-whisper>=1.0.0
openai>=1.0.0
sentence-transformers>=3.0.0
psycopg2-binary>=2.9.0
pgvector>=0.3.0
streamlit>=1.28.0
numpy>=1.24.0
torch>=2.0.0
```

- [ ] **Step 3: 安装依赖并验证**

```bash
cd /home/justin/stream-script-kb
pip install -r requirements.txt
python -c "import config; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /home/justin/stream-script-kb && git init && git add config.py requirements.txt && git commit -m "feat: project scaffold with config and dependencies"
```

---

### Task 2: step1_transcribe.py — faster-whisper 转录

**Files:**
- Create: `step1_transcribe.py`

**Interfaces:**
- Consumes: `config.CHUNK_DIR`, `config.TRANSCRIPT_DIR`, `config.WHISPER_MODEL`, `config.WHISPER_DEVICE`, `config.WHISPER_COMPUTE_TYPE`, `config.WHISPER_BEAM_SIZE`, `config.WHISPER_LANGUAGE`
- Produces: `data/transcripts/*.vtt` — 每个输入 .m4a 对应一个 VTT 文件

- [ ] **Step 1: 编写 step1_transcribe.py**

```python
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
        start_ts = f"{int(start//3600):02d}:{int(start%3600//60):02d}:{start%60:06.3f}"
        end_ts = f"{int(end//3600):02d}:{int(end%3600//60):02d}:{end%60:06.3f}"
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
            print(f"[STEP1] [{i}/{total}] ❌ {chunk} failed: {e}", file=sys.stderr)

        # 显存清理
        import torch
        torch.cuda.empty_cache()

    print(f"[STEP1] All done. VTT files in {config.TRANSCRIPT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本语法**

```bash
cd /home/justin/stream-script-kb
python -c "import py_compile; py_compile.compile('step1_transcribe.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/justin/stream-script-kb
git add step1_transcribe.py
git commit -m "feat: step1 faster-whisper transcription to VTT"
```

---

### Task 3: step2_chunk.py — VTT 字幕切块

**Files:**
- Create: `step2_chunk.py`
- Create: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `config.TRANSCRIPT_DIR`, `config.MIN_CHARS`, `config.MAX_CHARS`, `config.CHUNKS_FILE`
- Produces: `config.CHUNKS_FILE` — JSON 数组, 每项 `{chunk_id, source_file, start_time, end_time, char_count, text}`

- [ ] **Step 1: 编写测试 tests/test_chunk.py**

```python
"""Tests for step2_chunk.py"""
import json
import tempfile
from pathlib import Path

from step2_chunk import parse_vtt, build_chunks


def test_parse_vtt_single_entry():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.500
大家好欢迎来到直播间

00:00:04.000 --> 00:00:08.200
今天给大家带来一款非常好用的产品
"""
    entries = parse_vtt(vtt)
    assert len(entries) == 2
    assert entries[0]["text"] == "大家好欢迎来到直播间"
    assert entries[0]["start"] == 1.0
    assert entries[0]["end"] == 3.5


def test_parse_vtt_empty():
    entries = parse_vtt("WEBVTT\n")
    assert entries == []


def test_build_chunks_single():
    entries = [{"start": 0.0, "end": 5.0, "text": "这是一个测试句子。" * 50}]  # ~400 chars
    # Add enough text to trigger a chunk
    entries[0]["text"] = "这是一个测试句子。" * 80  # ~640 chars
    chunks = build_chunks(entries, source_file="test.vtt", min_chars=400, max_chars=800)
    assert len(chunks) >= 1
    assert 400 <= chunks[0]["char_count"] <= 800
    assert chunks[0]["source_file"] == "test.vtt"


def test_build_chunks_merges_short_tail():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "A" * 600 + "。"},
        {"start": 3.0, "end": 4.0, "text": "B" * 50},  # too short, should merge
    ]
    chunks = build_chunks(entries, source_file="t.vtt", min_chars=500, max_chars=1000)
    assert len(chunks) == 1
    assert "B" * 50 in chunks[0]["text"]


def test_chunk_id_format():
    entries = [{"start": 0.0, "end": 2.0, "text": "X" * 600 + "。"}]
    chunks = build_chunks(entries, source_file="part_003.vtt", min_chars=500, max_chars=1000)
    assert chunks[0]["chunk_id"] == "part_003_001"
```

- [ ] **Step 2: 运行测试, 预期全部 FAIL (函数未定义)**

```bash
cd /home/justin/stream-script-kb
pip install pytest -q
python -m pytest tests/test_chunk.py -v
```
Expected: 4 FAIL (函数未定义)

- [ ] **Step 3: 编写 step2_chunk.py**

```python
#!/usr/bin/env python3
"""Step 2: 将 VTT 字幕文件按字数切块 (500-1000 字/块)。

输入: data/transcripts/*.vtt
输出: data/chunks.json
"""
import json
import re
from pathlib import Path

import config


def _parse_timestamp(ts: str) -> float:
    """将 VTT 时间戳 'HH:MM:SS.mmm' 转为秒数。"""
    h, m, s = ts.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_vtt(content: str) -> list[dict]:
    """解析 VTT 文本, 返回字幕条目列表。

    返回: [{"start": float, "end": float, "text": str}, ...]
    """
    entries = []
    lines = content.strip().split("\n")
    i = 0

    # 跳过 WEBVTT 头和元数据
    while i < len(lines):
        line = lines[i].strip()
        if line == "WEBVTT":
            i += 1
            continue
        if line.startswith("Language:") or line.startswith("Kind:") or line == "":
            i += 1
            continue
        # 匹配时间戳行: "00:00:01.000 --> 00:00:03.500"
        if "-->" in line:
            parts = line.split("-->")
            if len(parts) == 2:
                start = _parse_timestamp(parts[0])
                end = _parse_timestamp(parts[1])
                # 文本行: 时间戳之后直到空行, 可能跨多行
                text_parts = []
                i += 1
                while i < len(lines) and lines[i].strip() != "":
                    text_parts.append(lines[i].strip())
                    i += 1
                entries.append({
                    "start": start,
                    "end": end,
                    "text": " ".join(text_parts),
                })
                continue
        i += 1

    return entries


_SENTENCE_END = re.compile(r"[。！？.!?]$")


def build_chunks(
    entries: list[dict],
    source_file: str,
    min_chars: int = 500,
    max_chars: int = 1000,
) -> list[dict]:
    """将字幕条目按字数拼接为 chunk。

    Args:
        entries: parse_vtt 的输出
        source_file: 源 VTT 文件名
        min_chars: 最小字数
        max_chars: 最大字数

    Returns:
        chunk 列表
    """
    if not entries:
        return []

    chunks = []
    buffer_entries = []  # 当前累积的字幕条目
    buffer_chars = 0
    chunk_idx = 0

    for entry in entries:
        text = entry["text"]
        char_count = len(text)

        # 如果当前条目本身超过 max_chars, 直接作为一个 chunk
        if char_count > max_chars:
            # 先保存当前 buffer
            if buffer_entries:
                chunks.append(_make_chunk(buffer_entries, source_file, chunk_idx))
                chunk_idx += 1
                buffer_entries = []
                buffer_chars = 0
            # 硬切: 在 max_chars 处切断
            pos = max_chars
            while pos < char_count:
                chunk_text = text[pos - max_chars:pos]
                chunks.append({
                    "chunk_id": f"{Path(source_file).stem}_{chunk_idx + 1:03d}",
                    "source_file": source_file,
                    "start_time": entry["start"],
                    "end_time": entry["end"],
                    "char_count": len(chunk_text),
                    "text": chunk_text,
                })
                chunk_idx += 1
                pos += max_chars
            continue

        buffer_entries.append(entry)
        buffer_chars += char_count

        # 到达切分阈值
        if buffer_chars >= min_chars:
            # 在句末标点处切
            combined = "".join(e["text"] for e in buffer_entries)
            if _SENTENCE_END.search(combined):
                chunks.append(_make_chunk(buffer_entries, source_file, chunk_idx))
                chunk_idx += 1
                buffer_entries = []
                buffer_chars = 0

    # 处理尾部残片
    if buffer_entries:
        if buffer_chars < min_chars and chunks:
            # 合并到上一个 chunk
            chunks[-1]["text"] += "".join(e["text"] for e in buffer_entries)
            chunks[-1]["char_count"] = len(chunks[-1]["text"])
            chunks[-1]["end_time"] = buffer_entries[-1]["end"]
        else:
            chunks.append(_make_chunk(buffer_entries, source_file, chunk_idx))

    return chunks


def _make_chunk(entries: list[dict], source_file: str, idx: int) -> dict:
    """从多个字幕条目组装一个 chunk。"""
    text = "".join(e["text"] for e in entries)
    return {
        "chunk_id": f"{Path(source_file).stem}_{idx + 1:03d}",
        "source_file": source_file,
        "start_time": entries[0]["start"],
        "end_time": entries[-1]["end"],
        "char_count": len(text),
        "text": text,
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
        print(f"[STEP2] [{i}/{total}] {source} → {len(chunks)} chunks "
              f"({len(entries)} subtitle entries)")

    config.CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"[STEP2] Done. {len(all_chunks)} chunks → {config.CHUNKS_FILE}")

    # 打印分布统计
    if all_chunks:
        lengths = [c["char_count"] for c in all_chunks]
        print(f"[STEP2] Chunk stats: min={min(lengths)}, max={max(lengths)}, "
              f"avg={sum(lengths)/len(lengths):.0f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试, 预期全部 PASS**

```bash
cd /home/justin/stream-script-kb
python -m pytest tests/test_chunk.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /home/justin/stream-script-kb
git add step2_chunk.py tests/test_chunk.py
git commit -m "feat: step2 VTT chunking by character count"
```

---

### Task 4: step3_deepseek.py — DeepSeek 并发提取结构化信息

**Files:**
- Create: `step3_deepseek.py`

**Interfaces:**
- Consumes: `config.CHUNKS_FILE`, `config.DEEPSEEK_API_KEY`, `config.DEEPSEEK_MODEL`, `config.DEEPSEEK_BASE_URL`, `config.DEEPSEEK_CONCURRENCY`, `config.DEEPSEEK_RETRIES`, `config.ENRICHED_FILE`, `config.ERRORS_LOG`
- Produces: `config.ENRICHED_FILE` + `config.ERRORS_LOG`

- [ ] **Step 1: 编写 step3_deepseek.py**

```python
#!/usr/bin/env python3
"""Step 3: 使用 DeepSeek API 对每个 chunk 做结构化信息提取。

输入: data/chunks.json
输出: data/enriched.json + data/errors.log
并发: asyncio + aiohttp, 并发数 config.DEEPSEEK_CONCURRENCY
"""
import asyncio
import json
import logging
import sys
from datetime import datetime

from openai import AsyncOpenAI

import config

# --- Prompt ---
SYSTEM_PROMPT = """你是直播话术分析专家。对给定的直播话术文本块，提取以下结构化信息。

严格按照 JSON 格式返回，不要输出任何 JSON 之外的内容。

字段说明:
- refined_script: 去噪润色后的流畅话术文本（修复口语重复、语气词过多、转录错误）
- summary: 这段话术的一句话摘要
- sales_stage: 销售阶段，只能是以下之一: 开场暖场/产品引入/价值塑造/逼单促单/互动留人/转款过渡/结尾收场
- strategy_types: 话术策略标签列表，从以下选择: ["限时稀缺", "价格锚定", "信任背书", "从众效应", "痛点放大", "场景代入", "算账对比", "亲身试用", "福利诱导", "互动引导", "悬念制造", "情感共鸣"]
- product_mentions: 提到的商品或品类名称列表
- selling_points: 提取的卖点列表（每条 10-20 字）
- target_audience: 目标人群描述，如 "宝妈"、"学生党"、"上班族"、"中老年" 等

返回格式:
{
  "refined_script": "...",
  "summary": "...",
  "sales_stage": "...",
  "strategy_types": [...],
  "product_mentions": [...],
  "selling_points": [...],
  "target_audience": "..."
}

如果文本内容不包含明确的话术（纯闲聊、无意义重复），sales_stage 填 "无法识别"，其他字段填空。"""


def _build_user_prompt(chunk_text: str) -> str:
    return f"请分析以下直播话术文本:\n\n{chunk_text}"


async def _process_one(
    client: AsyncOpenAI,
    chunk: dict,
    sem: asyncio.Semaphore,
    pbar: dict,  # mutable progress tracking
) -> dict | None:
    """处理单个 chunk, 成功返回 enriched dict, 失败返回 None 并记日志。"""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    for attempt in range(1, config.DEEPSEEK_RETRIES + 1):
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(text)},
                    ],
                    temperature=config.DEEPSEEK_TEMPERATURE,
                    max_tokens=config.DEEPSEEK_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)

                # 合并原始字段 + DeepSeek 产出
                result = {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "refined_script": parsed.get("refined_script", text),
                    "summary": parsed.get("summary", ""),
                    "sales_stage": parsed.get("sales_stage", ""),
                    "strategy_types": parsed.get("strategy_types", []),
                    "product_mentions": parsed.get("product_mentions", []),
                    "selling_points": parsed.get("selling_points", []),
                    "target_audience": parsed.get("target_audience", ""),
                }
                return result

            except (json.JSONDecodeError, KeyError) as e:
                if attempt == config.DEEPSEEK_RETRIES:
                    _log_error(chunk_id, text, content if 'content' in dir() else str(e))
                    return None
                await asyncio.sleep(1 * attempt)

            except Exception as e:
                if attempt == config.DEEPSEEK_RETRIES:
                    _log_error(chunk_id, text, str(e))
                    return None
                await asyncio.sleep(2 * attempt)

    return None


def _log_error(chunk_id: str, original_text: str, error_info: str) -> None:
    """将失败的 chunk 写入 errors.log。"""
    config.ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ERRORS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Chunk ID: {chunk_id}\n")
        f.write(f"Error: {error_info}\n")
        f.write(f"Text ({len(original_text)} chars):\n{original_text}\n")
        f.write(f"{'=' * 60}\n\n")


async def main_async():
    if not config.DEEPSEEK_API_KEY:
        print("[STEP3] ERROR: DEEPSEEK_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"[STEP3] Loaded {total} chunks from {config.CHUNKS_FILE}")

    # 清空 errors.log
    config.ERRORS_LOG.write_text("", encoding="utf-8")

    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    sem = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)

    pbar = {"done": 0, "failed": 0}

    async def _process_with_progress(chunk):
        result = await _process_one(client, chunk, sem, pbar)
        pbar["done"] += 1
        if result is None:
            pbar["failed"] += 1
        status = "OK" if result else "FAIL"
        print(f"[STEP3] [{pbar['done']}/{total}] {chunk['chunk_id']} {status} "
              f"({pbar['failed']} failed so far)")
        return result

    tasks = [_process_with_progress(c) for c in chunks]
    results = await asyncio.gather(*tasks)

    enriched = [r for r in results if r is not None]

    with open(config.ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    failed = total - len(enriched)
    print(f"[STEP3] Done. {len(enriched)}/{total} succeeded, {failed} failed")
    if failed > 0:
        print(f"[STEP3] Failed chunks logged to {config.ERRORS_LOG}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证语法**

```bash
cd /home/justin/stream-script-kb
python -c "import py_compile; py_compile.compile('step3_deepseek.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/justin/stream-script-kb
git add step3_deepseek.py
git commit -m "feat: step3 DeepSeek concurrent structured extraction"
```

---

### Task 5: step4_vectorize.py — 向量化入库 PostgreSQL

**Files:**
- Create: `step4_vectorize.py`
- Create: `tests/test_vectorize.py`

**Interfaces:**
- Consumes: `config.ENRICHED_FILE`, `config.EMBEDDING_MODEL`, `config.PG_HOST`, `config.PG_PORT`, `config.PG_USER`, `config.PG_PASSWORD`, `config.PG_DB`
- Produces: PostgreSQL 表 `scripts` (含 HNSW 索引)

- [ ] **Step 1: 编写测试 tests/test_vectorize.py**

```python
"""Tests for step4_vectorize.py"""
from step4_vectorize import get_connection, ensure_db, create_table, DB_CONFIG


def test_db_config_present():
    """数据库配置项存在。"""
    assert DB_CONFIG["host"] is not None
    assert DB_CONFIG["port"] is not None
    assert DB_CONFIG["dbname"] is not None
```

注意: 数据库相关测试需要 PostgreSQL 运行。如果 PG 未安装, 跳过集成测试。

- [ ] **Step 2: 运行测试 (预期 1 PASS)**

```bash
cd /home/justin/stream-script-kb
python -m pytest tests/test_vectorize.py -v
```

- [ ] **Step 3: 编写 step4_vectorize.py**

```python
#!/usr/bin/env python3
"""Step 4: 对 refined_script 生成 embedding 并写入 PostgreSQL。

输入: data/enriched.json
输出: PostgreSQL 表 scripts (含 embedding 向量和 HNSW 索引)

前置条件:
  - PostgreSQL 已安装运行
  - pgvector 扩展已可用 (CREATE EXTENSION IF NOT EXISTS vector)
  - 数据库 stream_scripts 已存在 (脚本自动创建)
"""
import json
import sys

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

import config

DB_CONFIG = {
    "host": config.PG_HOST,
    "port": config.PG_PORT,
    "user": config.PG_USER,
    "password": config.PG_PASSWORD,
    "dbname": config.PG_DB,
}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scripts (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    source_file TEXT DEFAULT '',
    start_time REAL DEFAULT 0.0,
    end_time REAL DEFAULT 0.0,
    refined_script TEXT NOT NULL,
    summary TEXT DEFAULT '',
    sales_stage TEXT DEFAULT '',
    strategy_types TEXT DEFAULT '[]',
    product_mentions TEXT DEFAULT '[]',
    selling_points TEXT DEFAULT '[]',
    target_audience TEXT DEFAULT '',
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_scripts_sales_stage ON scripts(sales_stage);
CREATE INDEX IF NOT EXISTS idx_scripts_source_file ON scripts(source_file);

-- HNSW 语义检索索引 (仅在不存在时创建, 避免重复报错)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_embedding_hnsw'
    ) THEN
        EXECUTE 'CREATE INDEX idx_scripts_embedding_hnsw ON scripts
                 USING hnsw (embedding vector_cosine_ops)
                 WITH (m = 16, ef_construction = 200)';
    END IF;
END $$;
"""


def get_connection():
    """获取 PostgreSQL 连接 (需提前 ensure_db)。"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    return conn


def ensure_db():
    """确保数据库存在。如果不存在则创建。"""
    # 连接默认 postgres 数据库来创建目标库
    admin_config = {**DB_CONFIG, "dbname": "postgres"}
    conn = psycopg2.connect(**admin_config)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (config.PG_DB,),
    )
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {config.PG_DB}")
        print(f"[STEP4] Created database: {config.PG_DB}")
    cur.close()
    conn.close()


def create_table(conn):
    """创建表和索引 (幂等)。"""
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    cur.execute(INDEX_SQL)
    conn.commit()
    cur.close()
    print("[STEP4] Schema and indexes ensured")


def main():
    print("[STEP4] Loading enriched data...")
    with open(config.ENRICHED_FILE, "r", encoding="utf-8") as f:
        enriched = json.load(f)
    total = len(enriched)
    print(f"[STEP4] {total} records to process")

    print("[STEP4] Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("[STEP4] Generating embeddings...")
    refined_texts = [item["refined_script"] for item in enriched]
    embeddings = model.encode(
        refined_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vecs = [e.tolist() for e in embeddings]
    print(f"[STEP4] {len(vecs)} embeddings generated (dim={len(vecs[0])})")

    print("[STEP4] Setting up database...")
    ensure_db()
    conn = get_connection()
    create_table(conn)

    print("[STEP4] Inserting records...")
    cur = conn.cursor()
    inserted = 0
    for item, vec in zip(enriched, vecs):
        cur.execute(
            """INSERT INTO scripts
               (chunk_id, source_file, start_time, end_time,
                refined_script, summary, sales_stage,
                strategy_types, product_mentions, selling_points,
                target_audience, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                item["chunk_id"],
                item["source_file"],
                item["start_time"],
                item["end_time"],
                item["refined_script"],
                item["summary"],
                item["sales_stage"],
                json.dumps(item.get("strategy_types", []), ensure_ascii=False),
                json.dumps(item.get("product_mentions", []), ensure_ascii=False),
                json.dumps(item.get("selling_points", []), ensure_ascii=False),
                item.get("target_audience", ""),
                vec,
            ),
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"[STEP4] Done. {inserted} records inserted into {config.PG_DB}.scripts")
    print(f"[STEP4] Verify: psql -d {config.PG_DB} -c 'SELECT COUNT(*) FROM scripts;'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试**

```bash
cd /home/justin/stream-script-kb
python -m pytest tests/test_vectorize.py -v
```

- [ ] **Step 5: 验证语法**

```bash
python -c "import py_compile; py_compile.compile('step4_vectorize.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 6: Commit**

```bash
cd /home/justin/stream-script-kb
git add step4_vectorize.py tests/test_vectorize.py
git commit -m "feat: step4 embedding generation and pgvector insertion"
```

---

### Task 6: app.py — Streamlit 检索界面

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `config` DB 连接信息, `config.STREAMLIT_PAGE_SIZE`, `config.STREAMLIT_TITLE`
- Produces: 浏览器 Web UI

- [ ] **Step 1: 编写 app.py**

```python
#!/usr/bin/env python3
"""Streamlit 直播话术语义检索界面。

用法: streamlit run app.py
"""
import json

import psycopg2
import streamlit as st
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

import config

# --- 页面配置 ---
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🎙️",
    layout="wide",
)
st.title(f"🎙️ {config.STREAMLIT_TITLE}")

# --- 缓存资源 ---
@st.cache_resource
def load_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource
def get_db_connection():
    conn = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    )
    register_vector(conn)
    return conn


# --- 筛选选项 (从 DB 动态获取) ---
@st.cache_data(ttl=300)
def get_filter_options():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT source_file FROM scripts ORDER BY source_file")
    sources = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT DISTINCT sales_stage FROM scripts WHERE sales_stage != '' ORDER BY sales_stage")
    stages = [r[0] for r in cur.fetchall()]

    # strategy_types 是 JSON 数组, 需要展开
    cur.execute("SELECT DISTINCT strategy_types FROM scripts WHERE strategy_types != '[]'")
    all_strategies = set()
    for (val,) in cur.fetchall():
        try:
            arr = json.loads(val)
            all_strategies.update(arr)
        except json.JSONDecodeError:
            pass
    strategies = sorted(all_strategies)

    # product_mentions 同 JSON 数组
    cur.execute("SELECT DISTINCT product_mentions FROM scripts WHERE product_mentions != '[]'")
    all_products = set()
    for (val,) in cur.fetchall():
        try:
            arr = json.loads(val)
            all_products.update(arr)
        except json.JSONDecodeError:
            pass
    products = sorted(all_products)

    cur.close()
    return sources, stages, strategies, products


# --- 页面渲染 ---
sources, stages, strategies, products = get_filter_options()

# 顶部筛选栏
st.subheader("🔍 筛选条件")
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_source = st.selectbox(
        "主播/来源", ["(全部)"] + sources, key="filter_source"
    )
with col2:
    selected_stage = st.selectbox(
        "销售阶段", ["(全部)"] + stages, key="filter_stage"
    )
with col3:
    selected_strategy = st.selectbox(
        "话术策略", ["(全部)"] + strategies, key="filter_strategy"
    )
with col4:
    selected_product = st.selectbox(
        "品类", ["(全部)"] + products, key="filter_product"
    )

# 中间搜索框
st.subheader("💬 语义搜索")
query_text = st.text_input(
    "输入想要查找的话术描述...",
    placeholder="例如：适合敏感肌的洗面奶推荐话术、逼单催付话术",
    key="search_query",
)

top_k = st.slider("返回条数", min_value=5, max_value=100, value=20, step=5)

# --- 查询逻辑 ---
def build_query(selected_source, selected_stage, selected_strategy, selected_product):
    """构建 SQL WHERE 子句和参数。"""
    conditions = []
    params = []

    if selected_source != "(全部)":
        conditions.append("source_file = %s")
        params.append(selected_source)
    if selected_stage != "(全部)":
        conditions.append("sales_stage = %s")
        params.append(selected_stage)
    if selected_strategy != "(全部)":
        conditions.append("strategy_types LIKE %s")
        params.append(f"%{selected_strategy}%")
    if selected_product != "(全部)":
        conditions.append("product_mentions LIKE %s")
        params.append(f"%{selected_product}%")

    where_clause = " AND ".join(conditions) if conditions else "TRUE"
    return where_clause, params


def semantic_search(embedding, where_clause, params, top_k):
    """执行语义搜索 (向量余弦相似度)。"""
    conn = get_db_connection()
    cur = conn.cursor()
    query = f"""
        SELECT chunk_id, source_file, sales_stage, strategy_types,
               product_mentions, selling_points, target_audience,
               refined_script, summary,
               embedding <=> %s::vector AS distance
        FROM scripts
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    all_params = [str(embedding)] + params + [str(embedding)] + [top_k]
    cur.execute(query, all_params)
    rows = cur.fetchall()
    cur.close()
    return rows


if st.button("🔎 搜索", type="primary", use_container_width=True) or query_text:
    model = load_model()
    query_embedding = model.encode(
        query_text if query_text else "直播话术",
        normalize_embeddings=True,
    ).tolist()
    embedding_str = json.dumps(query_embedding)

    where_clause, params = build_query(
        selected_source, selected_stage, selected_strategy, selected_product
    )

    results = semantic_search(embedding_str, where_clause, params, top_k)

    st.subheader(f"📋 搜索结果 ({len(results)} 条)")

    if not results:
        st.info("无匹配结果，请调整筛选条件或搜索词。")
    else:
        for row in results:
            (chunk_id, source_file, stage, strategy_json, product_json,
             selling_json, audience, refined, summary, distance) = row

            similarity = max(0, 1 - distance) if distance is not None else 0

            # 解析 JSON 字段
            try:
                strategies_list = json.loads(strategy_json) if isinstance(strategy_json, str) else strategy_json
            except json.JSONDecodeError:
                strategies_list = []
            try:
                products_list = json.loads(product_json) if isinstance(product_json, str) else product_json
            except json.JSONDecodeError:
                products_list = []

            with st.container():
                st.markdown("---")
                # 标签行
                tags = [stage] if stage else []
                tags.extend(strategies_list[:3])
                tag_html = " ".join(
                    f'<span style="background:#e8f0fe;color:#1a73e8;padding:2px 8px;'
                    f'border-radius:4px;font-size:12px;margin-right:4px;">{t}</span>'
                    for t in tags if t
                )
                st.markdown(tag_html, unsafe_allow_html=True)

                # 主体文本
                st.markdown(f"**📝 话术文本**")
                st.text(refined if refined else "(无文本)")

                # 元信息行
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.caption(f"📎 来源: {source_file}")
                with meta_col2:
                    st.caption(f"🏷️ 品类: {', '.join(products_list) if products_list else '—'}")
                with meta_col3:
                    st.caption(f"🎯 相似度: {similarity:.3f}")

                # 摘要
                if summary:
                    st.caption(f"💡 {summary}")

                # 卖点
                try:
                    selling_list = json.loads(selling_json) if isinstance(selling_json, str) else selling_json
                except json.JSONDecodeError:
                    selling_list = []
                if selling_list:
                    st.caption(f"✨ 卖点: {' · '.join(selling_list[:5])}")

                # 目标人群
                if audience:
                    st.caption(f"👥 目标人群: {audience}")
else:
    # 初始状态: 显示统计信息
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM scripts")
    count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT source_file) FROM scripts")
    src_count = cur.fetchone()[0]
    cur.close()
    st.info(f"📊 数据库中共 {count} 条话术记录, 来自 {src_count} 个来源。输入搜索词开始检索。")
```

- [ ] **Step 2: 验证语法**

```bash
python -c "import py_compile; py_compile.compile('app.py', doraise=True); print('Syntax OK')"
```
Expected: `Syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /home/justin/stream-script-kb
git add app.py
git commit -m "feat: step5 Streamlit semantic search UI"
```

---

## 自检清单 (Self-Review)

1. **Spec coverage:** 5 个 script + 1 个 app 全覆盖。数据流 m4a→vtt→chunks.json→enriched.json→PostgreSQL→Streamlit 闭环。
2. **Placeholder scan:** 无 TBD/TODO, 所有代码完整可运行。
3. **Type consistency:** config 属性名在全部 4 个脚本 + app.py 中一致; enriched JSON 的 8 个字段与 step3 输出一致, 与 step4 INSERT 一致, 与 app.py 展示一致。

✅ Plan does not contain issues.
