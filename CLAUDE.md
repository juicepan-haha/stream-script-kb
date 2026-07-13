# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

直播话术知识库 (Livestream Script Knowledge Base) — AI 驱动的直播销售话术分析、重写与 SOP 生成系统。主攻长尾市场中小主播群体。

**三线分支架构:**

| Branch | Role | Key Files |
|--------|------|-----------|
| `main` | 稳定批处理管道 | `step0-4` → `app.py` |
| `industrial-refactor` | 全内存流式反应堆 | `server_v2.py` + step5 RAG + step6 SOP |
| `Commercial-version` | SaaS 商用版 | `server_v2.py` + `app_frontend.py` + 卡密 |

## Commands

```bash
# === Batch 管道 (main branch) ===
python step0_download.py --url "..." --name "name" --cookie "..."  # 16并发下载
python step1_transcribe.py               # GPU流式转录 (零磁盘I/O)
python step2_chunk.py                     # 语义切块
python step3_deepseek.py [--limit N]     # DeepSeek 四段式富化
python step4_vectorize.py                 # pgvector 批量入库
streamlit run app.py                      # 语义搜索UI

# === 流式反应堆 (industrial-refactor / Commercial-version) ===
python server_v2.py                       # FastAPI 异步反应堆
streamlit run app_frontend.py             # SaaS 前端 (Commercial only)

# === 测试 ===
python -m pytest tests/ -v
python tests/test_memory_leak.py --duration 600  # 内存泄漏检测
```

## Architecture

See `docs/ARCHITECTURE.md` for full diagrams.

### 流式反应堆 (server_v2.py)

```
URL → Worker1 (ffmpeg→numpy) →q→ Worker2 (Whisper GPU) →q→ Worker3 (切块) →q→ Worker4 (DeepSeek)
                                                                                      │
                                              POST /api/v1/rewrite ← RAG+SOP ←──────┘
```

- **4 Worker 常驻内存**，asyncio.Queue 背压防御 (maxsize=20/50/100/100)
- **零磁盘 I/O**: ffmpeg stdout → numpy → faster-whisper → 内存 dict
- **硬件自适应**: 自动检测 GPU VRAM 选最优模型和 batch_size
- **卡密系统** (Commercial): `card_codes.txt` 热点加载 + `verify_card_code()` 异步校验
- **user_key 阅后即焚**: 用户的 DeepSeek Key 仅存活在内存任务生命周期中

### Batch 管道 (step0-4)

```
m3u8 → step0 (16并发下载+ffmpeg合并) → .m4a
    → step1 (ffmpeg→numpy→whisper) → .vtt
    → step2 (时间停顿1.5s切块) → chunks.json
    → step3 (DeepSeek四段式+动态校验) → enriched.json
    → step4 (流式分批embedding+批量INSERT ON CONFLICT) → pgvector
    → app.py (Streamlit语义搜索)
```

## Configuration

All config in `config.py`. Environment variables:

```bash
DEEPSEEK_API_KEY="sk-..."    # Step3/4/Server 默认 Key
PG_HOST="" PG_USER="" PG_DB=""  # PostgreSQL (空字符串=Unix socket)
HF_ENDPOINT="https://hf-mirror.com"  # 国内镜像
```

## API Endpoints (server_v2.py)

See `docs/API.md` for full reference.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/v1/analyze?url=...&user_key=...&card_code=...` | 流式分析 |
| POST | `/api/v1/rewrite?my_product=...&target_style=...&user_key=...&card_code=...` | RAG+SOP |
| GET | `/api/v1/progress/{id}` | 任务进度 |
| GET | `/api/v1/health` | 队列状态 |

## Database

PostgreSQL + pgvector. Table `scripts` with JSONB columns + HNSW index. `ON CONFLICT (chunk_id) DO UPDATE` 幂等写入.

## Key Design Decisions

- Chinese-language pipeline throughout
- pgvector over Milvus (small data volume)
- `-acodec copy` for merge (秒级), avoid re-encoding
- JSONB over TEXT for queryable JSON columns
- `@> ::jsonb` exact matching instead of `LIKE %value%`
- Semaphore + batch-size for API/memory control
- `aiofiles` for all async file I/O
