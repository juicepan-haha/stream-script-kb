# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

直播话术知识库 (Livestream Script Knowledge Base) — a 5-step batch pipeline that transcribes Taobao livestream audio, extracts structured sales-script metadata via DeepSeek, vectorizes the scripts, and serves them through a Streamlit semantic search UI.

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension installed
- ffmpeg (for step0 audio download)
- CUDA-capable GPU optional (Whisper falls back to CPU)

**Before first run**, edit `config.py` and change `CHUNK_DIR` from `/home/justin/audio_chunks` to a path on your machine. Also set environment variables: `DEEPSEEK_API_KEY`, `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DB`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (sequential, each step reads the prior step's output)
python step0_download.py --url "..." --name "主播名_20260712"       # ffmpeg → audio_chunks/*.m4a
python step0_download.py --batch data/stream_urls.json --duration 3600  # batch mode
python step1_transcribe.py   # .m4a → data/transcripts/*.vtt (faster-whisper, skips existing)
python step2_chunk.py        # .vtt → data/chunks.json (500-1000 char chunks)
python step3_deepseek.py [--limit N]  # chunks.json → data/enriched.json + data/errors.log
python step4_vectorize.py    # enriched.json → PostgreSQL scripts table + HNSW index

# Run the search UI
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

## Architecture

**Pipeline data flow (file-based intermediate state):**
```
m3u8/直播URL  →  step0 (ffmpeg download)  →  audio_chunks/*.m4a
  → step1 (faster-whisper, language=zh)  →  data/transcripts/*.vtt
  → step2 (char-based chunking at sentence boundaries)  →  data/chunks.json
  → step3 (DeepSeek async enrichment)  →  data/enriched.json + data/errors.log
  → step4 (bge embedding + pgvector)  →  PostgreSQL scripts table
  → app.py (Streamlit)  →  browser
```

Each step is an independent script. Intermediate results are persisted to disk, so failed steps can be re-run without redoing earlier work. Step 1 supports resume (skips existing VTT files). Step 3 supports `--limit N` for testing on a subset.

**Step 3 (DeepSeek enrichment)** is the critical path — uses `asyncio` + `AsyncOpenAI` with a semaphore (configurable via `DEEPSEEK_CONCURRENCY`, default 400) and 3 retries per chunk. Failed chunks are logged to `data/errors.log` with full original text for later debugging. The prompt extracts: refined_script, summary, sales_stage, strategy_types, product_mentions, selling_points, target_audience.

**Step 4 (vectorization)** uses `BAAI/bge-small-zh-v1.5` via sentence-transformers, with `normalize_embeddings=True`. The schema declares `vector(512)` but the model outputs 384 dimensions — pgvector's `vector(N)` is a max constraint, not fixed. HNSW index with `vector_cosine_ops` (m=16, ef_construction=200). The DB and table are created automatically if absent.

**Streamlit app** uses `@st.cache_resource` for the embedding model and DB connection, `@st.cache_data(ttl=300)` for filter dropdowns. Search uses pgvector's cosine distance operator `<=>` with optional faceted filters (source, sales stage, strategy type, product).

## Key Design Decisions

- **Chinese-language pipeline throughout** (Whisper `language=zh`, bge-small-zh-v1.5, DeepSeek with Chinese prompts)
- **pgvector over Milvus** (current data volume is small enough)
- **No real-time ingestion**, no VAD/audio preprocessing, no multi-user/auth
- **500-1000 character chunking** at sentence boundaries (句末标点 `。！？.!?`), hard-split on oversize entries, short tails merged into previous chunk
- **JSON arrays stored as JSON strings** in PostgreSQL (`strategy_types`, `product_mentions`, `selling_points`). Faceted filtering uses `LIKE %value%` — this is simple but fragile: substring matches mean "刀" would match "剪刀". Acceptable for current data volume; switch to `jsonb` with `?` / `@>` operators if this causes problems.

## Known Limitations

- `config.CHUNK_DIR` is hardcoded to `/home/justin/audio_chunks` — must be changed per-machine
- `streamlink` is listed in `requirements.txt` but unused by any pipeline step (only `experiments/test_ytdlp.py` explores alternative download approaches)
- Step 0 and step 1 are single-file-at-a-time; no parallel processing within each step
- No incremental update mechanism — re-running step4 re-inserts all records (no upsert/merge)
- Test coverage is minimal: only step2 chunking logic and step4 DB config presence
