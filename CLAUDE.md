# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

直播话术知识库 (Livestream Script Knowledge Base) — a 5-step batch pipeline that transcribes Taobao livestream audio, extracts structured sales-script metadata via DeepSeek, vectorizes the scripts, and serves them through a Streamlit semantic search UI.

## Prerequisites

- Python 3.12+
- PostgreSQL with pgvector extension installed
- ffmpeg (for step0 audio download)
- CUDA-capable GPU optional (Whisper falls back to CPU)

**Before first run**, review `shared/config.py` and set environment variables: `DEEPSEEK_API_KEY`, `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DB`.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (sequential, each step reads the prior step's output)
python3 -m pipeline.step0_download --url "..." --name "主播名_20260712"
python3 -m pipeline.step1_transcribe   # .m4a → data/transcripts/*.vtt
python3 -m pipeline.step2_chunk        # .vtt → data/chunks.json
python3 -m pipeline.step3_deepseek --limit N
python3 -m pipeline.step4_vectorize    # enriched.json → PostgreSQL + pgvector

# Run the search UI
python3 -m streamlit run web/app.py

# Run the streaming API
python3 -m streaming.server

# Run tests
python3 -m pytest tests/ -v
```

## Architecture

**Pipeline data flow (file-based intermediate state):**
```
m3u8/直播URL  →  step0 (ffmpeg download)  →  audio_chunks/*.m4a
  → step1 (faster-whisper, language=zh)  →  data/transcripts/*.vtt
  → step2 (char-based chunking at sentence boundaries)  →  data/chunks.json
  → step3 (DeepSeek async enrichment)  →  data/enriched.json + data/errors.log
  → step4 (bge embedding + pgvector)  →  PostgreSQL scripts table
  → web/app.py (Streamlit)  →  browser
```

Each step is an independent script. Intermediate results are persisted to disk, so failed steps can be re-run without redoing earlier work. Step 1 supports resume (skips existing VTT files). Step 3 supports `--limit N` for testing on a subset.

**Step 3 (DeepSeek enrichment)** is the critical path — uses `asyncio` + `AsyncOpenAI` with a semaphore (configurable via `DEEPSEEK_CONCURRENCY`, default 400) and 3 retries per chunk. Failed chunks are logged to `data/errors.log` with full original text for later debugging. The prompt extracts: refined_script, summary, sales_stage, strategy_types, product_mentions, selling_points, target_audience.

**Step 4 (vectorization)** uses `BAAI/bge-small-zh-v1.5` via sentence-transformers, with `normalize_embeddings=True`. The schema declares `vector(512)` but the model outputs 384 dimensions — pgvector's `vector(N)` is a max constraint, not fixed. HNSW index with `vector_cosine_ops` (m=16, ef_construction=200). The DB and table are created automatically if absent.

**Streamlit app** uses `@st.cache_resource` for the embedding model and DB connection, `@st.cache_data(ttl=300)` for filter dropdowns. Search uses pgvector's cosine distance operator `<=>` with optional faceted filters (source, sales stage, strategy type, product).

**Streaming API** is assembled in `streaming/server.py`. Shared queues and results live in `streaming/state.py`; HTTP endpoints live in `streaming/routes/`; the four queue-connected workers live in `streaming/workers/`; RAG retrieval and rewriting live in `streaming/services/rag.py`. Transient analysis results stay in memory, while the RAG rewrite endpoint reads the PostgreSQL knowledge base created by the offline pipeline.

## Key Design Decisions

- **Chinese-language pipeline throughout** (Whisper `language=zh`, bge-small-zh-v1.5, DeepSeek with Chinese prompts)
- **pgvector over Milvus** (current data volume is small enough)
- **Two explicit processing modes**: file-backed offline ingestion and queue-backed real-time analysis
- **500-1000 character chunking** at sentence boundaries (句末标点 `。！？.!?`), hard-split on oversize entries, short tails merged into previous chunk
- **JSON arrays stored as JSONB** in PostgreSQL (`strategy_types`, `product_mentions`, `selling_points`). Faceted filtering uses native JSONB containment queries.

## Known Limitations

- `streamlink` is listed in `requirements.txt` but unused by any pipeline step (only `experiments/test_ytdlp.py` explores alternative download approaches)
- Step 0 and step 1 are single-file-at-a-time; no parallel processing within each step
- The real-time analysis result store is process memory only and is lost on service restart
- Test coverage is minimal: only step2 chunking logic and step4 DB config presence
