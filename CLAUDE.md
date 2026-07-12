# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

直播话术知识库 (Livestream Script Knowledge Base) — a 5-step batch pipeline that transcribes Taobao livestream audio, extracts structured sales-script metadata via DeepSeek, vectorizes the scripts, and serves them through a Streamlit semantic search UI.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (sequential, each step reads the prior step's output)
python step0_download.py --url "..."          # streamlink → audio_chunks/*.m4a (带 cookie)
python step1_transcribe.py   # .m4a → data/transcripts/*.vtt (faster-whisper, CUDA, skips existing)
python step2_chunk.py        # .vtt → data/chunks.json (500-1000 char chunks)
python step3_deepseek.py     # chunks.json → data/enriched.json + data/errors.log (DeepSeek API, async)
python step4_vectorize.py    # enriched.json → PostgreSQL scripts table + HNSW index (bge-small-zh-v1.5)

# Run the search UI
streamlit run app.py

# Run tests (psycopg2 must be installed for test_vectorize.py to import)
python -m pytest tests/ -v
```

## Architecture

**Pipeline data flow (file-based intermediate state):**
```
m3u8/直播URL  →  step0 (streamlink + ffmpeg)  →  audio_chunks/*.m4a
  → step1 (faster-whisper)  →  data/transcripts/*.vtt
  → step2 (char-based chunking)  →  data/chunks.json
  → step3 (DeepSeek async enrichment)  →  data/enriched.json + data/errors.log
  → step4 (bge embedding + pgvector)  →  PostgreSQL scripts table
  → app.py (Streamlit)  →  browser
```

Each step is an independent script. Intermediate results are persisted to disk, so failed steps can be re-run without redoing earlier work. Step 1 supports resume (skips existing VTT files).

**Step 3 (DeepSeek enrichment)** is the critical path — it uses `asyncio` + `AsyncOpenAI` with a semaphore (default concurrency 5) and 3 retries per chunk. Failed chunks are logged to `data/errors.log` with full original text for later debugging. Supports `--limit N` for testing on a subset.

**Step 4 (vectorization)** uses `BAAI/bge-small-zh-v1.5` via sentence-transformers, with `normalize_embeddings=True`. The schema declares `vector(512)` but the model outputs 384 dimensions (pgvector's `vector(N)` is a max constraint, not fixed). HNSW index with `vector_cosine_ops`. The DB and table are created automatically if absent.

**Streamlit app** uses `@st.cache_resource` for the embedding model and DB connection, `@st.cache_data(ttl=300)` for filter dropdowns. Search uses pgvector's cosine distance operator `<=>` with optional faceted filters (source, sales stage, strategy type, product). JSON array columns (strategy_types, product_mentions, selling_points) are stored as JSON strings in PostgreSQL and parsed at query time — faceted filtering on these uses `LIKE %value%`.

**All configuration** lives in `config.py`. API keys and DB credentials come from environment variables (`DEEPSEEK_API_KEY`, `PG_HOST`, `PG_PORT`, `PG_USER`, `PG_PASSWORD`, `PG_DB`). The audio input directory is hardcoded to `/home/justin/audio_chunks`.

**Key design decisions (from spec):**
- Chinese-language pipeline throughout (Whisper `language=zh`, bge-small-zh-v1.5, DeepSeek with Chinese prompts)
- pgvector over Milvus (current data volume is small enough)
- No real-time ingestion, no VAD/audio preprocessing, no multi-user/auth
- 500-1000 character chunking at sentence boundaries (句末标点), hard-split on oversize entries, short tails merged into previous chunk
