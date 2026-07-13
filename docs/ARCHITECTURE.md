# 架构文档

## 系统全景

```
                          ┌─── 路径 A: Batch 管道 (main) ───┐
                          │                                  │
  m3u8 URL ──→ step0 ──→ step1 ──→ step2 ──→ step3 ──→ step4 ──→ app.py (Streamlit 搜索)
              下载     转录      切块      DeepSeek   向量入库
              .m4a    .vtt    chunks.json enriched.json  pgvector
                          │                                  │
                          └── 文件中介，每步落盘 ──────────────┘


                          ┌─── 路径 B: 流式反应堆 (industrial-refactor / Commercial) ───┐
                          │                                                             │
  URL ──→ Worker1 ──q──→ Worker2 ──q──→ Worker3 ──q──→ Worker4 ──→ enriched_results     │
         ffmpeg      Whisper    滑动窗口     DeepSeek      内存 dict                      │
         stdout      GPU常驻    语义切块     提词器富化                                     │
            │                                                                             │
            └── 零磁盘 I/O，全程 asyncio.Queue 内存管道 ──────────────────────────────────┘
```

## Worker 管线详解

```
download_queue (maxsize=20)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Worker 1 [下载]                               │
│ ffmpeg subprocess → stdout pipe               │
│ 每 30 秒读 960KB s16le → numpy float32        │
│ 推送 text_queue.put({audio, offset, user_key}) │
└──────────────────────────────────────────────┘
    │
    ▼
  text_queue (maxsize=50)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Worker 2 [转录]                               │
│ faster-whisper medium 常驻 GPU 显存            │
│ 硬件自适应: 8GB→medium+batch16, 24GB→large-v3 │
│ VAD 过滤静音                                  │
│ 输出带时间戳段列表 → chunk_queue               │
└──────────────────────────────────────────────┘
    │
    ▼
  chunk_queue (maxsize=100)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Worker 3 [切块]                               │
│ 滑动窗口缓冲区 (per task_id)                   │
│ 触发条件:                                     │
│   1. buffer_chars >= max_chars (1000)        │
│   2. buffer_chars >= min_chars (500) AND      │
│      (句末标点 OR 段间停顿 >1.5s)              │
│ 短尾合并到上一块                              │
└──────────────────────────────────────────────┘
    │
    ▼
  enriched_queue (maxsize=100)
    │
    ▼
┌──────────────────────────────────────────────┐
│ Worker 4 [DeepSeek]                          │
│ 使用用户自己的 API Key (阅后即焚)              │
│ 四段式 Prompt: icebreaker/painpoint/          │
│                mechanism/close_order          │
│ 动态关键词提取 + 代码级校验                    │
│ Semaphore 并发控制                            │
└──────────────────────────────────────────────┘
    │
    ▼
  enriched_results (内存 dict)
```

## Step 5+6: RAG 重写 + SOP 仪表盘

```
POST /api/v1/rewrite
    │
    ├─ verify_card_code(card_code)   ← 卡密校验
    │
    ├─ pgvector 余弦检索 TOP 5       ← 历史爆款 reference
    │
    └─ DeepSeek (一次 API 调用)       ← 用户自备 Key
         │
         ├─ rewritten_script         ← 四段式口语话术
         └─ sop_timeline[]           ← 秒级执行表
              ├─ time_range
              ├─ stage
              ├─ host_action
              ├─ operation_action
              └─ verbal_keywords
```

## 背压防御

```
Queue 满 → await put() 阻塞 → 上游自动暂停
  │
  ├─ download_queue (20): CDN 下载 > GPU 转录时触发
  ├─ text_queue (50):     下载过快时触发
  ├─ chunk_queue (100):   转录堆积时触发
  └─ enriched_queue (100): DeepSeek API 慢时触发
```

## 硬件自适应

```
启动 → 检测 GPU VRAM
  │
  ├─ ≥8GB (RTX 4070):    medium + float16 + batch=16
  ├─ ≥24GB (RTX 4090):   large-v3 + float16 + batch=32
  └─ 无 GPU:             base + CPU int8 + 4 threads
```

## 数据模型

```sql
scripts (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT UNIQUE,
    icebreaker TEXT,      -- 破冰留人
    painpoint TEXT,       -- 痛点植入
    mechanism TEXT,       -- 产品卖点
    close_order TEXT,     -- 逼单催单
    refined_script TEXT,  -- 合并文本 (for embedding)
    summary TEXT,
    sales_stage TEXT,
    strategy_types JSONB,
    product_mentions JSONB,
    selling_points JSONB,
    target_audience TEXT,
    embedding vector(512),
    created_at TIMESTAMP
)
-- HNSW index: vector_cosine_ops, m=16, ef_construction=200
```
