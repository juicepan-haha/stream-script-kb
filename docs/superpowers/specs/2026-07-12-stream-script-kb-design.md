# Stream Script KB — 直播话术知识库

## 概述

将淘宝直播音频转录、切块、DeepSeek 结构化提取、向量化入库，最后用 Streamlit 提供筛选+语义搜索的检索界面。

## 架构

5 个独立脚本 + 1 个 Streamlit 应用，顺序执行，中间结果落文件。

```
audio_chunks/*.m4a
    │  step1_transcribe.py (faster-whisper, CUDA)
    ▼
data/transcripts/*.vtt
    │  step2_chunk.py (按字数切块, 500-1000字)
    ▼
data/chunks.json
    │  step3_deepseek.py (DeepSeek API 并发, asyncio)
    ▼
data/enriched.json + data/errors.log
    │  step4_vectorize.py (bge-small-zh-v1.5 → pgvector)
    ▼
PostgreSQL (scripts 表 + HNSW 索引)
    │  app.py (Streamlit)
    ▼
浏览器 (筛选 + 语义搜索 + 卡片展示)
```

## Step 1: 转录

- 输入: `audio_chunks/*.m4a` (41 个)
- 输出: `data/transcripts/*.vtt` (一一对应)
- 模型: `faster-whisper` large-v3, device=cuda, compute_type=float16
- 断点续跑: 已有 .vtt 的跳过
- 进度打印: `[3/41] part_003.m4a → part_003.vtt`

检验: `ls data/transcripts/*.vtt | wc -l` 预期 41; `head -30 data/transcripts/part_000.vtt` 看格式

## Step 2: 切块

- 输入: `data/transcripts/*.vtt`
- 输出: `data/chunks.json` (数组)
- 策略: 累积到 500-1000 字, 在句末标点切分; 超 1000 字硬切; 尾片 <500 字合并到上一块
- 每个 chunk: `{chunk_id, source_file, start_time, end_time, char_count, text}`

检验: 打印每块的字数和来源文件

## Step 3: DeepSeek 结构化提取

- 输入: `data/chunks.json`
- 输出: `data/enriched.json` + `data/errors.log`
- API: `deepseek-chat`, 环境变量 `DEEPSEEK_API_KEY`
- 并发: asyncio + aiohttp, 并发数 5, 单条重试 3 次
- 字段: `refined_script, summary, sales_stage, strategy_types, product_mentions, selling_points, target_audience`
- 失败 chunk 写入 errors.log (chunk_id + 原始文本 + API 返回)

检验: 抽查 enriched.json 某条看字段完整性; `wc -l errors.log` 看失败数

## Step 4: 向量化入库

- 输入: `data/enriched.json`
- 输出: PostgreSQL 数据库 `stream_scripts`, 表 `scripts`
- Embedding: `BAAI/bge-small-zh-v1.5` (384 维, normalize)
- 对 `refined_script` 生成向量
- 建 HNSW 索引 (vector_cosine_ops)
- 脚本自动建库建表 (如不存在)

检验: psql SELECT 看字段和向量维度

## Step 5: Streamlit 检索

- 顶部: 筛选栏 (source_file, product_mentions, sales_stage, strategy_types)
- 中间: 语义搜索框
- 底部: 卡片流 (每页 20 条, 滚动加载)
- 卡片内容: sales_stage 标签, refined_script 文本, 来源信息

检验: 浏览器操作 — 筛选、搜索、组合查询

## 技术栈

- faster-whisper (large-v3, CUDA)
- DeepSeek API (deepseek-chat, OpenAI SDK 兼容)
- BAAI/bge-small-zh-v1.5 (sentence-transformers)
- PostgreSQL + pgvector
- Streamlit

## 配置

所有可调参数集中在 `config.py`: 路径、模型名、API key、并发数、DB 连接信息、切块字数阈值。

## 错误处理

- Step 1: m4a 文件损坏 → 打印错误, 继续下一个
- Step 3: API 超时/限流 → 重试 3 次, 仍失败记入 errors.log
- Step 3: JSON 解析失败 → 记入 errors.log, 不阻塞其他 chunk
- Step 4: DB 连接失败 → 终止, 打印连接信息供排查
- Step 5: 无结果 → 显示 "无匹配结果", 不报错

## 非目标

- 不涉及实时采集 (那是 live-script-collector 的职责)
- 不涉及音频预处理 (VAD/降噪)
- 不涉及 Milvus (当前数据量用 pgvector 足够)
- 不涉及多用户/权限
