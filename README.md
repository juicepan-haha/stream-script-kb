# 直播话术知识库

项目分为离线知识库构建、实时流式分析和知识库查询三个边界清晰的部分。

```text
stream-script-kb/
├── pipeline/                  # 方式一：离线批处理，结果写入文件和数据库
│   ├── step0_download.py      # m3u8 → audio_chunks/*.m4a
│   ├── step1_transcribe.py    # 音频 → data/transcripts/*.vtt
│   ├── step2_chunk.py         # 字幕 → data/chunks.json
│   ├── step3_deepseek.py      # 话术富化 → data/enriched.json
│   └── step4_vectorize.py     # 向量化 → PostgreSQL/pgvector
├── streaming/                 # 方式二：实时流式处理
│   ├── server.py              # FastAPI 应用装配和服务启动入口
│   ├── state.py               # 有界队列、任务进度和内存结果
│   ├── routes/
│   │   ├── analysis.py        # 分析、进度、结果和健康检查接口
│   │   └── rewrite.py         # RAG 话术改写接口
│   ├── workers/
│   │   ├── download.py        # ffmpeg 下载和 PCM 解码
│   │   ├── transcription.py   # Whisper 转录
│   │   ├── chunking.py        # 增量语义切块
│   │   └── enrichment.py      # DeepSeek 四段式话术富化
│   └── services/
│       └── rag.py             # 向量检索、脚本改写和 SOP 生成
├── web/                       # 知识库查询
│   └── app.py                 # Streamlit 主入口
├── shared/
│   └── config.py              # 三部分共用的路径、模型和数据库配置
├── data/                      # 离线流程的中间结果
├── audio_chunks/              # 下载后的音频
├── tests/                     # 自动化测试
├── experiments/               # 非正式实验代码
└── docs/                      # 设计和计划文档
```

## 方式一：离线知识库构建

这条处理链使用磁盘文件衔接各步骤，适合批量导入和断点重跑：

```text
m3u8 → 音频 → VTT 字幕 → 话术块 → DeepSeek 富化 → pgvector
```

从项目根目录依次运行：

```bash
python3 -m pipeline.step0_download --url "m3u8 地址" --name "主播名_日期"
python3 -m pipeline.step1_transcribe
python3 -m pipeline.step2_chunk
python3 -m pipeline.step3_deepseek
python3 -m pipeline.step4_vectorize
```

## 方式二：实时流式分析

这条处理链使用 `asyncio.Queue` 连接四个常驻 Worker：

```text
URL → 下载/解码 → Whisper 转录 → 增量切块 → DeepSeek 富化 → 内存结果
```

启动服务：

```bash
python3 -m streaming.server
```

也可以通过 Uvicorn 启动：

```bash
uvicorn streaming.server:app --host 0.0.0.0 --port 8000
```

实时分析结果保存在进程内存中。`/api/v1/rewrite` 会另外读取离线流程建立的 PostgreSQL 向量知识库。

## 查询页面

离线入库完成后启动 Streamlit：

```bash
python3 -m streamlit run web/app.py
```

全局参数位于 `shared/config.py`。运行前需要配置 DeepSeek API Key 和 PostgreSQL 连接环境变量。
