# 直播话术知识库 (Livestream Script Knowledge Base)

AI 驱动的直播销售话术分析、重写与 SOP 生成系统。主攻长尾市场中小主播群体。

## 分支策略

| 分支 | 定位 | 适合场景 |
|------|------|---------|
| `main` | 稳定批处理版 | 本地离线分析，文件接力管道 |
| `industrial-refactor` | 高性能流式版 | 全内存零磁盘反应堆 + RAG + SOP |
| `Commercial-version` | SaaS 商用版 | 卡密系统 + 阅后即焚 + Streamlit 前端 |

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置环境变量
export DEEPSEEK_API_KEY="sk-..."
export PG_HOST="" PG_USER="wentworth" PG_DB="stream_scripts"

# 启动后端 (Commercial-version)
python server_v2.py

# 启动前端 (另一个终端)
streamlit run app_frontend.py
```

## 核心模块

| 文件 | 功能 |
|------|------|
| `server_v2.py` | FastAPI 异步反应堆（4 Worker + RAG + SOP + 卡密） |
| `app_frontend.py` | Streamlit SaaS 前端（双模式 + Key 盲测 + 进度轮询） |
| `app.py` | Streamlit 语义搜索界面（batch 管道用） |
| `step0_download.py` | m3u8 并发下载（ThreadPoolExecutor 16 并发） |
| `step1_transcribe.py` | GPU 流式转录（ffmpeg→numpy→faster-whisper 零磁盘） |
| `step2_chunk.py` | 语义切块（时间停顿 1.5s + 句末标点） |
| `step3_deepseek.py` | DeepSeek 提词器级富化（四段式 + 动态校验） |
| `step4_vectorize.py` | pgvector 向量化入库（批量插入 + JSONB + ON CONFLICT） |
| `config.py` | 全局配置（硬件自适应 + HF 镜像 + 所有参数） |

## 系统要求

- Python 3.12+
- PostgreSQL 16 + pgvector
- ffmpeg
- NVIDIA GPU (可选，CPU 自动降级)
