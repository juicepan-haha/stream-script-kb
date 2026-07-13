"""stream-script-kb 全局配置"""
import os
from pathlib import Path

# --- 路径 ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNK_DIR = BASE_DIR / "audio_chunks"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 0: 下载 ---
FFMPEG_BITRATE = "128k"
FFMPEG_PATH = "/usr/bin/ffmpeg"

# --- Step 1: 转录 ---
WHISPER_MODEL = "medium"         # tiny/base/small/medium/large-v3
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"     # GPU 上 float16 更快，且省显存
WHISPER_BEAM_SIZE = 5
WHISPER_LANGUAGE = "zh"
WHISPER_BATCH_SIZE = 16          # 批量推理，提速 30-50%
WHISPER_VAD_FILTER = True        # 跳过静音段，提速 20-30%
WHISPER_CPU_THREADS = 4          # 限制 CPU 线程数，防止 WSL 资源耗尽
WHISPER_GPU_MEMORY_FRACTION = 0.6   # 限制显存使用比例，留余量给 WSL2
WHISPER_DEVICE_INDEX = 0             # GPU 索引: 0=单卡, [0,1]=双卡负载均衡, 设 -1=自动检测全部
SEGMENT_DURATION = 1800              # 流式切片时长（秒），30 分钟一段

# --- Step 2: 切块 ---
MIN_CHARS = 500
MAX_CHARS = 1000
SILENCE_GAP_SEC = 1.5             # 字幕间隔 > 此值视为换气/停顿，允许自然切分
CHUNKS_FILE = DATA_DIR / "chunks.json"

# --- Step 3: DeepSeek ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_CONCURRENCY = 400
DEEPSEEK_BATCH_SIZE = 50          # 每批 Task 数量，控制内存占用
DEEPSEEK_RETRIES = 3
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_MAX_TOKENS = 2048
ENRICHED_FILE = DATA_DIR / "enriched.json"
ERRORS_LOG = DATA_DIR / "errors.log"

# --- HuggingFace (国内需镜像) ---
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")      # 模型已缓存时跳过在线检查
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # transformers 库的离线模式

# --- Step 4: 向量化 ---
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
PG_HOST = os.environ.get("PG_HOST", "")  # 空字符串 = Unix socket
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "wentworth")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_DB = os.environ.get("PG_DB", "stream_scripts")

# --- Step 5: Streamlit ---
STREAMLIT_PAGE_SIZE = 20
STREAMLIT_TITLE = "直播话术知识库"
