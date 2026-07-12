"""stream-script-kb 全局配置"""
import os
from pathlib import Path

# --- 路径 ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHUNK_DIR = Path("/home/justin/audio_chunks")
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

# --- Step 0: 下载 ---
FFMPEG_BITRATE = "128k"
FFMPEG_PATH = "/usr/bin/ffmpeg"

# --- Step 1: 转录 ---
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
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
DEEPSEEK_CONCURRENCY = 400
DEEPSEEK_RETRIES = 3
DEEPSEEK_TEMPERATURE = 0.3
DEEPSEEK_MAX_TOKENS = 2048
ENRICHED_FILE = DATA_DIR / "enriched.json"
ERRORS_LOG = DATA_DIR / "errors.log"

# --- Step 4: 向量化 ---
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
PG_HOST = os.environ.get("PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("PG_PORT", "5432"))
PG_USER = os.environ.get("PG_USER", "postgres")
PG_PASSWORD = os.environ.get("PG_PASSWORD", "")
PG_DB = os.environ.get("PG_DB", "stream_scripts")

# --- Step 5: Streamlit ---
STREAMLIT_PAGE_SIZE = 20
STREAMLIT_TITLE = "直播话术知识库"
