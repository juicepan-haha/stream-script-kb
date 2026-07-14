"""
server_v2.py — AI 直播话术流式分析反应堆 (路径 B)

架构: 5 级内存队列 + 4 个常驻 Worker + 零磁盘 I/O

  download_queue ──→ Worker 1 (下载+解码)
       │
  text_queue ──→ Worker 2 (Whisper 转录)
       │
  chunk_queue ──→ Worker 3 (流式语义切块)
       │
  enriched_queue ──→ Worker 4 (DeepSeek 富化)
       │
  结果存入 transcript_results / enriched_results

与路径 A 的本质区别:
  - 不产生任何 .m4a / .wav / .vtt 中间文件
  - 所有数据通过 asyncio.Queue 在 Worker 之间传递
  - 背压防御: 每级队列 maxsize 限制，下游慢则上游自动暂停
"""
import asyncio
import json
import re
import time

import numpy as np
from fastapi import FastAPI
from openai import AsyncOpenAI
import uvicorn

import config
from supabase_client import supabase


# =========================================================================
# 节流进度更新器（status 纯洁 FSM，result_text 载人话日志）
# =========================================================================

class ThrottledProgressUpdater:
    """
    工业级节流更新器（数据库条件更新防线）：
      - 自动节流: 间隔 < min_interval 秒跳过
      - force=True: 绕过节流立刻尝试写入
      - DB 级终态锁: WHERE status NOT IN ('success','failed')
        多进程/Serverless 环境下也不会被竞态覆盖
      - 应用层 is_terminal 仅作缓存优化，数据源在 DB
    """

    def __init__(self, task_id: str, min_interval: float = 5.0):
        self.task_id = task_id
        self.min_interval = min_interval
        self.last_update: float = 0.0
        self._terminal_cached: bool = False  # 仅缓存，非权威

    async def update(self, status: str, detail: str,
                     force: bool = False) -> bool:
        """
        :return: True=实际写入了数据库, False=被跳过/被DB防线拦截
        """
        # 1. 缓存加速：已知终态直接跳过，省一次网络往返
        if self._terminal_cached:
            return False

        now = time.time()

        # 2. 节流判断
        if not force and (now - self.last_update < self.min_interval):
            return False

        self.last_update = now

        # 3. DB 级条件更新：只有当前 status 不是 success/failed 才允许写入
        #    这是跨进程/跨容器的终极防线，内存 is_terminal 只是缓存
        try:
            loop = asyncio.get_running_loop()

            def _do_update():
                return (
                    supabase.table("fish_box")
                    .update({
                        "status": status,
                        "result_text": detail[:50000],
                    })
                    .eq("id", self.task_id)
                    .not_.in_("status", ["success", "failed"])
                    .execute()
                )

            response = await loop.run_in_executor(None, _do_update)

            # 4. 如果 DB 返回空（已被其他进程锁死），更新缓存
            if not response.data:
                self._terminal_cached = True
                return False

            print(f"[Supabase] [{self.task_id}] → {status}")
            return True

        except Exception as e:
            print(f"[Supabase] [{self.task_id}] ⚠️ 写入失败: {e}")
            return False

# =========================================================================
# 内存传送带（背压防御：逐级 maxsize）
# =========================================================================
download_queue  = asyncio.Queue(maxsize=20)    # URL → Worker 1
text_queue      = asyncio.Queue(maxsize=50)    # numpy 音频块 → Worker 2
chunk_queue     = asyncio.Queue(maxsize=100)   # 转录段列表 → Worker 3
enriched_queue  = asyncio.Queue(maxsize=100)   # 话术 chunk → Worker 4

# 结果集
transcript_results: dict[str, list[dict]] = {}
enriched_results: dict[str, list[dict]] = {}
task_progress: dict[str, dict] = {}

# =========================================================================
# 工具函数
# =========================================================================

_SENTENCE_END = re.compile(r"[。！？.!?]$")
_CHUNK_SECONDS = 30

# =========================================================================
# 卡密系统（阅后即焚，不落库）
# =========================================================================

_CARD_CACHE: dict[str, dict] = {}  # code → {expires_at, max_uses, used}
_CARD_LOCK = asyncio.Lock()


def _load_card_codes(path: str = "card_codes.txt") -> None:
    """从文件加载卡密到内存。"""
    global _CARD_CACHE
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    code = parts[0].strip()
                    _CARD_CACHE[code] = {
                        "expires_at": int(parts[1].strip()),
                        "max_uses": int(parts[2].strip()),
                        "used": 0,
                    }
    except FileNotFoundError:
        print("[卡密] ⚠️ card_codes.txt 不存在，使用默认测试卡密")
        _CARD_CACHE["TEST-FREE"] = {"expires_at": 9999999999, "max_uses": 999, "used": 0}


async def verify_card_code(code: str) -> bool:
    """异步校验卡密有效性，更新使用计数。"""
    async with _CARD_LOCK:
        if code not in _CARD_CACHE:
            return False
        card = _CARD_CACHE[code]
        if int(time.time()) > card["expires_at"]:
            return False
        if card["used"] >= card["max_uses"]:
            return False
        card["used"] += 1
        return True
_BYTES_PER_CHUNK = _CHUNK_SECONDS * 16000 * 2  # s16le mono 16kHz


def _parse_cookies(netscape_path: str) -> str:
    """Netscape 格式 cookie → HTTP Cookie 头字符串。"""
    cookies = []
    try:
        with open(netscape_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 6:
                    cookies.append(f"{parts[5]}={parts[6] if len(parts) >= 7 else ''}")
    except FileNotFoundError:
        pass
    return "; ".join(cookies)


# =========================================================================
# Worker 1: 流式下载 + 内存解码
# =========================================================================

async def audio_download_and_decode_worker():
    """
    [Worker 1][下载] 启动 ffmpeg 子进程，通过 stdout 读取 raw PCM，
    切成 30 秒 NumPy 块推入 text_queue。全程不落盘。
    """
    cookie_str = _parse_cookies("cookies.txt")
    header_line = f"Cookie: {cookie_str}\r\n" if cookie_str else ""

    print("[Worker 1][下载] 就绪，等待任务...")

    while True:
        task_data = await download_queue.get()
        url = task_data["url"]
        task_id = task_data["task_id"]
        print(f"[Worker 1][下载] [{task_id}] 开始流式捕获: {url}")

        task_progress.setdefault(task_id, {})["stage"] = "downloading"

        cmd = [
            config.FFMPEG_PATH,
            "-headers", header_line,
            "-i", url,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            chunk_idx = 0
            offset_sec = 0.0
            t_start = time.time()

            while True:
                data = await proc.stdout.read(_BYTES_PER_CHUNK)
                if not data:
                    break

                samples = (
                    np.frombuffer(data, dtype=np.int16)
                    .astype(np.float32) / 32768.0
                )

                await text_queue.put({
                    "task_id": task_id,
                    "audio": samples,
                    "chunk_idx": chunk_idx,
                    "offset_sec": offset_sec,
                    "user_key": task_data.get("user_key", ""),
                })

                chunk_idx += 1
                offset_sec += _CHUNK_SECONDS

                if chunk_idx % 10 == 0:
                    elapsed = time.time() - t_start
                    print(f"[Worker 1][下载] [{task_id}] "
                          f"已推送 {chunk_idx} 块 ({offset_sec/60:.0f} 分钟), "
                          f"{(offset_sec/elapsed):.1f}x 实时率")

            # EOF
            await text_queue.put({
                "task_id": task_id,
                "done": True,
                "total_chunks": chunk_idx,
                "user_key": task_data.get("user_key", ""),
            })

            stderr_data = await proc.stderr.read()
            await proc.wait()
            if proc.returncode != 0:
                err_msg = stderr_data.decode(errors="replace")[-500:]
                print(f"[Worker 1][下载] [{task_id}] ⚠️ ffmpeg rc={proc.returncode}: "
                      f"{err_msg}")

            elapsed = time.time() - t_start
            print(f"[Worker 1][下载] [{task_id}] ✅ 完成: "
                  f"{chunk_idx} 块, {elapsed:.0f}s, "
                  f"{(offset_sec/elapsed):.1f}x 实时率")

        except Exception as e:
            print(f"[Worker 1][下载] [{task_id}] ❌ 错误: {e}")
            await text_queue.put({"task_id": task_id, "error": str(e)})

        finally:
            download_queue.task_done()


# =========================================================================
# Worker 2: 常驻显存的 Whisper 转录
# =========================================================================

async def whisper_transcribe_worker():
    """
    [Worker 2][转录] GPU 常驻，从 text_queue 消费 NumPy 数组，
    直接喂给 faster-whisper。输出带时间戳的段列表推入 chunk_queue。
    """
    from faster_whisper import WhisperModel
    import torch

    if torch.cuda.is_available():
        gpu_idx = config.WHISPER_DEVICE_INDEX
        if gpu_idx == -1:
            gpu_idx = 0
        total_vram = (
            torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3
        )
        usable = total_vram * config.WHISPER_GPU_MEMORY_FRACTION * 0.6
        model_map = {
            "large-v3": 4.5, "medium": 2.5, "small": 1.5,
            "base": 0.7, "tiny": 0.6,
        }
        best = "tiny"
        for m, vram in model_map.items():
            if vram <= usable:
                best = m
                break
        device, compute = "cuda", "float16"
        batch = max(4, min(32, int((usable - model_map[best]) / 0.1)))
        torch.cuda.set_per_process_memory_fraction(
            config.WHISPER_GPU_MEMORY_FRACTION, gpu_idx
        )
        print(f"[Worker 2][转录] 🖥️ GPU: {total_vram:.1f}GB → "
              f"model={best}, batch_size={batch}")
    else:
        best, device, compute, batch = "base", "cpu", "int8", 1
        print(f"[Worker 2][转录] ⚠️ 无 GPU，降级 CPU base")

    print(f"[Worker 2][转录] 加载模型: {best} ...")
    t0 = time.time()
    model = WhisperModel(best, device=device, compute_type=compute)
    print(f"[Worker 2][转录] 模型就绪 ({time.time() - t0:.1f}s)，等待音频...")

    while True:
        chunk = await text_queue.get()

        try:
            task_id = chunk["task_id"]
            task_progress.setdefault(task_id, {})["stage"] = "transcribing"

            if chunk.get("done"):
                print(f"[Worker 2][转录] [{task_id}] ✅ 转录完成 "
                      f"({chunk.get('total_chunks', 0)} 块)")
                await chunk_queue.put({
                    "task_id": task_id,
                    "done": True,
                    "user_key": chunk.get("user_key", ""),
                })
                text_queue.task_done()
                continue

            if chunk.get("error"):
                print(f"[Worker 2][转录] [{task_id}] ❌ 上游错误: {chunk['error']}")
                text_queue.task_done()
                continue

            audio = chunk["audio"]
            offset = chunk["offset_sec"]
            cidx = chunk["chunk_idx"]

            print(f"[Worker 2][转录] [{task_id}] 块 {cidx} "
                  f"(偏移 {offset:.0f}s)...", end=" ", flush=True)

            t1 = time.time()
            segments, _ = model.transcribe(
                audio,
                beam_size=config.WHISPER_BEAM_SIZE,
                language=config.WHISPER_LANGUAGE,
                vad_filter=config.WHISPER_VAD_FILTER,
            )

            seg_list = []
            for seg in segments:
                seg_list.append({
                    "start": seg.start + offset,
                    "end": seg.end + offset,
                    "text": seg.text.strip(),
                })

            elapsed = time.time() - t1
            ratio = len(audio) / 16000 / elapsed if elapsed > 0 else 0
            print(f"OK ({elapsed:.1f}s, {ratio:.1f}x, {len(seg_list)} 段)")

            # 存原始转录（API 可查询）
            if task_id not in transcript_results:
                transcript_results[task_id] = []
            transcript_results[task_id].extend(seg_list)

            # 推入切块队列
            await chunk_queue.put({
                "task_id": task_id,
                "segments": seg_list,
                "user_key": chunk.get("user_key", ""),
            })

        except Exception as e:
            print(f"[Worker 2][转录] [{chunk.get('task_id', '?')}] ❌ 错误: {e}")

        finally:
            text_queue.task_done()


# =========================================================================
# Worker 3: 流式语义切块（时间停顿 + 句末标点）
# =========================================================================

async def chunking_worker():
    """
    [Worker 3][切块] 从 chunk_queue 消费转录段，用滑动窗口做语义切块。
    触发条件: buffer >= min_chars 且 (句末标点 或 段间停顿 > 1.5s)

    与 step2_chunk.py 逻辑一致，但改为流式增量处理。
    """
    # 每个 task_id 维护一个独立的滑动缓冲区
    buffers: dict[str, list[dict]] = {}      # 段列表
    buffer_chars: dict[str, int] = {}         # 累计字数
    chunk_indices: dict[str, int] = {}        # chunk 序号

    min_chars = config.MIN_CHARS
    max_chars = config.MAX_CHARS
    silence_gap = config.SILENCE_GAP_SEC

    def _emit_chunk(task_id, entries):
        """将缓冲区内容输出为一个 chunk。"""
        idx = chunk_indices.get(task_id, 0)
        combined = "".join(e["text"] for e in entries)
        chunk_data = {
            "chunk_id": f"{task_id}_chunk_{idx + 1:04d}",
            "source_file": task_id,
            "start_time": entries[0]["start"],
            "end_time": entries[-1]["end"],
            "char_count": len(combined),
            "text": combined,
        }
        chunk_indices[task_id] = idx + 1
        return chunk_data

    print("[Worker 3][切块] 就绪，等待转录段...")

    while True:
        item = await chunk_queue.get()

        try:
            task_id = item["task_id"]

            if item.get("done"):
                # 上游结束，清空残片
                user_key = item.get("user_key", "")
                if task_id in buffers and buffers[task_id]:
                    chunk = _emit_chunk(task_id, buffers[task_id])
                    await enriched_queue.put({
                        "task_id": task_id, "chunk": chunk, "user_key": user_key,
                    })
                    print(f"[Worker 3][切块] [{task_id}] 尾部残片 → {chunk['chunk_id']} "
                          f"({chunk['char_count']} 字)")
                user_key = item.get("user_key", "")
                await enriched_queue.put({
                    "task_id": task_id, "done": True, "user_key": user_key,
                })
                task_progress.setdefault(task_id, {})["stage"] = "chunked"
                buffers.pop(task_id, None)
                buffer_chars.pop(task_id, None)
                print(f"[Worker 3][切块] [{task_id}] ✅ 切块完成")
                continue

            segments = item["segments"]
            if not segments:
                continue

            # 初始化缓冲区
            if task_id not in buffers:
                buffers[task_id] = []
                buffer_chars[task_id] = 0
                chunk_indices[task_id] = 0

            buffer = buffers[task_id]

            for i, seg in enumerate(segments):
                text = seg["text"]
                char_count = len(text)

                # 单条超 max_chars：先清 buffer，再硬切这条
                if char_count > max_chars:
                    if buffer:
                        chunk = _emit_chunk(task_id, buffer)
                        await enriched_queue.put({
                            "task_id": task_id, "chunk": chunk,
                            "user_key": item.get("user_key", ""),
                        })
                        print(f"[Worker 3][切块] [{task_id}] "
                              f"清缓冲区 → {chunk['chunk_id']} ({chunk['char_count']} 字)")
                        buffer.clear()
                        buffer_chars[task_id] = 0

                    pos = 0
                    while pos < char_count:
                        seg_text = text[pos:pos + max_chars]
                        hard_chunk = {
                            "chunk_id": f"{task_id}_chunk_{chunk_indices[task_id] + 1:04d}",
                            "source_file": task_id,
                            "start_time": seg["start"],
                            "end_time": seg["end"],
                            "char_count": len(seg_text),
                            "text": seg_text,
                        }
                        chunk_indices[task_id] += 1
                        await enriched_queue.put({
                            "task_id": task_id, "chunk": hard_chunk,
                            "user_key": item.get("user_key", ""),
                        })
                        pos += max_chars
                    continue

                buffer.append(seg)
                buffer_chars[task_id] += char_count

                # 判断是否该切
                should_split = False
                if buffer_chars[task_id] >= max_chars:
                    should_split = True
                elif buffer_chars[task_id] >= min_chars:
                    combined = "".join(e["text"] for e in buffer)
                    if _SENTENCE_END.search(combined):
                        should_split = True
                    elif i + 1 < len(segments):
                        next_seg = segments[i + 1]
                        if next_seg["start"] - seg["end"] > silence_gap:
                            should_split = True

                if should_split:
                    chunk = _emit_chunk(task_id, buffer)
                    await enriched_queue.put({
                        "task_id": task_id, "chunk": chunk,
                        "user_key": item.get("user_key", ""),
                    })
                    buffer.clear()
                    buffer_chars[task_id] = 0

        except Exception as e:
            print(f"[Worker 3][切块] [{item.get('task_id', '?')}] ❌ 错误: {e}")

        finally:
            chunk_queue.task_done()


# =========================================================================
# Worker 4: DeepSeek 提词器级富化
# =========================================================================

# Prompt 复用 step3 的设计
SYSTEM_PROMPT = """你是精通带货心理学和直播控场的话术提词专家。
你的任务是把口语流水账文本改造成主播可以直接上场朗读的"提词器级逐字稿"。

【硬性改造规则】：
1. 彻底过滤：删掉所有语病、重复、废话、口头禅（如"好不好"、"是不是"、"然后最后"、"那个那个"）
2. 逻辑分段：将混乱的叙述按照直播销售逻辑进行切片
3. 语气标注：在关键动作和语气转换处，使用中括号 [] 标注主播的情绪和动作提示
4. 保留干货：所有涉及价格、赠品、规格、色号、使用方法的具体数字和专有名词，绝对不准改

你必须输出合法的 json 对象（不要 Markdown 包裹）：
{
  "icebreaker": "破冰留人话术",
  "painpoint": "痛点植入话术",
  "mechanism": "产品卖点话术（必须锁定所有数字数据）",
  "close_order": "逼单催单话术"
}

如果文本内容不包含明确话术（纯闲聊、无意义重复），所有字段填空字符串。"""


# 动态关键词提取（复用 step3）
def _extract_critical_keywords(text: str) -> list[str]:
    keywords = set()
    patterns = [
        r"\d+[\.\d]*\s*(?:元|块|钱|折|%|％|ml|毫升|g|克|kg|斤|片|盒|瓶|支|包|袋|件|条|双|套|色|号|码|寸|英寸|分钟|小时|天|月|年|代|版|次)",
        r"(?:买|送|赠|减|省|便宜|优惠|只要|仅需|原价|现价|到手|券后)\s*\d+[\.\d]*",
        r"\d+[\.\d]*\s*(?:万|千|百|十|亿)?\s*(?:粉丝|销量|回购|好评|单)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            kw = m.group().strip()
            if len(kw) >= 2:
                keywords.add(kw)
    numeric = sorted(k for k in keywords if re.search(r"\d", k))
    return numeric[:15]


async def deepseek_enrich_worker():
    """
    [Worker 4][DeepSeek] 从 enriched_queue 消费话术 chunk。
    使用用户的 DeepSeek Key 动态初始化客户端（阅后即焚，不落盘）。
    """
    sem = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)
    # 缓存已创建的客户端，避免每个 chunk 都重新初始化
    client_cache: dict[str, AsyncOpenAI] = {}

    print("[Worker 4][DeepSeek] 就绪，等待话术 chunk...")

    while True:
        item = await enriched_queue.get()

        try:
            task_id = item["task_id"]

            if item.get("done"):
                task_progress.setdefault(task_id, {})["stage"] = "completed"
                print(f"[Worker 4][DeepSeek] [{task_id}] ✅ 全部富化完成 "
                      f"({len(enriched_results.get(task_id, []))} 条)")
                # 清理该任务的客户端缓存
                client_cache.pop(task_id, None)
                enriched_queue.task_done()
                continue

            chunk = item["chunk"]
            text = chunk["text"]
            critical_kw = _extract_critical_keywords(text)

            # 动态获取用户的 DeepSeek Key
            user_key = item.get("user_key", config.DEEPSEEK_API_KEY)
            if task_id not in client_cache:
                client_cache[task_id] = AsyncOpenAI(
                    api_key=user_key,
                    base_url=config.DEEPSEEK_BASE_URL,
                )
            client = client_cache[task_id]

            async with sem:
                for attempt in range(1, config.DEEPSEEK_RETRIES + 1):
                    try:
                        response = await client.chat.completions.create(
                            model=config.DEEPSEEK_MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": (
                                    f"【核心数据保卫战：以下关键词必须保留 — "
                                    f"{'、'.join(critical_kw) if critical_kw else '无'}】\n\n"
                                    f"{text}"
                                )},
                            ],
                            temperature=config.DEEPSEEK_TEMPERATURE,
                            max_tokens=config.DEEPSEEK_MAX_TOKENS,
                            response_format={"type": "json_object"},
                        )
                        content = response.choices[0].message.content.strip()
                        parsed = __import__("json").loads(content)
                        break
                    except Exception:
                        if attempt == config.DEEPSEEK_RETRIES:
                            parsed = {
                                "icebreaker": text[:200],
                                "painpoint": "",
                                "mechanism": "",
                                "close_order": "",
                                "_error": "API 重试全部失败",
                            }
                        else:
                            await asyncio.sleep(1 * attempt)

            # 组装输出
            full_parts = []
            for k in ["icebreaker", "painpoint", "mechanism", "close_order"]:
                if parsed.get(k):
                    full_parts.append(parsed[k])

            enriched = {
                "chunk_id": chunk["chunk_id"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "icebreaker": parsed.get("icebreaker", ""),
                "painpoint": parsed.get("painpoint", ""),
                "mechanism": parsed.get("mechanism", ""),
                "close_order": parsed.get("close_order", ""),
                "refined_script": "\n\n".join(full_parts) if full_parts else text,
                "selling_points": critical_kw[:10],
            }

            if task_id not in enriched_results:
                enriched_results[task_id] = []
            enriched_results[task_id].append(enriched)

            idx = len(enriched_results[task_id])
            print(f"[Worker 4][DeepSeek] [{task_id}] {chunk['chunk_id']} OK "
                  f"(#{idx}, {chunk['char_count']} 字)")

        except Exception as e:
            print(f"[Worker 4][DeepSeek] [{item.get('task_id', '?')}] ❌ 错误: {e}")

        finally:
            enriched_queue.task_done()


# =========================================================================
# FastAPI 接单大厅
# =========================================================================

app = FastAPI(title="AI直播话术流式分析反应堆 (路径B)")


@app.post("/api/v1/analyze")
async def start_analysis(
    task_id: str,
    video_url: str,
    user_deepseek_key: str = "",
    card_code: str = "",
):
    """
    Supabase 云版接口：前端先在 fish_box INSERT → 获得 task_id，
    然后调用此接口。后端 0ms 返回 queued，后台异步处理。
    """
    # 1. 卡密校验
    if not await verify_card_code(card_code):
        return {"status": "error", "message": "❌ 卡密无效或已过期！"}
    if not user_deepseek_key or not user_deepseek_key.startswith("sk-"):
        return {"status": "error", "message": "❌ DeepSeek API Key 格式无效！"}

    # 2. 立即返回，前端不转圈（BackgroundTasks 管理生命周期）
    asyncio.create_task(_run_pipeline(
        task_id, video_url, user_deepseek_key,
    ))

    return {"status": "queued", "task_id": task_id}


# =========================================================================
# 管道编排器：分阶段执行 + 节流更新器 + force 最终态
# =========================================================================

async def _run_pipeline(task_id: str, url: str, user_key: str):
    """
    后台协程 — 完整 4 阶段管道。
    updater.update("processing", ...) 自动节流 >=5 秒，
    updater.update("success", ..., force=True) 立刻写入最终结果。
    """
    updater = ThrottledProgressUpdater(task_id, min_interval=5.0)
    print(f"[Pipeline] [{task_id}] 启动: {url}")

    try:
        # ---- Stage 1: 下载 + 解码 ----
        await updater.update("processing",
                             "正在从直播源下载音频流并解码...", force=True)
        await download_queue.put({
            "task_id": task_id, "url": url, "user_key": user_key,
        })

        # ---- Stage 2: 等待转录有产出 ----
        while True:
            segs = len(transcript_results.get(task_id, []))
            if segs > 0:
                break
            await asyncio.sleep(5)

        # ---- Stage 3 & 4: 富化中，高频调用 updater 自动节流 ----
        last_report = ""
        while True:
            if task_progress.get(task_id, {}).get("stage") == "completed":
                break

            segs = len(transcript_results.get(task_id, []))
            chunks = len(enriched_results.get(task_id, []))
            report = f"GPU转录中 (已转录 {segs} 段) | AI提炼中 (已富化 {chunks} 块)"

            if report != last_report:
                # 高频调用也没事，updater 内部节流 >=5 秒
                await updater.update("processing", report)
                last_report = report

            await asyncio.sleep(3)

        # ---- 成功：force=True 立刻写入，绕过节流 ----
        enriched = enriched_results.get(task_id, [])
        if enriched:
            result_json = json.dumps(enriched, ensure_ascii=False, indent=2)
            await updater.update("success", result_json, force=True)
        else:
            await updater.update("success",
                                 "分析完成，但未提取到有效话术",
                                 force=True)
        print(f"[Pipeline] [{task_id}] ✅ ({len(enriched)} chunks)")

    except Exception as e:
        friendly_msg = _wrap_error_for_user(e)
        await updater.update("failed", friendly_msg, force=True)
        print(f"[Pipeline] [{task_id}] ❌ {friendly_msg}")


def _wrap_error_for_user(e: Exception) -> str:
    """防呆包装：将 Python 异常转为人话，不暴露源码。"""
    msg = str(e)
    if "TimeoutError" in type(e).__name__ or "timeout" in msg.lower():
        return "任务处理超时，请稍后重试或联系客服。"
    if "Connection" in type(e).__name__ or "connect" in msg.lower():
        return "网络连接异常，服务器无法访问直播源。"
    if "ffmpeg" in msg.lower():
        return "音频解码失败，直播源可能已失效。"
    if "CUDA" in msg or "out of memory" in msg.lower():
        return "服务器 GPU 资源不足，请稍后重试。"
    # 默认兜底
    return f"任务处理异常（错误代码: {task_id[-8:]}），请联系客服。"


@app.get("/api/v1/transcript/{task_id}")
async def get_transcript(task_id: str):
    """查询原始转录结果。"""
    segs = transcript_results.get(task_id, [])
    return {
        "task_id": task_id,
        "segments": len(segs),
        "text": "".join(s["text"] for s in segs),
        "details": segs,
    }


@app.get("/api/v1/enriched/{task_id}")
async def get_enriched(task_id: str):
    """查询 DeepSeek 富化后的提词器级话术。"""
    items = enriched_results.get(task_id, [])
    return {
        "task_id": task_id,
        "chunks": len(items),
        "results": items,
    }


@app.get("/api/v1/progress/{task_id}")
async def get_progress(task_id: str):
    """查询任务处理进度。"""
    progress = task_progress.get(task_id, {})
    return {
        "task_id": task_id,
        "stage": progress.get("stage", "unknown"),
        "transcript_segments": len(transcript_results.get(task_id, [])),
        "enriched_chunks": len(enriched_results.get(task_id, [])),
    }


@app.get("/api/v1/health")
async def health():
    return {
        "download_queue": download_queue.qsize(),
        "text_queue": text_queue.qsize(),
        "chunk_queue": chunk_queue.qsize(),
        "enriched_queue": enriched_queue.qsize(),
        "active_tasks": list(transcript_results.keys()),
    }


# =========================================================================
# Step 5: RAG "像素级平替" 爆款脚本重写
# =========================================================================

# 全局 Embedding 模型 + DB 连接（常驻内存）
_rag_model = None
_rag_db = None

# RAG + SOP Prompt 模板（合二为一，省一次 API 调用）
RAG_SYSTEM_PROMPT = """你是深谙中国直播电商（抖音、快手、淘宝）底层人性逻辑的顶级黄金卖货操盘手，
也是单兵作战的中小主播运营顾问。

TASK:
参考【历史爆款结构参考】的话术节奏，将新产品【{my_product}】重写为
【{target_style}】风格的口语话术脚本，同时输出一份秒级执行 SOP 仪表盘。

你输出的对象是"单兵作战"的中小主播——没有场控、没有助播、一个人全包。
SOP 必须精确到秒级动作指引，包含视觉和操作层面的提示，贴电脑屏幕旁就能无脑执行。

你必须输出一个合法的 json 对象，不要 Markdown 包裹：

{{
  "rewritten_script": "完整的四段式口语话术脚本（带 [破冰留人][痛点植入][产品卖点][逼单催单] 标注）",
  "sop_timeline": [
    {{
      "time_range": "00:00 - 00:30",
      "stage": "Icebreaker (开场憋单)",
      "host_action": "主播的肢体动作、表情、道具使用",
      "operation_action": "后台操作（弹链接/改价/发券/贴纸）",
      "verbal_keywords": "这个阶段必须喊的关键词"
    }}
  ]
}}

STRICT RULES:
1. 绝对不用书面语！多用"家人们、别划走、听我的、最后3单、拼手速、没了直接下播"等口语。
2. SOP 时间轴必须覆盖完整话术流程，每段 20-60 秒，总时长 2-5 分钟。
3. host_action 要具体到"眼睛看哪里、手做什么、用什么道具、身体姿态"。
4. operation_action 遵循"准备→触发→收尾"逻辑，单品直播链路完整。
5. 不要包含任何 AI 前言和客套话。
6. sop_timeline 是必填字段，必须包含 4-6 个时间节点，覆盖从开场到促单的完整链路。"""


def _init_rag():
    """延迟初始化 RAG 所需模型和 DB 连接。"""
    global _rag_model, _rag_db
    if _rag_model is None:
        from sentence_transformers import SentenceTransformer
        import psycopg2
        from pgvector.psycopg2 import register_vector

        print("[Step 5][RAG] 加载 Embedding 模型...")
        _rag_model = SentenceTransformer(config.EMBEDDING_MODEL)

        print("[Step 5][RAG] 连接向量数据库...")
        _rag_db = psycopg2.connect(
            host=config.PG_HOST, port=config.PG_PORT,
            user=config.PG_USER, password=config.PG_PASSWORD,
            dbname=config.PG_DB,
        )
        register_vector(_rag_db)
        print("[Step 5][RAG] ✅ 就绪")


def _vector_search(query: str, top_k: int = 5) -> list[dict]:
    """pgvector 余弦相似度检索历史爆款话术。"""
    _init_rag()

    vec = _rag_model.encode(query, normalize_embeddings=True).tolist()
    vec_str = json.dumps(vec)

    cur = _rag_db.cursor()
    cur.execute("""
        SELECT icebreaker, painpoint, mechanism, close_order,
               refined_script, sales_stage, strategy_types,
               product_mentions, selling_points,
               embedding <=> %s::vector AS distance
        FROM scripts
        WHERE refined_script != ''
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (vec_str, vec_str, top_k))

    rows = cur.fetchall()
    cur.close()

    results = []
    for row in rows:
        (ice, pain, mech, close, refined, stage,
         strategies, products, selling, distance) = row

        similarity = max(0.0, 1.0 - float(distance)) if distance else 0.0

        # JSONB 字段 psycopg2 直接返回 list
        if isinstance(strategies, str):
            strategies = json.loads(strategies)
        if isinstance(products, str):
            products = json.loads(products)

        results.append({
            "similarity": round(similarity, 4),
            "icebreaker": ice or "",
            "painpoint": pain or "",
            "mechanism": mech or "",
            "close_order": close or "",
            "refined_script": refined or "",
            "sales_stage": stage or "",
            "strategy_types": strategies or [],
            "product_mentions": products or [],
            "selling_points": selling or [],
        })

    return results


def _build_rag_context(retrieved: list[dict]) -> str:
    """把检索结果组装为 prompt context 字符串。"""
    parts = []
    for i, item in enumerate(retrieved, 1):
        sim_pct = item["similarity"] * 100
        parts.append(
            f"━━━ 爆款参考 #{i} (相似度 {sim_pct:.0f}%) ━━━\n"
            f"● 销售阶段: {item['sales_stage']}\n"
            f"● 话术策略: {', '.join(item['strategy_types'][:5])}\n"
            f"● 涉及品类: {', '.join(item['product_mentions'][:5])}\n"
            f"\n[破冰留人]:\n{item['icebreaker']}\n"
            f"\n[痛点植入]:\n{item['painpoint']}\n"
            f"\n[产品卖点]:\n{item['mechanism']}\n"
            f"\n[逼单催单]:\n{item['close_order']}\n"
        )
    return "\n".join(parts)


@app.post("/api/v1/rewrite")
async def rewrite_script(
    my_product: str,
    target_style: str = "呐喊憋单流",
    user_key: str = "",
    card_code: str = "",
):
    """
    Step 5: RAG 像素级平替 — 用户输入产品 → 向量检索历史爆款 → DeepSeek 重写。

    参数:
      my_product: 你要卖的产品（如"多功能不粘锅"）
      target_style: 目标话术风格（如"呐喊憋单流"、"温柔种草流"、"硬核测评流"）
      user_key: 用户自带的 DeepSeek API Key（阅后即焚）
      card_code: 本站激活卡密
    """
    # 卡密校验
    if not await verify_card_code(card_code):
        return {"status": "error", "message": "❌ 卡密无效或已过期！"}
    if not user_key or not user_key.startswith("sk-"):
        return {"status": "error", "message": "❌ DeepSeek API Key 格式无效！"}

    print(f"[Step 5][RAG] 检索请求: product={my_product}, style={target_style}")

    # 1. RAG 检索
    retrieved = _vector_search(my_product, top_k=5)
    if not retrieved:
        return {"status": "no_results", "message": "数据库中没有匹配的话术参考"}

    # 2. 组装 context
    context = _build_rag_context(retrieved)

    # 3. DeepSeek 重写（使用用户的 Key）
    client = AsyncOpenAI(
        api_key=user_key,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    user_prompt = (
        f"━━━ 历史爆款结构参考 ━━━\n{context}\n\n"
        f"━━━ 新任务 ━━━\n"
        f"产品: {my_product}\n"
        f"风格: {target_style}\n"
        f"请严格按照 SYSTEM ROLE 的所有规则，输出可直接念的话术脚本。"
    )

    raw_content = ""
    for attempt in range(config.DEEPSEEK_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT.format(
                        my_product=my_product, target_style=target_style
                    )},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            break
        except Exception as e:
            if attempt == config.DEEPSEEK_RETRIES - 1:
                print(f"[Step 5][RAG] ❌ DeepSeek 调用失败: {e}")
                return {"status": "error", "message": f"AI 重写失败: {e}"}
            await asyncio.sleep(1 * (attempt + 1))

    # 解析合二为一的输出：script + SOP timeline
    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        result = {
            "rewritten_script": raw_content,
            "sop_timeline": [],
        }

    # SOP 兜底：如果 DeepSeek 漏了 sop_timeline，追加一次专项请求
    if not result.get("sop_timeline"):
        print("[Step 5][RAG] SOP 为空，触发兜底重试...")
        sop_prompt = (
            f"产品: {my_product}\n风格: {target_style}\n"
            f"话术脚本:\n{result.get('rewritten_script', '')[:1500]}\n\n"
            f"请为上述话术脚本生成一份秒级 SOP 时间轴 JSON 数组。"
        )
        for attempt in range(2):
            try:
                sop_resp = await client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": (
                            "你是直播运营SOP专家。输出一个 json 对象: "
                            '{"sop_timeline": [{"time_range":"...","stage":"...",'
                            '"host_action":"...","operation_action":"...",'
                            '"verbal_keywords":"..."}]}'
                        )},
                        {"role": "user", "content": sop_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                sop_data = json.loads(
                    sop_resp.choices[0].message.content.strip()
                )
                result["sop_timeline"] = sop_data.get("sop_timeline", [])
                if result["sop_timeline"]:
                    break
            except Exception:
                await asyncio.sleep(1)

    return {
        "status": "success",
        "my_product": my_product,
        "target_style": target_style,
        "retrieved_references": [
            {
                "similarity": r["similarity"],
                "sales_stage": r["sales_stage"],
                "strategy_types": r["strategy_types"][:3],
            }
            for r in retrieved
        ],
        "rewritten_script": result.get("rewritten_script", raw_content),
        "sop_timeline": result.get("sop_timeline", []),
    }


@app.on_event("startup")
async def startup_event():
    _load_card_codes()
    print(f"[卡密] 已加载 {len(_CARD_CACHE)} 个卡密")
    asyncio.create_task(audio_download_and_decode_worker())
    asyncio.create_task(whisper_transcribe_worker())
    asyncio.create_task(chunking_worker())
    asyncio.create_task(deepseek_enrich_worker())
    print("🚀 异步反应堆 4 级流水线全部就位！")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
