#!/usr/bin/env python3
"""Step 3: 使用 DeepSeek API 对每个 chunk 做结构化信息提取。

输入: data/chunks.json
输出: data/enriched.json + data/errors.log
并发: asyncio + 异步 OpenAI client, 并发数 config.DEEPSEEK_CONCURRENCY
"""
import asyncio
import json
import logging
import sys
from datetime import datetime

from openai import AsyncOpenAI

import config

# --- Prompt ---
SYSTEM_PROMPT = """你是直播话术分析专家。对给定的直播话术文本块，提取以下结构化信息。

严格按照 JSON 格式返回，不要输出任何 JSON 之外的内容。

字段说明:
- refined_script: 去噪润色后的流畅话术文本（修复口语重复、语气词过多、转录错误）
- summary: 这段话术的一句话摘要
- sales_stage: 销售阶段，只能是以下之一: 开场暖场/产品引入/价值塑造/逼单促单/互动留人/转款过渡/结尾收场
- strategy_types: 话术策略标签列表，从以下选择: ["限时稀缺", "价格锚定", "信任背书", "从众效应", "痛点放大", "场景代入", "算账对比", "亲身试用", "福利诱导", "互动引导", "悬念制造", "情感共鸣"]
- product_mentions: 提到的商品或品类名称列表
- selling_points: 提取的卖点列表（每条 10-20 字）
- target_audience: 目标人群描述，如 "宝妈"、"学生党"、"上班族"、"中老年" 等

返回格式:
{
  "refined_script": "...",
  "summary": "...",
  "sales_stage": "...",
  "strategy_types": [...],
  "product_mentions": [...],
  "selling_points": [...],
  "target_audience": "..."
}

如果文本内容不包含明确的话术（纯闲聊、无意义重复），sales_stage 填 "无法识别"，其他字段填空。"""


def _build_user_prompt(chunk_text: str) -> str:
    return f"请分析以下直播话术文本:\n\n{chunk_text}"


async def _process_one(
    client: AsyncOpenAI,
    chunk: dict,
    sem: asyncio.Semaphore,
) -> dict | None:
    """处理单个 chunk, 成功返回 enriched dict, 失败返回 None 并记日志。"""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    for attempt in range(1, config.DEEPSEEK_RETRIES + 1):
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(text)},
                    ],
                    temperature=config.DEEPSEEK_TEMPERATURE,
                    max_tokens=config.DEEPSEEK_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()
                parsed = json.loads(content)

                return {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "refined_script": parsed.get("refined_script", text),
                    "summary": parsed.get("summary", ""),
                    "sales_stage": parsed.get("sales_stage", ""),
                    "strategy_types": parsed.get("strategy_types", []),
                    "product_mentions": parsed.get("product_mentions", []),
                    "selling_points": parsed.get("selling_points", []),
                    "target_audience": parsed.get("target_audience", ""),
                }

            except (json.JSONDecodeError, KeyError) as e:
                raw = content if "content" in dir() else str(e)
                if attempt == config.DEEPSEEK_RETRIES:
                    _log_error(chunk_id, text, raw)
                    return None
                await asyncio.sleep(1 * attempt)

            except Exception as e:
                if attempt == config.DEEPSEEK_RETRIES:
                    _log_error(chunk_id, text, str(e))
                    return None
                await asyncio.sleep(2 * attempt)

    return None


def _log_error(chunk_id: str, original_text: str, error_info: str) -> None:
    """将失败的 chunk 写入 errors.log。"""
    config.ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.ERRORS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Chunk ID: {chunk_id}\n")
        f.write(f"Error: {error_info}\n")
        f.write(f"Text ({len(original_text)} chars):\n{original_text}\n")
        f.write(f"{'=' * 60}\n\n")


async def main_async():
    if not config.DEEPSEEK_API_KEY:
        print("[STEP3] ERROR: DEEPSEEK_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个 chunk (0=全部)")
    args = parser.parse_args()

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if args.limit > 0 and args.limit < len(chunks):
        chunks = chunks[:args.limit]

    total = len(chunks)
    print(f"[STEP3] Loaded {total} chunks from {config.CHUNKS_FILE}")

    # 清空 errors.log
    config.ERRORS_LOG.write_text("", encoding="utf-8")

    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    sem = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)

    failed = 0
    done = 0

    async def _process_with_progress(chunk):
        nonlocal done, failed
        result = await _process_one(client, chunk, sem)
        done += 1
        if result is None:
            failed += 1
        status = "OK" if result else "FAIL"
        print(
            f"[STEP3] [{done}/{total}] {chunk['chunk_id']} {status} "
            f"({failed} failed so far)"
        )
        return result

    tasks = [_process_with_progress(c) for c in chunks]
    results = await asyncio.gather(*tasks)

    enriched = [r for r in results if r is not None]

    with open(config.ENRICHED_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    failed_count = total - len(enriched)
    print(f"[STEP3] Done. {len(enriched)}/{total} succeeded, {failed_count} failed")
    if failed_count > 0:
        print(f"[STEP3] Failed chunks logged to {config.ERRORS_LOG}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
