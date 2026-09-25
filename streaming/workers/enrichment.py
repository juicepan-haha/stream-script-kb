"""DeepSeek 话术富化 Worker。"""

import asyncio
import json
import re

from shared import config
from streaming.state import enriched_queue, enriched_results, task_progress


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


def extract_critical_keywords(text: str) -> list[str]:
    """提取富化时必须原样保留的数字、价格和规格信息。"""
    keywords = set()
    patterns = [
        r"\d+[\.\d]*\s*(?:元|块|钱|折|%|％|ml|毫升|g|克|kg|斤|片|盒|瓶|支|包|袋|件|条|双|套|色|号|码|寸|英寸|分钟|小时|天|月|年|代|版|次)",
        r"(?:买|送|赠|减|省|便宜|优惠|只要|仅需|原价|现价|到手|券后)\s*\d+[\.\d]*",
        r"\d+[\.\d]*\s*(?:万|千|百|十|亿)?\s*(?:粉丝|销量|回购|好评|单)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            keyword = match.group().strip()
            if len(keyword) >= 2:
                keywords.add(keyword)
    numeric = sorted(keyword for keyword in keywords if re.search(r"\d", keyword))
    return numeric[:15]


async def deepseek_enrich_worker():
    """消费话术块，调用 DeepSeek 生成四段式提词稿。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    semaphore = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)

    print("[Worker 4][DeepSeek] 就绪，等待话术 chunk...")

    while True:
        item = await enriched_queue.get()
        try:
            task_id = item["task_id"]

            if item.get("done"):
                task_progress.setdefault(task_id, {})["stage"] = "completed"
                print(
                    f"[Worker 4][DeepSeek] [{task_id}] ✅ 全部富化完成 "
                    f"({len(enriched_results.get(task_id, []))} 条)"
                )
                continue

            chunk = item["chunk"]
            text = chunk["text"]
            critical_keywords = extract_critical_keywords(text)

            async with semaphore:
                for attempt in range(1, config.DEEPSEEK_RETRIES + 1):
                    try:
                        response = await client.chat.completions.create(
                            model=config.DEEPSEEK_MODEL,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {
                                    "role": "user",
                                    "content": (
                                        "【核心数据保卫战：以下关键词必须保留 — "
                                        f"{'、'.join(critical_keywords) if critical_keywords else '无'}】\n\n"
                                        f"{text}"
                                    ),
                                },
                            ],
                            temperature=config.DEEPSEEK_TEMPERATURE,
                            max_tokens=config.DEEPSEEK_MAX_TOKENS,
                            response_format={"type": "json_object"},
                        )
                        content = response.choices[0].message.content.strip()
                        parsed = json.loads(content)
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
                            await asyncio.sleep(attempt)

            full_parts = [
                parsed[key]
                for key in ["icebreaker", "painpoint", "mechanism", "close_order"]
                if parsed.get(key)
            ]
            enriched = {
                "chunk_id": chunk["chunk_id"],
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"],
                "icebreaker": parsed.get("icebreaker", ""),
                "painpoint": parsed.get("painpoint", ""),
                "mechanism": parsed.get("mechanism", ""),
                "close_order": parsed.get("close_order", ""),
                "refined_script": "\n\n".join(full_parts) if full_parts else text,
                "selling_points": critical_keywords[:10],
            }

            enriched_results.setdefault(task_id, []).append(enriched)
            result_index = len(enriched_results[task_id])
            print(
                f"[Worker 4][DeepSeek] [{task_id}] {chunk['chunk_id']} OK "
                f"(#{result_index}, {chunk['char_count']} 字)"
            )
        except Exception as exc:
            print(
                f"[Worker 4][DeepSeek] [{item.get('task_id', '?')}] "
                f"❌ 错误: {exc}"
            )
        finally:
            enriched_queue.task_done()
