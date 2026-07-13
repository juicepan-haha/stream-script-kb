#!/usr/bin/env python3
"""Step 3: 使用 DeepSeek API 将口语流水账改造为提词器级逐字稿。

输入: data/chunks.json
输出: data/enriched.json + data/errors.log
并发: asyncio + 异步 OpenAI client, 并发数 config.DEEPSEEK_CONCURRENCY
"""
import asyncio
import json
import re
import sys
from datetime import datetime

import aiofiles
from openai import AsyncOpenAI

import config

# --- 动态提取校验关键词 ---


def _extract_critical_keywords(chunk_text: str, selling_points: list[str]) -> list[str]:
    """从原始文本和卖点中自动提取核心商业数据（价格、规格、数字、专有名词）。

    策略:
      1. 匹配所有含数字的词组（179元、30ml、5分钟、第3代 等）
      2. 匹配卖点中的核心规格词（正装、试用装、色卡、下颌线 等）
      3. 去重排序，限制数量避免 prompt 过长
    """
    keywords = set()

    # 1. 从原始文本提取含数字的词组
    #    匹配: 数字+常见单位/后缀 构成的短词组
    patterns = [
        r"\d+[\.\d]*\s*(?:元|块|钱|折|%|％|ml|毫升|g|克|kg|斤|片|盒|瓶|支|包|袋|件|条|双|套|色|号|码|寸|英寸|分钟|小时|天|月|年|代|版|次)",
        r"(?:买|送|赠|减|省|便宜|优惠|只要|仅需|原价|现价|到手|券后)\s*\d+[\.\d]*",
        r"\d+[\.\d]*\s*(?:万|千|百|十|亿)?\s*(?:粉丝|销量|回购|好评|单)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, chunk_text):
            kw = m.group().strip()
            if len(kw) >= 2:
                keywords.add(kw)

    # 2. 从 selling_points 提取数字部分
    for sp in selling_points:
        nums = re.findall(r"\d+[\.\d]*\s*[a-zA-Z]*[^\w\s]*", sp)
        for n in nums:
            clean = n.strip()
            if len(clean) >= 1:
                keywords.add(clean)

    # 3. 从原始文本提取专有名词（品牌、成分、色号等）
    #    中文 + 数字/字母 的组合
    compound_terms = re.findall(
        r"[一-鿿]+(?:[#＃]\d+)?(?:[a-zA-Z]+[\d\.]+)?",
        chunk_text,
    )
    # 滤过短的、过长的、纯数字的
    for term in compound_terms:
        term = term.strip()
        if 3 <= len(term) <= 12 and not term.isdigit():
            keywords.add(term)

    # 4. 加入 selling_points 中提取的核心品类词
    for sp in selling_points:
        words = re.findall(r"[一-鿿]{2,8}", sp)
        for w in words:
            if len(w) >= 3:  # 至少 3 个汉字才有意义
                keywords.add(w)

    # 限制数量: 最多 20 个，优先保留含数字的
    numeric_kw = [k for k in keywords if re.search(r"\d", k)]
    text_kw = [k for k in keywords if not re.search(r"\d", k)]

    final = sorted(set(numeric_kw))[:15] + sorted(text_kw)[:10]
    return final[:20]


# --- Prompt ---

SYSTEM_PROMPT = """你是精通带货心理学和直播控场的话术提词专家。
你的任务是把口语流水账文本改造成主播可以直接上场朗读的"提词器级逐字稿"。

【硬性改造规则】：
1. 彻底过滤：删掉所有语病、重复、废话、口头禅（如"好不好"、"是不是"、"然后最后"、"那个那个"）
2. 逻辑分段：将混乱的叙述按照直播销售逻辑进行切片
3. 语气标注：在关键动作和语气转换处，使用中括号 [] 标注主播的情绪和动作提示
4. 保留干货：所有涉及价格、赠品、规格、色号、使用方法的具体数字和专有名词，绝对不准改

【输出 JSON 字段说明】：
- icebreaker: 破冰留人话术（吸引停留的开场，高亮逐字稿，带情绪标注）
- painpoint: 痛点植入话术（用户痛点+产品解决方案，带情绪标注）
- mechanism: 产品卖点话术（核心机制、价格、优惠、规格演示，必须锁定所有数字数据）
- close_order: 逼单催单话术（制造紧迫感、促进下单，带情绪标注）

你必须输出合法的 json 对象（不要 Markdown 包裹）：
{
  "icebreaker": "...",
  "painpoint": "...",
  "mechanism": "...",
  "close_order": "..."
}

如果文本内容不包含明确话术（纯闲聊、无意义重复），所有字段填空字符串。"""


def _build_user_prompt(chunk_text: str, critical_keywords: list[str]) -> str:
    kw_block = "、".join(critical_keywords) if critical_keywords else "（无特殊关键词）"
    return (
        f"【🔥绝对死命令 - 核心数据保卫战】\n"
        f"以下关键词必须原封不动保留在输出话术中，不准改数字、不准删除、不准捏造新数字：\n"
        f"{kw_block}\n\n"
        f"【输入文本】：\n{chunk_text}"
    )


# --- 备用正则提取器 ---

_FIELD_PATTERNS = {
    "icebreaker": re.compile(
        r'"[iI]cebreaker"\s*:\s*"((?:[^"\\]|\\.)*)"'
    ),
    "painpoint": re.compile(
        r'"[pP]ainpoint"\s*:\s*"((?:[^"\\]|\\.)*)"'
    ),
    "mechanism": re.compile(
        r'"[mM]echanism"\s*:\s*"((?:[^"\\]|\\.)*)"'
    ),
    "close_order": re.compile(
        r'"[cC]lose_order"\s*:\s*"((?:[^"\\]|\\.)*)"'
    ),
}


def _regex_fallback_parser(raw_content: str, original_text: str) -> dict:
    """JSON 解析失败时的备用正则提取器。

    逐字段用正则匹配 JSON key-value，处理转义字符。
    如果连正则都救不回来，返回原始文本作为 refined_script 的兜底。
    """
    result = {}
    for field, pattern in _FIELD_PATTERNS.items():
        m = pattern.search(raw_content)
        if m:
            # 还原转义字符
            val = m.group(1).replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
            result[field] = val
        else:
            result[field] = ""

    # 标记为 regex 救回来的
    result["_regex_recovered"] = True

    # 如果四个字段全空，用原始文本兜底
    if not any(result.get(k) for k in _FIELD_PATTERNS):
        result["icebreaker"] = original_text
        result["_regex_recovered"] = True
        result["_fallback_reason"] = "all_fields_empty"

    return result


# --- 后置校验 ---

def _validate_output(parsed: dict, critical_keywords: list[str]) -> dict:
    """代码级强校验：检查核心关键词是否被 AI 弄丢，丢失则注入强制修正标记。"""
    if not critical_keywords:
        return parsed

    full_text = " ".join(v for v in parsed.values() if isinstance(v, str))
    missing = [kw for kw in critical_keywords if kw not in full_text]

    if missing:
        parsed["validation_warnings"] = missing
        # 强制修正注入到 mechanism 字段
        parsed["mechanism"] = (
            parsed.get("mechanism", "")
            + f"\n[⚠️ 系统校验提醒：请主播确认以下数据 — {', '.join(missing[:8])}]"
        )

    return parsed


# --- 处理单个 chunk ---

async def _process_one(
    client: AsyncOpenAI,
    chunk: dict,
    sem: asyncio.Semaphore,
) -> dict | None:
    """处理单个 chunk, 成功返回 enriched dict, 失败返回 None 并记日志。"""
    chunk_id = chunk["chunk_id"]
    text = chunk["text"]

    # 动态提取校验关键词
    critical_keywords = _extract_critical_keywords(text, [])

    for attempt in range(1, config.DEEPSEEK_RETRIES + 1):
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _build_user_prompt(text, critical_keywords)},
                    ],
                    temperature=config.DEEPSEEK_TEMPERATURE,
                    max_tokens=config.DEEPSEEK_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content.strip()

                # === JSON 解析（带备用正则兜底，绝不卡死流水线）===
                parsed = None
                recovered_via_regex = False

                # 第一关：标准 JSON 解析
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as json_err:
                    # 第二关：备用正则提取器
                    parsed = _regex_fallback_parser(content, text)
                    if parsed.get("_regex_recovered"):
                        recovered_via_regex = True
                        flag = "REGEX"
                    else:
                        # 第三关：正则也救不回来，重试 API
                        raw_preview = content[:200] if "content" in dir() else str(json_err)
                        if attempt < config.DEEPSEEK_RETRIES:
                            await asyncio.sleep(1 * attempt)
                            continue
                        else:
                            await _log_error(chunk_id, text, raw_preview)
                            return None

                # 后置校验
                parsed = _validate_output(parsed, critical_keywords)

                # 拼接完整话术文本（兼容 step2/step4 的 refined_script 字段）
                full_script_parts = []
                for key in ["icebreaker", "painpoint", "mechanism", "close_order"]:
                    val = parsed.get(key, "")
                    if val:
                        full_script_parts.append(val)
                refined_script = "\n\n".join(full_script_parts) if full_script_parts else text

                # 从话术内容推断销售阶段
                sales_stage = _infer_stage(parsed)

                return {
                    "chunk_id": chunk["chunk_id"],
                    "source_file": chunk["source_file"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    # 新字段
                    "icebreaker": parsed.get("icebreaker", ""),
                    "painpoint": parsed.get("painpoint", ""),
                    "mechanism": parsed.get("mechanism", ""),
                    "close_order": parsed.get("close_order", ""),
                    "validation_warnings": parsed.get("validation_warnings", []),
                    # 兼容旧字段
                    "refined_script": refined_script,
                    "summary": _extract_summary(parsed),
                    "sales_stage": sales_stage,
                    "strategy_types": _extract_strategies(parsed),
                    "product_mentions": _extract_products(text, parsed),
                    "selling_points": critical_keywords[:10],
                    "target_audience": _infer_audience(parsed),
                    # 内部标记
                    "_regex_recovered": recovered_via_regex,
                }

            except Exception as e:
                if attempt == config.DEEPSEEK_RETRIES:
                    await _log_error(chunk_id, text, str(e))
                    return None
                await asyncio.sleep(2 * attempt)

    return None


# --- 辅助函数 ---

def _infer_stage(parsed: dict) -> str:
    """根据 4 段话术的内容推断销售阶段。"""
    if parsed.get("icebreaker"):
        return "开场暖场"
    if parsed.get("close_order"):
        return "逼单促单"
    if parsed.get("mechanism") and parsed.get("painpoint"):
        return "价值塑造"
    if parsed.get("painpoint"):
        return "痛点放大"
    return "互动留人"


def _extract_summary(parsed: dict) -> str:
    """从 4 段话术生成一句话摘要。"""
    parts = []
    for key, label in [("mechanism", ""), ("painpoint", ""), ("icebreaker", "")]:
        val = parsed.get(key, "")
        if val:
            # 取第一句作为摘要
            first_sentence = re.split(r"[。！？\n]", val)[0][:60]
            parts.append(first_sentence)
            break
    return parts[0] if parts else ""


def _extract_strategies(parsed: dict) -> list[str]:
    """从话术内容推断策略标签。"""
    strategies = []
    full_text = " ".join(v for v in parsed.values() if isinstance(v, str))

    strategy_map = {
        "限时稀缺": ["限时", "限量", "最后", "抢光", "库存", "错过", "马上", "赶紧", "倒计时", "秒杀"],
        "价格锚定": ["原价", "平时", "专柜", "官方价", "便宜", "只要", "划算", "性价比"],
        "信任背书": ["认证", "检测", "报告", "专利", "专家", "医生", "明星", "同款"],
        "从众效应": ["都在买", "好评", "回购", "粉丝", "销量", "万人", "爆款"],
        "痛点放大": ["烦恼", "困扰", "难受", "尴尬", "担心", "害怕", "痛点"],
        "场景代入": ["想象", "如果", "每天", "早上", "晚上", "出门", "上班", "约会"],
        "算账对比": ["省了", "等于", "相当于", "平均", "每天只要", "折合"],
        "亲身试用": ["我自己", "我用", "亲测", "实测", "你看", "展示"],
        "福利诱导": ["送", "赠", "福利", "白送", "免费", "加赠", "额外"],
        "互动引导": ["扣", "弹幕", "评论", "点赞", "关注", "告诉"],
        "情感共鸣": ["懂你", "理解", "姐妹", "我们", "一起", "陪伴"],
    }

    for strategy, keywords in strategy_map.items():
        if any(kw in full_text for kw in keywords):
            strategies.append(strategy)

    return strategies[:5]


def _extract_products(text: str, parsed: dict) -> list[str]:
    """提取商品/品类名。"""
    full_text = " ".join(v for v in parsed.values() if isinstance(v, str))
    combined = text + " " + full_text

    # 常见品类/商品模式
    patterns = [
        r"(?:这款|这个|我们的|今天)\s*([一-鿿]{2,8}(?:霜|乳|液|水|粉|油|笔|膜|膏|露|胶|精华|喷雾|面膜|洗面奶|口红|唇釉|眼影|腮红|粉底|遮瑕|散粉|气垫|隔离|防晒|卸妆|眉笔|眼线|睫毛|香水|洗发|沐浴|身体乳|护手霜|面霜|爽肤水|乳液|精华液))",
        r"([一-鿿]{2,6}(?:系列|套装|组合|礼盒))",
    ]
    products = set()
    for pat in patterns:
        for m in re.finditer(pat, combined):
            products.add(m.group(1))

    return sorted(products)[:10]


def _infer_audience(parsed: dict) -> str:
    """推断目标人群。"""
    full_text = " ".join(v for v in parsed.values() if isinstance(v, str))

    audience_map = {
        "宝妈": ["宝宝", "孩子", "孕期", "产后", "哺乳", "带娃"],
        "学生党": ["学生", "宿舍", "平价", "生活费", "开学", "考试"],
        "上班族": ["上班", "通勤", "办公室", "加班", "同事", "会议"],
        "中老年": ["年纪", "老了", "皱纹", "松弛", "保养"],
        "敏感肌": ["敏感", "泛红", "刺痛", "不耐受", "温和"],
    }

    for audience, keywords in audience_map.items():
        if any(kw in full_text for kw in keywords):
            return audience
    return ""


# --- 日志 ---

async def _log_error(chunk_id: str, original_text: str, error_info: str) -> None:
    """异步写入 errors.log，不阻塞事件循环。"""
    config.ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(config.ERRORS_LOG, "a", encoding="utf-8") as f:
        await f.write(f"{'=' * 60}\n")
        await f.write(f"Time: {datetime.now().isoformat()}\n")
        await f.write(f"Chunk ID: {chunk_id}\n")
        await f.write(f"Error: {error_info}\n")
        await f.write(f"Text ({len(original_text)} chars):\n{original_text}\n")
        await f.write(f"{'=' * 60}\n\n")


# --- 主流程 ---

async def main_async():
    if not config.DEEPSEEK_API_KEY:
        print("[STEP3] ERROR: DEEPSEEK_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 个 chunk (0=全部)")
    args = parser.parse_args()

    async with aiofiles.open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.loads(await f.read())

    if args.limit > 0 and args.limit < len(chunks):
        chunks = chunks[:args.limit]

    total = len(chunks)
    print(f"[STEP3] Loaded {total} chunks from {config.CHUNKS_FILE}")

    # 清空 errors.log
    async with aiofiles.open(config.ERRORS_LOG, "w", encoding="utf-8") as f:
        await f.write("")

    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    sem = asyncio.Semaphore(config.DEEPSEEK_CONCURRENCY)

    failed = 0
    done = 0
    warned = 0
    regex_recovered = 0

    async def _process_with_progress(chunk):
        nonlocal done, failed, warned, regex_recovered
        result = await _process_one(client, chunk, sem)
        done += 1
        if result is None:
            failed += 1
            status = "FAIL"
        elif result.get("_regex_recovered"):
            regex_recovered += 1
            status = "REGEX"
        elif result.get("validation_warnings"):
            warned += 1
            status = "WARN"
        else:
            status = "OK"
        print(
            f"[STEP3] [{done}/{total}] {chunk['chunk_id']} {status} "
            f"({failed} failed, {warned} warned, {regex_recovered} regex)"
        )
        return result

    results = []
    batch_size = config.DEEPSEEK_BATCH_SIZE
    for batch_start in range(0, total, batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_tasks = [_process_with_progress(c) for c in batch]
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    enriched = [r for r in results if r is not None]

    async with aiofiles.open(config.ENRICHED_FILE, "w", encoding="utf-8") as f:
        await f.write(json.dumps(enriched, ensure_ascii=False, indent=2))

    failed_count = total - len(enriched)
    print(f"[STEP3] Done. {len(enriched)}/{total} succeeded, "
          f"{failed_count} failed, {warned} warned, {regex_recovered} regex-recovered")
    if failed_count > 0:
        print(f"[STEP3] Failed chunks logged to {config.ERRORS_LOG}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
