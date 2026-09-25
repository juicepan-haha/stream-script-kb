"""基于 pgvector 检索的直播话术改写服务。"""

import asyncio
import json
from typing import Any

from shared import config


_rag_model = None
_rag_db = None


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


def init_rag():
    """延迟初始化向量模型和数据库连接。"""
    global _rag_model, _rag_db
    if _rag_model is not None:
        return

    import psycopg2
    from pgvector.psycopg2 import register_vector
    from sentence_transformers import SentenceTransformer

    print("[RAG] 加载 Embedding 模型...")
    _rag_model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("[RAG] 连接向量数据库...")
    _rag_db = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    )
    register_vector(_rag_db)
    print("[RAG] ✅ 就绪")


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """使用 pgvector 余弦距离检索历史话术。"""
    init_rag()
    vector = _rag_model.encode(query, normalize_embeddings=True).tolist()
    vector_json = json.dumps(vector)

    cursor = _rag_db.cursor()
    try:
        cursor.execute(
            """
            SELECT icebreaker, painpoint, mechanism, close_order,
                   refined_script, sales_stage, strategy_types,
                   product_mentions, selling_points,
                   embedding <=> %s::vector AS distance
            FROM scripts
            WHERE refined_script != ''
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (vector_json, vector_json, top_k),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()

    results = []
    for row in rows:
        (
            icebreaker,
            painpoint,
            mechanism,
            close_order,
            refined_script,
            sales_stage,
            strategies,
            products,
            selling_points,
            distance,
        ) = row

        similarity = max(0.0, 1.0 - float(distance)) if distance else 0.0
        if isinstance(strategies, str):
            strategies = json.loads(strategies)
        if isinstance(products, str):
            products = json.loads(products)

        results.append({
            "similarity": round(similarity, 4),
            "icebreaker": icebreaker or "",
            "painpoint": painpoint or "",
            "mechanism": mechanism or "",
            "close_order": close_order or "",
            "refined_script": refined_script or "",
            "sales_stage": sales_stage or "",
            "strategy_types": strategies or [],
            "product_mentions": products or [],
            "selling_points": selling_points or [],
        })
    return results


def build_rag_context(retrieved: list[dict]) -> str:
    """把向量检索结果转换成 DeepSeek 的参考上下文。"""
    parts = []
    for index, item in enumerate(retrieved, 1):
        similarity_percent = item["similarity"] * 100
        parts.append(
            f"━━━ 爆款参考 #{index} (相似度 {similarity_percent:.0f}%) ━━━\n"
            f"● 销售阶段: {item['sales_stage']}\n"
            f"● 话术策略: {', '.join(item['strategy_types'][:5])}\n"
            f"● 涉及品类: {', '.join(item['product_mentions'][:5])}\n"
            f"\n[破冰留人]:\n{item['icebreaker']}\n"
            f"\n[痛点植入]:\n{item['painpoint']}\n"
            f"\n[产品卖点]:\n{item['mechanism']}\n"
            f"\n[逼单催单]:\n{item['close_order']}\n"
        )
    return "\n".join(parts)


async def rewrite_script(
    my_product: str,
    target_style: str = "呐喊憋单流",
) -> dict:
    """检索历史话术并让 DeepSeek 生成新脚本和执行 SOP。"""
    from openai import AsyncOpenAI

    print(f"[RAG] 检索请求: product={my_product}, style={target_style}")

    retrieved = vector_search(my_product, top_k=5)
    if not retrieved:
        return {"status": "no_results", "message": "数据库中没有匹配的话术参考"}

    context = build_rag_context(retrieved)
    client = AsyncOpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )
    user_prompt = (
        f"━━━ 历史爆款结构参考 ━━━\n{context}\n\n"
        "━━━ 新任务 ━━━\n"
        f"产品: {my_product}\n"
        f"风格: {target_style}\n"
        "请严格按照 SYSTEM ROLE 的所有规则，输出可直接念的话术脚本。"
    )

    raw_content = ""
    for attempt in range(config.DEEPSEEK_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": RAG_SYSTEM_PROMPT.format(
                            my_product=my_product,
                            target_style=target_style,
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            break
        except Exception as exc:
            if attempt == config.DEEPSEEK_RETRIES - 1:
                print(f"[RAG] ❌ DeepSeek 调用失败: {exc}")
                return {"status": "error", "message": f"AI 重写失败: {exc}"}
            await asyncio.sleep(attempt + 1)

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        result = {"rewritten_script": raw_content, "sop_timeline": []}

    if not result.get("sop_timeline"):
        print("[RAG] SOP 为空，触发兜底重试...")
        await _fill_missing_sop(
            client,
            result,
            my_product=my_product,
            target_style=target_style,
        )

    return {
        "status": "success",
        "my_product": my_product,
        "target_style": target_style,
        "retrieved_references": [
            {
                "similarity": item["similarity"],
                "sales_stage": item["sales_stage"],
                "strategy_types": item["strategy_types"][:3],
            }
            for item in retrieved
        ],
        "rewritten_script": result.get("rewritten_script", raw_content),
        "sop_timeline": result.get("sop_timeline", []),
    }


async def _fill_missing_sop(
    client: Any,
    result: dict,
    *,
    my_product: str,
    target_style: str,
) -> None:
    """在主响应缺少 SOP 时追加一次专项请求。"""
    sop_prompt = (
        f"产品: {my_product}\n风格: {target_style}\n"
        f"话术脚本:\n{result.get('rewritten_script', '')[:1500]}\n\n"
        "请为上述话术脚本生成一份秒级 SOP 时间轴 JSON 数组。"
    )
    for _ in range(2):
        try:
            response = await client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是直播运营SOP专家。输出一个 json 对象: "
                            '{"sop_timeline": [{"time_range":"...","stage":"...",'
                            '"host_action":"...","operation_action":"...",'
                            '"verbal_keywords":"..."}]}'
                        ),
                    },
                    {"role": "user", "content": sop_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )
            sop_data = json.loads(response.choices[0].message.content.strip())
            result["sop_timeline"] = sop_data.get("sop_timeline", [])
            if result["sop_timeline"]:
                return
        except Exception:
            await asyncio.sleep(1)
