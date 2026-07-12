#!/usr/bin/env python3
"""Streamlit 直播话术语义检索界面。

用法: streamlit run app.py
"""
import json

import psycopg2
import streamlit as st
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

import config

# --- 页面配置 ---
st.set_page_config(
    page_title=config.STREAMLIT_TITLE,
    page_icon="🎙️",
    layout="wide",
)
st.title(f"🎙️ {config.STREAMLIT_TITLE}")


# --- 缓存资源 ---
@st.cache_resource
def load_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource
def get_db_connection():
    conn = psycopg2.connect(
        host=config.PG_HOST,
        port=config.PG_PORT,
        user=config.PG_USER,
        password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    )
    conn.autocommit = True
    register_vector(conn)
    return conn


# --- 筛选选项 (从 DB 动态获取, 缓存 5 分钟) ---
@st.cache_data(ttl=300)
def get_filter_options():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT source_file FROM scripts ORDER BY source_file")
    sources = [r[0] for r in cur.fetchall()]

    cur.execute(
        "SELECT DISTINCT sales_stage FROM scripts WHERE sales_stage != '' ORDER BY sales_stage"
    )
    stages = [r[0] for r in cur.fetchall()]

    # strategy_types 是 JSON 数组, 需要展开去重
    cur.execute("SELECT DISTINCT strategy_types FROM scripts WHERE strategy_types != '[]'")
    all_strategies = set()
    for (val,) in cur.fetchall():
        try:
            arr = json.loads(val)
            all_strategies.update(arr)
        except json.JSONDecodeError:
            pass
    strategies = sorted(all_strategies)

    # product_mentions 同样
    cur.execute("SELECT DISTINCT product_mentions FROM scripts WHERE product_mentions != '[]'")
    all_products = set()
    for (val,) in cur.fetchall():
        try:
            arr = json.loads(val)
            all_products.update(arr)
        except json.JSONDecodeError:
            pass
    products = sorted(all_products)

    cur.close()
    return sources, stages, strategies, products


# --- 渲染 ---
try:
    sources, stages, strategies, products = get_filter_options()
except Exception:
    st.warning("⚠️ 无法连接数据库。请确认 PostgreSQL 已启动且 step4_vectorize.py 已执行。")
    st.stop()

# 顶部筛选栏
st.subheader("🔍 筛选条件")
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_source = st.selectbox("主播/来源", ["(全部)"] + sources)
with col2:
    selected_stage = st.selectbox("销售阶段", ["(全部)"] + stages)
with col3:
    selected_strategy = st.selectbox("话术策略", ["(全部)"] + strategies)
with col4:
    selected_product = st.selectbox("品类", ["(全部)"] + products)

# 搜索框
st.subheader("💬 语义搜索")
query_text = st.text_input(
    "输入想要查找的话术描述...",
    placeholder="例如：适合敏感肌的洗面奶推荐话术、逼单催付话术",
)
top_k = st.slider("返回条数", min_value=5, max_value=100, value=20, step=5)


# --- 查询构建 ---
def build_where(selected_source, selected_stage, selected_strategy, selected_product):
    conditions = []
    params = []
    if selected_source != "(全部)":
        conditions.append("source_file = %s")
        params.append(selected_source)
    if selected_stage != "(全部)":
        conditions.append("sales_stage = %s")
        params.append(selected_stage)
    if selected_strategy != "(全部)":
        conditions.append("strategy_types LIKE %s")
        params.append(f"%{selected_strategy}%")
    if selected_product != "(全部)":
        conditions.append("product_mentions LIKE %s")
        params.append(f"%{selected_product}%")
    return (" AND ".join(conditions) if conditions else "TRUE", params)


def semantic_search(embedding_str, where_clause, params, top_k):
    conn = get_db_connection()
    cur = conn.cursor()
    query = f"""
        SELECT chunk_id, source_file, sales_stage, strategy_types,
               product_mentions, selling_points, target_audience,
               refined_script, summary,
               embedding <=> %s::vector AS distance
        FROM scripts
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    all_params = [embedding_str] + params + [embedding_str, top_k]
    cur.execute(query, all_params)
    rows = cur.fetchall()
    cur.close()
    return rows


# --- 搜索按钮 ---
search_clicked = st.button("🔎 搜索", type="primary", use_container_width=True)

if search_clicked or query_text:
    model = load_model()
    query_embedding = model.encode(
        query_text if query_text else "直播话术",
        normalize_embeddings=True,
    ).tolist()
    embedding_str = json.dumps(query_embedding)

    where_clause, params = build_where(
        selected_source, selected_stage, selected_strategy, selected_product
    )
    results = semantic_search(embedding_str, where_clause, params, top_k)

    st.subheader(f"📋 搜索结果 ({len(results)} 条)")

    if not results:
        st.info("无匹配结果，请调整筛选条件或搜索词。")
    else:
        for row in results:
            (chunk_id, source_file, stage, strategy_json, product_json,
             selling_json, audience, refined, summary, distance) = row

            similarity = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0

            try:
                strategies_list = (
                    json.loads(strategy_json)
                    if isinstance(strategy_json, str)
                    else strategy_json
                )
            except json.JSONDecodeError:
                strategies_list = []
            try:
                products_list = (
                    json.loads(product_json)
                    if isinstance(product_json, str)
                    else product_json
                )
            except json.JSONDecodeError:
                products_list = []
            try:
                selling_list = (
                    json.loads(selling_json)
                    if isinstance(selling_json, str)
                    else selling_json
                )
            except json.JSONDecodeError:
                selling_list = []

            with st.container():
                st.markdown("---")
                # 标签行
                tags = [stage] if stage else []
                tags.extend(strategies_list[:3] if isinstance(strategies_list, list) else [])
                tag_html = " ".join(
                    f'<span style="background:#e8f0fe;color:#1a73e8;padding:2px 8px;'
                    f'border-radius:4px;font-size:12px;margin-right:4px;">{t}</span>'
                    for t in tags if t
                )
                st.markdown(tag_html, unsafe_allow_html=True)

                # 主体文本
                st.markdown("**📝 话术文本**")
                st.text(refined if refined else "(无文本)")

                # 元信息
                mc1, mc2, mc3 = st.columns(3)
                with mc1:
                    st.caption(f"📎 来源: {source_file}")
                with mc2:
                    prod_display = ", ".join(products_list) if products_list else "—"
                    st.caption(f"🏷️ 品类: {prod_display}")
                with mc3:
                    st.caption(f"🎯 相似度: {similarity:.3f}")

                if summary:
                    st.caption(f"💡 {summary}")
                if selling_list:
                    st.caption(f"✨ 卖点: {' · '.join(selling_list[:5])}")
                if audience:
                    st.caption(f"👥 目标人群: {audience}")
else:
    # 初始状态: 显示统计
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM scripts")
        count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT source_file) FROM scripts")
        src_count = cur.fetchone()[0]
        cur.close()
        st.info(
            f"📊 数据库中共 **{count}** 条话术记录, 来自 **{src_count}** 个来源。\n\n"
            "输入搜索词或点击搜索开始检索。"
        )
    except Exception:
        st.info("📊 数据库暂无数据。请先运行 step4_vectorize.py 入库。")
