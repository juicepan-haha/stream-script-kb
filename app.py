#!/usr/bin/env python3
"""Streamlit 直播话术语义检索界面。

用法: streamlit run app.py
"""
# ⚠️ 必须在导入 sentence_transformers 之前设置，否则会卡在 HF 网络请求
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

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


# --- 筛选选项（JSONB 原生查询，不再用 LIKE）---
@st.cache_data(ttl=300)
def get_filter_options():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT source_file FROM scripts ORDER BY source_file")
    sources = [r[0] for r in cur.fetchall()]

    cur.execute(
        "SELECT DISTINCT sales_stage FROM scripts WHERE sales_stage != '' "
        "ORDER BY sales_stage"
    )
    stages = [r[0] for r in cur.fetchall()]

    # JSONB 原生展开，数据库层面去重
    cur.execute(
        "SELECT DISTINCT elem FROM scripts, "
        "jsonb_array_elements_text(strategy_types) AS elem "
        "ORDER BY elem"
    )
    strategies = [r[0] for r in cur.fetchall()]

    cur.execute(
        "SELECT DISTINCT elem FROM scripts, "
        "jsonb_array_elements_text(product_mentions) AS elem "
        "ORDER BY elem"
    )
    products = [r[0] for r in cur.fetchall()]

    cur.close()
    return sources, stages, strategies, products


# --- 渲染筛选栏 ---
try:
    sources, stages, strategies, products = get_filter_options()
except Exception:
    st.warning("⚠️ 无法连接数据库。请确认 PostgreSQL 已启动且 step4_vectorize.py 已执行。")
    st.stop()

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


# --- 查询构建（JSONB @> 精准匹配）---
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
        conditions.append("strategy_types @> %s::jsonb")
        params.append(json.dumps([selected_strategy]))
    if selected_product != "(全部)":
        conditions.append("product_mentions @> %s::jsonb")
        params.append(json.dumps([selected_product]))
    return (" AND ".join(conditions) if conditions else "TRUE", params)


def semantic_search(embedding_str, where_clause, params, top_k):
    conn = get_db_connection()
    cur = conn.cursor()
    query = f"""
        SELECT chunk_id, source_file, sales_stage, strategy_types,
               product_mentions, selling_points, target_audience,
               icebreaker, painpoint, mechanism, close_order,
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


# --- 话术分段渲染 ---
def _display_script_section(label: str, text: str, emoji: str):
    """渲染单个话术分段，有内容才显示。"""
    if not text or not text.strip():
        return
    with st.expander(f"{emoji} {label}", expanded=True):
        st.markdown(text.replace("\n", "<br>"), unsafe_allow_html=True)


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
            (chunk_id, source_file, stage, strategy_types, product_mentions,
             selling_points, audience,
             icebreaker, painpoint, mechanism, close_order,
             refined, summary, distance) = row

            similarity = max(0.0, 1.0 - float(distance)) if distance is not None else 0.0

            # JSONB 列 psycopg2 直接返回 Python list
            strategies_list = strategy_types if isinstance(strategy_types, list) else []
            products_list = product_mentions if isinstance(product_mentions, list) else []
            selling_list = selling_points if isinstance(selling_points, list) else []

            with st.container():
                st.markdown("---")

                # 标签行
                tags = [stage] if stage else []
                tags.extend(strategies_list[:3])
                tag_html = " ".join(
                    f'<span style="background:#e8f0fe;color:#1a73e8;padding:2px 8px;'
                    f'border-radius:4px;font-size:12px;margin-right:4px;">{t}</span>'
                    for t in tags if t
                )
                st.markdown(tag_html, unsafe_allow_html=True)

                # 元信息
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.caption(f"📎 来源: {source_file}")
                with mc2:
                    prod_display = ", ".join(products_list) if products_list else "—"
                    st.caption(f"🏷️ 品类: {prod_display}")
                with mc3:
                    st.caption(f"🎯 相似度: {similarity:.3f}")
                with mc4:
                    audience_display = audience if audience else "—"
                    st.caption(f"👥 {audience_display}")

                if summary:
                    st.caption(f"💡 {summary}")
                if selling_list:
                    st.caption(f"✨ 卖点: {' · '.join(selling_list[:5])}")

                # 黄金四段式话术展开显示
                _display_script_section("开场破冰", icebreaker, "🎤")
                _display_script_section("痛点植入", painpoint, "🎯")
                _display_script_section("产品卖点", mechanism, "💎")
                _display_script_section("逼单催单", close_order, "🔥")
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
