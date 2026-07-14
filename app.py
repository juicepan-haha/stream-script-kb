#!/usr/bin/env python3
"""
app.py — Ultimate Cloud Edition 双页面路由系统

Page A (📥 数据采集入库): URL → 扣10点 → 后台管道 → 任务状态表
Page B (🔍 直播话术知识库): 筛选 → 语义搜索 → 1点重写
"""
import json
import time
import datetime as dt
import requests
import streamlit as st

# =========================================================================
# Supabase 客户端
# =========================================================================
@st.cache_resource(ttl=60)
def _get_clients():
    import os
    from dotenv import load_dotenv
    from supabase import create_client
    load_dotenv()
    url = os.getenv("SUPABASE_URL", "https://wpnaupyqqiiwjbmcucio.supabase.co")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")
    auth_sb = create_client(url, anon_key)
    from supabase_client import supabase as data_sb
    return auth_sb, data_sb

auth_supabase, db_supabase = _get_clients()

# =========================================================================
# 全局 Session State
# =========================================================================
_defaults = {
    "logged_in": False, "user_uuid": None, "user_email": None,
    "api_key": "", "card_code": "",
    "balance": 0, "is_vip": False, "vip_until": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================================
# 共享侧边栏
# =========================================================================
def _load_profile():
    uid = st.session_state.get("user_uuid")
    if not uid:
        return
    try:
        prof = db_supabase.table("user_profiles").select(
            "balance_count, vip_until"
        ).eq("id", uid).single().execute().data
        if prof:
            st.session_state.balance = prof.get("balance_count", 0)
            vu = prof.get("vip_until")
            if vu:
                vu_dt = dt.datetime.fromisoformat(str(vu).replace("Z", "+00:00"))
                st.session_state.is_vip = vu_dt > dt.datetime.now(dt.timezone.utc)
                st.session_state.vip_until = vu_dt
    except Exception:
        pass


def render_sidebar():
    _load_profile()
    with st.sidebar:
        st.header("⚙️ 个人中心")
        st.write(f"👤 `{st.session_state.get('user_email', '未登录')}`")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("📊 点数", st.session_state.balance)
        with c2:
            st.metric("👑", "VIP" if st.session_state.is_vip else "普通")

        st.caption("提炼10点 | 重写1点 | VIP免费")

        st.divider()

        st.session_state.api_key = st.text_input(
            "🔑 DeepSeek API Key",
            value=st.session_state.api_key, type="password",
            placeholder="sk-... 阅后即焚",
        ).strip()

        st.divider()

        exchange_code = st.text_input("🎫 兑换卡密", placeholder="输入卡密", key="ex_code")
        if st.button("立即兑换", use_container_width=True):
            if not exchange_code:
                st.warning("请输入卡密")
            else:
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/recharge",
                        params={"user_id": st.session_state.user_uuid, "card_code": exchange_code},
                        timeout=10,
                    )
                    d = resp.json()
                    if d.get("status") == "success":
                        st.success(d.get("message", "兑换成功"))
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(d.get("message", "兑换失败"))
                except requests.ConnectionError:
                    st.error("后端未启动")

        st.divider()

        if st.button("退出登录", use_container_width=True):
            auth_supabase.auth.sign_out()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# =========================================================================
# 工具函数
# =========================================================================
def _verify_key(key: str) -> tuple:
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=10,
        )
        if r.status_code == 200: return True, ""
        if r.status_code == 401: return False, "Key 无效"
        if r.status_code == 429: return False, "余额不足/限频"
        if r.status_code >= 500: return False, "DeepSeek 繁忙"
        return False, f"异常 (HTTP {r.status_code})"
    except requests.Timeout: return False, "网络超时"
    except Exception: return False, "网络错误"


# =========================================================================
# 页面 A：数据采集入库
# =========================================================================
def page_ingestion():
    render_sidebar()
    st.title("📥 视频源数据入库舱")
    st.caption("发送拉流任务 → 扣 10 点 → 后台 4 级提炼流水线")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        video_url = st.text_input("直播间 URL (m3u8)", placeholder="https://...merge.m3u8")
    with col2:
        product_name = st.text_input("🎯 指定核心产品（选填）", placeholder="如：多功能不粘锅")

    fire = st.button("🔥 发射！启动后台 4 级提炼流水线（-10点）", use_container_width=True)

    if fire:
        ak = st.session_state.api_key
        uid = st.session_state.user_uuid
        bal = st.session_state.balance
        is_vip = st.session_state.is_vip

        if not ak or not ak.startswith("sk-"):
            st.error("❌ 请先在侧边栏配置 DeepSeek API Key！")
        elif not video_url:
            st.warning("⚠️ 请输入直播间 URL！")
        elif not is_vip and bal < 10:
            st.error(f"❌ 点数不足！需要 10 点，当前 {bal} 点。")
        else:
            with st.spinner("🔄 校验 Key..."):
                ok, err = _verify_key(ak)
                if not ok:
                    st.error(f"❌ {err}")
                else:
                    # 插入任务 + 调后端
                    try:
                        resp = db_supabase.table("fish_box").insert({
                            "user_id": uid,
                            "video_url": video_url,
                            "result_text": "排队中...",
                        }).execute()
                        tid = resp.data[0]["id"]
                    except Exception as e:
                        st.error(f"❌ {str(e)[:200]}")
                        st.stop()

                    try:
                        requests.post(
                            "http://127.0.0.1:8000/api/v1/analyze",
                            params={
                                "task_id": str(tid), "video_url": video_url,
                                "user_id": uid, "user_deepseek_key": ak,
                            }, timeout=10,
                        )
                        st.success(f"🚀 任务已提交！Task ID: {tid}")
                        st.info("可立即前往「话术知识库」页面工作，或提交下一个链接。")
                    except requests.ConnectionError:
                        st.error("❌ 后端未启动！请运行 server_v2.py")

    # ---- 任务队列看板 ----
    st.markdown("---")
    st.subheader("📋 近期任务队列")
    try:
        tasks = db_supabase.table("fish_box").select(
            "id, video_url, status, result_text, created_at"
        ).eq("user_id", st.session_state.user_uuid).order(
            "created_at", desc=True
        ).limit(10).execute().data
        if tasks:
            rows = []
            for t in tasks:
                badge = {"queued": "🔵", "processing": "🟡", "success": "🟢", "failed": "🔴"}
                rows.append({
                    "ID": t["id"],
                    "状态": f"{badge.get(t['status'], '⚪')} {t['status']}",
                    "URL": str(t.get("video_url", ""))[:50],
                    "时间": str(t.get("created_at", ""))[:19],
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("暂无任务记录")
    except Exception as e:
        st.caption(f"加载失败: {e}")


# =========================================================================
# 页面 B：话术知识库
# =========================================================================
def page_knowledge_base():
    render_sidebar()
    st.title("🔍 直播话术知识库")
    st.caption("筛选 → 语义搜索 → 1 点一键重写")

    st.markdown("---")

    # 1. 筛选条件
    try:
        # 已成功的 source_file
        src_cur = db_supabase.select("*").execute  # wrong API — use supabase-py correctly
        # Actually use pgvector queries via the existing FastAPI or direct
        # For now, simple dropdowns from Supabase
        pass
    except Exception:
        pass

    try:
        src_data = db_supabase.table("scripts").select("source_file").execute().data
        sources = sorted(set(r["source_file"] for r in src_data if r.get("source_file")))
    except Exception:
        sources = []

    st.subheader("🔍 筛选条件")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_source = st.selectbox("主播/来源", ["(全部)"] + sources)
    with c2:
        sel_stage = st.selectbox("销售阶段", ["(全部)", "开场暖场", "产品引入", "价值塑造", "逼单促单", "互动留人"])
    with c3:
        sel_strategy = st.selectbox("话术策略", ["(全部)", "限时稀缺", "价格锚定", "信任背书", "痛点放大", "场景代入", "福利诱导"])
    with c4:
        sel_product = st.selectbox("品类", ["(全部)", "美妆个护", "厨房电器", "家居百货"])

    # 2. 语义搜索
    st.subheader("💬 语义检索")
    qcol1, qcol2 = st.columns([3, 1])
    with qcol1:
        query = st.text_input("输入想要查找的话术描述...", placeholder="适合敏感肌的洗面奶推荐话术")
    with qcol2:
        top_k = st.slider("返回条数", 5, 50, 20, 5)

    search_clicked = st.button("🔎 语义搜索", use_container_width=True, type="primary")

    st.markdown("---")

    # 3. 搜索执行 + 结果 + 重写
    search_results = st.session_state.get("_search_results", None)

    if search_clicked and query:
        ak = st.session_state.api_key
        if not ak:
            st.error("❌ 请先在侧边栏配置 API Key！")
        else:
            with st.spinner("🔍 语义检索中..."):
                # 调 FastAPI pgvector 搜索
                # 简化：用 POST 到后端搜索端点
                st.session_state._search_results = []
                st.info("搜索功能需后端 pgvector 端点支持，建设中...")

    # 4. 重写区
    st.subheader("🪄 一键重写（-1 点）")
    rewrite_style = st.selectbox("话术风格", ["呐喊憋单流", "温柔种草流", "硬核测评流", "剧情代入流"])
    rewrite_clicked = st.button("🪄 重写选中话术（-1点）", use_container_width=True)

    if rewrite_clicked:
        ak = st.session_state.api_key
        if not ak:
            st.error("❌ 请先配置 API Key！")
        elif not st.session_state.is_vip and st.session_state.balance < 1:
            st.error(f"❌ 点数不足！当前 {st.session_state.balance} 点。")
        else:
            st.info("重写功能建设中...（对接 /api/v1/rewrite-material）")


# =========================================================================
# 登录页面
# =========================================================================
def page_login():
    st.set_page_config(page_title="登录", page_icon="🔐", layout="centered")
    st.title("🔐 Ultimate Cloud Edition")

    tab1, tab2 = st.tabs(["登录", "注册"])
    with tab1:
        e = st.text_input("邮箱", key="li_e")
        p = st.text_input("密码", type="password", key="li_p")
        if st.button("登录", use_container_width=True):
            try:
                r = auth_supabase.auth.sign_in_with_password({"email": e, "password": p})
                st.session_state.logged_in = True
                st.session_state.user_uuid = r.user.id
                st.session_state.user_email = r.user.email
                st.rerun()
            except Exception as ex:
                st.error(f"❌ {str(ex)[:200]}")
    with tab2:
        e2 = st.text_input("邮箱", key="re_e")
        p2 = st.text_input("密码(≥6位)", type="password", key="re_p")
        if st.button("注册", use_container_width=True):
            try:
                auth_supabase.auth.sign_up({"email": e2, "password": p2})
                st.success("注册成功！请切换到登录。")
            except Exception as ex:
                st.error(f"❌ {str(ex)[:200]}")


# =========================================================================
# 路由
# =========================================================================
if not st.session_state.get("logged_in"):
    page_login()
else:
    pg1 = st.Page(page_ingestion, title="📥 数据采集入库", icon="📥")
    pg2 = st.Page(page_knowledge_base, title="🔍 直播话术知识库", icon="🔍")
    pg = st.navigation([pg1, pg2])
    pg.run()
