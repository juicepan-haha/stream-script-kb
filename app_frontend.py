#!/usr/bin/env python3
"""
app_frontend.py — Ultimate Cloud Edition 前端控制台

Supabase Auth 登录/注册 + 3 秒轮询状态渲染。
"""
import json
import os
import re
import time
import requests
import streamlit as st

# =========================================================================
# Supabase 客户端（anon key 用于 Auth，service_role 用于数据读写）
# =========================================================================

@st.cache_resource(ttl=60)  # 60秒过期，防止缓存旧 key
def _get_clients():
    from supabase import create_client
    from dotenv import load_dotenv
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL", "https://wpnaupyqqiiwjbmcucio.supabase.co")
    anon_key = os.getenv("SUPABASE_ANON_KEY", "")

    auth_sb = create_client(supabase_url, anon_key)
    from supabase_client import supabase as data_sb
    return auth_sb, data_sb

auth_supabase, db_supabase = _get_clients()

# =========================================================================
# Session State 初始化
# =========================================================================
_defaults = {
    "logged_in": False, "user_uuid": None, "user_email": None,
    "api_key": "", "url_or_product": "", "card_code": "",
    "task_result": None, "task_error": None,
    "is_running": False, "running_task_id": None,
    "running_mode": None, "poll_count": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =========================================================================
# 登录/注册页面
# =========================================================================

def show_login_page():
    st.set_page_config(page_title="Ultimate Cloud Edition - 登录", page_icon="🔐", layout="centered")
    st.title("🔐 Ultimate Cloud Edition - 用户认证")

    tab1, tab2 = st.tabs(["用户登录", "新用户注册"])

    with tab1:
        st.subheader("登录到您的账户")
        login_email = st.text_input("邮箱地址", key="login_email")
        login_password = st.text_input("密码", type="password", key="login_password")

        if st.button("立即登录", use_container_width=True):
            try:
                resp = auth_supabase.auth.sign_in_with_password({
                    "email": login_email, "password": login_password,
                })
                st.session_state["logged_in"] = True
                st.session_state["user_uuid"] = resp.user.id
                st.session_state["user_email"] = resp.user.email
                st.success("🎉 登录成功！")
                st.rerun()
            except Exception as e:
                st.error(f"❌ 登录失败：{str(e)[:200]}")

    with tab2:
        st.subheader("创建新账户")
        reg_email = st.text_input("邮箱地址", key="reg_email")
        reg_password = st.text_input("密码（至少6位）", type="password", key="reg_password")

        if st.button("立即注册", use_container_width=True):
            try:
                auth_supabase.auth.sign_up({"email": reg_email, "password": reg_password})
                st.success("🚀 注册成功！请切换到登录选项卡进行登录。")
            except Exception as e:
                st.error(f"❌ 注册失败：{str(e)[:200]}")

    # 用户已登录自动跳过
    if st.session_state.get("logged_in") and st.session_state.get("user_uuid"):
        st.rerun()


# =========================================================================
# Key 盲测
# =========================================================================

def _verify_deepseek_key(key: str) -> tuple:
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=10,
        )
        if resp.status_code == 200:   return True, ""
        if resp.status_code == 401:   return False, "Key 无效，请检查是否复制完整。"
        if resp.status_code == 429:   return False, "余额不足或频次限制，请充值后重试。"
        if resp.status_code >= 500:   return False, "DeepSeek 服务器繁忙。"
        return False, f"DeepSeek 异常 (HTTP {resp.status_code})"
    except requests.Timeout:          return False, "网络超时。"
    except requests.ConnectionError:   return False, "无法连接 DeepSeek。"
    except Exception as e:            return False, f"错误: {str(e)[:100]}"[:100]


# =========================================================================
# 主业务控制台
# =========================================================================

def show_main_app():
    st.set_page_config(page_title="AI直播话术爆款平替系统", page_icon="🚀", layout="centered")

    # ==================== 🔒 侧边栏：资产与配置区 ====================
    with st.sidebar:
        st.header("⚙️ 个人中心")
        st.write(f"👤 账号：`{st.session_state.get('user_email', '未登录')}`")

        # --- 动态拉取用户资产 ---
        import datetime as _dt
        uid = st.session_state["user_uuid"]
        balance = 0
        is_vip = False
        vip_until_str = None
        try:
            prof = db_supabase.table("user_profiles").select(
                "balance_count, vip_until"
            ).eq("id", uid).single().execute().data
            if prof:
                balance = prof.get("balance_count", 0)
                vip_until_str = prof.get("vip_until")
                if vip_until_str:
                    vu = _dt.datetime.fromisoformat(
                        str(vip_until_str).replace("Z", "+00:00")
                    )
                    is_vip = vu > _dt.datetime.now(_dt.timezone.utc)
        except Exception:
            pass

        # --- 资产看板 ---
        c1, c2 = st.columns(2)
        with c1:
            st.metric("📊 剩余额度", f"{balance} 次")
        with c2:
            st.metric("👑 会员", "尊贵VIP" if is_vip else "普通用户")
        if is_vip and vip_until_str:
            try:
                vu = _dt.datetime.fromisoformat(
                    str(vip_until_str).replace("Z", "+00:00")
                )
                st.caption(f"📅 有效期至 {vu.strftime('%Y-%m-%d')}")
            except Exception:
                pass

        st.divider()

        # --- API Key ---
        st.session_state.api_key = st.text_input(
            "🔑 DeepSeek API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="sk-... 阅后即焚",
            disabled=st.session_state.is_running,
        ).strip()

        if st.session_state.api_key:
            st.caption("🟢 已就绪（暂存内存）")
        else:
            st.caption("🔴 请先配置 Key")

        st.divider()

        # --- 卡密兑换 ---
        exchange_code = st.text_input(
            "🎫 兑换卡密 (次数/VIP)",
            placeholder="输入卡密兑换",
            disabled=st.session_state.is_running,
            key="exchange_code",
        )
        if st.button("立即兑换", use_container_width=True,
                     disabled=st.session_state.is_running):
            if not exchange_code:
                st.warning("请输入卡密！")
            else:
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/recharge",
                        params={
                            "user_id": uid,
                            "card_code": exchange_code,
                        }, timeout=10,
                    )
                    data = resp.json()
                    if data.get("status") == "success":
                        st.success("🎉 兑换成功！资产已更新。")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(data.get("message", "兑换失败"))
                except requests.ConnectionError:
                    st.error("❌ 后端未启动！")

        st.divider()

        if st.button("退出登录", use_container_width=True):
            auth_supabase.auth.sign_out()
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    # ==================== 🎬 主页面：极致清爽业务区 ====================
    st.title("🚀 爆款视频全自动提炼器")
    st.caption("后端 4 级流水线 + Supabase 实时状态流")

    st.markdown("---")

    mode = st.radio(
        "选择功能模式",
        ["🎙️ 直播间分析", "✍️ 爆款重写"],
        horizontal=True, disabled=st.session_state.is_running,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.url_or_product = st.text_input(
            "直播间URL 或 产品名称",
            value=st.session_state.url_or_product,
            placeholder="https://...m3u8" if "URL" in mode else "例如：多功能不粘锅",
            disabled=st.session_state.is_running,
        ).strip()
    with col2:
        target_style = st.selectbox(
            "话术风格（仅重写模式）",
            ["呐喊憋单流", "温柔种草流", "硬核测评流", "剧情代入流", "快节奏秒杀流"],
            disabled="URL" in mode or st.session_state.is_running,
        )

    # ---- 点火 ----
    fire = st.button("🔥 开始全自动提炼与脚本重写", type="primary",
                     use_container_width=True, disabled=st.session_state.is_running)

    if fire:
        p = st.session_state.url_or_product
        ak = st.session_state.api_key
        cc = st.session_state.card_code
        st.session_state.task_result = None
        st.session_state.task_error = None

        # 重新拉取资产（防止过期）
        uid = st.session_state["user_uuid"]
        balance = 0
        is_vip = False
        try:
            prof = db_supabase.table("user_profiles").select(
                "balance_count, vip_until"
            ).eq("id", uid).single().execute().data
            if prof:
                balance = prof.get("balance_count", 0)
                vu_str = prof.get("vip_until")
                if vu_str:
                    import datetime as _dt2
                    vu = _dt2.datetime.fromisoformat(
                        str(vu_str).replace("Z", "+00:00")
                    )
                    is_vip = vu > _dt2.datetime.now(_dt2.timezone.utc)
        except Exception:
            pass

        # 🚨 三条防线
        if not ak or ak.strip() == "":
            st.error("❌ 请先在【左侧边栏】配置您的 DeepSeek API Key！")
        elif not ak.startswith("sk-"):
            st.error("❌ Key 格式无效！应以 sk- 开头。")
        elif not is_vip and balance <= 0:
            st.error("❌ 免费额度已用尽！请先在【左侧边栏】兑换卡密充值。")
        elif not p:
            st.warning("⚠️ 请输入直播间 URL 或产品名称！")
        elif not cc:
            st.error("❌ 请输入激活卡密！")
        else:
            with st.spinner("🔄 校验 Key..."):
                ok, err = _verify_deepseek_key(ak)
                if not ok:
                    st.error(f"❌ {err}")
                else:
                    st.success("✅ 验证通过！")
                    st.session_state.is_running = True
                    st.session_state.running_mode = "analyze" if "URL" in mode else "rewrite"
                    st.session_state.poll_count = 0
                    st.rerun()

    # ---- 运行态：提交任务 ----
    if st.session_state.is_running and st.session_state.running_task_id is None:
        cm = st.session_state.running_mode
        st.markdown("---")
        with st.spinner("📤 创建任务..."):
            try:
                uid = st.session_state["user_uuid"]
                resp = db_supabase.table("fish_box").insert({
                    "user_id": uid,
                    "video_url": st.session_state.url_or_product,
                    "result_text": "排队中...",
                }).execute()
                tid = resp.data[0]["id"]
                st.session_state.running_task_id = str(tid)
            except Exception as e:
                st.session_state.task_error = f"❌ {str(e)[:200]}"
                st.session_state.is_running = False
                st.rerun()

            if cm == "analyze":
                try:
                    requests.post(
                        "http://127.0.0.1:8000/api/v1/analyze",
                        params={
                            "task_id": str(tid),
                            "video_url": st.session_state.url_or_product,
                            "user_id": st.session_state.user_uuid,
                            "user_deepseek_key": st.session_state.api_key,
                            "card_code": st.session_state.card_code,
                        }, timeout=10,
                    )
                except requests.ConnectionError:
                    st.session_state.task_error = "❌ 后端未启动！"
                    st.session_state.is_running = False
                    st.rerun()
            else:
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/rewrite",
                        params={
                            "my_product": st.session_state.url_or_product,
                            "target_style": target_style,
                            "user_key": st.session_state.api_key,
                            "card_code": st.session_state.card_code,
                        }, timeout=120,
                    )
                    data = resp.json()
                    if data.get("status") == "success":
                        st.session_state.task_result = {"mode": "rewrite", "data": data}
                    else:
                        st.session_state.task_error = data.get("message", "错误")
                    st.session_state.is_running = False
                    st.rerun()
                except requests.ConnectionError:
                    st.session_state.task_error = "❌ 后端未启动！"
                    st.session_state.is_running = False
                    st.rerun()

        # 提交后刷新一次进入轮询态
        st.rerun()

    # ---- 轮询态：@st.fragment 局部刷新，不闪烁全页 ----
    if st.session_state.is_running and st.session_state.running_task_id:
        task_id = st.session_state.running_task_id

        @st.fragment(run_every=3)
        def _poll_status():
            st.markdown("---")
            try:
                row = db_supabase.table("fish_box").select(
                    "status,result_text"
                ).eq("id", task_id).single().execute().data
            except Exception:
                row = {}

            sts = row.get("status", "processing")
            txt = row.get("result_text", "")
            badge = {"queued": "🔵", "processing": "🟡", "success": "🟢", "failed": "🔴"}
            st.subheader(f"{badge.get(sts, '⚪')} 状态: {sts.upper()}")

            if sts == "processing":
                st.warning("后端正在疯狂榨干服务器...")
                m = re.match(r"PROGRESS:(\d+)\\|", txt or "")
                if m:
                    pct = int(m.group(1)) / 100.0
                    log = txt[txt.index("|") + 1:]
                else:
                    pct = min(0.8, st.session_state.poll_count / 80)
                    log = txt
                st.progress(pct, text=f"管道运转中... ({int(pct*100)}%)")
                st.caption(f"⏱️ {log}")
                st.session_state.poll_count += 1

            elif sts in ("success", "failed"):
                # 终态：写入 session_state，跳出轮询
                if sts == "success":
                    st.session_state.task_result = {"mode": "analyze", "result_text": txt}
                else:
                    st.session_state.task_error = txt or "任务失败"
                st.session_state.is_running = False
                st.rerun()  # 最后一次全页重载，渲染最终结果

            elif sts == "queued":
                st.info(f"🔵 排队中... {txt}")
                st.session_state.poll_count += 1

            # 超时保底
            if st.session_state.poll_count > 200:
                st.session_state.task_error = "⏰ 超时（10分钟），请刷新重试。"
                st.session_state.is_running = False
                st.rerun()

        _poll_status()

    # ---- 结果渲染 ----
    if st.session_state.task_error:
        st.error(st.session_state.task_error)

    if st.session_state.task_result:
        r = st.session_state.task_result
        st.markdown("---")

        if r["mode"] == "analyze":
            st.success("✅ 分析完成！")
            st.subheader("📝 富化话术")
            txt = r.get("result_text", "")
            try:
                items = json.loads(txt) if txt.strip().startswith("[") else []
            except Exception:
                st.markdown(txt); items = []
            if isinstance(items, list):
                for it in items[:10]:
                    with st.expander(it.get("chunk_id", f"chunk_{id(it)}")):
                        st.markdown("🎤 " + str(it.get("icebreaker", "")))
                        st.markdown("🎯 " + str(it.get("painpoint", "")))
                        st.markdown("💎 " + str(it.get("mechanism", "")))
                        st.markdown("🔥 " + str(it.get("close_order", "")))

        elif r["mode"] == "rewrite":
            d = r["data"]
            st.success("✅ 重写完成！")
            if d.get("retrieved_references"):
                st.subheader("📚 历史爆款参考")
                rcols = st.columns(len(d["retrieved_references"]))
                for i, ref in enumerate(d["retrieved_references"]):
                    with rcols[i]:
                        st.metric(f"#{i+1}", f"{ref['similarity']*100:.0f}%")
            st.subheader("📝 话术脚本")
            st.markdown(d.get("rewritten_script", "").replace("[", "\n\n**[").replace("]", "]** "))
            sop = d.get("sop_timeline", [])
            if sop:
                st.subheader("⏱️ SOP 仪表盘")
                st.dataframe([{
                    "时间": s.get("time_range",""), "阶段": s.get("stage",""),
                    "主播": s.get("host_action",""), "操作": s.get("operation_action",""),
                    "关键词": s.get("verbal_keywords",""),
                } for s in sop], use_container_width=True)

        # 导出
        ext = ""
        if r["mode"] == "rewrite":
            ext = r["data"].get("rewritten_script", "")
        elif r["mode"] == "analyze":
            ext = r.get("result_text", "")
        if ext:
            st.download_button("📥 一键下载 TXT", data=ext,
                               file_name=f"话术_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                               mime="text/plain", use_container_width=True)

        if st.button("🔄 开始新任务", use_container_width=True):
            for k in ("task_result", "task_error", "is_running",
                      "running_task_id", "running_mode", "poll_count"):
                st.session_state[k] = None if k in ("task_result","task_error",
                    "running_task_id","running_mode") else (False if k=="is_running" else 0)
            st.rerun()

    st.markdown("---")
    st.caption("💡 卡密请前往发卡网购买 | Key 在 platform.deepseek.com | 本站不储存任何 Key")


# =========================================================================
# 路由
# =========================================================================
if not st.session_state["logged_in"]:
    show_login_page()
else:
    show_main_app()
