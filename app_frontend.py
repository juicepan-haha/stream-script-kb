#!/usr/bin/env python3
"""
app_frontend.py — Ultimate Cloud Edition 前端控制台

Streamlit 轻量级看板，直连 Supabase 3 秒轮询 + 状态徽章实时渲染。
防刷新：所有输入和结果通过 st.session_state 持久化。
"""
import json
import time
import requests
import streamlit as st

# =========================================================================
# Supabase 客户端单例（防轮询时连接数爆炸）
# =========================================================================
@st.cache_resource
def _get_supabase():
    from supabase_client import supabase as sb
    return sb

supabase = _get_supabase()

# =========================================================================
# 页面配置
# =========================================================================
st.set_page_config(
    page_title="AI直播话术爆款平替系统",
    page_icon="🚀",
    layout="centered",
)

st.title("🎬 Ultimate Cloud Edition 视频分析控制台")
st.caption("主攻长尾市场 — 后端 4 级流水线 + Supabase 实时状态流")

st.markdown("---")

# =========================================================================
# 1. 持久化内存初始化
# =========================================================================
_defaults = {
    "api_key": "", "url_or_product": "", "card_code": "",
    "task_result": None, "task_error": None,
    "is_running": False, "running_task_id": None,
    "running_mode": None, "poll_count": 0,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================================================================
# 2. 输入区域（绑定 session_state，刷新不丢）
# =========================================================================
mode = st.radio(
    "选择功能模式",
    ["🎙️ 直播间分析（URL → 转录 → 话术提炼）",
     "✍️ 爆款重写（输入产品 → RAG 检索 → SOP 脚本）"],
    horizontal=True,
    disabled=st.session_state.is_running,
)

col1, col2 = st.columns(2)
with col1:
    st.session_state.url_or_product = st.text_input(
        "1. 直播间URL 或 产品名称",
        value=st.session_state.url_or_product,
        placeholder=(
            "https://livenging.alicdn.com/.../merge.m3u8"
            if "URL" in mode else "例如：多功能不粘锅"
        ),
        disabled=st.session_state.is_running,
    ).strip()
with col2:
    target_style = st.selectbox(
        "2. 目标话术风格（仅爆款重写模式）",
        ["呐喊憋单流", "温柔种草流", "硬核测评流", "剧情代入流", "快节奏秒杀流"],
        disabled="URL" in mode or st.session_state.is_running,
    )

st.session_state.api_key = st.text_input(
    "3. 您的 DeepSeek API Key（本站不储存，阅后即焚）",
    type="password", value=st.session_state.api_key,
    placeholder="sk-...", disabled=st.session_state.is_running,
).strip()

st.session_state.card_code = st.text_input(
    "4. 激活卡密（前往发卡网购买）",
    value=st.session_state.card_code,
    placeholder="BETA-TEST-001", disabled=st.session_state.is_running,
).strip()


# =========================================================================
# 3. DeepSeek Key 盲测
# =========================================================================
def _verify_deepseek_key(key: str) -> tuple[bool, str]:
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1},
            timeout=10,
        )
        if resp.status_code == 200:   return True, ""
        if resp.status_code == 401:   return False, "Key 无效，请检查是否复制完整（sk- 开头）。"
        if resp.status_code == 429:   return False, "余额不足或频次限制，请充值后重试。"
        if resp.status_code >= 500:   return False, "DeepSeek 服务器繁忙，请稍后重试。"
        return False, f"DeepSeek 返回异常 (HTTP {resp.status_code})，请稍后重试。"
    except requests.Timeout:          return False, "网络超时，请检查网络。"
    except requests.ConnectionError:   return False, "无法连接 DeepSeek，请检查网络或代理。"
    except Exception as e:            return False, f"未知错误: {str(e)[:100]}"


# =========================================================================
# 4. 业务点火
# =========================================================================
fire_clicked = st.button(
    "🔥 开始全自动提炼与脚本重写", type="primary",
    use_container_width=True, disabled=st.session_state.is_running,
)

if fire_clicked:
    product = st.session_state.url_or_product
    api_key = st.session_state.api_key
    card = st.session_state.card_code

    st.session_state.task_result = None
    st.session_state.task_error = None

    if not product or not api_key or not card:
        st.error("❌ 请完整填写以上所有参数！")
    elif not api_key.startswith("sk-"):
        st.error("❌ DeepSeek API Key 格式无效！应以 'sk-' 开头。")
    else:
        with st.spinner("🔄 正在校验 DeepSeek Key..."):
            ok, err_msg = _verify_deepseek_key(api_key)
            if not ok:
                st.error(f"❌ {err_msg}")
            else:
                st.success("✅ DeepSeek Key 验证通过！")
                st.session_state.is_running = True
                st.session_state.running_mode = "analyze" if "URL" in mode else "rewrite"
                st.session_state.poll_count = 0
                st.rerun()

# =========================================================================
# 5. 运行态：轮询 + 状态渲染
# =========================================================================
if st.session_state.is_running:
    current_mode = st.session_state.running_mode
    product = st.session_state.url_or_product
    api_key = st.session_state.api_key
    card = st.session_state.card_code

    # --- 用 st.empty() 局部刷新，避免整页闪烁 ---
    status_holder = st.empty()

    with status_holder.container():
        st.markdown("---")

        # 第一步提交
        if st.session_state.running_task_id is None:
            with st.spinner("📤 正在创建任务..."):
                # 前端 INSERT Supabase（RLS 自动绑定 user_id）
                try:
                    # user_deepseek_key 只通过 API 传后端，绝不落库
                    TEST_USER_UUID = "7b889093-c96a-4348-9157-42d81b2fd284"
                    sb_resp = supabase.table("fish_box").insert({
                        "user_id": TEST_USER_UUID,
                        "video_url": product,
                        "result_text": "排队中，等待处理...",
                    }).execute()
                    task_id = sb_resp.data[0]["id"] if sb_resp.data else None
                    if not task_id:
                        st.session_state.task_error = "❌ 任务创建失败"
                        st.session_state.is_running = False
                        st.rerun()
                except Exception as e:
                    st.session_state.task_error = f"❌ Supabase: {str(e)[:200]}"
                    st.session_state.is_running = False
                    st.rerun()

                st.session_state.running_task_id = str(task_id)

                # 调后端启动管道
                if current_mode == "analyze":
                    try:
                        resp = requests.post(
                            "http://127.0.0.1:8000/api/v1/analyze",
                            params={
                                "task_id": str(task_id), "video_url": product,
                                "user_deepseek_key": api_key, "card_code": card,
                            }, timeout=10,
                        )
                        data = resp.json()
                        if data.get("status") not in ("queued", "accepted"):
                            st.session_state.task_error = data.get("message", "错误")
                            st.session_state.is_running = False
                            st.rerun()
                    except requests.ConnectionError:
                        st.session_state.task_error = "❌ 后端未启动！请运行 server_v2.py。"
                        st.session_state.is_running = False
                        st.rerun()

                elif current_mode == "rewrite":
                    try:
                        resp = requests.post(
                            "http://127.0.0.1:8000/api/v1/rewrite",
                            params={
                                "my_product": product, "target_style": target_style,
                                "user_key": api_key, "card_code": card,
                            }, timeout=120,
                        )
                        data = resp.json()
                        if data.get("status") == "success":
                            st.session_state.task_result = {
                                "mode": "rewrite", "data": data,
                            }
                        else:
                            st.session_state.task_error = data.get("message", "错误")
                        st.session_state.is_running = False
                        st.rerun()
                    except requests.ConnectionError:
                        st.session_state.task_error = "❌ 后端未启动！"
                        st.session_state.is_running = False
                        st.rerun()

        # --- 轮询 Supabase 状态 ---
        task_id = st.session_state.running_task_id

        if current_mode == "analyze" and task_id:
            st.session_state.poll_count += 1

            try:
                sb_resp = supabase.table("fish_box").select(
                    "status, result_text"
                ).eq("id", task_id).single().execute()
                row = sb_resp.data if sb_resp.data else {}
            except Exception:
                row = {}

            db_status = row.get("status", "processing")
            db_text = row.get("result_text", "")

            # 状态徽章
            badge = {"queued": "🔵", "processing": "🟡", "success": "🟢", "failed": "🔴"}
            st.subheader(f"{badge.get(db_status, '⚪')} 状态: {db_status.upper()}")

            if db_status == "processing":
                st.warning("后端正在疯狂榨干服务器，请勿关闭网页...")

                # 真·进度条：解析 result_text 中的 PROGRESS:XX| 前缀
                import re as _re
                pct_match = _re.match(r"PROGRESS:(\d+)\|", db_text or "")
                if pct_match:
                    real_pct = int(pct_match.group(1)) / 100.0
                    log_text = db_text[db_text.index("|") + 1:]
                else:
                    real_pct = min(0.8, st.session_state.poll_count / 80)
                    log_text = db_text

                st.progress(real_pct, text=f"管道运转中... ({int(real_pct*100)}%)")
                st.caption(f"⏱️ 实时日志: {log_text}")

            elif db_status == "success":
                st.success("🟢 分析成功！")
                st.session_state.task_result = {
                    "mode": "analyze", "result_text": db_text,
                }
                st.session_state.is_running = False
                st.rerun()

            elif db_status == "failed":
                st.error(f"🔴 任务失败: {db_text}")
                st.session_state.task_error = db_text or "未知错误"
                st.session_state.is_running = False
                st.rerun()

            elif db_status == "queued":
                st.info(f"🔵 排队中... {db_text}")

            if st.session_state.poll_count > 200:
                st.session_state.task_error = (
                    "⏰ 检测到云端处理超时（超过 10 分钟）。"
                    "系统后台已自动重置该任务，请刷新页面或重新发起分析。"
                )
                st.session_state.is_running = False
                st.rerun()

    # 3 秒后自动重绘（Streamlit 标准轮询模式）
    if st.session_state.is_running:
        time.sleep(3)
        st.rerun()

# =========================================================================
# 6. 结果渲染层
# =========================================================================
if st.session_state.task_error:
    st.error(st.session_state.task_error)

if st.session_state.task_result:
    result = st.session_state.task_result
    st.markdown("---")

    if result["mode"] == "analyze":
        st.success("✅ 直播间分析完成！")
        st.subheader("📝 富化话术结果")

        result_text = result.get("result_text", "")
        try:
            items = json.loads(result_text) if isinstance(result_text, str) and result_text.strip().startswith("[") else []
        except (json.JSONDecodeError, TypeError):
            st.markdown(result_text)
            items = []

        if isinstance(items, list):
            for item in items[:10]:
                cid = item.get("chunk_id", f"chunk_{id(item)}")
                with st.expander(f"{cid}"):
                    st.markdown("**🎤 破冰:**\n" + str(item.get("icebreaker", "")))
                    st.markdown("**🎯 痛点:**\n" + str(item.get("painpoint", "")))
                    st.markdown("**💎 卖点:**\n" + str(item.get("mechanism", "")))
                    st.markdown("**🔥 逼单:**\n" + str(item.get("close_order", "")))

    elif result["mode"] == "rewrite":
        data = result["data"]
        st.success("✅ 爆款话术重写完成！")

        if data.get("retrieved_references"):
            st.subheader("📚 检索到的历史爆款参考")
            refs = data["retrieved_references"]
            rcols = st.columns(len(refs))
            for i, ref in enumerate(refs):
                with rcols[i]:
                    st.metric(f"参考 #{i+1}", f"{ref['similarity']*100:.0f}%")
                    st.caption(ref.get("sales_stage", ""))

        st.subheader("📝 完整话术脚本")
        script = data.get("rewritten_script", "")
        st.markdown(script.replace("[", "\n\n**[").replace("]", "]** "))

        sop = data.get("sop_timeline", [])
        if sop:
            st.subheader("⏱️ 秒级执行 SOP 仪表盘")
            st.dataframe(
                [{
                    "时间": s.get("time_range", ""), "阶段": s.get("stage", ""),
                    "主播动作": s.get("host_action", ""), "后台操作": s.get("operation_action", ""),
                    "关键词": s.get("verbal_keywords", ""),
                } for s in sop],
                use_container_width=True,
            )

    # 导出
    export_text = ""
    if result["mode"] == "rewrite":
        d = result["data"]
        parts = [f"爆款话术 — {d.get('my_product','')} ({d.get('target_style','')})", "="*50, d.get("rewritten_script","")]
        for s in d.get("sop_timeline", []):
            parts.append(f"\n[{s.get('time_range','')}] {s.get('stage','')}\n  主播: {s.get('host_action','')}\n  操作: {s.get('operation_action','')}")
        export_text = "\n".join(parts)
    elif result["mode"] == "analyze":
        export_text = result.get("result_text", "")

    if export_text:
        st.download_button(
            "📥 一键下载话术文案 (TXT)", data=export_text,
            file_name=f"爆款话术_{time.strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain", use_container_width=True,
        )

    if st.button("🔄 开始新任务", use_container_width=True):
        for k in ("task_result", "task_error", "is_running",
                  "running_task_id", "running_mode", "poll_count"):
            st.session_state[k] = None if k in ("task_result", "task_error",
                                                  "running_task_id", "running_mode") else (False if k == "is_running" else 0)
        st.rerun()

st.markdown("---")
st.caption("💡 卡密请前往发卡网购买 | DeepSeek Key 在 platform.deepseek.com | 本站不储存任何 Key")
