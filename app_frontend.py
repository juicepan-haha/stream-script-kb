#!/usr/bin/env python3
"""
app_frontend.py — 中小主播爆款话术"平替"全自动反应堆 (SaaS 前端)

纯 Streamlit 极简界面，对接后端 server_v2.py FastAPI。

防刷新：所有输入和结果通过 st.session_state 持久化。
"""
import json
import time
import requests
import streamlit as st
from supabase_client import supabase

# =========================================================================
# 页面配置
# =========================================================================
st.set_page_config(
    page_title="AI直播话术爆款平替系统",
    page_icon="🚀",
    layout="centered",
)

st.title("🚀 中小主播爆款话术平替全自动反应堆")
st.caption("主攻长尾市场版 — 达到飞书 80% 效果，只需 1% 成本")

st.markdown("---")

# =========================================================================
# 1. 持久化内存初始化（防刷新数据暴毙）
# =========================================================================
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "url_or_product" not in st.session_state:
    st.session_state.url_or_product = ""
if "card_code" not in st.session_state:
    st.session_state.card_code = ""
if "task_result" not in st.session_state:
    st.session_state.task_result = None
if "task_error" not in st.session_state:
    st.session_state.task_error = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "running_task_id" not in st.session_state:
    st.session_state.running_task_id = None
if "running_mode" not in st.session_state:
    st.session_state.running_mode = None
if "poll_count" not in st.session_state:
    st.session_state.poll_count = 0

# =========================================================================
# 输入区域（绑定到 session_state）
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
            if "URL" in mode else
            "例如：多功能不粘锅"
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
    type="password",
    value=st.session_state.api_key,
    placeholder="sk-...",
    disabled=st.session_state.is_running,
).strip()

st.session_state.card_code = st.text_input(
    "4. 激活卡密（前往发卡网购买，单次仅需几毛钱）",
    value=st.session_state.card_code,
    placeholder="BETA-TEST-001",
    disabled=st.session_state.is_running,
).strip()


# =========================================================================
# DeepSeek Key 盲测
# =========================================================================
def _verify_deepseek_key(key: str) -> tuple[bool, str]:
    """轻量级盲测 Key 有效性，返回 (成功, 人话提示)。"""
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True, ""
        elif resp.status_code == 401:
            return False, "校验失败：您的 DeepSeek Key 无效，请检查是否复制完整（sk- 开头）。"
        elif resp.status_code == 429:
            return False, "访问受限：您的 DeepSeek 账户余额不足或触发频次限制，请充值后重试。"
        elif resp.status_code >= 500:
            return False, "DeepSeek 服务器繁忙，请稍后重试。"
        else:
            return False, f"DeepSeek 返回异常 (HTTP {resp.status_code})，请稍后重试。"
    except requests.Timeout:
        return False, "网络连接超时，请检查网络后重试。"
    except requests.ConnectionError:
        return False, "无法连接 DeepSeek 服务器，请检查网络或使用代理。"
    except Exception as e:
        return False, f"未知错误: {str(e)[:100]}"


# =========================================================================
# 2. 业务点火控制逻辑（st.session_state.is_running 锁）
# =========================================================================
fire_clicked = st.button(
    "🔥 开始全自动提炼与脚本重写",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.is_running,
)

if fire_clicked or st.session_state.is_running:
    product = st.session_state.url_or_product
    api_key = st.session_state.api_key
    card = st.session_state.card_code

    # ---- 校验 ----
    if fire_clicked:
        st.session_state.task_result = None
        st.session_state.task_error = None

        if not product or not api_key or not card:
            st.error("❌ 请完整填写以上所有参数！")
        elif not api_key.startswith("sk-"):
            st.error("❌ DeepSeek API Key 格式无效！应以 'sk-' 开头。")
        else:
            # Key 盲测
            with st.spinner("🔄 正在安全校验您的 DeepSeek Key 状态（本站不记录）..."):
                ok, err_msg = _verify_deepseek_key(api_key)
                if not ok:
                    st.error(f"❌ {err_msg}")
                else:
                    st.success("✅ DeepSeek Key 验证通过！")
                    # 点火！
                    st.session_state.is_running = True
                    st.session_state.running_mode = "analyze" if "URL" in mode else "rewrite"
                    st.session_state.poll_count = 0
                    st.rerun()

    # ---- 长耗时任务执行 ----
    if st.session_state.is_running:
        current_mode = st.session_state.running_mode

        with st.spinner(
            "🎙️ 后端正在疯狂榨干服务器，正在下载并提炼话术，请绝对不要刷新网页..."
            if current_mode == "analyze" else
            "✍️ RAG 检索历史爆款 + DeepSeek 重写 + SOP 生成中..."
        ):
            try:
                if current_mode == "analyze":
                    # --- 直播间分析模式：提交 + 轮询 ---

                    # 第一次提交：INSERT Supabase → 调用后端
                    if st.session_state.running_task_id is None:
                        # 先写入 Supabase（前端 INSERT，RLS 自动绑定 user_id）
                        try:
                            sb_resp = supabase.table("fish_box").insert({
                                "video_url": product,
                                "user_deepseek_key": api_key,
                                "status": "queued",
                                "result_text": "排队中，等待处理...",
                            }).execute()
                            task_id = sb_resp.data[0]["id"] if sb_resp.data else None
                            if not task_id:
                                st.session_state.task_error = "❌ 创建任务失败，请重试。"
                                st.session_state.is_running = False
                                st.rerun()
                        except Exception as e:
                            st.session_state.task_error = f"❌ Supabase 写入失败: {str(e)[:200]}"
                            st.session_state.is_running = False
                            st.rerun()

                        # 调后端启动管道
                        try:
                            resp = requests.post(
                                "http://127.0.0.1:8000/api/v1/analyze",
                                params={
                                    "task_id": str(task_id),
                                    "video_url": product,
                                    "user_deepseek_key": api_key,
                                    "card_code": card,
                                },
                                timeout=10,
                            )
                            data = resp.json()
                            if data.get("status") not in ("queued", "accepted"):
                                st.session_state.task_error = data.get("message", "未知错误")
                                st.session_state.is_running = False
                                st.rerun()
                        except requests.ConnectionError:
                            st.session_state.task_error = "❌ 后端未启动！请运行 server_v2.py。"
                            st.session_state.is_running = False
                            st.rerun()

                        st.session_state.running_task_id = str(task_id)

                    # --- 轮询 Supabase 实时状态 ---
                    task_id = st.session_state.running_task_id
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
                    badge_color = {
                        "queued": "🔵", "processing": "🟡",
                        "success": "🟢", "failed": "🔴",
                    }
                    st.write(f"{badge_color.get(db_status, '⚪')} **{db_status.upper()}**")
                    st.caption(db_text[:300] if db_status == "processing" else "")

                    if db_status == "success":
                        st.session_state.task_result = {
                            "mode": "analyze",
                            "result_text": db_text,
                        }
                        st.session_state.is_running = False
                        st.rerun()
                    elif db_status == "failed":
                        st.session_state.task_error = db_text or "任务失败，请重试。"
                        st.session_state.is_running = False
                        st.rerun()
                    elif st.session_state.poll_count > 240:
                        st.session_state.task_error = "⏰ 分析超时（20 分钟），请稍后重试。"
                        st.session_state.is_running = False
                        st.rerun()
                    else:
                        time.sleep(3)
                        st.rerun()

                else:
                    # --- 爆款重写模式 ---
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/rewrite",
                        params={
                            "my_product": product,
                            "target_style": target_style,
                            "user_key": api_key,
                            "card_code": card,
                        },
                        timeout=120,
                    )
                    data = resp.json()

                    if data.get("status") == "success":
                        st.session_state.task_result = {
                            "mode": "rewrite",
                            "data": data,
                        }
                    else:
                        st.session_state.task_error = data.get("message", "未知错误")

                    st.session_state.is_running = False
                    st.rerun()

            except requests.ConnectionError:
                st.session_state.task_error = "❌ 无法连接后端服务！请确认 server_v2.py 已启动。"
                st.session_state.is_running = False
                st.rerun()
            except Exception as e:
                st.session_state.task_error = f"❌ 运行异常: {str(e)[:200]}"
                st.session_state.is_running = False
                st.rerun()

# =========================================================================
# 3. 结果渲染层（持久化内存驱动，刷新不掉）
# =========================================================================
if st.session_state.task_error:
    st.error(st.session_state.task_error)

if st.session_state.task_result:
    result = st.session_state.task_result

    if result["mode"] == "analyze":
        st.success("✅ 直播间分析完成！")
        st.subheader("📝 富化话术结果")

        # result_text 是后端塞进来的 JSON 话术，尝试解析
        result_text = result.get("result_text", "")
        try:
            if isinstance(result_text, str):
                items = json.loads(result_text) if result_text.strip().startswith("[") else []
            else:
                items = result_text
        except (json.JSONDecodeError, TypeError):
            # 纯文本结果
            st.markdown(result_text)
            items = []

        if isinstance(items, list):
            for item in items[:10]:
                chunk_id = item.get("chunk_id", f"chunk_{items.index(item)}")
                with st.expander(f"{chunk_id} ({len(str(item))} 字)"):
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
            ref_cols = st.columns(len(refs))
            for i, ref in enumerate(refs):
                with ref_cols[i]:
                    st.metric(f"参考 #{i+1}", f"{ref['similarity']*100:.0f}%")
                    st.caption(ref.get("sales_stage", ""))

        st.subheader("📝 完整话术脚本")
        script = data.get("rewritten_script", "")
        st.markdown(script.replace("[", "\n\n**[").replace("]", "]** "))

        sop = data.get("sop_timeline", [])
        if sop:
            st.subheader("⏱️ 秒级执行 SOP 仪表盘")
            sop_data = [
                {
                    "时间": s.get("time_range", ""),
                    "阶段": s.get("stage", ""),
                    "主播动作": s.get("host_action", ""),
                    "后台操作": s.get("operation_action", ""),
                    "关键词": s.get("verbal_keywords", ""),
                }
                for s in sop
            ]
            st.dataframe(sop_data, use_container_width=True)

    # =====================================================================
    # 5. 一键导出模块（无需数据库，纯内存生成 TXT）
    # =====================================================================
    export_text = ""
    if result["mode"] == "rewrite":
        data = result["data"]
        export_parts = [
            f"爆款话术脚本 — {data.get('my_product', '')} ({data.get('target_style', '')})",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            data.get("rewritten_script", ""),
            "",
            "=" * 50,
            "秒级执行 SOP 仪表盘",
            "=" * 50,
        ]
        for step in data.get("sop_timeline", []):
            export_parts.extend([
                f"\n[{step.get('time_range', '')}] {step.get('stage', '')}",
                f"  主播动作: {step.get('host_action', '')}",
                f"  后台操作: {step.get('operation_action', '')}",
                f"  关键词: {step.get('verbal_keywords', '')}",
            ])
        export_text = "\n".join(export_parts)

    elif result["mode"] == "analyze":
        results = result.get("results", [])
        export_parts = [
            f"直播间话术分析报告 — {st.session_state.url_or_product}",
            f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"共 {len(results)} 条话术切片",
            "=" * 50,
        ]
        for item in results:
            export_parts.extend([
                f"\n━━━ {item.get('chunk_id', '')} ({item.get('char_count', 0)}字) ━━━",
                f"[破冰留人] {item.get('icebreaker', '')}",
                f"[痛点植入] {item.get('painpoint', '')}",
                f"[产品卖点] {item.get('mechanism', '')}",
                f"[逼单催单] {item.get('close_order', '')}",
            ])
        export_text = "\n".join(export_parts)

    if export_text:
        file_name = f"爆款话术_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        st.download_button(
            label="📥 一键下载话术文案 (TXT)",
            data=export_text,
            file_name=file_name,
            mime="text/plain",
            use_container_width=True,
        )

    # 重置按钮
    if st.button("🔄 开始新任务", use_container_width=True):
        for key in ("task_result", "task_error", "is_running",
                     "running_task_id", "running_mode", "poll_count"):
            st.session_state[key] = None if key in ("task_result", "task_error",
                                                      "running_task_id", "running_mode") else False
        st.rerun()

st.markdown("---")
st.caption(
    "💡 提示：卡密请前往发卡网购买 | DeepSeek Key 在 platform.deepseek.com 获取 | "
    "本站绝不记录您的 API Key"
)
