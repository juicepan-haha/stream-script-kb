#!/usr/bin/env python3
"""
app_frontend.py — 中小主播爆款话术"平替"全自动反应堆 (SaaS 前端)

纯 Streamlit 极简界面，对接后端 server_v2.py FastAPI。
"""
import time
import requests
import streamlit as st

# =========================================================================
# 页面配置
# =========================================================================
st.set_page_config(
    page_title="AI直播话术爆款平替系统",
    page_icon="🚀",
    layout="centered",
)

st.title("🚀 中小主播爆款话术"平替"全自动反应堆")
st.caption("主攻长尾市场版 — 达到飞书 80% 效果，只需 1% 成本")

st.markdown("---")

# =========================================================================
# 模式选择
# =========================================================================
mode = st.radio(
    "选择功能模式",
    ["🎙️ 直播间分析（URL → 转录 → 话术提炼）",
     "✍️ 爆款重写（输入产品 → RAG 检索 → SOP 脚本）"],
    horizontal=True,
)

# =========================================================================
# 输入区域
# =========================================================================
col1, col2 = st.columns(2)

with col1:
    url_or_product = st.text_input(
        "1. 直播间URL 或 产品名称",
        placeholder=(
            "https://livenging.alicdn.com/.../merge.m3u8"
            if "URL" in mode else
            "例如：多功能不粘锅"
        ),
    )

with col2:
    target_style = st.selectbox(
        "2. 目标话术风格（仅爆款重写模式）",
        ["呐喊憋单流", "温柔种草流", "硬核测评流", "剧情代入流", "快节奏秒杀流"],
        disabled="URL" in mode,
    )

user_key = st.text_input(
    "3. 您的 DeepSeek API Key（本站不储存，阅后即焚）",
    type="password",
    placeholder="sk-...",
)

card_code = st.text_input(
    "4. 激活卡密（前往发卡网购买，单次仅需几毛钱）",
    placeholder="BETA-TEST-001",
)

# =========================================================================
# DeepSeek Key 盲测（锅甩在最前面）
# =========================================================================

def _test_deepseek_key(key: str) -> bool:
    """轻量级盲测：用最小的请求验证 Key 是否有效。"""
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
        return resp.status_code == 200
    except Exception:
        return False


# =========================================================================
# 提交按钮
# =========================================================================
if st.button("🔥 开始全自动提炼与脚本重写", type="primary", use_container_width=True):
    if not url_or_product or not user_key or not card_code:
        st.error("❌ 请完整填写以上所有参数！")
    elif not user_key.startswith("sk-"):
        st.error("❌ DeepSeek API Key 格式无效！应以 'sk-' 开头。")
    else:
        # 1. 盲测 Key
        with st.spinner("🔄 正在安全校验您的 DeepSeek Key 状态（本站不记录）..."):
            if not _test_deepseek_key(user_key):
                st.error("❌ DeepSeek Key 校验失败！请检查 Key 是否正确、余额是否充足。")
                st.stop()
        st.success("✅ DeepSeek Key 验证通过！")

        # 2. 根据模式发送请求
        if "URL" in mode:
            # 直播间分析模式
            with st.spinner("🎙️ 系统正在全流式内存转录、切块并调用 RAG 库分析中... 请勿关闭网页"):
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/analyze",
                        params={
                            "url": url_or_product,
                            "user_key": user_key,
                            "card_code": card_code,
                        },
                        timeout=10,
                    )
                    data = resp.json()
                except requests.ConnectionError:
                    st.error("❌ 无法连接后端服务！请确认 server_v2.py 已启动。")
                    st.stop()

                if data["status"] == "accepted":
                    task_id = data["task_id"]
                    st.success(f"✅ {data['message']}")
                    st.info(f"📋 任务 ID: `{task_id}`")

                    # 轮询进度
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for _ in range(120):  # 最多等 10 分钟
                        time.sleep(5)
                        try:
                            prog = requests.get(
                                f"http://127.0.0.1:8000/api/v1/progress/{task_id}"
                            ).json()
                        except Exception:
                            continue

                        stage = prog.get("stage", "unknown")
                        segs = prog.get("transcript_segments", 0)
                        chunks = prog.get("enriched_chunks", 0)

                        status_text.text(
                            f"阶段: {stage} | 转录段: {segs} | 富化块: {chunks}"
                        )
                        progress_bar.progress(min(0.95, chunks / max(1, segs / 100)))

                        if stage == "completed":
                            progress_bar.progress(1.0)
                            status_text.text("✅ 分析完成！正在拉取结果...")

                            # 拉取富化结果
                            enriched = requests.get(
                                f"http://127.0.0.1:8000/api/v1/enriched/{task_id}"
                            ).json()

                            st.subheader("📝 富化话术结果")
                            for item in enriched.get("results", [])[:10]:
                                with st.expander(
                                    f"{item.get('chunk_id', '')} "
                                    f"({item.get('char_count', 0)} 字)"
                                ):
                                    st.markdown("**🎤 破冰:**\n" + item.get("icebreaker", ""))
                                    st.markdown("**🎯 痛点:**\n" + item.get("painpoint", ""))
                                    st.markdown("**💎 卖点:**\n" + item.get("mechanism", ""))
                                    st.markdown("**🔥 逼单:**\n" + item.get("close_order", ""))
                            break
                    else:
                        st.warning("⏰ 分析超时，请稍后通过任务 ID 查询结果。")
                else:
                    st.error(data.get("message", "未知错误"))

        else:
            # 爆款重写模式
            with st.spinner("✍️ RAG 检索历史爆款 + DeepSeek 重写 + SOP 生成中..."):
                try:
                    resp = requests.post(
                        "http://127.0.0.1:8000/api/v1/rewrite",
                        params={
                            "my_product": url_or_product,
                            "target_style": target_style,
                            "user_key": user_key,
                            "card_code": card_code,
                        },
                        timeout=120,
                    )
                    data = resp.json()
                except requests.ConnectionError:
                    st.error("❌ 无法连接后端服务！请确认 server_v2.py 已启动。")
                    st.stop()
                except requests.Timeout:
                    st.error("⏰ 请求超时，请重试。")
                    st.stop()

                if data["status"] == "success":
                    st.success("✅ 爆款话术重写完成！")

                    # 检索参考
                    if data.get("retrieved_references"):
                        st.subheader("📚 检索到的历史爆款参考")
                        ref_cols = st.columns(len(data["retrieved_references"]))
                        for i, ref in enumerate(data["retrieved_references"]):
                            with ref_cols[i]:
                                st.metric(
                                    f"参考 #{i+1}",
                                    f"{ref['similarity']*100:.0f}%",
                                )
                                st.caption(ref.get("sales_stage", ""))

                    # 话术脚本
                    st.subheader("📝 完整话术脚本")
                    script = data.get("rewritten_script", "")
                    st.markdown(script.replace("[", "\n\n**[").replace("]", "]** "))

                    # SOP 仪表盘
                    sop = data.get("sop_timeline", [])
                    if sop:
                        st.subheader("⏱️ 秒级执行 SOP 仪表盘")
                        sop_data = []
                        for step in sop:
                            sop_data.append({
                                "时间": step.get("time_range", ""),
                                "阶段": step.get("stage", ""),
                                "主播动作": step.get("host_action", ""),
                                "后台操作": step.get("operation_action", ""),
                                "关键词": step.get("verbal_keywords", ""),
                            })
                        st.dataframe(sop_data, use_container_width=True)
                    else:
                        st.info("SOP 时间轴生成中，请重试或调整产品名称。")

                elif data["status"] == "no_results":
                    st.warning("📭 数据库中没有匹配的话术参考。请先导入更多直播数据。")
                else:
                    st.error(data.get("message", "未知错误"))

st.markdown("---")
st.caption(
    "💡 提示：卡密请前往发卡网购买 | DeepSeek Key 在 platform.deepseek.com 获取 | "
    "本站绝不记录您的 API Key"
)
