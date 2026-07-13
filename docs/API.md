# API 文档

Base URL: `http://localhost:8000`

## 端点总览

| 方法 | 路径 | 认证 | 用途 |
|------|------|------|------|
| POST | `/api/v1/analyze` | 卡密 | 提交直播间 URL → 全管道分析 |
| POST | `/api/v1/rewrite` | 卡密 + Key | RAG 检索 + 爆款重写 + SOP |
| GET | `/api/v1/transcript/{id}` | 无 | 查询原始转录 |
| GET | `/api/v1/enriched/{id}` | 无 | 查询富化话术 |
| GET | `/api/v1/progress/{id}` | 无 | 查询任务进度 |
| GET | `/api/v1/health` | 无 | 队列状态 |

---

## POST /api/v1/analyze

提交直播间 URL 进行全管道流式分析。

**参数 (Query String):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | string | 是 | 直播间 m3u8 URL |
| `user_key` | string | 是 | 用户自备的 DeepSeek API Key (sk-...) |
| `card_code` | string | 是 | 本站激活卡密 |

**响应:**

```json
{
  "status": "accepted",
  "task_id": "task_1783948529775",
  "message": "🚀 验证通过！已送入流式反应堆..."
}
```

**错误:**

```json
{"status": "error", "message": "❌ 卡密无效、已被使用或已过期！"}
{"status": "error", "message": "❌ DeepSeek API Key 格式无效！"}
```

---

## POST /api/v1/rewrite

RAG 检索 + DeepSeek 爆款重写 + SOP 仪表盘。

**参数 (Query String):**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `my_product` | string | 是 | 产品名称 (如"多功能不粘锅") |
| `target_style` | string | 否 | 话术风格 (默认"呐喊憋单流") |
| `user_key` | string | 是 | 用户自备的 DeepSeek API Key |
| `card_code` | string | 是 | 本站激活卡密 |

**target_style 可选值:**
`呐喊憋单流` `温柔种草流` `硬核测评流` `剧情代入流` `快节奏秒杀流`

**响应:**

```json
{
  "status": "success",
  "my_product": "多功能不粘锅",
  "target_style": "呐喊憋单流",
  "retrieved_references": [
    {"similarity": 0.435, "sales_stage": "开场暖场", "strategy_types": ["痛点放大", "亲身试用"]}
  ],
  "rewritten_script": "[破冰留人]家人们！别划走！...",
  "sop_timeline": [
    {
      "time_range": "00:00-00:05",
      "stage": "破冰留人",
      "host_action": "拿起锅敲两下，眼神坚定看镜头",
      "operation_action": "镜头推近，展示锅体",
      "verbal_keywords": "别划走"
    }
  ]
}
```

---

## GET /api/v1/progress/{task_id}

查询流式分析任务进度。

**响应:**

```json
{
  "task_id": "task_1783948529775",
  "stage": "transcribing",
  "transcript_segments": 3228,
  "enriched_chunks": 26
}
```

**stage 生命周期:** `downloading` → `transcribing` → `chunked` → `completed`

---

## GET /api/v1/transcript/{task_id}

查询原始转录结果。

**响应:**

```json
{
  "task_id": "task_1783948529775",
  "segments": 3228,
  "text": "家人们欢迎来到直播间...",
  "details": [
    {"start": 0.0, "end": 2.5, "text": "家人们欢迎来到直播间"}
  ]
}
```

---

## GET /api/v1/enriched/{task_id}

查询 DeepSeek 富化后的提词器级话术。

**响应:**

```json
{
  "task_id": "task_1783948529775",
  "chunks": 26,
  "results": [
    {
      "chunk_id": "task_..._chunk_0001",
      "icebreaker": "家人们！...",
      "painpoint": "你们有没有...",
      "mechanism": "看好了！这款...",
      "close_order": "最后20单！..."
    }
  ]
}
```

---

## GET /api/v1/health

系统队列健康检查。

**响应:**

```json
{
  "download_queue": 0,
  "text_queue": 0,
  "chunk_queue": 0,
  "enriched_queue": 0,
  "active_tasks": ["task_1783948529775"]
}
```

队列值接近 maxsize 时说明下游处理跟不上，系统背压正在生效。
