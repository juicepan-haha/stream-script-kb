"""流式服务的进程内队列和任务结果。"""

import asyncio


# 有界队列让下游处理变慢时自然形成背压。
download_queue = asyncio.Queue(maxsize=20)
text_queue = asyncio.Queue(maxsize=50)
chunk_queue = asyncio.Queue(maxsize=100)
enriched_queue = asyncio.Queue(maxsize=100)

# 实时分析结果仅保存在当前服务进程中。
transcript_results: dict[str, list[dict]] = {}
enriched_results: dict[str, list[dict]] = {}
task_progress: dict[str, dict] = {}
