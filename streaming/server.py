"""AI 直播话术流式分析服务的 FastAPI 应用入口。

数据通过四个有界内存队列在下载、转录、切块和富化 Worker 之间传递。
具体业务实现分别位于 ``workers/``、``routes/`` 和 ``services/``。
"""

import asyncio

import uvicorn
from fastapi import FastAPI

from streaming.routes.analysis import router as analysis_router
from streaming.routes.rewrite import router as rewrite_router
from streaming.workers import (
    audio_download_and_decode_worker,
    chunking_worker,
    deepseek_enrich_worker,
    whisper_transcribe_worker,
)


app = FastAPI(title="AI直播话术流式分析反应堆 (路径B)")
app.include_router(analysis_router)
app.include_router(rewrite_router)


@app.on_event("startup")
async def startup_event():
    """启动四个常驻流水线 Worker。"""
    asyncio.create_task(audio_download_and_decode_worker())
    asyncio.create_task(whisper_transcribe_worker())
    asyncio.create_task(chunking_worker())
    asyncio.create_task(deepseek_enrich_worker())
    print("🚀 异步反应堆 4 级流水线全部就位！")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
