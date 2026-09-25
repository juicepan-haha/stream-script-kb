"""实时分析任务和结果查询接口。"""

import time

from fastapi import APIRouter

from streaming.state import (
    chunk_queue,
    download_queue,
    enriched_queue,
    enriched_results,
    task_progress,
    text_queue,
    transcript_results,
)


router = APIRouter(prefix="/api/v1")


@router.post("/analyze")
async def start_analysis(url: str):
    """接收直播地址并把任务送入流式流水线。"""
    task_id = f"task_{int(time.time() * 1000)}"
    await download_queue.put({"url": url, "task_id": task_id})
    return {
        "status": "accepted",
        "task_id": task_id,
        "message": "已送入流式传送带，全程无盘化分析中...",
    }


@router.get("/transcript/{task_id}")
async def get_transcript(task_id: str):
    """查询原始转录结果。"""
    segments = transcript_results.get(task_id, [])
    return {
        "task_id": task_id,
        "segments": len(segments),
        "text": "".join(segment["text"] for segment in segments),
        "details": segments,
    }


@router.get("/enriched/{task_id}")
async def get_enriched(task_id: str):
    """查询 DeepSeek 富化后的话术。"""
    items = enriched_results.get(task_id, [])
    return {"task_id": task_id, "chunks": len(items), "results": items}


@router.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """查询任务当前处理阶段。"""
    progress = task_progress.get(task_id, {})
    return {
        "task_id": task_id,
        "stage": progress.get("stage", "unknown"),
        "transcript_segments": len(transcript_results.get(task_id, [])),
        "enriched_chunks": len(enriched_results.get(task_id, [])),
    }


@router.get("/health")
async def health():
    """返回队列积压和当前任务概况。"""
    return {
        "download_queue": download_queue.qsize(),
        "text_queue": text_queue.qsize(),
        "chunk_queue": chunk_queue.qsize(),
        "enriched_queue": enriched_queue.qsize(),
        "active_tasks": list(transcript_results.keys()),
    }
