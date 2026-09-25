"""RAG 话术改写接口。"""

from fastapi import APIRouter

from streaming.services.rag import rewrite_script


router = APIRouter(prefix="/api/v1")


@router.post("/rewrite")
async def rewrite(my_product: str, target_style: str = "呐喊憋单流"):
    """根据历史知识库为新产品生成直播话术和执行 SOP。"""
    return await rewrite_script(my_product, target_style)
