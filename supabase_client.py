"""
supabase_client.py — 全局唯一的 Supabase 管理客户端

使用 service_role_key 初始化，拥有不受 RLS 限制的完整读写权限，
专门供后端异步任务默默更新 fish_box 表的状态与结果。

用法:
  from supabase_client import supabase
  supabase.table("fish_box").update({...}).eq("id", task_id).execute()
"""
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. 自动加载项目根目录下的 .env 文件
load_dotenv()

# 2. 从环境变量中读取核心密钥
supabase_url: str = os.getenv("SUPABASE_URL", "")
supabase_key: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

if not supabase_url or not supabase_key:
    raise ValueError(
        "❌ SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY 未设置！\n"
        "请检查项目根目录的 .env 文件，确保包含:\n"
        "  SUPABASE_URL=https://xxxxx.supabase.co\n"
        "  SUPABASE_SERVICE_ROLE_KEY=sb_secret_..."
    )

# 3. 创建全局唯一的 Supabase 客户端实例
supabase: Client = create_client(supabase_url, supabase_key)
print(f"[Supabase] ✅ 已连接: {supabase_url}")
