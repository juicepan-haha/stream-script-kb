-- ============================================================
-- Ultimate Cloud Edition — fish_box 表结构加固
-- 在 Supabase SQL Editor 中运行
-- ============================================================

-- 1. status 字段：设默认值 'queued' + NOT NULL
ALTER TABLE public.fish_box
  ALTER COLUMN status SET DEFAULT 'queued',
  ALTER COLUMN status SET NOT NULL;

-- 2. result_text 字段：确保是 text 类型（能存大段话术 JSON + 人话日志）
ALTER TABLE public.fish_box
  ALTER COLUMN result_text TYPE text;

-- 3. 已有 NULL status 的行，统一补为 'queued'
UPDATE public.fish_box SET status = 'queued' WHERE status IS NULL;

-- 4. 确保有 created_at 列（超时检测依赖）
ALTER TABLE public.fish_box
  ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
