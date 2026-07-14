-- ============================================================
-- Ultimate Cloud Edition — 原子卡密兑换系统 + 用户素材库
-- 在 Supabase SQL Editor 中运行
-- ============================================================

-- 1. 建卡密表
CREATE TABLE IF NOT EXISTS public.recharge_cards (
    id SERIAL PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    max_uses INT NOT NULL DEFAULT 1,
    used INT NOT NULL DEFAULT 0,
    bonus_times INT NOT NULL DEFAULT 5,
    bonus_vip_days INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. 种子卡密（测试用）
INSERT INTO public.recharge_cards (code, max_uses, bonus_times)
VALUES
    ('BETA-TEST-001', 999, 50),
    ('DEMO-FREE-2026', 50, 50),
    ('VIP-ALPHA-100', 100, 100)
ON CONFLICT (code) DO NOTHING;

-- 3. 原子兑换函数（SELECT FOR UPDATE 排他锁，防无限套现）
CREATE OR REPLACE FUNCTION atomic_redeem_card(
    p_code TEXT,
    p_user_id UUID
) RETURNS TABLE(success BOOLEAN, bonus_times INT, bonus_vip_days INT, message TEXT) AS $$
DECLARE
    card RECORD;
    profile RECORD;
BEGIN
    -- Step 1: 排他锁锁定卡密行，防止并发
    SELECT * INTO card
    FROM public.recharge_cards
    WHERE code = p_code
    FOR UPDATE;

    IF NOT FOUND THEN
        RETURN QUERY SELECT FALSE, 0, 0, '卡密不存在'::TEXT;
        RETURN;
    END IF;

    -- Step 2: 检查是否还有可用次数
    IF card.used >= card.max_uses THEN
        RETURN QUERY SELECT FALSE, 0, 0, '卡密已被用完'::TEXT;
        RETURN;
    END IF;

    -- Step 3: 原子扣减卡密次数
    UPDATE public.recharge_cards
    SET used = used + 1
    WHERE id = card.id;

    -- Step 4: 为用户加额度/VIP
    INSERT INTO public.user_profiles (id, balance_count)
    VALUES (p_user_id, card.bonus_times)
    ON CONFLICT (id) DO UPDATE
    SET balance_count = user_profiles.balance_count + card.bonus_times;

    -- Step 5: 如果卡密带 VIP 天数
    IF card.bonus_vip_days > 0 THEN
        UPDATE public.user_profiles
        SET vip_until = COALESCE(
            GREATEST(vip_until, NOW()),
            NOW()
        ) + (card.bonus_vip_days || ' days')::INTERVAL
        WHERE id = p_user_id;
    END IF;

    RETURN QUERY SELECT TRUE, card.bonus_times, card.bonus_vip_days,
        ('兑换成功！+' || card.bonus_times || ' 次额度')::TEXT;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;


-- ============================================================
-- 4. 用户结构化素材库（历史重写免 GPU 成本）
-- ============================================================
CREATE TABLE IF NOT EXISTS public.user_materials (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    title TEXT DEFAULT '',
    tags TEXT[] DEFAULT '{}',
    video_url TEXT DEFAULT '',
    source_task_id INT DEFAULT NULL,
    key_selling_points JSONB DEFAULT '[]'::jsonb,
    full_script TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RLS
ALTER TABLE public.user_materials ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own materials"
ON public.user_materials FOR SELECT
TO authenticated
USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own materials"
ON public.user_materials FOR INSERT
TO authenticated
WITH CHECK (auth.uid() = user_id);
