#!/usr/bin/env python3
"""Step 4: 工业级流式向量化并批量写入 PostgreSQL。

规范化改进：
  1. 流式分批生成 Embedding + 批量插入（Bulk Insert），内存可控
  2. ON CONFLICT 幂等写入，支持重跑不重复
  3. 单批次异常隔离，批次回滚后继续处理其他批次
  4. JSONB 替代 TEXT，支持原生 JSON 查询操作符
  5. 全链 Context Manager，杜绝连接泄露
  6. chunk_id UNIQUE 约束，数据完整性保障
"""
import json
import sys
from pathlib import Path
from typing import Any

import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

import config

# ====== 配置 ======

BATCH_SIZE = 256                     # 每批处理条数，低配机器可调小

# ====== 建表（JSONB + UNIQUE）======

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scripts (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    source_file TEXT DEFAULT '',
    start_time REAL DEFAULT 0.0,
    end_time REAL DEFAULT 0.0,

    -- 黄金四段式话术切片（对齐 Step 3 输出）
    icebreaker TEXT DEFAULT '',
    painpoint TEXT DEFAULT '',
    mechanism TEXT DEFAULT '',
    close_order TEXT DEFAULT '',

    refined_script TEXT NOT NULL,
    summary TEXT DEFAULT '',
    sales_stage TEXT DEFAULT '',
    strategy_types JSONB DEFAULT '[]'::jsonb,
    product_mentions JSONB DEFAULT '[]'::jsonb,
    selling_points JSONB DEFAULT '[]'::jsonb,
    target_audience TEXT DEFAULT '',
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_scripts_sales_stage ON scripts(sales_stage);
CREATE INDEX IF NOT EXISTS idx_scripts_source_file ON scripts(source_file);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes WHERE indexname = 'idx_scripts_embedding_hnsw'
    ) THEN
        EXECUTE 'CREATE INDEX idx_scripts_embedding_hnsw ON scripts
                 USING hnsw (embedding vector_cosine_ops)
                 WITH (m = 16, ef_construction = 200)';
    END IF;
END $$;
"""

INSERT_QUERY = """INSERT INTO scripts
    (chunk_id, source_file, start_time, end_time,
     icebreaker, painpoint, mechanism, close_order,
     refined_script, summary, sales_stage,
     strategy_types, product_mentions, selling_points,
     target_audience, embedding)
    VALUES %s
    ON CONFLICT (chunk_id) DO UPDATE SET
        icebreaker       = EXCLUDED.icebreaker,
        painpoint        = EXCLUDED.painpoint,
        mechanism        = EXCLUDED.mechanism,
        close_order      = EXCLUDED.close_order,
        refined_script   = EXCLUDED.refined_script,
        summary          = EXCLUDED.summary,
        sales_stage      = EXCLUDED.sales_stage,
        strategy_types   = EXCLUDED.strategy_types,
        product_mentions = EXCLUDED.product_mentions,
        selling_points   = EXCLUDED.selling_points,
        target_audience  = EXCLUDED.target_audience,
        embedding        = EXCLUDED.embedding"""


# ====== 数据库工具 ======

def ensure_db() -> None:
    """确保目标数据库存在。"""
    conn = psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT,
        user=config.PG_USER, password=config.PG_PASSWORD,
        dbname="postgres",
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (config.PG_DB,),
            )
            if not cur.fetchone():
                cur.execute(f'CREATE DATABASE "{config.PG_DB}"')
                print(f"[STEP4] Created database: {config.PG_DB}")
    finally:
        conn.close()


def init_schema() -> None:
    """初始化表结构和 HNSW 索引。"""
    ensure_db()
    with psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT,
        user=config.PG_USER, password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(INDEX_SQL)
    print("[STEP4] Schema and indexes ready")


# ====== 流式分批处理 ======

def process_batches(enriched: list[dict[str, Any]], model: SentenceTransformer) -> int:
    """流式分批：生成 embedding → 批量插入，单批异常隔离。"""
    total = len(enriched)
    inserted = 0
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    with psycopg2.connect(
        host=config.PG_HOST, port=config.PG_PORT,
        user=config.PG_USER, password=config.PG_PASSWORD,
        dbname=config.PG_DB,
    ) as conn:
        register_vector(conn)

        for batch_idx in range(0, total, BATCH_SIZE):
            batch = enriched[batch_idx:batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1
            print(f"[STEP4] Batch {batch_num}/{num_batches} "
                  f"({len(batch)} records) ...", end=" ", flush=True)

            try:
                # 1. 生成 embedding
                texts = [item.get("refined_script", "") for item in batch]
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                # 2. 构造批量数据
                rows = []
                for item, emb in zip(batch, embeddings):
                    rows.append((
                        item["chunk_id"],
                        item.get("source_file", ""),
                        float(item.get("start_time", 0.0)),
                        float(item.get("end_time", 0.0)),
                        item.get("icebreaker", ""),
                        item.get("painpoint", ""),
                        item.get("mechanism", ""),
                        item.get("close_order", ""),
                        item.get("refined_script", ""),
                        item.get("summary", ""),
                        item.get("sales_stage", ""),
                        json.dumps(item.get("strategy_types", []), ensure_ascii=False),
                        json.dumps(item.get("product_mentions", []), ensure_ascii=False),
                        json.dumps(item.get("selling_points", []), ensure_ascii=False),
                        item.get("target_audience", ""),
                        emb.tolist(),
                    ))

                # 3. 批量写入（ON CONFLICT 幂等）
                with conn.cursor() as cur:
                    execute_values(cur, INSERT_QUERY, rows)
                conn.commit()

                inserted += len(batch)
                print("OK")

            except Exception as e:
                conn.rollback()
                print(f"SKIP ({e})", file=sys.stderr)
                continue

    return inserted


# ====== 主入口 ======

def main():
    print("[STEP4] Loading enriched data...")
    enriched_file = Path(config.ENRICHED_FILE)
    if not enriched_file.exists():
        print(f"[STEP4] ERROR: {config.ENRICHED_FILE} not found. "
              f"Run step3_deepseek.py first.", file=sys.stderr)
        sys.exit(1)

    with open(enriched_file, "r", encoding="utf-8") as f:
        enriched = json.load(f)

    total = len(enriched)
    print(f"[STEP4] {total} records loaded")

    if total == 0:
        print("[STEP4] No records. Run step3_deepseek.py first.")
        return

    init_schema()

    print(f"[STEP4] Loading model: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    inserted = process_batches(enriched, model)

    print(f"[STEP4] Done. {inserted}/{total} records inserted "
          f"into {config.PG_DB}.scripts")
    print(f"[STEP4] Verify: psql -d {config.PG_DB} "
          f"-c 'SELECT COUNT(*) FROM scripts;'")


if __name__ == "__main__":
    main()
