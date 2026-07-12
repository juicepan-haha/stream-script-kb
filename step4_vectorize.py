#!/usr/bin/env python3
"""Step 4: 对 refined_script 生成 embedding 并写入 PostgreSQL。

输入: data/enriched.json
输出: PostgreSQL 表 scripts (含 embedding 向量和 HNSW 索引)

前置条件:
  - PostgreSQL 已安装运行
  - pgvector 扩展已可用
  - 数据库 stream_scripts 已存在 (脚本自动创建)
"""
import json
import sys

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

import config

DB_CONFIG = {
    "host": config.PG_HOST,
    "port": config.PG_PORT,
    "user": config.PG_USER,
    "password": config.PG_PASSWORD,
    "dbname": config.PG_DB,
}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS scripts (
    id SERIAL PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    source_file TEXT DEFAULT '',
    start_time REAL DEFAULT 0.0,
    end_time REAL DEFAULT 0.0,
    refined_script TEXT NOT NULL,
    summary TEXT DEFAULT '',
    sales_stage TEXT DEFAULT '',
    strategy_types TEXT DEFAULT '[]',
    product_mentions TEXT DEFAULT '[]',
    selling_points TEXT DEFAULT '[]',
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


def get_connection():
    """获取 PostgreSQL 连接（不注册 vector 类型，等建表后再注册）。"""
    conn = psycopg2.connect(**DB_CONFIG)
    return conn


def ensure_db():
    """确保数据库存在。如果不存在则创建。"""
    admin_config = {**DB_CONFIG, "dbname": "postgres"}
    conn = psycopg2.connect(**admin_config)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (config.PG_DB,),
    )
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {config.PG_DB}")
        print(f"[STEP4] Created database: {config.PG_DB}")
    cur.close()
    conn.close()


def create_table(conn):
    """创建表和索引 (幂等)。"""
    cur = conn.cursor()
    cur.execute(SCHEMA_SQL)
    cur.execute(INDEX_SQL)
    conn.commit()
    cur.close()
    print("[STEP4] Schema and indexes ensured")


def main():
    print("[STEP4] Loading enriched data...")
    with open(config.ENRICHED_FILE, "r", encoding="utf-8") as f:
        enriched = json.load(f)
    total = len(enriched)
    print(f"[STEP4] {total} records to process")

    if total == 0:
        print("[STEP4] No records. Run step3_deepseek.py first.")
        return

    print("[STEP4] Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("[STEP4] Generating embeddings...")
    refined_texts = [item["refined_script"] for item in enriched]
    embeddings = model.encode(
        refined_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vecs = [e.tolist() for e in embeddings]
    print(f"[STEP4] {len(vecs)} embeddings generated (dim={len(vecs[0])})")

    print("[STEP4] Setting up database...")
    ensure_db()
    conn = get_connection()
    create_table(conn)
    # pgvector 类型在 CREATE EXTENSION 之后才可用
    register_vector(conn)

    print("[STEP4] Inserting records...")
    cur = conn.cursor()
    inserted = 0
    for item, vec in zip(enriched, vecs):
        cur.execute(
            """INSERT INTO scripts
               (chunk_id, source_file, start_time, end_time,
                refined_script, summary, sales_stage,
                strategy_types, product_mentions, selling_points,
                target_audience, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                item["chunk_id"],
                item["source_file"],
                item["start_time"],
                item["end_time"],
                item["refined_script"],
                item["summary"],
                item["sales_stage"],
                json.dumps(item.get("strategy_types", []), ensure_ascii=False),
                json.dumps(item.get("product_mentions", []), ensure_ascii=False),
                json.dumps(item.get("selling_points", []), ensure_ascii=False),
                item.get("target_audience", ""),
                vec,
            ),
        )
        inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"[STEP4] Done. {inserted} records inserted into {config.PG_DB}.scripts")
    print(f"[STEP4] Verify: psql -d {config.PG_DB} -c 'SELECT COUNT(*) FROM scripts;'")


if __name__ == "__main__":
    main()
