"""Tests for step4_vectorize.py"""
from step4_vectorize import DB_CONFIG


def test_db_config_present():
    """数据库配置项存在且非空。"""
    assert DB_CONFIG["host"] is not None
    assert DB_CONFIG["port"] is not None
    assert DB_CONFIG["dbname"] is not None
    assert len(DB_CONFIG["dbname"]) > 0
