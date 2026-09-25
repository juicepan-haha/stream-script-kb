"""Tests for the vectorization database configuration."""
from shared import config


def test_db_config_present():
    """数据库配置项存在且非空。"""
    assert config.PG_HOST is not None
    assert config.PG_PORT is not None
    assert config.PG_DB is not None
    assert len(config.PG_DB) > 0
