"""Tests for step2_chunk.py"""
from step2_chunk import parse_vtt, build_chunks


def test_parse_vtt_single_entry():
    vtt = """WEBVTT

00:00:01.000 --> 00:00:03.500
大家好欢迎来到直播间

00:00:04.000 --> 00:00:08.200
今天给大家带来一款非常好用的产品
"""
    entries = parse_vtt(vtt)
    assert len(entries) == 2
    assert entries[0]["text"] == "大家好欢迎来到直播间"
    assert entries[0]["start"] == 1.0
    assert entries[0]["end"] == 3.5


def test_parse_vtt_empty():
    entries = parse_vtt("WEBVTT\n")
    assert entries == []


def test_build_chunks_single():
    long_text = "这是一个测试句子。" * 80  # ~640 chars
    entries = [{"start": 0.0, "end": 5.0, "text": long_text}]
    chunks = build_chunks(entries, source_file="test.vtt", min_chars=400, max_chars=800)
    assert len(chunks) >= 1
    assert 400 <= chunks[0]["char_count"] <= 800
    assert chunks[0]["source_file"] == "test.vtt"


def test_build_chunks_merges_short_tail():
    entries = [
        {"start": 0.0, "end": 2.0, "text": "A" * 600 + "。"},
        {"start": 3.0, "end": 4.0, "text": "B" * 50},
    ]
    chunks = build_chunks(entries, source_file="t.vtt", min_chars=500, max_chars=1000)
    assert len(chunks) == 1
    assert "B" * 50 in chunks[0]["text"]


def test_chunk_id_format():
    entries = [{"start": 0.0, "end": 2.0, "text": "X" * 600 + "。"}]
    chunks = build_chunks(entries, source_file="part_003.vtt", min_chars=500, max_chars=1000)
    assert chunks[0]["chunk_id"] == "part_003_001"
