"""流式服务拆分后的纯函数测试。"""

from streaming.services.rag import build_rag_context
from streaming.workers.download import parse_cookies
from streaming.workers.enrichment import extract_critical_keywords


def test_parse_cookies(tmp_path):
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text(
        ".example.com\tTRUE\t/\tFALSE\t0\tsession\tabc123\n",
        encoding="utf-8",
    )

    assert parse_cookies(str(cookie_file)) == "session=abc123"


def test_extract_critical_keywords_keeps_price_and_offer():
    keywords = extract_critical_keywords("原价199元，今天到手99元，买2送1。")

    assert "199元" in keywords
    assert "99元" in keywords
    assert "买2" in keywords


def test_build_rag_context_contains_retrieved_fields():
    context = build_rag_context([
        {
            "similarity": 0.91,
            "sales_stage": "逼单",
            "strategy_types": ["稀缺感"],
            "product_mentions": ["不粘锅"],
            "icebreaker": "先留人",
            "painpoint": "普通锅容易粘",
            "mechanism": "涂层导热均匀",
            "close_order": "最后三单",
        }
    ])

    assert "相似度 91%" in context
    assert "不粘锅" in context
    assert "最后三单" in context
