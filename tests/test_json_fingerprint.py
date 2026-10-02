import json

from signalcast.data import json_sha256


def test_news_fingerprint_survives_git_line_endings_and_json_formatting(tmp_path):
    path = tmp_path / "news.json"
    content = {"published_at": "2024-06-07", "title": "Решение Банка России", "rate": 16.0}
    path.write_bytes(json.dumps(content, ensure_ascii=False, indent=2).replace("\n", "\r\n").encode("utf-8"))
    before = json_sha256(path)
    reformatted = dict(reversed(list(content.items())))
    path.write_bytes((json.dumps(reformatted, ensure_ascii=False) + "\n").encode("utf-8"))
    assert json_sha256(path) == before


def test_news_fingerprint_changes_when_available_information_changes(tmp_path):
    path = tmp_path / "news.json"
    content = {"published_at": "2024-06-07", "rate": 16.0}
    path.write_text(json.dumps(content), encoding="utf-8")
    before = json_sha256(path)
    content["published_at"] = "2024-07-26"
    path.write_text(json.dumps(content), encoding="utf-8")
    assert json_sha256(path) != before
