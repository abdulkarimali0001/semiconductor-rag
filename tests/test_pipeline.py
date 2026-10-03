"""End-to-end test with a fake Ollama server and a tiny corpus.  Run: pytest -q"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
import mock_ollama  # noqa: E402

srv, url = mock_ollama.start()
os.environ["OLLAMA_HOST"] = url

import build_corpus  # noqa: E402
import rag  # noqa: E402

DOCS = [
    ("ko", "고대역폭 메모리", "HBM 고대역폭 메모리 는 DRAM 을 수직으로 쌓아 대역폭을 높인 메모리 이다. " * 4),
    ("en", "Photolithography", "Photolithography transfers a circuit pattern onto a wafer using light and a photoresist. " * 4),
    ("en", "Etching", "Etching removes material from the wafer surface after lithography defines the pattern. " * 4),
]


def setup_corpus(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks.jsonl"; index = tmp_path / "index.npz"
    with chunks.open("w", encoding="utf-8") as f:
        for i, (lang, title, text) in enumerate(DOCS):
            f.write(json.dumps({"id": f"{lang}:{title}:0", "lang": lang, "title": title, "url": "u", "text": text}, ensure_ascii=False) + "\n")
    monkeypatch.setattr(rag, "CHUNKS", chunks); monkeypatch.setattr(rag, "INDEX", index)
    return chunks


def test_chunking_respects_size():
    text = "\n".join(["문장 " * 60] * 10)
    parts = build_corpus.chunk(text, size=600)
    assert len(parts) > 1 and all(len(p) < 900 for p in parts)


def test_index_retrieve_answer(tmp_path, monkeypatch):
    setup_corpus(tmp_path, monkeypatch)
    rag.build_index()
    r = rag.Retriever()
    hits = r.search("Photolithography transfers a circuit pattern onto a wafer", k=2)
    assert hits[0]["title"] == "Photolithography"
    out = rag.answer("What is photolithography?", r, k=2)
    assert out["answer"] and len(out["sources"]) == 2


def test_evaluate_runs(tmp_path, monkeypatch):
    setup_corpus(tmp_path, monkeypatch)
    rag.build_index()
    data = tmp_path / "data"; data.mkdir()
    (data / "eval.jsonl").write_text(json.dumps({"question": "Photolithography transfers a circuit pattern onto a wafer?",
                                                 "answer": "light", "gold_chunk": "en:Photolithography:0",
                                                 "question_lang": "en", "source_lang": "en", "cross_lingual": False}) + "\n" +
                                     json.dumps({"question": "HBM 고대역폭 메모리 는 무엇인가?", "answer": "stacked DRAM",
                                                 "gold_chunk": "ko:고대역폭 메모리:0", "question_lang": "ko", "source_lang": "ko",
                                                 "cross_lingual": True}, ensure_ascii=False) + "\n", encoding="utf-8")
    import evaluate
    monkeypatch.setattr(evaluate, "ROOT", tmp_path); monkeypatch.setattr(evaluate, "R", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--judge", "mock"])
    evaluate.main()
    m = json.loads((tmp_path / "results" / "metrics.json").read_text())
    assert m["questions"] == 2 and 0 <= m["retrieval_all"]["hit@5"] <= 1


def test_make_eval_draft(tmp_path, monkeypatch):
    setup_corpus(tmp_path, monkeypatch)
    import make_eval
    monkeypatch.setattr(make_eval, "ROOT", tmp_path); (tmp_path / "data").mkdir(exist_ok=True)
    monkeypatch.setattr(make_eval, "load_chunks", rag.load_chunks)
    monkeypatch.setattr(sys, "argv", ["make_eval", "--n", "2"])
    make_eval.main()
    rows = [json.loads(x) for x in (tmp_path / "data" / "eval_draft.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[1]["cross_lingual"] and rows[0]["question"]
