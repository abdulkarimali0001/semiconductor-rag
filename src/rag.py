"""Korean-English RAG: embed chunks, retrieve the most relevant ones, answer with citations.

Usage:
    python src/rag.py index                       # embed data/chunks.jsonl -> data/index.npz
    python src/rag.py ask "HBM은 왜 AI에 필요한가요?"
    python src/rag.py ask "What is EUV lithography?" --k 5

Design:
  - bge-m3 embeddings: one multilingual model, so a Korean question can find an English
    passage and vice versa (cross-lingual retrieval).
  - Plain NumPy cosine search: the corpus is a few thousand chunks, so a vector database
    would add complexity without benefit. Swapping in FAISS or Chroma later is easy.
  - The prompt forces the model to cite sources as [1], [2] and to say when the
    sources don't contain the answer, which reduces hallucination.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ollama_client as ol

ROOT = Path(__file__).resolve().parent.parent
CHUNKS, INDEX = ROOT / "data" / "chunks.jsonl", ROOT / "data" / "index.npz"
EMBED_MODEL = "bge-m3"
LLM = "qwen2.5:3b"

SYSTEM = """You answer questions about semiconductors using ONLY the numbered sources provided.
Rules:
- Answer in the same language as the question (Korean question -> Korean answer).
- After each fact, cite the source number in square brackets, e.g. [2].
- If the sources do not contain the answer, say so plainly (in Korean: "제공된 자료에서 답을 찾을 수 없습니다.").
- Be concise: at most 4 sentences."""


def load_chunks():
    return [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]


def build_index(batch=32):
    chunks = load_chunks()
    vecs = []
    for i in range(0, len(chunks), batch):
        vecs += ol.embed(EMBED_MODEL, [c["text"] for c in chunks[i:i + batch]])
        print(f"embedded {min(i + batch, len(chunks))}/{len(chunks)}", end="\r", flush=True)
    v = np.asarray(vecs, dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.savez(INDEX, vectors=v, ids=np.array([c["id"] for c in chunks]))
    print(f"\nindex: {v.shape[0]} chunks x {v.shape[1]} dims -> {INDEX}")


class Retriever:
    def __init__(self):
        d = np.load(INDEX)
        self.v, self.ids = d["vectors"], list(d["ids"])
        by_id = {c["id"]: c for c in load_chunks()}
        self.chunks = [by_id[i] for i in self.ids]

    def search(self, query, k=5):
        q = np.asarray(ol.embed(EMBED_MODEL, [query])[0], dtype=np.float32)
        scores = self.v @ (q / np.linalg.norm(q))
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def answer(question, retriever, k=5, model=LLM):
    hits = retriever.search(question, k)
    sources = "\n\n".join(f"[{n}] ({h['title']}) {h['text']}" for n, h in enumerate(hits, 1))
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"}]
    text = ol.chat(model, msg, options={"temperature": 0, "seed": 1})["message"]["content"]
    return {"question": question, "answer": text, "sources": hits}


def answer_without_rag(question, model=LLM):
    """Baseline: the same model with no sources, to measure what retrieval adds."""
    msg = [{"role": "user", "content": f"Answer concisely in the question's language.\n\nQuestion: {question}"}]
    return ol.chat(model, msg, options={"temperature": 0, "seed": 1})["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    a = sub.add_parser("ask"); a.add_argument("question"); a.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "index":
        ol.ensure([EMBED_MODEL]); build_index()
    else:
        ol.ensure([EMBED_MODEL, LLM])
        r = answer(args.question, Retriever(), args.k)
        print(r["answer"], "\n\nSources:")
        for n, h in enumerate(r["sources"], 1):
            print(f"[{n}] {h['title']} ({h['lang']}, score {h['score']:.2f})  {h['url']}")


if __name__ == "__main__":
    main()
