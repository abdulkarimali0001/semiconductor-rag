"""Evaluate retrieval and answers on the reviewed test set (data/eval.jsonl).

Usage:  python src/evaluate.py [--judge llama3.1:8b]

Metrics
  Retrieval   hit@1, hit@5 (is the gold chunk among the top results?), MRR
              split into same-language vs cross-lingual questions
  Answers     correctness judged by a DIFFERENT model than the one answering (avoids
              a model grading itself), compared against the reference answer:
              correct / partial / wrong.  RAG vs the same model with no retrieval.
  Citations   share of RAG answers that cite the source containing the gold chunk
Writes results/metrics.json, results/answers.jsonl, results/eval_summary.png
"""
import argparse
import json
import re

import numpy as np

import ollama_client as ol
from rag import LLM, ROOT, Retriever, answer, answer_without_rag

R = ROOT / "results"
JUDGE_PROMPT = """You are grading an answer to a question about semiconductors.
Question: {q}
Reference answer: {ref}
Candidate answer: {cand}
Is the candidate answer factually consistent with the reference and does it answer the question?
Return JSON only: {{"verdict": "correct" | "partial" | "wrong"}}"""


def judge(model, q, ref, cand):
    r = ol.chat(model, [{"role": "user", "content": JUDGE_PROMPT.format(q=q, ref=ref, cand=cand)}],
                options={"temperature": 0, "seed": 1}, fmt="json")
    try:
        v = json.loads(r["message"]["content"]).get("verdict", "wrong")
    except json.JSONDecodeError:
        v = "wrong"
    return v if v in ("correct", "partial", "wrong") else "wrong"


def score(verdicts):
    return float(np.mean([{"correct": 1.0, "partial": 0.5, "wrong": 0.0}[v] for v in verdicts])) if verdicts else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--judge", default="llama3.1:8b"); ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    ol.ensure([LLM, args.judge, "bge-m3"])
    R.mkdir(exist_ok=True)
    items = [json.loads(line) for line in (ROOT / "data" / "eval.jsonl").open(encoding="utf-8")]
    ret = Retriever()
    rows = []
    # Pass 1: retrieval + answers (only the answer model and embedder are loaded)
    for i, it in enumerate(items, 1):
        hits = ret.search(it["question"], k=10)
        ids = [h["id"] for h in hits]
        rank = ids.index(it["gold_chunk"]) + 1 if it["gold_chunk"] in ids else None
        rag = answer(it["question"], ret, k=args.k)
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", rag["answer"])}
        gold_pos = next((n for n, h in enumerate(rag["sources"], 1) if h["id"] == it["gold_chunk"]), None)
        rows.append({**it, "rank": rank, "rag_answer": rag["answer"], "no_rag_answer": answer_without_rag(it["question"]),
                     "cited": sorted(cited), "cites_gold": bool(gold_pos and gold_pos in cited)})
        print(f"answering {i}/{len(items)}", end="\r", flush=True)
    # Pass 2: judging. Unload the other models first so an 8 GB Mac doesn't swap between models.
    ol.unload_all()
    for i, r in enumerate(rows, 1):
        r["rag_verdict"] = judge(args.judge, r["question"], r["answer"], r["rag_answer"])
        r["no_rag_verdict"] = judge(args.judge, r["question"], r["answer"], r["no_rag_answer"])
        print(f"judging {i}/{len(rows)}  ", end="\r", flush=True)

    def retrieval(sub):
        ranks = [r["rank"] for r in sub]
        return {"n": len(sub), "hit@1": float(np.mean([r == 1 for r in ranks])),
                "hit@5": float(np.mean([r is not None and r <= 5 for r in ranks])),
                "mrr": float(np.mean([1 / r if r else 0 for r in ranks]))}

    m = {
        "questions": len(rows),
        "retrieval_all": retrieval(rows),
        "retrieval_same_language": retrieval([r for r in rows if not r["cross_lingual"]]),
        "retrieval_cross_lingual": retrieval([r for r in rows if r["cross_lingual"]]),
        "answer_score_rag": score([r["rag_verdict"] for r in rows]),
        "answer_score_no_rag": score([r["no_rag_verdict"] for r in rows]),
        "rag_correct_rate": float(np.mean([r["rag_verdict"] == "correct" for r in rows])),
        "no_rag_correct_rate": float(np.mean([r["no_rag_verdict"] == "correct" for r in rows])),
        "citation_of_gold_source": float(np.mean([r["cites_gold"] for r in rows if r["rank"] and r["rank"] <= args.k] or [0])),
        "answer_model": LLM, "judge_model": args.judge, "embedding_model": "bge-m3", "k": args.k,
    }
    (R / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False))
    with (R / "answers.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    groups = ["same language", "cross-lingual"]
    for j, key in enumerate(["hit@1", "hit@5"]):
        ax[0].bar(np.arange(2) + j * 0.38, [m["retrieval_same_language"][key] * 100, m["retrieval_cross_lingual"][key] * 100], 0.36, label=key)
    ax[0].set_xticks(np.arange(2) + 0.19, groups); ax[0].set(title="Retrieval: gold passage found (%)", ylim=(0, 100)); ax[0].legend()
    ax[1].bar(["No retrieval", "RAG"], [m["no_rag_correct_rate"] * 100, m["rag_correct_rate"] * 100], color=["tab:gray", "tab:green"])
    ax[1].set(title="Answers judged correct (%)", ylim=(0, 100))
    for a in ax:
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(R / "eval_summary.png", dpi=150)
    print("\n" + json.dumps(m, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
