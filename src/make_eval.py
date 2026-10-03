"""Draft an evaluation set: one question + reference answer per sampled chunk.

Usage:  python src/make_eval.py --n 60
Writes data/eval_draft.jsonl. REVIEW IT BY HAND: delete bad questions, fix answers,
then save the result as data/eval.jsonl. A human-checked test set is what makes the
evaluation trustworthy.

Half the questions are written in the OTHER language from their source chunk
(Korean question for an English passage and vice versa) to test cross-lingual retrieval.
"""
import argparse
import json
import random

import ollama_client as ol
from rag import CHUNKS, LLM, ROOT, load_chunks

PROMPT = """Read the passage and write ONE factual question that the passage clearly answers, plus a short answer.
The question must make sense on its own (do not say "the passage" or "this text").
Write the question and the answer in {lang}.
Return JSON only: {{"question": "...", "answer": "..."}}

Passage:
{text}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ol.ensure([LLM])
    chunks = [c for c in load_chunks() if len(c["text"]) > 300]
    random.seed(args.seed)
    picked = random.sample(chunks, min(args.n, len(chunks)))
    out = ROOT / "data" / "eval_draft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(picked):
            cross = i % 2 == 1
            q_lang = {"ko": "English", "en": "Korean"}[c["lang"]] if cross else {"ko": "Korean", "en": "English"}[c["lang"]]
            r = ol.chat(LLM, [{"role": "user", "content": PROMPT.format(lang=q_lang, text=c["text"])}],
                        options={"temperature": 0.2, "seed": i}, fmt="json")
            try:
                qa = json.loads(r["message"]["content"])
            except json.JSONDecodeError:
                continue
            f.write(json.dumps({"question": qa.get("question", ""), "answer": qa.get("answer", ""), "gold_chunk": c["id"],
                                "question_lang": "ko" if q_lang == "Korean" else "en", "source_lang": c["lang"],
                                "cross_lingual": cross}, ensure_ascii=False) + "\n")
            print(f"{i + 1}/{len(picked)}", end="\r", flush=True)
    print(f"\ndraft written to {out}. Review it, then save as data/eval.jsonl  (chunks from {CHUNKS.name})")


if __name__ == "__main__":
    main()
