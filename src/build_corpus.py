"""Download Korean and English Wikipedia articles about semiconductors and split them into chunks.

Usage:  python src/build_corpus.py
Writes data/chunks.jsonl  (one chunk per line: id, lang, title, url, text)
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "chunks.jsonl"

TITLES = {
    "ko": ["반도체", "반도체 소자 제조", "DRAM", "낸드 플래시", "플래시 메모리", "고대역폭 메모리", "웨이퍼",
           "포토리소그래피", "식각", "화학 기상 증착", "이온 주입", "수율", "트랜지스터", "MOSFET",
           "집적 회로", "SK하이닉스", "삼성전자", "극자외선 리소그래피", "반도체 패키징", "파운드리"],
    "en": ["Semiconductor", "Semiconductor device fabrication", "Dynamic random-access memory", "Flash memory",
           "High Bandwidth Memory", "Wafer (electronics)", "Photolithography", "Etching (microfabrication)",
           "Chemical vapor deposition", "Ion implantation", "Extreme ultraviolet lithography", "MOSFET",
           "Integrated circuit", "SK Hynix", "Semiconductor fabrication plant", "Die (integrated circuit)"],
}


def fetch(lang, title, retries=5):
    """Fetch one article's plain text. Wikipedia rate-limits fast clients (HTTP 429),
    so we wait and retry with growing delays, respecting the Retry-After header."""
    q = urllib.parse.urlencode({"action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                                "redirects": 1, "titles": title})
    req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}",
                                 headers={"User-Agent": "semiconductor-rag-portfolio/1.0 (student project)"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                pages = json.loads(r.read())["query"]["pages"]
            page = next(iter(pages.values()))
            return page.get("title", title), page.get("extract", "")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 10 * 2 ** attempt
            print(f"  rate-limited, waiting {wait}s...", flush=True)
            time.sleep(wait)


def chunk(text, size=600, overlap=100):
    """Split on paragraph boundaries into ~size-character chunks with some overlap."""
    paras = [p.strip() for p in re.split(r"\n+", text) if len(p.strip()) > 40 and not p.startswith("==")]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur.strip()); cur = cur[-overlap:]
        cur += " " + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def main():
    OUT.parent.mkdir(exist_ok=True)
    n, seen = 0, set()
    with OUT.open("w", encoding="utf-8") as f:
        for lang, titles in TITLES.items():
            for t in titles:
                time.sleep(1.5)  # be polite to Wikipedia: about one request every 1.5 s
                try:
                    title, text = fetch(lang, t)
                except Exception as e:  # keep going if one page fails
                    print(f"skip {lang}:{t} ({e})"); continue
                if (lang, title) in seen or not text:  # two titles can redirect to the same page
                    print(f"skip {lang}:{t} (duplicate or empty)"); continue
                seen.add((lang, title))
                url = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                for i, c in enumerate(chunk(text)):
                    f.write(json.dumps({"id": f"{lang}:{title}:{i}", "lang": lang, "title": title, "url": url, "text": c},
                                       ensure_ascii=False) + "\n"); n += 1
                print(f"{lang}: {title}", flush=True)
    print(f"{n} chunks from {len(seen)} articles -> {OUT}")

if __name__ == "__main__":
    main()
