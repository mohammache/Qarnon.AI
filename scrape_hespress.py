#!/usr/bin/env python3
"""
Scrape Moroccan Darija comments from Hespress via WordPress REST API.
Saves raw comments to dataset/hespress_comments_raw.jsonl (incremental).

Usage:
    python scrape_hespress.py                  # default: 200 articles
    python scrape_hespress.py --articles 500
    python scrape_hespress.py --articles 500 --resume
"""

import json
import time
import random
import argparse
import re
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.hespress.com/wp-json/wp/v2"
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)
RAW_OUT = OUT_DIR / "hespress_comments_raw.jsonl"

TOTAL_POSTS = 527_710
PER_PAGE = 100

# Darija markers — Arabic-script words common in Darija but rare in MSA
DARIJA_MARKERS_AR = [
    "ديال", "ديالي", "ديالك", "ديالو", "ديالها", "ديالهم",
    "بزاف", "بززاف", "واش", "وش", "راه", "راك", "راني",
    "كيف داير", "مزيان", "مزيانة", "ماشي", "ماشيش",
    "هاد", "هادي", "هادا", "هاداك", "كاين", "كاينة",
    "خويا", "ختي", "عندي", "عندك", "عنده", "عندها",
    "كنقول", "كنعرف", "كنشوف", "كيتسمى", "كيگول",
    "غادي", "غير", "حتى", "فين", "منين",
]
DARIJA_MARKERS_LATIN = [
    "bzzaf", "bzaf", "wach", "dyal", "kayn", "machi",
    "hadi", "rah", "3ndi", "kif", "mezyan", "walo",
    "nta", "ntiya", "howa", "hiya", "7na",
]


def is_darija(text: str) -> bool:
    """Heuristic: check for Darija markers in Arabic or Latin script."""
    if len(text) < 25:
        return False
    tl = text.lower()
    for marker in DARIJA_MARKERS_AR:
        if marker in text:
            return True
    for marker in DARIJA_MARKERS_LATIN:
        if marker in tl:
            return True
    return False


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (compatible; MoroccoTone-research-scraper/1.0)"
    )
    return session


def fetch_json(session: requests.Session, url: str, params: dict) -> tuple[list, dict]:
    resp = session.get(url, params=params, timeout=20)
    resp.raise_for_status()
    total = int(resp.headers.get("X-WP-Total", 0))
    total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
    return resp.json(), {"total": total, "pages": total_pages}


def sample_post_pages(n_articles: int, total_posts: int) -> list[int]:
    """
    Sample page numbers across the full history for diversity.
    Blend of recent (low page numbers) and older (high page numbers).
    """
    max_page = total_posts // PER_PAGE
    n_pages_needed = (n_articles // PER_PAGE) + 10  # over-fetch to account for skips

    # 40% recent (pages 1-50), 60% spread across full history
    recent = random.sample(range(1, min(51, max_page)), min(n_pages_needed // 3, 50))
    spread = random.sample(range(51, max_page), min(n_pages_needed, max_page - 51))
    pages = list(set(recent + spread))
    random.shuffle(pages)
    return pages


def load_seen_ids() -> set[int]:
    if not RAW_OUT.exists():
        return set()
    seen = set()
    with open(RAW_OUT) as f:
        for line in f:
            try:
                obj = json.loads(line)
                seen.add(obj["article_id"])
            except Exception:
                pass
    return seen


def scrape(n_articles: int, resume: bool):
    session = make_session()
    seen_article_ids = load_seen_ids() if resume else set()

    if not resume and RAW_OUT.exists():
        RAW_OUT.unlink()

    pages = sample_post_pages(n_articles, TOTAL_POSTS)
    articles_done = 0
    total_comments = 0
    darija_comments = 0

    print(f"Target: {n_articles} articles | Resume: {resume}")
    print(f"Already seen: {len(seen_article_ids)} articles\n")

    with open(RAW_OUT, "a", encoding="utf-8") as out_f:
        for page_num in pages:
            if articles_done >= n_articles:
                break

            # Fetch article list for this page
            try:
                posts, _ = fetch_json(session, f"{BASE}/posts", {
                    "per_page": PER_PAGE,
                    "page": page_num,
                    "_fields": "id,title,categories,date,slug",
                })
            except Exception as e:
                print(f"  [skip page {page_num}] {e}")
                time.sleep(2)
                continue

            for post in posts:
                if articles_done >= n_articles:
                    break

                post_id = post["id"]
                if post_id in seen_article_ids:
                    continue

                title = strip_html(post.get("title", {}).get("rendered", ""))

                # Fetch comments for this article (up to 100)
                try:
                    comments_raw, meta = fetch_json(session, f"{BASE}/comments", {
                        "post": post_id,
                        "per_page": 100,
                        "page": 1,
                        "_fields": "id,content,author_name,date",
                    })
                except Exception as e:
                    print(f"  [skip article {post_id}] {e}")
                    time.sleep(1)
                    continue

                article_comments = []
                for c in comments_raw:
                    text = strip_html(c.get("content", "") or "")
                    if not text:
                        continue
                    total_comments += 1
                    if is_darija(text):
                        darija_comments += 1
                        article_comments.append({
                            "comment_id": c["id"],
                            "text": text,
                            "author": c.get("author_name", ""),
                            "date": c.get("date", ""),
                        })

                if article_comments:
                    record = {
                        "article_id": post_id,
                        "article_title": title,
                        "article_date": post.get("date", ""),
                        "article_slug": post.get("slug", ""),
                        "comments": article_comments,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()

                articles_done += 1
                seen_article_ids.add(post_id)

                if articles_done % 10 == 0:
                    pct = darija_comments / max(total_comments, 1) * 100
                    print(
                        f"  {articles_done}/{n_articles} articles | "
                        f"{darija_comments}/{total_comments} Darija comments ({pct:.0f}%)"
                    )

                time.sleep(0.4)

            time.sleep(0.3)

    print(f"\nDone.")
    print(f"  Articles scraped : {articles_done}")
    print(f"  Total comments   : {total_comments}")
    print(f"  Darija comments  : {darija_comments} ({darija_comments/max(total_comments,1)*100:.1f}%)")
    print(f"  Output           : {RAW_OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--articles", type=int, default=200,
                        help="Number of articles to scrape (default: 200)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()
    random.seed(42)
    scrape(args.articles, args.resume)


if __name__ == "__main__":
    main()
