#!/usr/bin/env python3
"""
Fetch all Moroccan Arabic (ary) sentences from Tatoeba API.
Produces: dataset/tatoeba_ary.jsonl
License: CC BY 2.0 FR (https://creativecommons.org/licenses/by/2.0/fr/)
"""

import json
import time
import requests
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Moroccan Darija linguistic assistant specialized in translation, "
    "spelling normalization, grammar, and cultural interpretation of Moroccan Arabic "
    "(Darija), which is frequently mixed with French and Modern Standard Arabic."
)

API_BASE = "https://tatoeba.org/en/api_v0/search"
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)


def ex(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# Only languages relevant to Moroccan communication context
LANG_NAMES = {
    "eng": "English",
    "fra": "French",
    "ara": "Modern Standard Arabic",
    "spa": "Spanish",
}


def fetch_page(page: int, target: str = "") -> dict:
    params = {"from": "ary", "page": page, "orphans": "no", "unapproved": "no"}
    if target:
        params["to"] = target
    resp = requests.get(API_BASE, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_sentences() -> list[dict]:
    """Fetch every ary sentence (no target filter = all translations included)."""
    sentences = []
    page = 1
    while True:
        data = fetch_page(page)
        results = data.get("results", [])
        if not results:
            break
        sentences.extend(results)
        paging = data.get("paging", {}).get("Sentences", {})
        print(f"  page {page}/{paging.get('pageCount', '?')}  "
              f"({len(sentences)}/{paging.get('count', '?')} sentences)")
        if not paging.get("nextPage"):
            break
        page += 1
        time.sleep(0.5)
    return sentences


def build_examples(sentences: list[dict]) -> list[dict]:
    examples = []
    seen_ary = set()

    for sent in sentences:
        ary_text = sent.get("text", "").strip()
        if not ary_text or ary_text in seen_ary:
            continue
        seen_ary.add(ary_text)

        # Flatten direct + indirect translations
        all_translations = []
        for group in sent.get("translations", []):
            all_translations.extend(group)

        # Group translations by lang — only keep relevant languages
        by_lang: dict[str, list[str]] = {}
        for tr in all_translations:
            lang = tr.get("lang", "")
            text = tr.get("text", "").strip()
            if lang in LANG_NAMES and text:
                by_lang.setdefault(lang, []).append(text)

        if not by_lang:
            continue

        # For each available translation language, build instruction pairs
        for lang, texts in by_lang.items():
            target_text = texts[0]  # use first (most direct)
            lang_name = LANG_NAMES.get(lang, lang)

            # ary → target
            examples.append(ex(
                f"Translate this Moroccan Darija sentence to {lang_name}: {ary_text}",
                target_text
            ))
            # target → ary  (only for eng/fra, most useful)
            if lang in ("eng", "fra"):
                examples.append(ex(
                    f"Translate this {lang_name} sentence to Moroccan Darija: {target_text}",
                    ary_text
                ))

        # Multi-language example: show all available translations
        if len(by_lang) >= 2:
            translations_block = "\n".join(
                f"  {LANG_NAMES.get(l, l)}: {ts[0]}"
                for l, ts in by_lang.items()
            )
            examples.append(ex(
                f"Give translations of this Moroccan Darija sentence in all available languages: {ary_text}",
                translations_block
            ))

    return examples


def main():
    print("Fetching Tatoeba ary sentences...")
    sentences = fetch_all_sentences()
    print(f"\nFetched {len(sentences)} unique sentences")

    examples = build_examples(sentences)
    print(f"Built {len(examples)} instruction examples")

    out_path = OUT_DIR / "tatoeba_ary.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Also save raw sentences for reference
    raw_path = OUT_DIR / "tatoeba_ary_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)

    print(f"\nOutput: {out_path} ({len(examples)} examples)")
    print(f"Raw:    {raw_path} ({len(sentences)} sentences)")


if __name__ == "__main__":
    main()
