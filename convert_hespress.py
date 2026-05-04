#!/usr/bin/env python3
"""
Convert raw Hespress comments → instruction JSONL using heuristic tasks.
Input : dataset/hespress_comments_raw.jsonl
Output: dataset/hespress_instructions.jsonl
"""

import json
import re
import random
from pathlib import Path

OUT_DIR = Path("dataset")
SYSTEM_PROMPT = (
    "You are a Moroccan Darija linguistic assistant specialized in translation, "
    "spelling normalization, grammar, and cultural interpretation of Moroccan Arabic "
    "(Darija), which is frequently mixed with French and Modern Standard Arabic."
)

# ── Language detection helpers ───────────────────────────────────────────────

ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿ]")
LATIN_RANGE  = re.compile(r"[a-zA-ZÀ-ÿ]")

DARIJA_MARKERS = [
    "ديال", "ديالي", "ديالك", "ديالو", "ديالها", "ديالهم",
    "بزاف", "بززاف", "واش", "راه", "راك", "راني", "راهم",
    "مزيان", "مزيانة", "ماشي", "ماشيش", "هاد", "هادي",
    "كاين", "كاينة", "غادي", "خويا", "ختي", "باش", "حيث",
    "كنقول", "كنعرف", "كيفاش",
]

FRENCH_WORDS = {
    "le", "la", "les", "de", "du", "des", "est", "pas", "que",
    "pour", "avec", "dans", "sur", "par", "qui", "mais", "ou",
    "donc", "car", "comme", "si", "tout", "très", "bien", "aussi",
    "même", "plus", "encore", "toujours", "jamais", "peut", "faut",
    "voilà", "voila", "merci", "bonjour", "monsieur", "madame",
    "bravo", "enfin", "alors", "quand", "après", "avant", "sans",
}

MSA_MARKERS = [
    "إن", "إنما", "لكن", "بل", "أيضاً", "حيث", "إذ", "إلا",
    "ومع ذلك", "على الرغم", "في حين", "بينما",
]


def arabic_ratio(text: str) -> float:
    ar = len(ARABIC_RANGE.findall(text))
    la = len(LATIN_RANGE.findall(text))
    total = ar + la
    return ar / total if total else 0.0


def darija_score(text: str) -> int:
    return sum(1 for m in DARIJA_MARKERS if m in text)


def french_words_in(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())
    return [t for t in tokens if t in FRENCH_WORDS]


def msa_score(text: str) -> int:
    return sum(1 for m in MSA_MARKERS if m in text)


def detect_language(text: str) -> str:
    ar_ratio = arabic_ratio(text)
    has_darija = darija_score(text) >= 1
    french = french_words_in(text)
    has_french = len(french) >= 2
    has_msa = msa_score(text) >= 2

    if ar_ratio > 0.85:
        if has_darija:
            return "Moroccan Darija (Arabic script)"
        elif has_msa:
            return "Modern Standard Arabic (MSA)"
        else:
            return "Arabic (Darija or MSA)"
    elif ar_ratio > 0.4:
        if has_french:
            return "Mixed Moroccan Darija and French"
        else:
            return "Mixed Arabic and Latin script (Darija)"
    else:
        if has_french:
            return "French (with some Arabic)"
        else:
            return "Darija in Latin script (3arabizi)"


def register_label(text: str) -> str:
    darija = darija_score(text)
    msa = msa_score(text)
    if darija >= 3:
        return "informal / colloquial Moroccan Darija"
    elif msa >= 2 and darija <= 1:
        return "formal / Modern Standard Arabic"
    elif darija >= 1 and msa >= 1:
        return "semi-formal mixed Darija and MSA"
    else:
        return "neutral / semi-formal Arabic"


def split_at_sentence(text: str, ratio: float = 0.5) -> tuple[str, str] | None:
    """Split on punctuation near the midpoint."""
    mid = int(len(text) * ratio)
    # Look for sentence boundary (Arabic/Latin punct) around midpoint
    punct = re.compile(r"[.،؟!?]\s+")
    best = None
    best_dist = len(text)
    for m in punct.finditer(text):
        dist = abs(m.end() - mid)
        if dist < best_dist:
            best_dist = dist
            best = m.end()
    if best and best > 20 and best < len(text) - 20:
        return text[:best].strip(), text[best:].strip()
    # Fallback: split at word boundary
    words = text.split()
    if len(words) < 6:
        return None
    half = len(words) // 2
    return " ".join(words[:half]), " ".join(words[half:])


def extract_french_segments(text: str) -> list[str]:
    """Extract Latin-script phrases (likely French) from Darija text."""
    segments = re.findall(r"[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\s''\-]{2,}", text)
    cleaned = [s.strip() for s in segments if len(s.strip()) > 3]
    return cleaned


def ex(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ── Task generators ───────────────────────────────────────────────────────────

def task_article_reaction(title: str, comment: str) -> dict:
    """Article title → natural Darija comment."""
    prompts = [
        f"Write a Moroccan comment reacting to this news: {title}",
        f"How might a Moroccan reader respond to this article? \"{title}\"",
        f"Give a typical Moroccan Darija reaction to: {title}",
    ]
    return ex(random.choice(prompts), comment)


def task_continuation(comment: str) -> dict | None:
    """First half of comment → second half."""
    if len(comment) < 80:
        return None
    result = split_at_sentence(comment)
    if not result:
        return None
    first, second = result
    if len(first) < 20 or len(second) < 20:
        return None
    prompts = [
        f"Continue this Moroccan text naturally:\n{first}",
        f"Complete this Darija comment:\n{first}",
    ]
    return ex(random.choice(prompts), second)


def task_language_detection(comment: str) -> dict:
    lang = detect_language(comment)
    prompts = [
        f"What language or dialect is this text written in?\n\n{comment}",
        f"Identify the language(s) used in this text:\n\n{comment}",
    ]
    return ex(random.choice(prompts), lang)


def task_french_extraction(comment: str) -> dict | None:
    """Identify French words/phrases in code-switched Darija."""
    if arabic_ratio(comment) < 0.3:
        return None
    french = french_words_in(comment)
    segments = extract_french_segments(comment)
    if not french and not segments:
        return None
    all_french = list(dict.fromkeys(segments + french))  # deduplicate, preserve order
    if not all_french:
        return None
    answer = ", ".join(f'"{s}"' for s in all_french[:8])
    prompts = [
        f"Identify the French words or phrases in this Moroccan Darija text:\n\n{comment}",
        f"This text mixes Darija and French. Extract the French parts:\n\n{comment}",
    ]
    return ex(random.choice(prompts), f"French elements found: {answer}")


def task_register(comment: str) -> dict:
    label = register_label(comment)
    prompts = [
        f"What is the register (formal/informal) of this Moroccan text?\n\n{comment}",
        f"How would you describe the formality level of this Arabic/Darija text?\n\n{comment}",
    ]
    return ex(random.choice(prompts), label)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    raw_path = OUT_DIR / "hespress_comments_raw.jsonl"
    out_path = OUT_DIR / "hespress_instructions.jsonl"

    examples: list[dict] = []
    task_counts = {
        "article_reaction": 0,
        "continuation": 0,
        "language_detection": 0,
        "french_extraction": 0,
        "register": 0,
    }

    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            article = json.loads(line)
            title = article.get("article_title", "").strip()
            comments = article.get("comments", [])

            for c in comments:
                text = c.get("text", "").strip()
                if not text or len(text) < 25:
                    continue

                # Task 1: article reaction (always, if title exists)
                if title:
                    examples.append(task_article_reaction(title, text))
                    task_counts["article_reaction"] += 1

                # Task 2: continuation (longer comments only)
                cont = task_continuation(text)
                if cont:
                    examples.append(cont)
                    task_counts["continuation"] += 1

                # Task 3: language detection (sample 60% to avoid noise)
                if random.random() < 0.6:
                    examples.append(task_language_detection(text))
                    task_counts["language_detection"] += 1

                # Task 4: French extraction (code-switched comments only)
                fr = task_french_extraction(text)
                if fr:
                    examples.append(fr)
                    task_counts["french_extraction"] += 1

                # Task 5: register (sample 40%)
                if random.random() < 0.4:
                    examples.append(task_register(text))
                    task_counts["register"] += 1

    random.shuffle(examples)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'Task':<25} {'Count':>7}")
    print("-" * 34)
    for task, count in task_counts.items():
        print(f"  {task:<23} {count:>7}")
    print("-" * 34)
    print(f"  {'TOTAL':<23} {len(examples):>7}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
