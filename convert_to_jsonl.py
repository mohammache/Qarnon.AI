#!/usr/bin/env python3
"""
Convert Moroccan Darija CSV files → JSONL instruction format for LLM fine-tuning.
Output: dataset/darija_foundation.jsonl
"""

import csv
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a Moroccan Darija linguistic assistant specialized in translation, "
    "spelling normalization, grammar, and cultural interpretation of Moroccan Arabic "
    "(Darija), which is frequently mixed with French and Modern Standard Arabic."
)

DATA_DIR = Path("darija-data")
OUT_DIR = Path("dataset")

PERSONS_PRESENT = ["ana", "nta", "nti", "howa", "hia", "7na", "ntoma", "homa"]
PERSONS_DESC = {
    "ana":   "first person singular (ana / I)",
    "nta":   "second person singular masculine (nta / you masc.)",
    "nti":   "second person singular feminine (nti / you fem.)",
    "howa":  "third person singular masculine (howa / he)",
    "hia":   "third person singular feminine (hia / she)",
    "7na":   "first person plural (7na / we)",
    "ntoma": "second person plural (ntoma / you all)",
    "homa":  "third person plural (homa / they)",
}


def ex(user_msg: str, assistant_msg: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def variants(row: dict, exclude: set[str]) -> list[str]:
    return [v.strip() for k, v in row.items() if k not in exclude and v.strip()]


# ── Handlers ────────────────────────────────────────────────────────────────

def handle_sentences(rows: list[dict], out: list):
    for row in rows:
        en = row.get("english", "").strip()
        da = row.get("darija", "").strip()
        if not en or not da:
            continue
        out.append(ex(
            random.choice([
                f"Translate to Darija: {en}",
                f"How do you say this in Moroccan Darija? {en}",
                f"Write this sentence in Darija: {en}",
            ]),
            da
        ))
        out.append(ex(
            random.choice([
                f"Translate this Darija sentence to English: {da}",
                f"What does this Darija sentence mean in English? {da}",
            ]),
            en
        ))


def handle_proverbs(rows: list[dict], out: list):
    for row in rows:
        da = row.get("darija", "").strip()
        en = row.get("eng", "").strip()
        if not da or not en:
            continue
        out.append(ex(
            f'What does this Moroccan proverb mean? "{da}"',
            f'This Moroccan proverb ("{da}") means: {en}'
        ))
        out.append(ex(
            f"Give a Moroccan Darija proverb related to: {en}",
            f'"{da}" — meaning: {en}'
        ))


def handle_idioms(rows: list[dict], out: list):
    for row in rows:
        en = row.get("eng", "").strip()
        vs = variants(row, {"eng"})
        if not vs or not en:
            continue
        for v in vs:
            out.append(ex(
                f'What does this Moroccan Darija expression mean? "{v}"',
                en
            ))
        out.append(ex(
            f"Give a Moroccan Darija expression that means: {en}",
            vs[0]
        ))


def handle_vocab(rows: list[dict], out: list):
    """Generic n1/n2/.../eng vocabulary files."""
    eng_keys = {"eng", "english"}
    for row in rows:
        eng = next((row[k].strip() for k in eng_keys if k in row and row[k] and row[k].strip()), "")
        vs = variants(row, eng_keys)
        if not vs:
            continue
        canonical = vs[0]

        if eng:
            for v in vs:
                out.append(ex(
                    random.choice([
                        f'Translate this Darija word to English: {v}',
                        f'What does "{v}" mean in English?',
                    ]),
                    eng
                ))
            out.append(ex(
                f"List the common Darija spellings for: {eng}",
                ", ".join(vs)
            ))

        if len(vs) > 1:
            for v in vs[1:]:
                out.append(ex(
                    f'Normalize this Darija spelling to its canonical form: {v}',
                    canonical
                ))


def handle_conjug_present(rows: list[dict], out: list):
    for row in rows:
        root = row.get("root", "").strip()
        if not root:
            continue
        forms = {p: row.get(p, "").strip() for p in PERSONS_PRESENT}
        valid = {p: f for p, f in forms.items() if f}
        if not valid:
            continue

        paradigm = "\n".join(
            f"  {PERSONS_DESC[p]}: {valid[p]}"
            for p in PERSONS_PRESENT if p in valid
        )
        out.append(ex(
            f'Conjugate the Darija verb "{root}" in the present tense for all persons.',
            f'Present tense of "{root}":\n{paradigm}'
        ))
        for p in PERSONS_PRESENT:
            if p in valid:
                out.append(ex(
                    f'Conjugate "{root}" in present tense for the {PERSONS_DESC[p]}.',
                    valid[p]
                ))


def handle_conjug_past(rows: list[dict], out: list, present_roots: list[str]):
    for idx, row in enumerate(rows):
        forms = {p: row.get(p, "").strip() for p in PERSONS_PRESENT}
        valid = {p: f for p, f in forms.items() if f}
        if not valid:
            continue
        root = present_roots[idx] if idx < len(present_roots) else (
            valid.get("howa") or valid.get("ana", f"verb#{idx}")
        )

        paradigm = "\n".join(
            f"  {PERSONS_DESC[p]}: {valid[p]}"
            for p in PERSONS_PRESENT if p in valid
        )
        out.append(ex(
            f'Conjugate the Darija verb "{root}" in the past tense for all persons.',
            f'Past tense of "{root}":\n{paradigm}'
        ))
        for p in PERSONS_PRESENT:
            if p in valid:
                out.append(ex(
                    f'Conjugate "{root}" in past tense for the {PERSONS_DESC[p]}.',
                    valid[p]
                ))


def handle_imperatives(rows: list[dict], out: list, present_roots: list[str]):
    imp_persons = {
        "nta":   "you masculine singular",
        "nti":   "you feminine singular",
        "ntoma": "you plural",
    }
    for idx, row in enumerate(rows):
        root = row.get("root_darija", "").strip()
        if not root:
            root = present_roots[idx] if idx < len(present_roots) else f"verb#{idx}"
        forms = {p: row.get(p, "").strip() for p in imp_persons}
        valid = {p: f for p, f in forms.items() if f}
        if not valid:
            continue

        summary = ", ".join(f"{imp_persons[p]}: {valid[p]}" for p in imp_persons if p in valid)
        out.append(ex(
            f'Give the imperative forms of the Darija verb "{root}".',
            summary
        ))
        for p, desc in imp_persons.items():
            if p in valid:
                out.append(ex(
                    f'What is the imperative of "{root}" for {desc} in Darija?',
                    valid[p]
                ))


def handle_masc_fem_plural(rows: list[dict], out: list):
    for row in rows:
        masc = row.get("masculine", "").strip()
        fem  = row.get("feminine", "").strip()
        mp   = row.get("masc_plural", "").strip()
        fp   = row.get("fem_plural", "").strip()
        if not (masc or fem):
            continue
        if masc and fem:
            out.append(ex(f'What is the feminine form of "{masc}" in Darija?', fem))
            out.append(ex(f'What is the masculine form of "{fem}" in Darija?', masc))
        if masc and mp:
            out.append(ex(f'What is the masculine plural of "{masc}" in Darija?', mp))
        if fem and fp:
            out.append(ex(f'What is the feminine plural of "{fem}" in Darija?', fp))
        if (masc or fem) and (mp or fp):
            base = masc or fem
            plural = mp or fp
            out.append(ex(f'What is the plural of "{base}" in Darija?', plural))


def handle_verb_to_noun(rows: list[dict], out: list):
    for row in rows:
        verb = row.get("verb", "").strip()
        noun = row.get("noun", "").strip()
        if not verb or not noun:
            continue
        out.append(ex(
            f'What is the verbal noun (masdar) derived from the Darija verb "{verb}"?',
            noun
        ))
        out.append(ex(
            f'Which Darija verb does the noun "{noun}" derive from?',
            verb
        ))


def handle_indefinite(rows: list[dict], out: list):
    for row in rows:
        indef = row.get("indef", "").strip()
        defn  = row.get("def", "").strip()
        if not indef or not defn:
            continue
        out.append(ex(f'Make this Darija word definite (add the article): "{indef}"', defn))
        out.append(ex(f'What is the indefinite form of "{defn}" in Darija?', indef))


def handle_imagenet(rows: list[dict], out: list):
    for row in rows:
        eng = row.get("english", "").strip().replace("_", " ")
        vs = [
            row.get("specific1_darija", "").strip(),
            row.get("specific2_darija", "").strip(),
            row.get("general_darija", "").strip(),
        ]
        ar = row.get("darija_ar", "").strip()
        vs = [v for v in vs if v]
        if not eng or not vs:
            continue
        answer = vs[0]
        if len(vs) > 1:
            answer += f" (also: {', '.join(vs[1:])})"
        if ar:
            answer += f" — in Arabic script: {ar}"
        out.append(ex(f'How do you say "{eng}" in Moroccan Darija?', answer))


# ── Router ───────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    OUT_DIR.mkdir(exist_ok=True)
    examples: list[dict] = []

    # Pre-load present-tense roots for alignment with past/imperatives
    present_roots: list[str] = []
    cp_path = DATA_DIR / "conjug_present.csv"
    if cp_path.exists():
        present_roots = [r.get("root", "").strip() for r in read_csv(cp_path)]

    ROUTES = {
        "sentences":              lambda rows: handle_sentences(rows, examples),
        "sentences2":             lambda rows: handle_sentences(rows, examples),
        "proverbs":               lambda rows: handle_proverbs(rows, examples),
        "idioms":                 lambda rows: handle_idioms(rows, examples),
        "weird":                  lambda rows: handle_idioms(rows, examples),
        "conjug_present":         lambda rows: handle_conjug_present(rows, examples),
        "conjug_past":            lambda rows: handle_conjug_past(rows, examples, present_roots),
        "imperatives":            lambda rows: handle_imperatives(rows, examples, present_roots),
        "masculine_feminine_plural": lambda rows: handle_masc_fem_plural(rows, examples),
        "verb-to-noun":           lambda rows: handle_verb_to_noun(rows, examples),
        "(in)definite":           lambda rows: handle_indefinite(rows, examples),
        "imagenet_b_darija":      lambda rows: handle_imagenet(rows, examples),
    }

    total_by_file = {}
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        stem = csv_path.stem
        rows = read_csv(csv_path)
        if not rows:
            continue
        before = len(examples)
        handler = ROUTES.get(stem, lambda rows: handle_vocab(rows, examples))
        handler(rows)
        total_by_file[stem] = len(examples) - before

    random.shuffle(examples)

    out_path = OUT_DIR / "darija_foundation.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ex_item in examples:
            f.write(json.dumps(ex_item, ensure_ascii=False) + "\n")

    print(f"\n{'File':<35} {'Examples':>8}")
    print("-" * 45)
    for stem, count in sorted(total_by_file.items(), key=lambda x: -x[1]):
        print(f"  {stem:<33} {count:>8}")
    print("-" * 45)
    print(f"  {'TOTAL':<33} {len(examples):>8}")
    print(f"\nOutput: {out_path}")


if __name__ == "__main__":
    main()
