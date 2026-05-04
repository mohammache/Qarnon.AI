#!/usr/bin/env python3
"""
Evaluate MoroccoTone fine-tuned model on held-out prompts.
Run on Kaggle after training, or locally with the downloaded adapter.

Usage:
    # On Kaggle (after Cell 7):
    python evaluate.py --adapter /kaggle/working/moroccoone-qwen/lora_adapter

    # Locally with merged model:
    python evaluate.py --adapter ./lora_adapter
"""

import argparse
import json
from pathlib import Path

EVAL_PROMPTS = [
    # ── Translation ────────────────────────────────────────────────────────
    {
        "category": "translation",
        "prompt": "Translate to Darija: I don't understand what you're saying.",
        "expected_keywords": ["mafhemtch", "mafhmt", "mafhmtch", "fahm"],
    },
    {
        "category": "translation",
        "prompt": "Translate this Darija sentence to English: واش عندك شي حاجة تاكل؟",
        "expected_keywords": ["eat", "food", "something", "anything"],
    },
    {
        "category": "translation",
        "prompt": "Translate this Darija sentence to French: بغيت نمشي للدار.",
        "expected_keywords": ["maison", "aller", "veux", "rentrer"],
    },

    # ── Grammar ────────────────────────────────────────────────────────────
    {
        "category": "grammar",
        "prompt": 'Conjugate the Darija verb "kla" in present tense for the third person plural (homa / they).',
        "expected_keywords": ["kayaklo", "kayklo"],
    },
    {
        "category": "grammar",
        "prompt": 'What is the feminine form of "kbir" in Darija?',
        "expected_keywords": ["kbira"],
    },
    {
        "category": "grammar",
        "prompt": 'What is the plural of "ktab" in Darija?',
        "expected_keywords": ["ktoba", "ktoub", "ktub"],
    },

    # ── Vocabulary / normalization ─────────────────────────────────────────
    {
        "category": "normalization",
        "prompt": "Normalize this Darija spelling to its canonical form: wahd",
        "expected_keywords": ["wa7d"],
    },
    {
        "category": "vocabulary",
        "prompt": "What does \"bzzaf\" mean in English?",
        "expected_keywords": ["much", "lot", "very", "many"],
    },
    {
        "category": "vocabulary",
        "prompt": "List the common Darija spellings for: coffee",
        "expected_keywords": ["9ahwa", "qahwa", "9hwa"],
    },

    # ── Cultural / expressions ─────────────────────────────────────────────
    {
        "category": "cultural",
        "prompt": 'What does this Moroccan proverb mean? "SSari7a ra7a"',
        "expected_keywords": ["frankness", "honest", "peace", "truth"],
    },
    {
        "category": "cultural",
        "prompt": 'What does this Moroccan Darija expression mean? "safi t8nna"',
        "expected_keywords": ["worry", "calm", "relax", "fine"],
    },
    {
        "category": "cultural",
        "prompt": "Write a Moroccan Darija message wishing someone a happy Eid.",
        "expected_keywords": ["عيد", "مبارك", "3id", "eid"],
    },

    # ── Code-switching / communication ────────────────────────────────────
    {
        "category": "communication",
        "prompt": "Write a polite message in Moroccan Darija mixed with French to ask a professor to reschedule a meeting.",
        "expected_keywords": ["professeur", "réunion", "rendez-vous", "3afak", "mumkin"],
    },
    {
        "category": "communication",
        "prompt": "Identify the French words or phrases in this Moroccan Darija text: كنخدم f un projet très important u ghadi nkun occupé had l-week.",
        "expected_keywords": ["projet", "très", "important", "occupé", "week"],
    },
    {
        "category": "communication",
        "prompt": "What language or dialect is this text written in?\n\nراك كيفاش كنقولك دايما بلي خاصك تمشي بكري، ديال هكذا كتوصل متأخر!",
        "expected_keywords": ["darija", "moroccan", "arabic"],
    },
]


def run_evaluation(adapter_path: str):
    import torch
    from unsloth import FastLanguageModel

    print(f"Loading model from: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = adapter_path,
        max_seq_length = 512,
        dtype          = None,
        load_in_4bit   = True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded.\n")

    results = []
    category_scores: dict[str, list[int]] = {}

    for item in EVAL_PROMPTS:
        messages = [{"role": "user", "content": item["prompt"]}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids      = inputs,
                max_new_tokens = 150,
                temperature    = 0.3,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        ).strip()

        # Keyword check
        response_lower = response.lower()
        hits = [kw for kw in item["expected_keywords"] if kw.lower() in response_lower]
        score = 1 if hits else 0

        results.append({
            "category": item["category"],
            "prompt":   item["prompt"],
            "output":   response,
            "keywords": item["expected_keywords"],
            "hits":     hits,
            "pass":     score,
        })

        category_scores.setdefault(item["category"], []).append(score)

        status = "✓" if score else "✗"
        print(f"[{status}] [{item['category']}]")
        print(f"  PROMPT: {item['prompt'][:80]}")
        print(f"  OUTPUT: {response[:120]}")
        if not score:
            print(f"  EXPECTED any of: {item['expected_keywords']}")
        print()

    # Summary
    print("=" * 60)
    print("RESULTS BY CATEGORY")
    print("=" * 60)
    total_pass = 0
    for cat, scores in category_scores.items():
        p = sum(scores)
        n = len(scores)
        total_pass += p
        print(f"  {cat:<20} {p}/{n}  ({100*p//n}%)")
    print("-" * 60)
    n_total = len(results)
    print(f"  {'TOTAL':<20} {total_pass}/{n_total}  ({100*total_pass//n_total}%)")

    # Save results
    out = Path("dataset") / "eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="/kaggle/working/moroccoone-qwen/lora_adapter")
    args = parser.parse_args()
    run_evaluation(args.adapter)


if __name__ == "__main__":
    main()
