"""
Task 1 — Data preparation for HotpotQA (distractor setting).

Outputs:
  data/corpus.jsonl       — all unique passages from the knowledge base
  data/train_pairs.jsonl  — (query, positive_passage) pairs from train split
  data/eval_set.jsonl     — held-out evaluation questions + answers
"""

import json
import random
from pathlib import Path
from datasets import load_dataset

RANDOM_STATE = 42
TRAIN_PAIRS    = 5000
EVAL_QUESTIONS = 500

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

random.seed(RANDOM_STATE)


def build_passage_text(title: str, sentences: list[str]) -> str:
    return f"{title}: {''.join(sentences)}"


def main():
    print("Loading HotpotQA …")
    ds_train = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds_val   = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    # ── corpus ────────────────────────────────────────────────────────────────
    print("Building corpus …")
    corpus: dict[str, str] = {}

    for split in (ds_train, ds_val):
        for ex in split:
            for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
                pid = title.replace(" ", "_")
                if pid not in corpus:
                    corpus[pid] = build_passage_text(title, sentences)

    with open(DATA_DIR / "corpus.jsonl", "w") as f:
        for pid, text in corpus.items():
            f.write(json.dumps({"passage_id": pid, "text": text}) + "\n")
    print(f"  corpus: {len(corpus):,} passages → data/corpus.jsonl")

    # ── train pairs (train split only — no eval leakage) ─────────────────────
    print("Building training pairs …")
    train_pairs = []
    for ex in ds_train:
        supporting_titles = set(ex["supporting_facts"]["title"])
        for title, sentences in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title in supporting_titles:
                train_pairs.append({
                    "query": ex["question"],
                    "positive_passage_id": title.replace(" ", "_"),
                    "positive_passage": build_passage_text(title, sentences),
                })

    random.shuffle(train_pairs)
    train_pairs = train_pairs[:TRAIN_PAIRS]

    with open(DATA_DIR / "train_pairs.jsonl", "w") as f:
        for pair in train_pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"  train pairs: {len(train_pairs):,} → data/train_pairs.jsonl")

    # ── eval set (validation split only) ─────────────────────────────────────
    print("Building eval set …")
    val_list = list(ds_val)
    random.shuffle(val_list)

    eval_set = []
    for ex in val_list[:EVAL_QUESTIONS]:
        supporting_ids = [t.replace(" ", "_") for t in ex["supporting_facts"]["title"]]
        eval_set.append({
            "question_id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "supporting_passage_ids": list(dict.fromkeys(supporting_ids)),
        })

    with open(DATA_DIR / "eval_set.jsonl", "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")
    print(f"  eval set: {len(eval_set):,} questions → data/eval_set.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
