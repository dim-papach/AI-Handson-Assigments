"""
Task 5 — Evaluation
Computes Recall@5, Recall@10, MRR@10 for all three configurations.
Saves results/comparison_table.csv.
"""

import csv
import json
from pathlib import Path

ROOT    = Path(__file__).parent.parent
RES_DIR = ROOT / "results"

CONFIG_OUTPUTS = {
    "Config A": RES_DIR / "config_a_outputs.jsonl",
    "Config B": RES_DIR / "config_b_outputs.jsonl",
    "Config C": RES_DIR / "config_c_outputs.jsonl",
}
TABLE_PATH = RES_DIR / "comparison_table.csv"


# ── retrieval metrics ─────────────────────────────────────────────────────────
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return float(any(r in retrieved[:k] for r in relevant))


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_config(output_path: Path) -> dict:
    if not output_path.exists():
        print(f"  {output_path.name} not found — skipping.")
        return {"Recall@5": None, "Recall@10": None, "MRR@10": None, "n": 0}

    r5_scores, r10_scores, mrr_scores = [], [], []

    with open(output_path) as f:
        for line in f:
            item      = json.loads(line)
            retrieved = item.get("retrieved_passage_ids", [])
            relevant  = item.get("supporting_passage_ids", [])

            if not relevant:
                continue

            r5_scores.append(recall_at_k(retrieved, relevant, 5))
            r10_scores.append(recall_at_k(retrieved, relevant, 10))
            mrr_scores.append(mrr_at_k(retrieved, relevant, 10))

    n = len(r5_scores)
    if n == 0:
        return {"Recall@5": None, "Recall@10": None, "MRR@10": None, "n": 0}

    return {
        "Recall@5":  round(sum(r5_scores)  / n, 4),
        "Recall@10": round(sum(r10_scores) / n, 4),
        "MRR@10":    round(sum(mrr_scores) / n, 4),
        "n":         n,
    }


# ── answer quality: EM and F1 (HotpotQA-style) ───────────────────────────────
import re
import string
from collections import Counter


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common      = Counter(pred_tokens) & Counter(gold_tokens)
    n_common    = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_answer_quality(output_path: Path) -> dict:
    if not output_path.exists():
        return {"Answer EM": None, "Answer F1": None}

    em_scores, f1_scores = [], []
    with open(output_path) as f:
        for line in f:
            item = json.loads(line)
            pred = item.get("predicted_answer", "")
            gold = item.get("gold_answer", "")
            if not gold:
                continue
            em_scores.append(exact_match(pred, gold))
            f1_scores.append(token_f1(pred, gold))

    n = len(em_scores)
    if n == 0:
        return {"Answer EM": None, "Answer F1": None}
    return {
        "Answer EM": round(sum(em_scores) / n, 4),
        "Answer F1": round(sum(f1_scores) / n, 4),
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    rows = []
    for config_name, path in CONFIG_OUTPUTS.items():
        print(f"Evaluating {config_name} …")
        retrieval = evaluate_config(path)
        quality   = evaluate_answer_quality(path)
        row = {"Config": config_name, **retrieval, **quality}
        rows.append(row)
        print(f"  Recall@5={retrieval['Recall@5']}  Recall@10={retrieval['Recall@10']}  "
              f"MRR@10={retrieval['MRR@10']}  EM={quality['Answer EM']}  F1={quality['Answer F1']}")

    # write CSV
    fieldnames = ["Config", "Recall@5", "Recall@10", "MRR@10", "Answer EM", "Answer F1", "n"]
    with open(TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison table saved to {TABLE_PATH}")


if __name__ == "__main__":
    main()
