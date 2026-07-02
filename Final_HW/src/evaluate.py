"""
Task 5: Evaluation
5.1 Universal retrieval metrics: Recall@5, Recall@10, MRR@10
5.2 HotpotQA official metrics: Answer EM/F1, Supporting Facts EM/F1, Joint EM/F1
5.3 RAGAS: Faithfulness, Context Precision, Context Recall (best config only)
Saves results/comparison_table.csv and results/ragas_scores.json.
"""

import csv
import json
import os
import re
import socket
import string
from collections import Counter
from pathlib import Path

# Force IPv4 resolution to prevent connection hangs on hosts with broken IPv6
# (same fix as hw2/src/config.py)
_old_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    return [res for res in _old_getaddrinfo(*args, **kwargs) if res[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

from dotenv import load_dotenv

ROOT    = Path(__file__).parent.parent
load_dotenv(ROOT.parent / ".env")

RES_DIR = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

CONFIG_OUTPUTS = {
    "Config A": RES_DIR / "config_a_outputs.jsonl",
    "Config B": RES_DIR / "config_b_outputs.jsonl",
    "Config C": RES_DIR / "config_c_outputs.jsonl",
}
TABLE_PATH      = RES_DIR / "comparison_table.csv"
RAGAS_PATH      = RES_DIR / "ragas_scores.json"
RAGAS_SAMPLE    = 50   # number of questions for RAGAS (API calls are expensive)


# 5.1 Retrieval metrics
def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Calculate Recall@k for a single query."""
    return float(any(r in retrieved[:k] for r in relevant))


def mrr_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Calculate Mean Reciprocal Rank@k for a single query."""
    for rank, pid in enumerate(retrieved[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(output_path: Path) -> dict:
    """Evaluate retrieval performance (Recall and MRR) over the entire output file."""
    if not output_path.exists():
        return {"Recall@5": None, "Recall@10": None, "MRR@10": None, "n": 0}

    r5, r10, mrr = [], [], []
    with open(output_path) as f:
        for line in f:
            item      = json.loads(line)
            retrieved = item.get("retrieved_passage_ids", [])
            relevant  = item.get("supporting_passage_ids", [])
            if not relevant:
                continue
            r5.append(recall_at_k(retrieved, relevant, 5))
            r10.append(recall_at_k(retrieved, relevant, 10))
            mrr.append(mrr_at_k(retrieved, relevant, 10))

    n = len(r5)
    if n == 0:
        return {"Recall@5": None, "Recall@10": None, "MRR@10": None, "n": 0}
    return {
        "Recall@5":  round(sum(r5)  / n, 4),
        "Recall@10": round(sum(r10) / n, 4),
        "MRR@10":    round(sum(mrr) / n, 4),
        "n": n,
    }


# 5.2 HotpotQA official metrics
def normalize(text: str) -> str:
    """Normalize text by lowercasing, removing articles, and stripping punctuation."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(c for c in text if c not in string.punctuation)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """Check if the normalized prediction exactly matches the normalized gold answer."""
    return float(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    """Calculate the token-level F1 score between prediction and gold answer."""
    pred_t  = normalize(pred).split()
    gold_t  = normalize(gold).split()
    common  = Counter(pred_t) & Counter(gold_t)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred_t)
    r = n_common / len(gold_t)
    return 2 * p * r / (p + r)


def supporting_facts_scores(retrieved: list[str], relevant: list[str]) -> tuple[float, float]:
    """EM and F1 at the passage level (approximation of sentence-level SF metric)."""
    relevant_set  = set(relevant)
    retrieved_set = set(retrieved)
    if not relevant_set:
        return 0.0, 0.0
    # EM: all relevant passages retrieved
    sf_em = float(relevant_set.issubset(retrieved_set))
    # F1: token overlap on passage-id sets
    tp = len(relevant_set & retrieved_set)
    if tp == 0:
        return sf_em, 0.0
    p  = tp / len(retrieved_set) if retrieved_set else 0.0
    r  = tp / len(relevant_set)
    sf_f1 = 2 * p * r / (p + r)
    return sf_em, sf_f1


def evaluate_official(output_path: Path) -> dict:
    """Evaluate HotpotQA official metrics (EM and F1) over the entire output file."""
    if not output_path.exists():
        return {"Answer EM": None, "Answer F1": None,
                "SF EM": None, "SF F1": None,
                "Joint EM": None, "Joint F1": None}

    ans_em, ans_f1, sf_em_l, sf_f1_l, j_em, j_f1 = [], [], [], [], [], []

    with open(output_path) as f:
        for line in f:
            item      = json.loads(line)
            pred      = item.get("predicted_answer", "")
            gold      = item.get("gold_answer", "")
            # Supporting Facts approximates "the evidence the system actually used," so it
            # must read context_passage_ids (what was shown to the generator), not the wider
            # retrieved_passage_ids window kept for Recall@5/10; same reasoning as the RAGAS
            # contexts fix above. Falls back to retrieved_passage_ids for older output rows
            # that predate the context_passage_ids field.
            evidence  = item.get("context_passage_ids", item.get("retrieved_passage_ids", []))
            relevant  = item.get("supporting_passage_ids", [])
            if not gold:
                continue

            a_em  = exact_match(pred, gold)
            a_f1  = token_f1(pred, gold)
            s_em, s_f1 = supporting_facts_scores(evidence, relevant)

            ans_em.append(a_em)
            ans_f1.append(a_f1)
            sf_em_l.append(s_em)
            sf_f1_l.append(s_f1)
            # Joint EM: both answer and SF exactly correct
            j_em.append(float(a_em == 1.0 and s_em == 1.0))
            # Joint F1: harmonic mean of answer F1 and SF F1
            j_f1.append(2 * a_f1 * s_f1 / (a_f1 + s_f1) if (a_f1 + s_f1) > 0 else 0.0)

    n = len(ans_em)
    if n == 0:
        return {"Answer EM": None, "Answer F1": None,
                "SF EM": None, "SF F1": None,
                "Joint EM": None, "Joint F1": None}
    return {
        "Answer EM": round(sum(ans_em) / n, 4),
        "Answer F1": round(sum(ans_f1) / n, 4),
        "SF EM":     round(sum(sf_em_l) / n, 4),
        "SF F1":     round(sum(sf_f1_l) / n, 4),
        "Joint EM":  round(sum(j_em)    / n, 4),
        "Joint F1":  round(sum(j_f1)    / n, 4),
    }


# 5.3 RAGAS
def evaluate_ragas(output_path: Path, config_name: str) -> dict | None:
    """Evaluate generation and retrieval quality using RAGAS."""
    try:
        import sys, types
        # langchain-community >=0.4 removed chat_models.vertexai; stub it so ragas doesn't crash
        if "langchain_community.chat_models.vertexai" not in sys.modules:
            _stub = types.ModuleType("langchain_community.chat_models.vertexai")
            _stub.ChatVertexAI = type("ChatVertexAI", (), {})  # empty stub class
            sys.modules["langchain_community.chat_models.vertexai"] = _stub
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, context_precision, context_recall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from datasets import Dataset
    except ImportError as e:
        print(f"  RAGAS skipped: {e}")
        return None

    if not output_path.exists():
        return None

    # build passage_id → text lookup from corpus
    corpus_path = ROOT / "data" / "corpus.jsonl"
    id_to_text: dict[str, str] = {}
    with open(corpus_path) as f:
        for line in f:
            p = json.loads(line)
            id_to_text[p["passage_id"]] = p["text"]

    with open(output_path) as f:
        items = [json.loads(line) for line in f][:RAGAS_SAMPLE]

    # RAGAS must see exactly the passages the generator was given, not the wider
    # retrieved_passage_ids window kept for Recall@5/10, so we read context_passage_ids.
    data = {
        "question":     [i["question"] for i in items],
        "answer":       [i["predicted_answer"] for i in items],
        "contexts":     [
            [id_to_text[pid] for pid in i.get("context_passage_ids", i.get("retrieved_passage_ids", [])) if pid in id_to_text] or [""]
            for i in items
        ],
        "ground_truth": [i["gold_answer"] for i in items],
    }

    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    ))
    emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    ))

    print(f"  Running RAGAS on {RAGAS_SAMPLE} samples …")
    run_cfg = RunConfig(timeout=180, max_retries=3, max_wait=60, max_workers=4)
    result = ragas_evaluate(
        Dataset.from_dict(data),
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
        embeddings=emb,
        run_config=run_cfg,
    )
    def _mean(val) -> float:
        """ragas >=0.2 returns per-sample lists; older versions return a scalar."""
        if isinstance(val, list):
            vals = [v for v in val if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        return float(val)

    scores = {
        "config":            config_name,
        "faithfulness":      round(_mean(result["faithfulness"]), 4),
        "context_precision": round(_mean(result["context_precision"]), 4),
        "context_recall":    round(_mean(result["context_recall"]), 4),
        "n_samples":         RAGAS_SAMPLE,
    }
    return scores


# Main
def main() -> None:
    """Calculate all evaluation metrics for all configurations and save results."""
    rows = []
    best_config, best_f1, best_path = None, -1.0, None

    for config_name, path in CONFIG_OUTPUTS.items():
        print(f"Evaluating {config_name} …")
        retrieval = evaluate_retrieval(path)
        official  = evaluate_official(path)
        row = {"Config": config_name, **retrieval, **official}
        rows.append(row)
        print(f"  Recall@5={retrieval['Recall@5']}  MRR@10={retrieval['MRR@10']}  "
              f"Ans EM={official['Answer EM']}  Ans F1={official['Answer F1']}  "
              f"SF EM={official['SF EM']}  SF F1={official['SF F1']}  "
              f"Joint EM={official['Joint EM']}  Joint F1={official['Joint F1']}")
        if official["Answer F1"] and official["Answer F1"] > best_f1:
            best_f1, best_config, best_path = official["Answer F1"], config_name, path

    # write CSV
    fieldnames = ["Config", "Recall@5", "Recall@10", "MRR@10",
                  "Answer EM", "Answer F1", "SF EM", "SF F1",
                  "Joint EM", "Joint F1", "n"]
    with open(TABLE_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nComparison table saved to {TABLE_PATH}")

    # RAGAS on best config (cached: RAGAS calls an LLM judge, so skip if already computed)
    if RAGAS_PATH.exists():
        print(f"\nRAGAS scores already exist at {RAGAS_PATH}, skipping "
              f"(delete the file to force a recompute).")
    else:
        print(f"\nRunning RAGAS on best config ({best_config}) ...")
        ragas_scores = evaluate_ragas(best_path, best_config)
        if ragas_scores:
            with open(RAGAS_PATH, "w") as f:
                json.dump(ragas_scores, f, indent=2)
            print(f"  Faithfulness={ragas_scores['faithfulness']}  "
                  f"Context Precision={ragas_scores['context_precision']}  "
                  f"Context Recall={ragas_scores['context_recall']}")
            print(f"  RAGAS scores saved to {RAGAS_PATH}")
        else:
            print("  RAGAS skipped.")


if __name__ == "__main__":
    main()
