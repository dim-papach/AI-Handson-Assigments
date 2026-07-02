"""
Interactive query tool — ask any question to one of the three pipeline configurations.

Usage:
    poetry run python Final_HW/src/query.py -q "Where was Christopher Nolan born?" --config B
    poetry run python Final_HW/src/query.py -q "Who founded the Dallas Texans?" --config C --show-passages

Configs:
    A  Baseline RAG (off-the-shelf bge-small embedder, top-5 retrieval)
    B  Improved RAG (fine-tuned embedder + bge-reranker-base cross-encoder, top-3 after reranking)  [default]
    C  ReAct agent  (Config B retrieval in an iterative loop, up to 5 retrieve calls)

Note: the corpus contains ~28 k Wikipedia passages drawn from the HotpotQA distractor setting.
Questions outside that topic space will still get an answer, but from whatever the retriever finds
closest — treat those results as best-effort, not authoritative.
"""

import argparse
import os
import socket
import sys
from pathlib import Path

# make sibling modules importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

# Force IPv4 resolution to prevent connection hangs on hosts with broken IPv6
# (same fix as hw2/src/config.py)
_old_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    return [res for res in _old_getaddrinfo(*args, **kwargs) if res[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT.parent / ".env")

_LLM_KWARGS = dict(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)


def _llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(**_LLM_KWARGS)


# ── Config A ──────────────────────────────────────────────────────────────────
def query_a(question: str, show_passages: bool) -> None:
    from baseline_rag import get_client, build_index, retrieve, generate_answer, TOP_K

    print("Loading Config A (off-the-shelf embedder) …")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    client   = get_client()
    build_index(client, embedder)
    llm      = _llm()

    passages = retrieve(question, embedder, client)
    answer   = generate_answer(question, passages[:TOP_K], llm)

    print(f"\nAnswer: {answer}")
    if show_passages:
        _print_passages(passages[:TOP_K])


# ── Config B ──────────────────────────────────────────────────────────────────
def query_b(question: str, show_passages: bool) -> None:
    from improved_retrieval import get_client, build_index, retrieve_and_rerank, generate_answer, TOP_K_RERANK

    MODEL_DIR = ROOT / "models" / "finetuned_embedder"
    emb_path  = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else "BAAI/bge-small-en-v1.5"

    print("Loading Config B (fine-tuned embedder + cross-encoder reranker) …")
    embedder = SentenceTransformer(emb_path)
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    client   = get_client()
    build_index(client, embedder, "corpus_config_b")
    llm      = _llm()

    # retrieve_and_rerank() returns the full reranked list (for Recall@10 bookkeeping
    # elsewhere); only the top TOP_K_RERANK go to the generator, same as the pipeline.
    ranked   = retrieve_and_rerank(question, embedder, reranker, client)
    passages = ranked[:TOP_K_RERANK]
    answer   = generate_answer(question, passages, llm)

    print(f"\nAnswer: {answer}")
    if show_passages:
        _print_passages(passages)


# ── Config C ──────────────────────────────────────────────────────────────────
def query_c(question: str, show_passages: bool) -> None:
    from agent import get_client, run_agent
    from improved_retrieval import build_index

    MODEL_DIR = ROOT / "models" / "finetuned_embedder"
    emb_path  = str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else "BAAI/bge-small-en-v1.5"

    print("Loading Config C (ReAct agent) …")
    embedder = SentenceTransformer(emb_path)
    reranker = CrossEncoder("BAAI/bge-reranker-base")
    client   = get_client()
    build_index(client, embedder, "corpus_config_b")
    llm      = _llm()

    result = run_agent(question, embedder, reranker, client, llm)
    answer = result["final_answer"]
    steps  = result["steps_used"]

    print(f"\nAnswer: {answer}")
    print(f"Retrieve calls used: {steps}")
    if show_passages:
        ids = result["retrieved_ids"]
        print(f"\nRetrieved passage IDs ({len(ids)} total across all steps):")
        for pid in ids:
            print(f"  {pid}")


# ── helpers ───────────────────────────────────────────────────────────────────
def _print_passages(passages: list[dict]) -> None:
    print("\nRetrieved passages:")
    for i, p in enumerate(passages, 1):
        text = p.get("text", "")
        print(f"  [{i}] {p['passage_id']}")
        print(f"      {text[:300]}{'…' if len(text) > 300 else ''}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the RAG pipeline interactively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-q", "--question", required=True, help="The question to ask.")
    parser.add_argument(
        "-c", "--config",
        choices=["A", "B", "C"],
        default="B",
        help="Configuration to use: A (baseline), B (fine-tuned + reranker) [default], C (agent).",
    )
    parser.add_argument(
        "--show-passages",
        action="store_true",
        help="Print the retrieved passages alongside the answer.",
    )
    args = parser.parse_args()

    print(f"\n{'─'*60}")
    print(f"Config {args.config}  |  {args.question}")
    print(f"{'─'*60}")

    if args.config == "A":
        query_a(args.question, args.show_passages)
    elif args.config == "B":
        query_b(args.question, args.show_passages)
    else:
        query_c(args.question, args.show_passages)

    print()


if __name__ == "__main__":
    main()
