"""
Configuration B: Improved Retrieval
Fine-tuned bi-encoder (BAAI/bge-small-en-v1.5) + cross-encoder reranking.
Step 1: fine_tune(), trains and saves models/finetuned_embedder/
Step 2: run(), builds index with fine-tuned embedder, reranks, generates answers
"""

import json
import os
import random
import socket
from pathlib import Path

import torch
from dotenv import load_dotenv

# Force IPv4 resolution to prevent connection hangs on hosts with broken IPv6
# (same fix as hw2/src/config.py)
_old_getaddrinfo = socket.getaddrinfo
def _ipv4_getaddrinfo(*args, **kwargs):
    return [res for res in _old_getaddrinfo(*args, **kwargs) if res[0] == socket.AF_INET]
socket.getaddrinfo = _ipv4_getaddrinfo

from langchain_google_genai import ChatGoogleGenerativeAI
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT     = Path(__file__).parent.parent
load_dotenv(ROOT.parent / ".env")

# Constants
BASE_EMBEDDER_ID   = "BAAI/bge-small-en-v1.5"
RERANKER_ID        = "BAAI/bge-reranker-base"
INDEX_NAME         = "corpus_config_b"
TOP_K_RETRIEVE     = 20   # bi-encoder fetches top-20
TOP_K_RERANK       = 3    # passages shown to the generator after reranking
RECALL_K           = 10   # reranked ids stored for Recall@10 / MRR@10 (real top-10 window)
BATCH_SIZE_EMBED   = 256
BATCH_SIZE_TRAIN   = 8
EPOCHS             = 2
RANDOM_STATE       = 42
DIAG_SLICE         = 200  # questions for Recall@5 diagnostic

DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models" / "finetuned_embedder"
RES_DIR   = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
TRAIN_PATH  = DATA_DIR / "train_pairs.jsonl"
EVAL_PATH   = DATA_DIR / "eval_set.jsonl"
OUTPUT_PATH = RES_DIR / "config_b_outputs.jsonl"


# Helpers
def get_client() -> OpenSearch:
    """Initialize and return an OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        timeout=60,
    )


def recall_at_k(embedder: SentenceTransformer, client: OpenSearch, index: str, k: int = 5) -> float:
    """Compute Recall@k on the diagnostic slice."""
    with open(EVAL_PATH) as f:
        eval_set = [json.loads(line) for line in f]
    eval_set = eval_set[:DIAG_SLICE]

    hits = 0
    for item in tqdm(eval_set, desc=f"Recall@{k} diagnostic"):
        q_emb = embedder.encode([item["question"]], normalize_embeddings=True)[0].tolist()
        resp  = client.search(
            index=index,
            body={
                "size":  k,
                "query": {"knn": {"embedding": {"vector": q_emb, "k": k}}},
                "_source": ["passage_id"],
            },
        )
        retrieved = {hit["_source"]["passage_id"] for hit in resp["hits"]["hits"]}
        if any(pid in retrieved for pid in item["supporting_passage_ids"]):
            hits += 1
    return hits / len(eval_set)


def build_index(client: OpenSearch, embedder: SentenceTransformer, index_name: str) -> None:
    """
    Build the OpenSearch index if it does not exist, and populate it with
    embedded passages from the corpus.
    """
    if client.indices.exists(index=index_name):
        print(f"Index '{index_name}' already exists, skipping build.")
        return

    dim = embedder.get_sentence_embedding_dimension()
    client.indices.create(
        index=index_name,
        body={
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": 100}},
            "mappings": {
                "properties": {
                    "passage_id": {"type": "keyword"},
                    "text":       {"type": "text"},
                    "embedding":  {
                        "type":      "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name":       "hnsw",
                            "space_type": "cosinesimil",
                            "engine":     "nmslib",
                            "parameters": {"ef_construction": 128, "m": 16},
                        },
                    },
                }
            },
        },
    )

    with open(CORPUS_PATH) as f:
        passages = [json.loads(line) for line in f]

    print(f"Embedding {len(passages):,} passages …")
    embeddings = embedder.encode(
        [p["text"] for p in passages],
        batch_size=BATCH_SIZE_EMBED,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    def _actions():
        for p, emb in zip(passages, embeddings):
            yield {
                "_index": index_name,
                "_id":    p["passage_id"],
                "_source": {"passage_id": p["passage_id"], "text": p["text"], "embedding": emb.tolist()},
            }

    print("Bulk-indexing …")
    helpers.bulk(client, _actions(), chunk_size=500, request_timeout=60)
    client.indices.refresh(index=index_name)
    print(f"Indexed {len(passages):,} passages into '{index_name}'.")


# Task 3a: Fine-tune bi-encoder
def fine_tune() -> None:
    """
    Fine-tune the base bi-encoder using MultipleNegativesRankingLoss on the HotpotQA
    training pairs.
    """
    if (MODEL_DIR / "config.json").exists():
        print("Fine-tuned model already exists, skipping training.")
        return

    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    print("Loading training pairs …")
    with open(TRAIN_PATH) as f:
        pairs = [json.loads(line) for line in f]

    examples = [
        InputExample(texts=[p["query"], p["positive_passage"]])
        for p in pairs
    ]

    model      = SentenceTransformer(BASE_EMBEDDER_ID)
    dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE_TRAIN)
    loss       = losses.MultipleNegativesRankingLoss(model)

    # diagnostic Recall@5 before fine-tuning
    client       = get_client()
    base_embedder = SentenceTransformer(BASE_EMBEDDER_ID)
    build_index(client, base_embedder, "corpus_config_a")   # reuse Config A index
    r5_before = recall_at_k(base_embedder, client, "corpus_config_a")
    print(f"Recall@5 BEFORE fine-tuning: {r5_before:.3f}")

    print(f"Fine-tuning for {EPOCHS} epoch(s) …")
    model.fit(
        train_objectives=[(dataloader, loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        output_path=str(MODEL_DIR),
        show_progress_bar=True,
    )
    print(f"Fine-tuned model saved to {MODEL_DIR}")

    # diagnostic Recall@5 after fine-tuning
    finetuned = SentenceTransformer(str(MODEL_DIR))
    build_index(client, finetuned, INDEX_NAME)
    r5_after = recall_at_k(finetuned, client, INDEX_NAME)
    print(f"Recall@5 AFTER  fine-tuning: {r5_after:.3f}  (Δ = {r5_after - r5_before:+.3f})")


# Task 3b: Retrieve + rerank
def retrieve_and_rerank(
    query: str,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    client: OpenSearch,
) -> list[dict]:
    """
    Retrieve top-k candidates using the bi-encoder, then re-rank all of them with the
    cross-encoder. Returns the full reranked list (up to TOP_K_RETRIEVE); callers slice
    the first TOP_K_RERANK passages for generation context and the first RECALL_K
    passage ids for retrieval-quality metrics, so Recall@10 reflects a real top-10
    window instead of the (much smaller) generator context.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    resp  = client.search(
        index=INDEX_NAME,
        body={
            "size":  TOP_K_RETRIEVE,
            "query": {"knn": {"embedding": {"vector": q_emb, "k": TOP_K_RETRIEVE}}},
            "_source": ["passage_id", "text"],
        },
    )
    candidates = [hit["_source"] for hit in resp["hits"]["hits"]]

    scores  = reranker.predict([(query, p["text"]) for p in candidates])
    ranked  = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked]


# Generation
PROMPT_TEMPLATE = """\
Answer the question using ONLY the passages below.
Be concise — one sentence or a short phrase.

Passages:
{context}

Question: {question}
Answer:"""

def generate_answer(question: str, passages: list[dict], llm: ChatGoogleGenerativeAI) -> str:
    """Generate an answer using the provided LLM based on retrieved passages."""
    context = "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return llm.invoke(PROMPT_TEMPLATE.format(context=context, question=question)).content.strip()


# Main
def run() -> None:
    """Execute the Improved Retrieval pipeline across the evaluation set."""
    done: set[str] = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["question_id"])
        print(f"Resuming, {len(done)} questions already answered.")

    embedder = SentenceTransformer(str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else BASE_EMBEDDER_ID)
    reranker = CrossEncoder(RERANKER_ID)
    client   = get_client()
    build_index(client, embedder, INDEX_NAME)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    with open(EVAL_PATH) as f:
        eval_set = [json.loads(line) for line in f]

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(eval_set, desc="Config B inference"):
            if item["question_id"] in done:
                continue

            ranked            = retrieve_and_rerank(item["question"], embedder, reranker, client)
            context_passages  = ranked[:TOP_K_RERANK]
            answer            = generate_answer(item["question"], context_passages, llm)

            result = {
                "question_id":            item["question_id"],
                "question":               item["question"],
                "predicted_answer":       answer,
                "retrieved_passage_ids":  [p["passage_id"] for p in ranked[:RECALL_K]],       # for Recall@5/10
                "context_passage_ids":    [p["passage_id"] for p in context_passages],        # what the generator saw
                "gold_answer":            item.get("answer", ""),
                "supporting_passage_ids": item.get("supporting_passage_ids", []),
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

    print(f"Done. Outputs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    fine_tune()
    run()
