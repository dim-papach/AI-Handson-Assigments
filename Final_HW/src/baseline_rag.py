"""
Configuration A — Baseline RAG
Off-the-shelf embedder (BAAI/bge-small-en-v1.5) + top-k HNSW retrieval via OpenSearch.
Generator: Google Gemini (via langchain-google-genai), temperature=0.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Constants
EMBEDDER_ID  = "BAAI/bge-small-en-v1.5"
INDEX_NAME   = "corpus_config_a"
TOP_K        = 5
BATCH_SIZE   = 256
RANDOM_STATE = 42

ROOT     = Path(__file__).parent.parent
load_dotenv(ROOT.parent / ".env")
DATA_DIR = ROOT / "data"
RES_DIR  = ROOT / "results"
RES_DIR.mkdir(exist_ok=True)

CORPUS_PATH  = DATA_DIR / "corpus.jsonl"
EVAL_PATH    = DATA_DIR / "eval_set.jsonl"
OUTPUT_PATH  = RES_DIR / "config_a_outputs.jsonl"


# OpenSearch client
def get_client() -> OpenSearch:
    """Initialize and return an OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        timeout=60,
    )


# Index building
def build_index(client: OpenSearch, embedder: SentenceTransformer) -> None:
    """
    Build the OpenSearch index if it does not exist, and populate it with
    embedded passages from the corpus.
    """
    if client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists — skipping build.")
        return

    dim = embedder.get_sentence_embedding_dimension()
    client.indices.create(
        index=INDEX_NAME,
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

    print("Loading corpus …")
    with open(CORPUS_PATH) as f:
        passages = [json.loads(line) for line in f]

    print(f"Embedding {len(passages):,} passages (CPU — this takes a while) …")
    texts      = [p["text"] for p in passages]
    embeddings = embedder.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    def _actions():
        for p, emb in zip(passages, embeddings):
            yield {
                "_index": INDEX_NAME,
                "_id":    p["passage_id"],
                "_source": {
                    "passage_id": p["passage_id"],
                    "text":       p["text"],
                    "embedding":  emb.tolist(),
                },
            }

    print("Bulk-indexing …")
    helpers.bulk(client, _actions(), chunk_size=500, request_timeout=60)
    client.indices.refresh(index=INDEX_NAME)
    print(f"Indexed {len(passages):,} passages into '{INDEX_NAME}'.")


# Retrieval
def retrieve(
    query: str,
    embedder: SentenceTransformer,
    client: OpenSearch,
    k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the top-k most relevant passages for a given query using
    cosine similarity in OpenSearch.
    """
    q_emb = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    resp  = client.search(
        index=INDEX_NAME,
        body={
            "size":  k,
            "query": {"knn": {"embedding": {"vector": q_emb, "k": k}}},
            "_source": ["passage_id", "text"],
        },
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


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
    msg     = PROMPT_TEMPLATE.format(context=context, question=question)
    return llm.invoke(msg).content.strip()


# Main
def run() -> None:
    """Execute the Baseline RAG pipeline across the evaluation set."""
    # load already-answered questions to avoid re-calling the API
    done: set[str] = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["question_id"])
        print(f"Resuming — {len(done)} questions already answered.")

    embedder = SentenceTransformer(EMBEDDER_ID)
    client   = get_client()
    build_index(client, embedder)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    with open(EVAL_PATH) as f:
        eval_set = [json.loads(line) for line in f]

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(eval_set, desc="Config A inference"):
            if item["question_id"] in done:
                continue

            passages = retrieve(item["question"], embedder, client)
            answer   = generate_answer(item["question"], passages, llm)

            result = {
                "question_id":           item["question_id"],
                "question":              item["question"],
                "predicted_answer":      answer,
                "retrieved_passage_ids": [p["passage_id"] for p in passages],
                "gold_answer":           item.get("answer", ""),
                "supporting_passage_ids": item.get("supporting_passage_ids", []),
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

    print(f"Done. Outputs saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
