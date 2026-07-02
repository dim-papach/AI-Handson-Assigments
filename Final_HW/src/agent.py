"""
Configuration C — Iterative-Retrieval Agent (ReAct loop)
Built on top of Config B's retrieval pipeline.
Saves execution traces to src/traces/.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

ROOT     = Path(__file__).parent.parent
load_dotenv(ROOT.parent / ".env")

# Constants
EMBEDDER_PATH  = ROOT / "models" / "finetuned_embedder"
BASE_EMBEDDER  = "BAAI/bge-small-en-v1.5"
RERANKER_ID    = "BAAI/bge-reranker-base"
INDEX_NAME     = "corpus_config_b"
TOP_K_RETRIEVE = 20
TOP_K_RERANK   = 3
MAX_STEPS      = 5
RANDOM_STATE   = 42

DATA_DIR   = ROOT / "data"
RES_DIR    = ROOT / "results"
TRACES_DIR = Path(__file__).parent / "traces"
RES_DIR.mkdir(exist_ok=True)
TRACES_DIR.mkdir(exist_ok=True)

EVAL_PATH   = DATA_DIR / "eval_set.jsonl"
OUTPUT_PATH = RES_DIR / "config_c_outputs.jsonl"


# Helpers
def get_client() -> OpenSearch:
    """Initialize and return an OpenSearch client."""
    return OpenSearch(
        hosts=[{"host": os.getenv("OPENSEARCH_HOST", "localhost"), "port": 9200}],
        use_ssl=False,
        verify_certs=False,
        timeout=60,
    )


def retrieve_tool(
    query: str,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    client: OpenSearch,
) -> list[dict]:
    """The agent's single tool: retrieve(query) -> list[passage]."""
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
    scores     = reranker.predict([(query, p["text"]) for p in candidates])
    ranked     = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [p for _, p in ranked[:TOP_K_RERANK]]


# ReAct prompts
SYSTEM_PROMPT = """\
You are a question-answering agent with access to a retrieval tool.
To answer a question you can call retrieve() one or more times.

At each step output EXACTLY one of:
  Thought: <your reasoning>
  Action: retrieve("<query>")

OR, when you have enough information:
  Thought: <final reasoning>
  Answer: <concise answer>

Rules:
- Use retrieve() whenever you need more information.
- You may call retrieve() at most {max_steps} times total.
- Once you write "Answer:", stop immediately.
- Be concise — answer in one sentence or a short phrase.
"""

STEP_TEMPLATE = """\
Question: {question}

{history}"""


def parse_action(text: str) -> str | None:
    """Extract query from 'Action: retrieve("<query>")' or None."""
    import re
    m = re.search(r'retrieve\s*\(\s*["\'](.+?)["\']\s*\)', text)
    return m.group(1) if m else None


def parse_answer(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip().lower().startswith("answer:"):
            return line.split(":", 1)[1].strip()
    return None


# Agent loop
def run_agent(
    question: str,
    embedder: SentenceTransformer,
    reranker: CrossEncoder,
    client: OpenSearch,
    llm: ChatGoogleGenerativeAI,
) -> dict:
    """
    Execute the iterative ReAct loop for a given question, allowing the agent to
    retrieve information multiple times before formulating a final answer.
    """
    history      = ""
    trace_steps  = []
    final_answer = ""
    retrieved_ids: list[str] = []

    system = SYSTEM_PROMPT.format(max_steps=MAX_STEPS)

    for step in range(MAX_STEPS + 1):
        prompt = STEP_TEMPLATE.format(question=question, history=history)
        # Gemini API occasionally drops the connection mid-session; retry with backoff
        for attempt in range(4):
            try:
                import time
                raw = llm.invoke([
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ]).content
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        response = (
            "".join(c if isinstance(c, str) else c.get("text", "") for c in raw)
            if isinstance(raw, list) else raw
        ).strip()

        # check for final answer
        answer = parse_answer(response)
        if answer:
            final_answer = answer
            trace_steps.append({"type": "answer", "content": response})
            break

        # check for retrieval action
        query = parse_action(response)
        if query and step < MAX_STEPS:
            passages     = retrieve_tool(query, embedder, reranker, client)
            retrieved_ids.extend(p["passage_id"] for p in passages)
            observation  = "\n".join(f"- {p['text'][:300]}" for p in passages)
            trace_steps.append({
                "type":        "thought_action",
                "content":     response,
                "query":       query,
                "observation": observation,
            })
            history += (
                f"{response}\n"
                f"Observation: {observation}\n\n"
            )
        else:
            # no valid action or budget exhausted — take best guess
            final_answer = response.replace("Thought:", "").strip()
            trace_steps.append({"type": "fallback", "content": response})
            break

    return {
        "final_answer": final_answer,
        "trace":        trace_steps,
        "retrieved_ids": list(dict.fromkeys(retrieved_ids)),
        "steps_used":   len([s for s in trace_steps if s["type"] == "thought_action"]),
    }


# Main
def run() -> None:
    """Execute the iterative-retrieval agent pipeline across the evaluation set."""
    done: set[str] = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH) as f:
            for line in f:
                done.add(json.loads(line)["question_id"])
        print(f"Resuming — {len(done)} questions already answered.")

    embedder_path = str(EMBEDDER_PATH) if (EMBEDDER_PATH / "config.json").exists() else BASE_EMBEDDER
    embedder = SentenceTransformer(embedder_path)
    reranker = CrossEncoder(RERANKER_ID)
    client   = get_client()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    with open(EVAL_PATH) as f:
        eval_set = [json.loads(line) for line in f]

    with open(OUTPUT_PATH, "a") as out:
        for item in tqdm(eval_set, desc="Config C inference"):
            if item["question_id"] in done:
                continue

            result_agent = run_agent(item["question"], embedder, reranker, client, llm)

            result = {
                "question_id":            item["question_id"],
                "question":               item["question"],
                "predicted_answer":       result_agent["final_answer"],
                "retrieved_passage_ids":  result_agent["retrieved_ids"],
                "gold_answer":            item.get("answer", ""),
                "supporting_passage_ids": item.get("supporting_passage_ids", []),
                "steps_used":             result_agent["steps_used"],
            }
            out.write(json.dumps(result) + "\n")
            out.flush()

            # save full trace as separate file
            trace_file = TRACES_DIR / f"{item['question_id']}.json"
            with open(trace_file, "w") as tf:
                json.dump({
                    "question":     item["question"],
                    "gold_answer":  item.get("answer", ""),
                    "trace":        result_agent["trace"],
                    "final_answer": result_agent["final_answer"],
                }, tf, indent=2)

    print(f"Done. Outputs saved to {OUTPUT_PATH}")
    print(f"Traces saved to {TRACES_DIR}/")


if __name__ == "__main__":
    run()
