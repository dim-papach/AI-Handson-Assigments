# FINAL ASSIGNMENT

## From RAG to a Tool-Using Retrieval Agent

Build and benchmark three progressively capable retrieval systems on a question-answering dataset:
- A baseline RAG pipeline: off-the-shelf embedder, naive top-k retrieval 
- An improved retrieval system with a fine-tuned embedder
- An iterative-retrieval agent
---

### Benchmark & Corpus Choice

For our project we chose the HotpotQA benchmark. It uses multi-hop reasoning, which means that the questions require reasoning over multiple sentences to answer. For example:

- "What is the population of the country where the director of Inception was born?"

- Hop 1: Passage "Inception" → finds that the director is Christopher Nolan.
- Hop 2: Passage "Christopher Nolan" → finds that he was born in London, UK.
- Hop 3 (or continuation of the 2nd): Passage "United Kingdom" → finds the population.

It also uses Sentence-level supporting facts, which means each context passage  is split into individual sentences, and the ground truth doesn't just say "this passage is relevant" — it points to the exact sentence(s) inside that passage that justify the answer. So the annotation is `(passage_title, sentence_index)`, not just `passage_title`. 

Those things make the HotpotQA benchmark particularly suitable for astrophysics questions, which are often multi-hop and require reasoning over multiple sentences to answer.

### Compute and model setup

**Hardware**: CPU-only laptop — Intel Core Ultra 7 165U (14 threads), 16 GB RAM, no discrete GPU (`torch.cuda.is_available() == False`). All embedding, fine-tuning, and reranking steps run on CPU; PyTorch 2.11 was installed with a CUDA build but falls back to CPU since no NVIDIA GPU is present.

**Generator**: Google Gemini (`gemini-2.5-flash`) via the `langchain-google-genai` API, `temperature=0`, API key loaded from `.env` (`GEMINI_API_KEY`).

With no GPU available, running a local 7B+ instruction model (e.g. Llama-3.1-8B, Qwen2.5-7B) for generation was not practical — CPU-only inference at that scale would be too slow for 500 evaluation questions × 3 configurations, especially for Config C where the agent issues up to 5 sequential LLM calls per question. Gemini Flash offloads generation to the cloud while keeping local compute free for the embedding/retrieval components.

For the same reason, all locally-run models were deliberately kept small: `BAAI/bge-small-en-v1.5` (33M params) as the embedder and `BAAI/bge-reranker-base` for reranking — both from the assignment's suggested CPU-friendly options, keeping bi-encoder fine-tuning (Task 3a) and index-building tractable on CPU within reasonable time.

We also set `temperature=0` and `random_state=42` throughout (data sampling, embedder training, generation) to keep results deterministic and reproducible.

### Configuration A

Configuration A uses `BAAI/bge-small-en-v1.5` (33M params) as an off-the-shelf embedder, with no fine-tuning applied. No chunking was used, since HotpotQA passages are already short — each one is built as `"{title}: {sentences}"` by concatenating the title with its sentences — so the assignment's guidance to chunk only long passages did not apply here. Passages are indexed in OpenSearch using HNSW (`nmslib` engine, `cosinesimil` space, `ef_construction=128, m=16`), and retrieval is a plain top-`k=5` vector kNN search with no reranking. The generator is `gemini-2.5-flash` at `temperature=0`. The exact prompt template used is:

```
Answer the question using ONLY the passages below.
Be concise — one sentence or a short phrase.

Passages:
{context}

Question: {question}
Answer:
```
where `{context}` is the top-5 retrieved passages, numbered `[1]`, `[2]`, and so on.

### Configuration B

## 4. Configuration B — Improved Retrieval

We fine-tune the same base model as Configuration A (`BAAI/bge-small-en-v1.5`) using `MultipleNegativesRankingLoss` with in-batch negatives, on 5,000 (query, positive-passage) pairs sampled from HotpotQA's train split only (`data/train_pairs.jsonl`), where the positive passage for each question is its supporting-fact passage. We train for 2 epochs, batch size 8, 100 warmup steps, `random_state=42`, and save the result to `models/finetuned_embedder/`.

We were surprised by the diagnostic Recall@5 result: on 200 held-out questions, using the bi-encoder alone with no reranking, Recall@5 actually dropped from 0.970 before fine-tuning to 0.930 after (Δ = −0.040). Fine-tuning made raw retrieval worse on this benchmark, not better — we dig into why in Section 7.

On top of the (worse) fine-tuned embedder we add cross-encoder reranking: the bi-encoder retrieves the top-20 candidates, and `BAAI/bge-reranker-base` re-scores all 20 as full query-passage pairs, keeping the top-3 for generation. We picked reranking specifically because it doesn't depend on embedding quality the way retrieval does — since fine-tuning wasn't helping the bi-encoder, we wanted a technique that could recover precision independently. And it does: on the full evaluation set, Configuration B's combined pipeline still beats Configuration A on Answer EM (0.406 → 0.436) and Answer F1 (0.551 → 0.580), and has the best MRR@10 of all three configurations (0.944). So the reranker, not the fine-tuning, looks like what's actually driving the improvement.

### Configuration C

### Evaluation results

### Findings and error analysis

### Installation and execution