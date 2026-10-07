# ai-lab: a measured, local RAG pipeline

A small RAG system built and evaluated locally (Ollama: `llama3.2:3b`, `nomic-embed-text`; ChromaDB). The goal was not a demo but a measured system: every design choice below was tested against an eval set.

## Pipeline
1. `01-rag/ingest.py`: sentence-window chunking (2 sentences, stride 1) -> embeddings -> ChromaDB
2. `02-structured/ask_structured.py`: retrieve -> distance threshold -> LLM returns validated JSON (`answer`, `answerable`, `citations`, `evidence`) -> checks -> retry or abstain
3. `03-evals/`: easy set (15 questions), hard set (10 questions; paraphrases and near-miss questions whose topic is in the docs but whose answer is not)

## Results (threshold 0.95)

| Configuration | Hard: answered | Hard: abstained | Easy: answered | Easy: abstained | Hard-set time |
|---|---|---|---|---|---|
| Evidence quote check only | 4/4 | 3/6 | 10/10 | 5/5 | 249s |
| + 3b LLM verifier | 1/4 | 6/6 | 6/10 | 5/5 | 361s |
| + 7b LLM verifier | 4/4 | 6/6 | 6/10 | 5/5 | 1410s |
| + lexical grounded check (default) | 3/4 | 6/6 | 9/10 | 5/5 | 198s |

## Findings
- **Retrieval was not the main problem.** 13 of 14 answerable questions had the right source at rank 1. But distance alone cannot separate answerable questions from near-misses: near-miss questions had closest-chunk distances of 0.57-0.65, while some answerable ones sat at 0.87-0.92. A single threshold only removes clearly off-topic questions cheaply (no LLM call).
- **The 3b model answers from its own knowledge** (parametric leakage): it returned correct-looking commands for questions the documents do not cover, and quoted an unrelated but real sentence as "evidence". A check that the quote exists in the context did not catch this.
- **LLM verifiers were noisy.** The 3b verifier rejected correct answers and accepted an irrelevant quote. The 7b verifier fixed the hard set but still rejected 4 of 10 easy questions and was about 7x slower than the lexical check.
- **A cheap lexical check worked best here:** require most answer tokens to appear in the quoted evidence. It matched the 7b verifier on abstention at roughly the cost of no verifier.

## Limitations
- Only 25 questions, written by me; design choices and thresholds were tuned on the same questions (no held-out set), so the numbers are optimistic.
- The grounded check is lexical: a correct paraphrased answer can be rejected (1 miss on each set).
- Tiny corpus (5 short documents); retrieval behavior on larger corpora is untested.

## Run

    ollama pull llama3.2:3b && ollama pull nomic-embed-text
    python 01-rag/ingest.py
    python 02-structured/ask_structured.py "How do I start a FastAPI server?"
    python 03-evals/run_evals.py --max-distance 0.95
    python 03-evals/run_evals.py --eval-file 03-evals/eval_set_hard.json --max-distance 0.95

## Agent experiment (04-agents)

A tool-calling agent (llama3.2:3b) with `search_docs` and `list_sources` tools, evaluated on the same 25 questions. Answerable = "searched and did not abstain" (looser than the pipeline metric, which also checks the citation); abstention is detected by string match.

| Configuration | Answerable answered | Abstention | Abstained without searching | Avg steps | Time |
|---|---|---|---|---|---|
| Baseline | 7/14 | 5/11 | n/a | 1.7 | 554s |
| Distance gate in tool | 8/14 | 6/11 | n/a | 1.7 | 535s |
| Gate + forced search | 10/14 | 3/11 | n/a | 2.3 | 685s |
| Tool = full RAG pipeline | 3/14 | 8/11 | 5 | 1.6 | 627s |
| Tool = pipeline + forced search | 7/14 | 5/11 | 2 | 2.4 | 978s |
| (reference) pipeline without agent | 12/14 | 11/11 | n/a | n/a | about 300s |

### Agent findings
- For this 3b model the fixed pipeline beat every agent configuration on recall, abstention and time. The agent's value is multi-step questions (for example comparing git and docker), which this eval set barely covers.
- The agent often skipped the search tool and answered or abstained directly. Forcing a search raised recall but lowered abstention in both tool variants.
- A high abstention score can be misleading: with the pipeline tool, 5 of the 8 correct abstentions happened without any search, and the same behaviour shows up as missed answerable questions.
- Without the gate, the agent answered near-miss questions from its own knowledge (for example pip uninstall) even after searching.

### Agent limitations
- Only the 3b model was tested; a larger model may behave differently.
- Pipeline and agent numbers use different metrics and are not strictly comparable.
- Abstention detection is a string match; the guardrails in the pipeline tool were tuned on these same questions.
