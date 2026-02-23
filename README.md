# StructAlignLiteRAG

Frozen **lite_a-only** pipeline extracted from StructAlignRAG for cleaner iteration.

## Environment

Run under your existing conda env:

```powershell
conda activate hipporag
```

`llm_key.txt` is read from the project root (gitignored). Alternatively you can set `OPENAI_API_KEY` in the environment.

## Smoke (sample)

Retrieval-only:

```powershell
python main.py --dataset sample --run_mode retrieval_only --embedding_name facebook/contriever --run_tag smoke_retrieval
```

RAG QA:

```powershell
python main.py --dataset sample --run_mode rag_qa --embedding_name facebook/contriever --run_tag smoke_qa
```

Outputs are written under:

`outputs/<dataset>/<llm>_<emb>/metrics/runs/<run_tag>/`

