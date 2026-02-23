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

## NV-Embed-v2 (other machine)

Recommended workflow: run `retrieval_only` first to check `Recall@2/5` before spending QA budget, then run `rag_qa` if it looks competitive.

Retrieval-only (build index once):

```powershell
python main.py --dataset 2wikimultihopqa --run_mode retrieval_only --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_retrieval --force_index_from_scratch true --offline_llm_workers 16 --online_qa_workers 8
python main.py --dataset hotpotqa         --run_mode retrieval_only --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_retrieval --force_index_from_scratch true --offline_llm_workers 16 --online_qa_workers 8
python main.py --dataset musique         --run_mode retrieval_only --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_retrieval --force_index_from_scratch true --offline_llm_workers 16 --online_qa_workers 8
```

RAG QA (reuse index):

```powershell
python main.py --dataset 2wikimultihopqa --run_mode rag_qa --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_rag_qa --force_index_from_scratch false --offline_llm_workers 16 --online_qa_workers 8
python main.py --dataset hotpotqa         --run_mode rag_qa --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_rag_qa --force_index_from_scratch false --offline_llm_workers 16 --online_qa_workers 8
python main.py --dataset musique         --run_mode rag_qa --embedding_name nvidia/NV-Embed-v2 --run_tag nv2_rag_qa --force_index_from_scratch false --offline_llm_workers 16 --online_qa_workers 8
```

### NV2: serial runner (7 datasets)

If you want a single command to run the full suite serially with NV2 on another machine (`sample` + `case_study_university` + 3 multi-hop + `nq_rear` + `popqa`), use:

```powershell
conda activate hipporag
cd C:\Project\StructAlignLiteRAG

# Fresh run (delete existing outputs for those datasets) + run retrieval_only then rag_qa
powershell -ExecutionPolicy Bypass -File tools\run_nv2_serial.ps1 -CleanOutputs -ForceRebuild
```

Useful flags:
- `-Datasets @("sample","case_study_university","2wikimultihopqa","hotpotqa","musique","nq_rear","popqa")` (override dataset list)
- `-RetrievalOnly` (only run `retrieval_only` for all datasets)
- `-SkipQA` (same as above; keep retrieval_only only)
- `-Resume` (skip runs when the target predictions file already exists)
- `-OfflineLLMWorkers 16 -OnlineQAWorkers 8` (defaults already match the project)
