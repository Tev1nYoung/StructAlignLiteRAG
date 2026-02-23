import argparse
import os
import time

from src.structalignlite.config import DEFAULT_EMB_NAME, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_NAME, StructAlignRAGConfig
from src.structalignlite.data.dataset_loader import (
    cap_samples,
    get_gold_answers,
    get_gold_docs,
    load_corpus,
    load_samples,
)
from src.structalignlite.structalignrag import StructAlignRAG
from src.structalignlite.utils.logging_utils import setup_logging


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_bool(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")


def main() -> None:
    parser = argparse.ArgumentParser(description="StructAlignLiteRAG (lite_a only) retrieval and QA")
    parser.add_argument("--dataset", type=str, default="sample", help="Dataset name (reproduce/dataset/*.json)")
    parser.add_argument(
        "--run_mode",
        type=str,
        default="rag_qa",
        help="Run mode: retrieval_only (retrieval metrics only) | rag_qa (retrieval+QA)",
    )
    parser.add_argument("--save_root", type=str, default="outputs", help="Save root directory")
    parser.add_argument(
        "--run_tag",
        type=str,
        default=None,
        help="Optional run tag: isolate predictions/metrics under outputs/<dataset>/<llm>_<emb>/metrics/runs/<tag>/",
    )
    parser.add_argument("--force_index_from_scratch", type=str, default="false", help="Rebuild offline index")

    # Models
    parser.add_argument("--llm_base_url", type=str, default=DEFAULT_LLM_BASE_URL, help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default=DEFAULT_LLM_NAME, help="LLM model name")
    parser.add_argument("--embedding_name", type=str, default=DEFAULT_EMB_NAME, help="Embedding model name")

    # Embedding runtime (optional overrides)
    parser.add_argument("--embedding_batch_size", type=int, default=None, help="Embedding batch size override")
    parser.add_argument("--embedding_max_seq_len", type=int, default=None, help="Embedding max sequence length override")
    parser.add_argument(
        "--embedding_dtype",
        type=str,
        default=None,
        help="Embedding dtype override: auto|float16|bfloat16|float32",
    )
    parser.add_argument(
        "--embedding_query_instruction",
        type=str,
        default=None,
        help="Optional query instruction for instruction-tuned embedders (e.g., NV-Embed-v2).",
    )

    # Parallelism
    parser.add_argument("--offline_llm_workers", type=int, default=16, help="Offline LLM parallel workers")
    parser.add_argument("--online_qa_workers", type=int, default=8, help="Online QA parallel workers (per-query)")

    # Optional quick checks
    parser.add_argument("--max_queries", type=int, default=None, help="Optional cap on number of queries")
    parser.add_argument("--query_offset", type=int, default=0, help="Optional starting offset")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="Optional deterministic shuffle seed")

    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)

    run_mode = str(args.run_mode or "").strip().lower()
    if run_mode not in ("retrieval_only", "rag_qa"):
        raise ValueError(f"Invalid --run_mode: {args.run_mode} (expected retrieval_only|rag_qa)")

    dataset = args.dataset
    corpus_path = os.path.join("reproduce", "dataset", f"{dataset}_corpus.json")
    samples_path = os.path.join("reproduce", "dataset", f"{dataset}.json")

    corpus = load_corpus(corpus_path)
    samples = load_samples(samples_path)
    samples = cap_samples(
        samples,
        max_queries=args.max_queries,
        query_offset=int(args.query_offset or 0),
        shuffle_seed=args.shuffle_seed,
    )

    queries = [s["question"] for s in samples]
    qids = [str(s.get("id", i)) for i, s in enumerate(samples)]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name=dataset)
        assert len(queries) == len(gold_docs) == len(gold_answers)
    except Exception:
        gold_docs = None

    run_tag = args.run_tag or f"smoke_{run_mode}"

    cfg = StructAlignRAGConfig(
        dataset=dataset,
        save_root=args.save_root,
        run_tag=run_tag,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=_parse_bool(args.force_index_from_scratch),
        offline_llm_workers=int(args.offline_llm_workers or 16),
        online_qa_workers=int(args.online_qa_workers or 8),
    )

    if args.embedding_batch_size is not None:
        cfg.embedding_batch_size = int(args.embedding_batch_size)
    if args.embedding_max_seq_len is not None:
        cfg.embedding_max_seq_len = int(args.embedding_max_seq_len)
    else:
        emb_name = str(args.embedding_name or "").lower()
        if "nv-embed-v2" in emb_name or "nv_embed_v2" in emb_name:
            cfg.embedding_max_seq_len = 4096
    if args.embedding_dtype is not None:
        cfg.embedding_dtype = str(args.embedding_dtype)
    if args.embedding_query_instruction is not None:
        cfg.embedding_query_instruction = str(args.embedding_query_instruction)

    t0 = time.time()
    rag = StructAlignRAG(cfg)
    t_init = time.time() - t0

    t1 = time.time()
    rag.index(corpus=corpus)
    t_index = time.time() - t1

    run_timing = {"init": float(t_init), "index": float(t_index)}

    if run_mode == "retrieval_only":
        rag.retrieval_only(queries=queries, gold_docs=gold_docs, qids=qids, run_timing_s=run_timing)
    else:
        rag.rag_qa(queries=queries, gold_docs=gold_docs, gold_answers=gold_answers, qids=qids, run_timing_s=run_timing)


if __name__ == "__main__":
    main()

