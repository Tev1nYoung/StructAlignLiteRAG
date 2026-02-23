from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.structalignlite.config import DEFAULT_EMB_NAME, DEFAULT_LLM_BASE_URL, DEFAULT_LLM_NAME, StructAlignLiteConfig  # noqa: E402
from src.structalignlite.data.dataset_loader import get_gold_docs, load_samples  # noqa: E402
from src.structalignlite.embed.factory import build_embedder  # noqa: E402
from src.structalignlite.metrics.metrics import retrieval_recall  # noqa: E402
from src.structalignlite.offline.indexer import OfflineIndexer  # noqa: E402
from src.structalignlite.online.retriever import StructAlignRetriever  # noqa: E402
from src.structalignlite.utils.naming import sanitize_model_name  # noqa: E402


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_md_table(path: Path, headers: List[str], rows: List[List[str]], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / max(len(xs), 1))


class _MemoizedEmbedder:
    def __init__(self, base) -> None:
        self._base = base
        self._cache: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

    def encode(self, texts, instruction: str = ""):
        if isinstance(texts, str):
            key_texts = (texts,)
        else:
            key_texts = tuple(str(x) for x in (texts or []))
        key = (str(instruction or ""), key_texts)
        if key in self._cache:
            return self._cache[key]
        out = self._base.encode(list(key_texts), instruction=str(instruction or ""))
        self._cache[key] = out
        return out

    @property
    def tokenizer(self):
        return getattr(self._base, "tokenizer", None)


def _run_one_variant(
    *,
    dataset: str,
    questions: List[str],
    query_dags: List[Dict[str, Any]],
    gold_docs: List[List[str]],
    index: Dict[str, Any],
    embedder,
    cfg: StructAlignLiteConfig,
) -> Dict[str, Any]:
    retriever = StructAlignRetriever(config=cfg)
    retrieved_docs_all: List[List[str]] = []

    latencies: List[float] = []
    evidence_tokens: List[int] = []
    selected_nodes_n: List[int] = []
    dijkstra_added_n: List[int] = []

    t0 = time.time()
    for q, dag in zip(questions, query_dags):
        qt0 = time.time()
        rr = retriever.retrieve(question=q, query_dag=dag, index=index, embedder=embedder, llm=None)
        latencies.append(float(time.time() - qt0))
        retrieved_docs_all.append(list(rr.retrieved_docs or []))
        dbg = rr.debug or {}
        evidence_tokens.append(int(dbg.get("evidence_tokens") or 0))
        selected_nodes_n.append(int(len(dbg.get("selected_nodes") or [])))
        dijkstra_added_n.append(int(dbg.get("dijkstra_added_nodes") or 0))
    total_s = float(time.time() - t0)

    rec = retrieval_recall(gold_docs=gold_docs, retrieved_docs=retrieved_docs_all, k_list=[2, 5])
    return {
        "Recall@2": float(rec.get("Recall@2") or 0.0),
        "Recall@5": float(rec.get("Recall@5") or 0.0),
        "LatencyAvgS": _mean(latencies),
        "EvidenceTokens": int(round(_mean([float(x) for x in evidence_tokens]))),
        "SelectedNodes": int(round(_mean([float(x) for x in selected_nodes_n]))),
        "DijkstraAddedNodes": int(round(_mean([float(x) for x in dijkstra_added_n]))),
        "TotalS": round(total_s, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-free ablation for PPR/Dijkstra on the evidence graph (Recall@2/5).")
    parser.add_argument("--save_root", type=str, default="outputs")
    parser.add_argument("--llm_base_url", type=str, default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm_name", type=str, default=DEFAULT_LLM_NAME)
    parser.add_argument("--embedding_name", type=str, default=DEFAULT_EMB_NAME)
    parser.add_argument("--embedding_batch_size", type=int, default=16)
    parser.add_argument("--embedding_max_seq_len", type=int, default=512)
    parser.add_argument("--embedding_dtype", type=str, default="auto")
    parser.add_argument("--embedding_query_instruction", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=str(_ROOT / "reports" / "contriever"))

    # Baseline qa_predictions.json run tags (must contain query_dag).
    parser.add_argument("--run_tag_2wiki", type=str, default="nocache_full_rag_qa")
    parser.add_argument("--run_tag_hotpot", type=str, default="lite_hotpot_full")
    parser.add_argument("--run_tag_musique", type=str, default="lite_musique_full")
    args = parser.parse_args()

    datasets_and_tags = [
        ("2wikimultihopqa", args.run_tag_2wiki),
        ("hotpotqa", args.run_tag_hotpot),
        ("musique", args.run_tag_musique),
    ]

    emb_instruction = args.embedding_query_instruction
    if emb_instruction is None:
        emb_instruction = "Given a question, retrieve relevant documents that best answer the question."

    base_embedder = build_embedder(
        model_name=args.embedding_name,
        batch_size=int(args.embedding_batch_size),
        max_length=int(args.embedding_max_seq_len),
        normalize=True,
        dtype=str(args.embedding_dtype),
    )

    llm_tag = sanitize_model_name(args.llm_name)
    emb_tag = sanitize_model_name(args.embedding_name)

    variants = [
        ("baseline", {"enable_local_propagation": True, "enable_dijkstra_connect": True}),
        ("wo_ppr", {"enable_local_propagation": False, "enable_dijkstra_connect": True}),
        ("wo_dijkstra", {"enable_local_propagation": True, "enable_dijkstra_connect": False}),
        ("wo_both", {"enable_local_propagation": False, "enable_dijkstra_connect": False}),
    ]

    rows: List[Dict[str, Any]] = []

    for dataset, run_tag in datasets_and_tags:
        print(f"[ablate] dataset={dataset} | loading predictions + gold docs ...", flush=True)
        embedder = _MemoizedEmbedder(base_embedder)
        meta_dir = _ROOT / args.save_root / dataset / f"{llm_tag}_{emb_tag}"
        preds_path = meta_dir / "metrics" / "runs" / run_tag / "qa_predictions.json"
        if not preds_path.exists():
            raise FileNotFoundError(str(preds_path))

        preds = _load_json(preds_path)
        if not isinstance(preds, list):
            raise ValueError(f"Invalid predictions JSON (expected list): {preds_path}")

        samples_path = _ROOT / "reproduce" / "dataset" / f"{dataset}.json"
        samples = load_samples(str(samples_path))
        gold_docs = get_gold_docs(samples, dataset_name=dataset)

        if len(preds) != len(gold_docs):
            raise ValueError(f"{dataset}: predictions length mismatch (preds={len(preds)} gold={len(gold_docs)})")

        questions = [str(p.get("question") or "") for p in preds]
        query_dags = [p.get("query_dag") or {} for p in preds]

        # Load offline index once per dataset.
        print(f"[ablate] dataset={dataset} | loading offline index ...", flush=True)
        reusable_dir = meta_dir / "reusable"
        cfg_base = StructAlignLiteConfig(
            dataset=dataset,
            save_root=str(args.save_root),
            run_tag=None,
            llm_base_url=str(args.llm_base_url),
            llm_name=str(args.llm_name),
            embedding_model_name=str(args.embedding_name),
            force_index_from_scratch=False,
            embedding_batch_size=int(args.embedding_batch_size),
            embedding_max_seq_len=int(args.embedding_max_seq_len),
            embedding_dtype=str(args.embedding_dtype),
        )
        cfg_base.embedding_query_instruction = str(emb_instruction or "")

        indexer = OfflineIndexer(config=cfg_base, meta_dir=str(reusable_dir))
        index = indexer.load_index()

        baseline_metrics: Optional[Dict[str, Any]] = None
        for variant_name, toggles in variants:
            print(f"[ablate] dataset={dataset} | variant={variant_name} | start", flush=True)
            cfg = replace(cfg_base)
            for k, v in toggles.items():
                setattr(cfg, k, v)

            m = _run_one_variant(
                dataset=dataset,
                questions=questions,
                query_dags=query_dags,
                gold_docs=gold_docs,
                index=index,
                embedder=embedder,
                cfg=cfg,
            )
            if variant_name == "baseline":
                baseline_metrics = m

            r2 = float(m.get("Recall@2") or 0.0)
            r5 = float(m.get("Recall@5") or 0.0)
            b2 = float((baseline_metrics or {}).get("Recall@2") or 0.0)
            b5 = float((baseline_metrics or {}).get("Recall@5") or 0.0)

            row = {
                "dataset": dataset,
                "variant": variant_name,
                "R@2": _fmt(r2),
                "R@5": _fmt(r5),
                "dR@2_vs_baseline": _fmt(r2 - b2) if baseline_metrics else "",
                "dR@5_vs_baseline": _fmt(r5 - b5) if baseline_metrics else "",
                "LatencyAvgS": _fmt(m.get("LatencyAvgS")),
                "EvidenceTokens": str(m.get("EvidenceTokens")),
                "SelectedNodes": str(m.get("SelectedNodes")),
                "DijkstraAddedNodes": str(m.get("DijkstraAddedNodes")),
                "TotalS": str(m.get("TotalS")),
                "predictions_tag": run_tag,
            }
            rows.append(row)

    out_dir = Path(args.out_dir)
    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(out_dir / "graph_component_ablation.csv", rows, fieldnames)

    md_rows = [[r[h] for h in fieldnames] for r in rows]
    _write_md_table(
        out_dir / "graph_component_ablation.md",
        headers=fieldnames,
        rows=md_rows,
        title="Graph Component Ablation (LLM-free, Recall@2/5)",
    )


if __name__ == "__main__":
    main()
