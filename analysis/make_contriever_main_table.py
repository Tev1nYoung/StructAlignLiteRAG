from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sanitize_model_name(name: str) -> str:
    if name is None:
        return "none"
    s = str(name).strip()
    s = s.replace("\\", "_").replace("/", "_")
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    s2 = "".join(out)
    while "__" in s2:
        s2 = s2.replace("__", "_")
    s2 = s2.strip("_")
    return s2 or "none"


def _load_json_blocks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8")
    dec = json.JSONDecoder()
    blocks: List[Dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(text, i)
        if isinstance(obj, dict):
            blocks.append(obj)
        i = j
    return blocks


def _pick_best_json_block(
    path: Path,
    *,
    expected_dataset: Optional[str] = None,
    expected_run_mode: Optional[str] = "rag_qa",
    expected_num_queries: Optional[int] = 1000,
    score_metric: str = "Recall@5",
) -> Dict[str, Any]:
    blocks = _load_json_blocks(path)
    if not blocks:
        raise ValueError(f"No JSON blocks found in {path}")

    candidates: List[Dict[str, Any]] = []
    for b in blocks:
        if expected_dataset is not None and b.get("dataset") != expected_dataset:
            continue
        if expected_run_mode is not None and b.get("run_mode") != expected_run_mode:
            continue
        if expected_num_queries is not None and b.get("num_queries") != expected_num_queries:
            continue
        candidates.append(b)

    if not candidates:
        candidates = blocks

    def _score(b: Dict[str, Any]) -> float:
        retrieval = b.get("retrieval_metrics") or {}
        v = retrieval.get(score_metric)
        try:
            return float(v)
        except Exception:
            return float("-inf")

    best_score = max(_score(b) for b in candidates)
    best = [b for b in candidates if _score(b) == best_score]
    return best[-1]


def _metrics_from_record(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    retrieval = record.get("retrieval_metrics") or {}
    qa = record.get("qa_metrics") or {}
    return {
        "R@2": retrieval.get("Recall@2"),
        "R@5": retrieval.get("Recall@5"),
        "EM": qa.get("ExactMatch"),
        "F1": qa.get("F1"),
    }


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.4f}"
    except Exception:
        return str(x)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Contriever main table (StructAlignLiteRAG vs HippoRAG).")
    parser.add_argument("--llm_name", type=str, default="meta/llama-3.3-70b-instruct")
    parser.add_argument("--embedding_name", type=str, default="facebook/contriever")
    parser.add_argument("--hipporag_root", type=str, default=str(Path(__file__).resolve().parents[1].parent / "HippoRAG"))
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "contriever"),
        help="Output directory for main_table.{md,csv}",
    )
    parser.add_argument("--run_tag_2wiki", type=str, default="nocache_full_rag_qa")
    parser.add_argument("--run_tag_hotpot", type=str, default="lite_hotpot_full")
    parser.add_argument("--run_tag_musique", type=str, default="lite_musique_full")
    args = parser.parse_args()

    struct_root = Path(__file__).resolve().parents[1]
    hippo_root = Path(args.hipporag_root).resolve()

    llm_tag = _sanitize_model_name(args.llm_name)
    emb_tag = _sanitize_model_name(args.embedding_name)
    nv2_tag = _sanitize_model_name("nvidia/NV-Embed-v2")

    run_tags = {
        "2wikimultihopqa": args.run_tag_2wiki,
        "hotpotqa": args.run_tag_hotpot,
        "musique": args.run_tag_musique,
    }

    rows_csv: List[Dict[str, Any]] = []
    rows_md: List[List[str]] = []

    for dataset, run_tag in run_tags.items():
        meta_dir = struct_root / "outputs" / dataset / f"{llm_tag}_{emb_tag}"
        our_metrics_path = meta_dir / "metrics" / "runs" / run_tag / "metrics_log.jsonl"
        our_record = _pick_best_json_block(
            our_metrics_path,
            expected_dataset=dataset,
            expected_run_mode="rag_qa",
            expected_num_queries=1000,
        )
        our_m = _metrics_from_record(our_record)

        hippo_meta_dir_c = hippo_root / "outputs" / dataset / f"{llm_tag}_{emb_tag}"
        hippo_record_c = _pick_best_json_block(
            hippo_meta_dir_c / "metrics_log.jsonl",
            expected_dataset=dataset,
            expected_run_mode="rag_qa",
            expected_num_queries=1000,
        )
        hippo_m_c = _metrics_from_record(hippo_record_c)

        hippo_meta_dir_nv2 = hippo_root / "outputs" / dataset / f"{llm_tag}_{nv2_tag}"
        hippo_record_nv2 = _pick_best_json_block(
            hippo_meta_dir_nv2 / "metrics_log.jsonl",
            expected_dataset=dataset,
            expected_run_mode="rag_qa",
            expected_num_queries=1000,
        )
        hippo_m_nv2 = _metrics_from_record(hippo_record_nv2)

        for method, m in [
            (f"StructAlignLiteRAG({args.embedding_name})", our_m),
            (f"HippoRAG({args.embedding_name})", hippo_m_c),
        ]:
            row = {
                "dataset": dataset,
                "method": method,
                "R@2": _fmt(m["R@2"]),
                "R@5": _fmt(m["R@5"]),
                "EM": _fmt(m["EM"]),
                "F1": _fmt(m["F1"]),
                "HippoRAG(NV2)_R@2": _fmt(hippo_m_nv2["R@2"]),
                "HippoRAG(NV2)_R@5": _fmt(hippo_m_nv2["R@5"]),
                "HippoRAG(NV2)_EM": _fmt(hippo_m_nv2["EM"]),
                "HippoRAG(NV2)_F1": _fmt(hippo_m_nv2["F1"]),
            }
            rows_csv.append(row)
            rows_md.append([row[h] for h in row.keys()])

    out_dir = Path(args.out_dir)
    headers = list(rows_csv[0].keys()) if rows_csv else []
    _write_csv(out_dir / "main_table.csv", rows_csv, headers)
    _write_md_table(out_dir / "main_table.md", headers, rows_md, title="Contriever Main Table (StructAlignLiteRAG vs HippoRAG)")


if __name__ == "__main__":
    main()
