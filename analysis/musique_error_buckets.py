from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Set, Tuple


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


def _normalize_answer(answer: str) -> str:
    answer = answer or ""
    answer = answer.lower()
    exclude = set(string.punctuation)
    answer = "".join(ch for ch in answer if ch not in exclude)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    answer = " ".join(answer.split())
    return answer


def _f1_single(gold: str, pred: str) -> float:
    gold_tokens = _normalize_answer(gold).split()
    pred_tokens = _normalize_answer(pred).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / max(len(pred_tokens), 1)
    recall = 1.0 * num_same / max(len(gold_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _qa_em_f1_single(gold_set: Set[str], pred: str) -> Tuple[float, float]:
    pred = pred or ""
    em = 0.0
    f1 = 0.0
    for g in list(gold_set):
        if _normalize_answer(g) == _normalize_answer(pred):
            em = 1.0
        f1 = max(f1, _f1_single(g, pred))
    return em, f1


def _get_gold_answers(samples: List[Dict[str, Any]]) -> List[Set[str]]:
    gold_answers: List[Set[str]] = []
    for sample in samples:
        gold_ans = None
        if "answer" in sample or "gold_ans" in sample:
            gold_ans = sample["answer"] if "answer" in sample else sample["gold_ans"]
        elif "reference" in sample:
            gold_ans = sample["reference"]
        elif "obj" in sample:
            gold_ans = set(
                [sample["obj"]]
                + [sample["possible_answers"]]
                + [sample["o_wiki_title"]]
                + [sample["o_aliases"]]
            )
            gold_ans = list(gold_ans)
        if gold_ans is None:
            raise ValueError("Cannot find gold answer in sample.")

        if isinstance(gold_ans, str):
            gold_list = [gold_ans]
        else:
            gold_list = list(gold_ans)
        gold_set = set(gold_list)
        if "answer_aliases" in sample and isinstance(sample["answer_aliases"], list):
            gold_set.update(sample["answer_aliases"])
        gold_answers.append(gold_set)
    return gold_answers


def _get_gold_docs(samples: List[Dict[str, Any]]) -> List[List[str]]:
    gold_docs: List[List[str]] = []
    for sample in samples:
        if "supporting_facts" in sample:
            # hotpotqa / 2wikimultihopqa style (title, sentence_id)
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [item for item in sample.get("context", []) if item[0] in gold_title]
            gold_doc = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            # musique alt export format
            gold_doc = [f"{item['title']}\n{item['text']}" for item in sample["contexts"] if item.get("is_supporting")]
        else:
            # musique in our repo uses paragraphs
            paragraphs = sample.get("paragraphs") or []
            gold_paragraphs = []
            for item in paragraphs:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [
                str(item.get("title") or "") + "\n" + str(item.get("text") if "text" in item else item.get("paragraph_text") or "")
                for item in gold_paragraphs
            ]
        gold_docs.append(list(set(gold_doc)))
    return gold_docs


def _title_from_doc_text(doc_text: str) -> str:
    if not doc_text:
        return ""
    return str(doc_text).split("\n", 1)[0].strip()


def _safe_mean(xs: Sequence[float]) -> float:
    import statistics

    return float(statistics.mean(xs)) if xs else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Musique error buckets for StructAlignLiteRAG.")
    parser.add_argument("--llm_name", type=str, default="meta/llama-3.3-70b-instruct")
    parser.add_argument("--embedding_name", type=str, default="facebook/contriever")
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--run_tag", type=str, default="lite_musique_full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_per_bucket", type=int, default=25)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "contriever"),
        help="Output directory for musique_error_summary.md + musique_cases.jsonl",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset = str(args.dataset)
    if dataset != "musique":
        raise ValueError("This script currently supports dataset=musique only.")

    samples_path = root / "reproduce" / "dataset" / f"{dataset}.json"
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    gold_answers = _get_gold_answers(samples)
    gold_docs = _get_gold_docs(samples)

    llm_tag = _sanitize_model_name(args.llm_name)
    emb_tag = _sanitize_model_name(args.embedding_name)
    pred_path = (
        root
        / "outputs"
        / dataset
        / f"{llm_tag}_{emb_tag}"
        / "metrics"
        / "runs"
        / str(args.run_tag)
        / "qa_predictions.json"
    )
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if len(preds) != len(samples):
        raise ValueError(f"Predictions length mismatch: preds={len(preds)} samples={len(samples)}")

    buckets: DefaultDict[str, List[int]] = defaultdict(list)
    per_bucket_stats: DefaultDict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for i, (p, gset, gdocs) in enumerate(zip(preds, gold_answers, gold_docs)):
        retrieved_top5 = list(p.get("retrieved_docs_top5") or [])
        hit = len(set(retrieved_top5) & set(gdocs)) > 0

        pred_ans = str(p.get("answer") or "")
        em, f1 = _qa_em_f1_single(gset, pred_ans)
        qa_ok = (em > 0.0) or (f1 > 0.0)

        if (not hit) and (not qa_ok):
            b = "retrieval_fail@5 & qa_fail"
        elif (not hit) and qa_ok:
            b = "retrieval_fail@5 & qa_ok"
        elif hit and (not qa_ok):
            b = "retrieval_ok@5 & qa_fail"
        else:
            b = "retrieval_ok@5 & qa_ok"

        buckets[b].append(i)
        dbg = p.get("debug") or {}
        per_bucket_stats[b]["em"].append(float(em))
        per_bucket_stats[b]["f1"].append(float(f1))
        per_bucket_stats[b]["subq_coverage"].append(float(dbg.get("subq_coverage") or 0.0))
        per_bucket_stats[b]["evidence_tokens"].append(float(dbg.get("evidence_tokens") or 0.0))
        per_bucket_stats[b]["num_groups"].append(float(dbg.get("num_groups") or 0.0))
        per_bucket_stats[b]["elapsed_s"].append(float(dbg.get("elapsed_s") or 0.0))

    rng = random.Random(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_path = out_dir / "musique_cases.jsonl"
    summary_path = out_dir / "musique_error_summary.md"

    selected_indices: List[int] = []
    for b, idxs in buckets.items():
        idxs2 = list(idxs)
        rng.shuffle(idxs2)
        take = min(int(args.sample_per_bucket), len(idxs2))
        selected_indices.extend(idxs2[:take])

    selected_indices = sorted(set(selected_indices))

    with open(cases_path, "w", encoding="utf-8") as f:
        for i in selected_indices:
            p = preds[i]
            dbg = p.get("debug") or {}
            gdocs = gold_docs[i]
            g_titles = sorted({_title_from_doc_text(x) for x in gdocs})
            r_titles = [_title_from_doc_text(x) for x in (p.get("retrieved_docs_top5") or [])]
            gset = gold_answers[i]
            em, f1 = _qa_em_f1_single(gset, str(p.get("answer") or ""))

            dbg_small = {
                "num_groups": dbg.get("num_groups"),
                "subq_coverage": dbg.get("subq_coverage"),
                "covered_groups": dbg.get("covered_groups"),
                "evidence_tokens": dbg.get("evidence_tokens"),
                "ppr_enabled": dbg.get("ppr_enabled"),
                "ppr_num_nodes": dbg.get("ppr_num_nodes"),
                "binding_best_score": dbg.get("binding_best_score"),
                "selected_docs_top20": (dbg.get("selected_docs") or [])[:20],
                "selected_nodes_n": len(dbg.get("selected_nodes") or []),
                "elapsed_s": dbg.get("elapsed_s"),
            }

            bucket = None
            for b, idxs in buckets.items():
                if i in idxs:
                    bucket = b
                    break

            row = {
                "qid": p.get("qid"),
                "question": p.get("question"),
                "bucket": bucket,
                "gold_answers": sorted(list(gset)),
                "pred_answer": p.get("answer"),
                "em": float(em),
                "f1": float(f1),
                "retrieval_ok@5": bool(len(set(p.get("retrieved_docs_top5") or []) & set(gdocs)) > 0),
                "gold_doc_titles": g_titles,
                "retrieved_top5_titles": r_titles,
                "debug": dbg_small,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(samples)
    lines: List[str] = []
    lines.append("# Musique Error Buckets (Contriever)")
    lines.append("")
    lines.append(f"- Dataset: `{dataset}` (n={total})")
    lines.append(f"- Run tag: `{args.run_tag}`")
    lines.append(f"- Cases: `{cases_path}`")
    lines.append("")

    headers = ["bucket", "n", "rate", "EM(avg)", "F1(avg)", "SubQCov(avg)", "EvidenceTokens(avg)", "NumGroups(avg)"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for b in [
        "retrieval_fail@5 & qa_fail",
        "retrieval_fail@5 & qa_ok",
        "retrieval_ok@5 & qa_fail",
        "retrieval_ok@5 & qa_ok",
    ]:
        n = len(buckets.get(b, []))
        rate = n / max(total, 1)
        st = per_bucket_stats.get(b) or {}
        row = [
            b,
            str(n),
            f"{rate:.3f}",
            f"{_safe_mean(st.get('em') or []):.3f}",
            f"{_safe_mean(st.get('f1') or []):.3f}",
            f"{_safe_mean(st.get('subq_coverage') or []):.3f}",
            f"{_safe_mean(st.get('evidence_tokens') or []):.1f}",
            f"{_safe_mean(st.get('num_groups') or []):.2f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `retrieval_ok@5 & qa_fail` suggests reader/evidence packaging issues (retrieval succeeded but final answer failed).")
    lines.append("- `retrieval_fail@5 & qa_ok` can indicate aliasing/eval quirks or that the model answered without retrieving a gold doc string match.")
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
