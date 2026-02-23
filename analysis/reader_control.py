from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.structalignlite.online.generator import minimal_extract  # noqa: E402


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
    # Match src/structalignlite/utils/text_utils.py normalize_answer:
    # - lowercase
    # - remove ASCII punctuation
    # - remove articles
    # - whitespace fix
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


def _qa_em_f1(gold_answers: Sequence[Set[str]], preds: Sequence[str]) -> Dict[str, float]:
    assert len(gold_answers) == len(preds)
    ems: List[float] = []
    f1s: List[float] = []
    for gs, p in zip(gold_answers, preds):
        p = p or ""
        em = 0.0
        f1 = 0.0
        for g in list(gs):
            if _normalize_answer(g) == _normalize_answer(p):
                em = 1.0
            f1 = max(f1, _f1_single(g, p))
        ems.append(em)
        f1s.append(f1)
    import statistics

    return {
        "ExactMatch": float(statistics.mean(ems) if ems else 0.0),
        "F1": float(statistics.mean(f1s) if f1s else 0.0),
    }


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


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reader-control report: final vs minimal answer extraction.")
    parser.add_argument("--llm_name", type=str, default="meta/llama-3.3-70b-instruct")
    parser.add_argument("--embedding_name", type=str, default="facebook/contriever")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "contriever"),
        help="Output directory for reader_control.{md,csv}",
    )
    parser.add_argument("--run_tag_2wiki", type=str, default="nocache_full_rag_qa")
    parser.add_argument("--run_tag_hotpot", type=str, default="lite_hotpot_full")
    parser.add_argument("--run_tag_musique", type=str, default="lite_musique_full")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    llm_tag = _sanitize_model_name(args.llm_name)
    emb_tag = _sanitize_model_name(args.embedding_name)
    run_tags = {
        "2wikimultihopqa": args.run_tag_2wiki,
        "hotpotqa": args.run_tag_hotpot,
        "musique": args.run_tag_musique,
    }

    rows: List[Dict[str, Any]] = []

    for dataset, run_tag in run_tags.items():
        samples_path = root / "reproduce" / "dataset" / f"{dataset}.json"
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        gold_answers = _get_gold_answers(samples)

        pred_path = (
            root
            / "outputs"
            / dataset
            / f"{llm_tag}_{emb_tag}"
            / "metrics"
            / "runs"
            / run_tag
            / "qa_predictions.json"
        )
        preds = json.loads(pred_path.read_text(encoding="utf-8"))
        assert len(preds) == len(gold_answers), f"{dataset}: predictions length mismatch"

        final_answers = [str(p.get("answer") or "") for p in preds]
        minimal_answers = [
            minimal_extract(((p.get("gen_meta") or {}).get("raw") or "")) for p in preds
        ]

        m_final = _qa_em_f1(gold_answers, final_answers)
        m_min = _qa_em_f1(gold_answers, minimal_answers)

        forced_choice = sum(1 for p in preds if bool((p.get("gen_meta") or {}).get("forced_choice")))
        forced_yesno = sum(1 for p in preds if bool((p.get("gen_meta") or {}).get("forced_yesno")))
        n = len(preds)

        row = {
            "dataset": dataset,
            "run_tag": run_tag,
            "n": n,
            "final_EM": round(float(m_final["ExactMatch"]), 4),
            "final_F1": round(float(m_final["F1"]), 4),
            "minimal_EM": round(float(m_min["ExactMatch"]), 4),
            "minimal_F1": round(float(m_min["F1"]), 4),
            "delta_EM_final_minus_min": round(float(m_final["ExactMatch"] - m_min["ExactMatch"]), 4),
            "delta_F1_final_minus_min": round(float(m_final["F1"] - m_min["F1"]), 4),
            "forced_choice_n": forced_choice,
            "forced_choice_rate": round(forced_choice / max(n, 1), 4),
            "forced_yesno_n": forced_yesno,
            "forced_yesno_rate": round(forced_yesno / max(n, 1), 4),
        }
        rows.append(row)

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "reader_control.csv"
    md_path = out_dir / "reader_control.md"

    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(csv_path, rows, fieldnames)

    lines: List[str] = []
    lines.append("# Reader-Control (final vs minimal)")
    lines.append("")
    lines.append("`final` uses `qa_predictions.json[].answer` (current pipeline).")
    lines.append(
        "`minimal` extracts only the first line after `Answer:` from `gen_meta.raw` (no forced-choice/yesno/date/nationality rules)."
    )
    lines.append("")
    lines.append("| " + " | ".join(fieldnames) + " |")
    lines.append("| " + " | ".join(["---"] * len(fieldnames)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k in fieldnames) + " |")
    lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
