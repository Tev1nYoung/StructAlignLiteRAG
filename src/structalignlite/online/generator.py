from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from ..config import StructAlignLiteConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def minimal_extract(raw: str) -> str:
    """
    Minimal answer extraction to avoid dataset-specific postprocessing.

    Rules:
    - Prefer the last explicit "Answer:" span if present.
    - Take the first line only.
    - Strip common wrappers/quotes and a trailing period.
    """
    raw = str(raw or "").strip()
    if not raw:
        return ""

    matches = re.findall(r"(?i)\banswer\s*:\s*(.+)", raw, flags=re.DOTALL)
    s = matches[-1].strip() if matches else raw
    s = s.splitlines()[0].strip()

    s = re.sub(r"(?i)^(the answer is|answer is|final answer is)\s*[:\-]?\s*", "", s).strip()
    s = s.strip(" \t\"'`")
    s = s.rstrip(".").strip()
    return s


class AnswerGenerator:
    def __init__(self, config: StructAlignLiteConfig, llm) -> None:
        self.config = config
        self.llm = llm

    def answer(self, question: str, passages: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        evidence_lines: List[str] = []
        for p in passages:
            title = p.get("title", "")
            text = p.get("text", "")
            evidence_lines.append(f"Wikipedia Title: {title}\n{text}")
        evidence = "\n\n".join(evidence_lines)

        system = (
            "As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. "
            'Your response must start after \"Thought: \", where you break down the reasoning process. '
            'Conclude with \"Answer: \" to present a concise, definitive response, without extra commentary.\n'
            "If the evidence does not explicitly contain the answer, make a best-effort inference. "
            "Do not answer with 'unknown' or 'none'.\n"
            "Formatting rules:\n"
            "- For yes/no questions, output exactly: yes or no (lowercase).\n"
            "- For date questions, output as: D Month YYYY (e.g., 20 March 851), without commas.\n"
            "- For entity answers, output the shortest canonical name seen in the evidence (avoid parentheses).\n"
        )

        one_shot_docs = (
            "Wikipedia Title: Example Person\nExample Person was born in Example City in 1900.\n\n"
            "Wikipedia Title: Example City\nExample City is a city in Exampleland.\n"
        )
        one_shot_user = f"{one_shot_docs}\n\nQuestion: Where was Example Person born?\nThought: "
        one_shot_assistant = "Example Person was born in Example City. \nAnswer: Example City."

        user = f"{evidence}\n\nQuestion: {question}\nThought: "
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": one_shot_user},
            {"role": "assistant", "content": one_shot_assistant},
            {"role": "user", "content": user},
        ]
        try:
            raw, meta = self.llm.infer(
                messages=messages,
                temperature=self.config.temperature,
                max_completion_tokens=self.config.max_new_tokens,
                seed=self.config.seed,
            )
        except Exception as e:
            logger.warning(f"[StructAlignLiteRAG] [GEN_FINAL] LLM infer failed | err={type(e).__name__}: {e}")
            raw, meta = "", {"error": f"{type(e).__name__}: {e}"}

        ans = minimal_extract(raw or "")
        out_meta = {"raw": raw, "llm_meta": meta, "extractor": "minimal_v1"}
        return ans, out_meta

