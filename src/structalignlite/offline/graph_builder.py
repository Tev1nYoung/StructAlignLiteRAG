from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from ..config import StructAlignRAGConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


EdgeRow = Dict[str, Any]


def _add_edge(adj: Dict[str, List[Tuple[str, float, str]]], edges: List[EdgeRow], u: str, v: str, typ: str, weight: float, meta: Dict[str, Any] | None = None) -> None:
    adj.setdefault(u, []).append((v, float(weight), typ))
    adj.setdefault(v, []).append((u, float(weight), typ))
    row: EdgeRow = {"src": u, "dst": v, "type": typ, "weight": float(weight)}
    if meta:
        row.update(meta)
    edges.append(row)


def build_evidence_graph(
    config: StructAlignRAGConfig,
    docs: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    canonical_capsules: List[Dict[str, Any]],
    canonical_embeddings: np.ndarray,
    out_dir: str,
    llm=None,
) -> Dict[str, Any]:
    """
    Builds:
    - graph_edges.jsonl
    - graph_adj.pkl
    """
    os.makedirs(out_dir, exist_ok=True)
    edge_path = os.path.join(out_dir, "graph_edges.jsonl")
    adj_path = os.path.join(out_dir, "graph_adj.pkl")

    adj: Dict[str, List[Tuple[str, float, str]]] = {}
    edges: List[EdgeRow] = []

    # Node init (optional; adjacency can be implicit)
    for d in docs:
        adj.setdefault(f"D:{int(d['doc_idx'])}", [])
    for p in passages:
        adj.setdefault(f"P:{p['passage_id']}", [])
    for e in entities:
        adj.setdefault(f"E:{e['entity_id']}", [])
    for c in canonical_capsules:
        adj.setdefault(f"C:{c['canonical_id']}", [])

    # Passage -> Doc edges
    for p in passages:
        _add_edge(adj, edges, f"P:{p['passage_id']}", f"D:{int(p['doc_idx'])}", "in_doc", config.w_passage_doc)

    # Capsule -> Passage / Entity edges
    for c in canonical_capsules:
        cid = f"C:{c['canonical_id']}"
        # provenance passages
        seen_p: Set[str] = set()
        for prov in c.get("provenance") or []:
            pid = prov.get("passage_id")
            if not pid:
                continue
            if pid in seen_p:
                continue
            seen_p.add(pid)
            _add_edge(adj, edges, cid, f"P:{pid}", "in_passage", config.w_capsule_passage)

        for eid in c.get("entity_ids") or []:
            _add_edge(adj, edges, cid, f"E:{eid}", "mentions", config.w_capsule_entity)

    # Similarity edges between canonical capsules
    if canonical_embeddings is not None and len(canonical_capsules) > 1 and config.sim_edge_topk > 0:
        emb = np.ascontiguousarray(canonical_embeddings.astype(np.float32))
        d = emb.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(emb)

        k = min(int(config.sim_edge_topk), len(canonical_capsules))
        sims, nbrs = index.search(emb, k)

        added = set()
        pbar = tqdm(range(len(canonical_capsules)), desc="Sim Edges", disable=False, ascii=True, dynamic_ncols=True)
        for i in pbar:
            ci = canonical_capsules[i]["canonical_id"]
            for rank in range(1, k):
                j = int(nbrs[i, rank])
                if j < 0 or j == i:
                    continue
                sim = float(sims[i, rank])
                if sim < float(config.sim_edge_threshold):
                    continue
                cj = canonical_capsules[j]["canonical_id"]
                a = (ci, cj) if ci < cj else (cj, ci)
                if a in added:
                    continue
                added.add(a)
                # Sim edges are intentionally "costly" to avoid drift in dense entity graphs.
                # Keep a mostly-constant traversal cost; store the actual similarity in edge metadata.
                w = float(config.w_sim_base)
                _add_edge(adj, edges, f"C:{ci}", f"C:{cj}", "sim", w, meta={"sim": sim})
            pbar.set_postfix({"edges": len(added)})

    # Deterministic adjacency ordering
    for u in list(adj.keys()):
        adj[u].sort(key=lambda x: (x[0], x[2], x[1]))

    # Persist
    edges.sort(key=lambda r: (str(r.get("src")), str(r.get("dst")), str(r.get("type"))))
    with open(edge_path, "w", encoding="utf-8") as f:
        for row in edges:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(adj_path, "wb") as f:
        pickle.dump(adj, f)

    logger.info(f"[StructAlignRAG] [OFFLINE_GRAPH] written | edges={len(edges)} path={edge_path}")
    return {"edge_path": edge_path, "adj_path": adj_path, "num_edges": len(edges), "num_nodes": len(adj)}
