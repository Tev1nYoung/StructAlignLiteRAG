import os
from dataclasses import dataclass
from typing import Optional


DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_NAME = "meta/llama-3.3-70b-instruct"
DEFAULT_EMB_NAME = "facebook/contriever"


@dataclass
class StructAlignLiteConfig:
    # I/O
    dataset: str = "sample"
    save_root: str = "outputs"
    # Optional run tag for isolating per-run metrics/predictions under outputs/<dataset>/<llm>_<emb>/metrics/runs/<tag>/
    run_tag: Optional[str] = None

    # Models
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_name: str = DEFAULT_LLM_NAME
    embedding_model_name: str = DEFAULT_EMB_NAME

    # Embedding runtime
    embedding_batch_size: int = 16
    # Contriever (BERT) supports up to 512 positions; we also hard-cap in the embedder.
    embedding_max_seq_len: int = 512
    embedding_return_normalized: bool = True
    embedding_dtype: str = "auto"
    # For instruction-tuned embedders (e.g., NV-Embed-v2). Ignored by plain encoders like Contriever.
    embedding_query_instruction: str = "Given a question, retrieve relevant documents that best answer the question."

    # Offline
    force_index_from_scratch: bool = False
    chunk_tokens: int = 320
    chunk_overlap: int = 64
    chunk_func: str = "by_token"  # by_token|by_word
    capsule_mode: str = "llm"  # llm|sentence
    max_capsules_per_passage: int = 6
    offline_llm_workers: int = 16
    # Online QA: parallelize across queries (safe; does not change ranking/logic).
    # 0 means "auto" (pick a conservative default based on available API keys).
    online_qa_workers: int = 8

    # Canonicalization
    enable_entity_canonicalization: bool = True
    enable_capsule_canonicalization: bool = True
    entity_person_merge: bool = True
    entity_min_freq_lowercase: int = 2
    entity_max_tokens_lowercase: int = 4
    capsule_ann_top_m: int = 32
    capsule_sim_threshold: float = 0.88
    sim_edge_topk: int = 32
    sim_edge_threshold: float = 0.82

    # Online: query decomposition
    enable_query_dag: bool = True
    query_dag_max_nodes: int = 7

    # Online: candidate generation
    subq_top_capsule: int = 50

    # Online: local propagation (mini-PPR on induced subgraph)
    enable_local_propagation: bool = True
    ppr_min_groups: int = 2
    ppr_seed_per_group: int = 4
    ppr_hops: int = 2
    ppr_max_nodes: int = 5000
    ppr_steps: int = 10
    ppr_alpha: float = 0.85
    ppr_rrf_k: int = 60
    ppr_rrf_pool: int = 200
    ppr_prize_weight: float = 2.0
    ppr_doc_weight: float = 5.0
    ppr_allow_sim_edges: bool = False
    ppr_strength_mentions: float = 1.0
    ppr_strength_in_passage: float = 1.0
    ppr_strength_in_doc: float = 0.7
    ppr_strength_entails: float = 0.5
    ppr_strength_contradicts: float = 0.0
    ppr_strength_sim: float = 0.2
    ppr_eps: float = 1e-6

    # Online: structure-aligned selection (binding/assignment)
    binding_candidate_k: int = 12  # per subQ, how many candidates to consider for binding
    # Lite binding uses greedy assignment (beam=1) without parent-overlap bonus/penalty.
    binding_doc_diversity_bonus: float = 0.08  # encourage selecting capsules from different docs across subQs
    binding_repeat_doc_penalty: float = 0.03  # soft penalty for reusing a doc already selected by earlier subQs
    binding_repeat_capsule_penalty: float = 0.02  # soft penalty for selecting the exact same capsule repeatedly
    doc_dense_score_weight: float = 0.9  # blend dense doc signal into structural doc ranking
    doc_rrf_k: int = 60  # RRF constant (larger -> flatter)
    doc_rrf_pool: int = 120  # how many top docs per query to include in RRF
    doc_rrf_weight: float = 5.0  # weight of RRF score when fusing with structural doc score

    # Online: entity jump expansion (capsule -> entity -> doc(title))
    entity_jump_top_m: int = 25
    entity_jump_bonus: float = 0.6
    # Genericness (unsupervised, corpus-derived; avoids hardcoded blocklists)
    genericness_skip_threshold: float = 0.85  # skip jumps (entity/mention) to overly-generic titles

    # Doc-mention jump: seed from top doc passages -> extract TitleCase mentions -> boost matching doc titles.
    enable_doc_mention_jump: bool = True
    doc_mention_jump_from_top_n: int = 2
    doc_mention_jump_top_m: int = 20
    doc_mention_jump_bonus: float = 0.35

    # Online: optional LLM doc rerank (improves top-2 / top-5 recall on multi-hop)
    enable_llm_doc_rerank: bool = True
    llm_doc_rerank_top_n: int = 10  # rerank inside the first N docs from structural rank
    llm_doc_rerank_select_k: int = 2  # how many docs to place at the top
    llm_doc_rerank_snippet_chars: int = 320  # per-candidate snippet budget

    # Retrieval output
    retrieval_top_k: int = 200  # used for recall@k list
    qa_top_k_passages: int = 5  # number of passages fed to LLM (match HippoRAG default)
    qa_ensure_top_docs: int = 2  # ensure evidence includes at least this many top-ranked docs (if available)
    passage_token_budget: int = 1800
    subq_coverage_top_m: int = 5

    # GCP parameters
    seed_top_s: int = 6
    root_strategy: str = "max_prize"  # max_prize|max_multi_coverage
    w_capsule_entity: float = 0.08
    w_capsule_passage: float = 0.06
    w_passage_doc: float = 0.05
    w_sim_base: float = 0.35

    # Prize shaping
    bonus_entity_match: float = 0.05
    penalty_long_capsule: float = 0.0  # per-token penalty (MVP off)

    # LLM generation
    temperature: float = 0.0
    max_new_tokens: int = 256
    seed: Optional[int] = None

    # Offline LLM extraction/NLI (separate budget)
    offline_temperature: float = 0.0
    offline_max_new_tokens: int = 800
    offline_store_llm_meta: bool = False  # keep outputs small on large corpora

    # Debug/observability
    debug_trace: bool = False
    debug_trace_top_docs: int = 20

    def dataset_dir(self) -> str:
        return os.path.join("reproduce", "dataset")

    def save_dir(self) -> str:
        # Keep the same "outputs/<dataset>" style
        return os.path.join(self.save_root, self.dataset)
