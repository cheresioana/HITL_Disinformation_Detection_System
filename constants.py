import os

EMB = "SBERT"
model_name = "google/gemma-3-4b-it"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_TOKENS = 8000
LANGUAGE = "EN"  # "EN" or "RO" — Romanian runner overrides this to "RO"


# Clustering parameters
CLUSTER_CHUNK_SIZE = 5  # Number of nodes processed per group in process_cluster
COMPRESSION_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity above which nodes merge
CLUSTER_MAX_DIST_NARRATIVE = 0.8  # the maximum distance to which the clustering to happen

# Classification parameters
# ── Cross-encoder for reranking candidates (lazy-loaded, thread-safe) ──
import threading

import numpy as np


def _resolve_device(env_var):
    """Resolve the torch device for a model.

    Priority: the model-specific env var (e.g. RERANK_DEVICE) → the global
    DEVICE env var → auto-detect ('cuda' when a GPU is visible, else 'cpu').
    Set DEVICE=cuda to move every model at once, or override one model with
    its own env var (EMBED_DEVICE, RERANK_DEVICE, STS_DEVICE, NLI_DEVICE).

    Note: never returns 'mps' from auto-detect — the SBERT/MiniLM stack hit
    MPS meta-tensor issues, so Apple GPUs must be opted into explicitly.
    The fp64 NaN workaround in get_reranker is only applied off-CUDA.
    """
    forced = os.environ.get(env_var) or os.environ.get("DEVICE")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# One device flag per model. All default to the global DEVICE env var, then
# to auto-detect. Override any single model via its own env var.
EMBED_DEVICE = _resolve_device("EMBED_DEVICE")    # SBERT sentence embeddings (LLM/BERT.py)
RERANK_DEVICE = _resolve_device("RERANK_DEVICE")  # MS-MARCO cross-encoder reranker
STS_DEVICE = _resolve_device("STS_DEVICE")        # STS-B cross-encoder
NLI_DEVICE = _resolve_device("NLI_DEVICE")        # NLI DeBERTa cross-encoder

_reranker = None
_reranker_lock = threading.Lock()

def get_reranker():
    """Return the cross-encoder reranker, loading it on first call (thread-safe)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:                       # double-checked locking
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(
                    'cross-encoder/ms-marco-MiniLM-L-12-v2',
                    device=RERANK_DEVICE,
                )
                # Workaround: torch 2.7 on CPU/MPS produces NaN inside the
                # MiniLM encoder layers in fp32 — verified, every rerank score
                # came back NaN despite clean weights. Casting to fp64 makes
                # the forward pass stable. fp64 is crippled on CUDA GPUs, so
                # keep fp32 there (the NaN bug was CPU/MPS-only).
                if not RERANK_DEVICE.startswith("cuda"):
                    _reranker.model = _reranker.model.double()
    return _reranker


# ── NLI cross-encoder — gated max ensemble companion ─────────────────────
# Loaded only when the hybrid scoring path in get_label_dual asks for it.
# Independent of get_reranker() so swapping either model is a one-line change.

class _NLIRerankerWrapper:
    """Bidirectional NLI scorer that returns a single ``support`` score per pair.

    The wrapped CrossEncoder emits 3 logits (entailment / neutral / contradiction).
    We softmax to probabilities and collapse to a scalar
    ``entail_prob - contradict_prob`` in ``[-1, +1]``.

    Bidirectional: each pair (a, b) is also scored as (b, a) and the per-pair
    maximum is returned. This rescues asymmetric paraphrases where one side is
    more specific than the other (e.g. ``"get involved in war"`` →
    ``"send soldiers to war"`` reads as neutral forward but as entailment
    in reverse). Contradictions stay negative in both directions.
    """

    def __init__(self, ce, entail_id, contradict_id):
        self._ce = ce
        self._entail_id = entail_id
        self._contradict_id = contradict_id

    @property
    def model(self):
        return self._ce.model

    def _score(self, pairs, **kwargs):
        # CrossEncoder returns raw logits for multi-class heads. Softmax them
        # explicitly so the wrapper behaviour is invariant to sentence-
        # transformers version defaults.
        raw = np.asarray(self._ce.predict(pairs, **kwargs), dtype=float)
        if raw.ndim == 1:
            return raw  # defensive: already collapsed
        shifted = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, self._entail_id] - probs[:, self._contradict_id]

    def predict(self, pairs, bidir=True, **kwargs):
        kwargs.setdefault('show_progress_bar', False)
        pairs = list(pairs)
        if not bidir or not pairs:
            return self._score(pairs, **kwargs)
        reverse_pairs = [(b, a) for a, b in pairs]
        scores = self._score(pairs + reverse_pairs, **kwargs)
        n = len(pairs)
        return np.maximum(scores[:n], scores[n:])


_nli_reranker = None
_nli_reranker_lock = threading.Lock()


class _STSScorerWrapper:
    """Adapter for an STS cross-encoder.

    Sentence-transformers' STS models output a single similarity score in
    ``[0, 1]`` (default sigmoid head for regression num_labels=1). We rescale
    to roughly ``[-10, +10]`` via ``(score - 0.5) * STS_SCALE`` so the
    existing MS-MARCO-tuned thresholds (NO_MATCH=-7, POSITIVE=0, GAP=3)
    continue to be correctly calibrated:
        STS=0.0 (no similarity) → -10
        STS=0.5 (borderline)    →   0
        STS=1.0 (near-identical)→ +10
    """

    def __init__(self, ce):
        self._ce = ce

    @property
    def model(self):
        return self._ce.model

    def predict(self, pairs, **kwargs):
        kwargs.setdefault('show_progress_bar', False)
        raw = np.asarray(self._ce.predict(pairs, **kwargs), dtype=float)
        return (raw - 0.5) * STS_SCALE


_sts_reranker = None
_sts_reranker_lock = threading.Lock()


def get_sts_reranker():
    """Return the STS cross-encoder reranker, loading it on first call."""
    global _sts_reranker
    if _sts_reranker is None:
        with _sts_reranker_lock:
            if _sts_reranker is None:
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder(
                    'cross-encoder/stsb-roberta-base',
                    device=STS_DEVICE,
                )
                _sts_reranker = _STSScorerWrapper(ce)
    return _sts_reranker


def get_nli_reranker():
    """Return the bidirectional NLI scorer (lazy-loaded, thread-safe)."""
    global _nli_reranker
    if _nli_reranker is None:
        with _nli_reranker_lock:
            if _nli_reranker is None:
                from sentence_transformers import CrossEncoder
                ce = CrossEncoder(
                    'cross-encoder/nli-deberta-v3-base',
                    device=NLI_DEVICE,
                )
                id2label = {
                    int(k): str(v).lower()
                    for k, v in ce.model.config.id2label.items()
                }
                entail_id = next(i for i, lbl in id2label.items() if 'entail' in lbl)
                contradict_id = next(
                    i for i, lbl in id2label.items() if 'contra' in lbl
                )
                _nli_reranker = _NLIRerankerWrapper(ce, entail_id, contradict_id)
    return _nli_reranker


# ── V3 tunable parameters ──
GAP_THRESHOLD = 3.0           # score gap to short-circuit without calling LLM judge
NO_MATCH_THRESHOLD = -7.0     # both fake & real below this → "no relevant match"
POSITIVE_MATCH_THRESHOLD = 0.0  # treat scores above this as a positive match

# Hybrid (gated-max) ensemble parameters: NLI lifts an MS-MARCO score only when
# it confidently rates the pair as entailment. NLI_SCALE maps NLI's [-1, +1]
# range onto MS-MARCO's so the thresholds above stay calibrated.
NLI_CONFIDENT_THRESHOLD = 0.5
NLI_SCALE = 10.0

# STS rescaling: maps STS-B's [0, 1] sigmoid output onto MS-MARCO's [-10, +10]
# range so the thresholds above keep firing the right short-circuits.
STS_SCALE = 20.0