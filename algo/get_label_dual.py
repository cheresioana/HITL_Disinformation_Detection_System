"""
Dual-tree classification: compare an article against a fake narrative tree
and a real news tree to determine if it is disinformation.

Main entry point: get_label_dual_v3()
  - Retrieves top matches from both trees via cosine + cross-encoder reranking
  - Extracts the most general disinformation narrative from the tree hierarchy
  - Asks an LLM judge whether the article supports the narrative
  - Short-circuits for clear-cut cases (very low scores or large gaps)

Legacy versions (V1, V2) are kept at the bottom for reference.
"""
import os
import threading

import numpy as np
from LLM.Gemma import gemma_judge_fake_or_real

from constants import (
    get_reranker, get_nli_reranker,
    GAP_THRESHOLD, NO_MATCH_THRESHOLD, POSITIVE_MATCH_THRESHOLD,
    NLI_CONFIDENT_THRESHOLD, NLI_SCALE,
)

# ──────────────────────────────────────────────────────────────────────
# Per-query diagnostic logging
# ──────────────────────────────────────────────────────────────────────
# Off by default — eval and live runs stay quiet. Flip with
# ``set_verbose(True)`` from the caller, or set the GET_LABEL_DUAL_VERBOSE=1
# environment variable. The eval CLI exposes this as ``--verbose``.
VERBOSE = os.environ.get("GET_LABEL_DUAL_VERBOSE", "1").lower() in ("1", "true", "yes")


def set_verbose(flag: bool = True) -> None:
    """Enable/disable the per-query [FAKE]/[REAL]/[decision] traces."""
    global VERBOSE
    VERBOSE = bool(flag)


def _vprint(*args, **kwargs) -> None:
    if VERBOSE:
        print(*args, **kwargs)


# Serializes access to the shared cross-encoder / NLI reranker models. A single
# torch model called concurrently from many threads (parallel eval) can contend
# or return wrong results, so reranker forward passes are run one at a time.
# The LLM judge (Ollama HTTP) stays outside this lock and runs fully parallel.
_reranker_lock = threading.Lock()

_cache = {}  # key: id(matches) -> (E_norm, nodes, idxs, length)


def invalidate_cache(matches=None):
    """Drop cached embedding matrices.

    If *matches* is given, only that entry is removed.
    Otherwise the entire cache is cleared.
    """
    if matches is not None:
        _cache.pop(id(matches), None)
    else:
        _cache.clear()


def _ensure_cache(matches):
    """Cache normalized embedding matrix for a tree's match list."""
    matches_id = id(matches)
    cached = _cache.get(matches_id)
    # Invalidate if the list grew or shrank since the cache was built
    if cached is not None and cached[3] == len(matches):
        return cached[:3]

    nodes, embs, idxs = zip(*matches)
    E = np.asarray(embs, dtype=np.float32)
    denom = np.linalg.norm(E, axis=1, keepdims=True)
    np.maximum(denom, 1e-12, out=denom)
    E = E / denom

    _cache[matches_id] = (E, nodes, idxs, len(matches))
    return E, nodes, idxs


def _retrieve_candidates(row, matches, top_k=10, tree_name=""):
    """Core retrieval: cosine top-K candidates with cross-encoder reranking.

    Returns (candidates, rerank_scores) where:
      - candidates: list of (node, cosine_sim, tree_id)
      - rerank_scores: numpy array of cross-encoder scores (same order)

    *tree_name* is a label ("FAKE" / "REAL") used only for log prefixes so
    that callers can tell the two retrieval passes apart in the output.
    """
    tag = f"[{tree_name}] " if tree_name else ""
    if not matches:
        return [], np.array([])

    E, nodes, idxs = _ensure_cache(matches)

    input_embedding = np.array(row['embedding'], dtype=np.float32)
    norm = float(np.linalg.norm(input_embedding))
    nan_in_input = bool(np.isnan(input_embedding).any())
    _vprint(
        f"  {tag}query_text={str(row.get('text', ''))[:80]!r} "
        f"emb_shape={input_embedding.shape} emb_norm={norm:.6g} emb_has_nan={nan_in_input}",
        flush=True,
    )

    v = input_embedding / max(norm, 1e-12)
    sims = E @ v
    sims_nan = int(np.isnan(sims).sum())
    if sims_nan < len(sims):
        _vprint(
            f"  {tag}cosine sims: n={len(sims)} nan_count={sims_nan} "
            f"min={float(np.nanmin(sims)):.6g} max={float(np.nanmax(sims)):.6g}",
            flush=True,
        )
    else:
        _vprint(f"  {tag}cosine sims: n={len(sims)} nan_count={sims_nan} (ALL NaN)", flush=True)

    if top_k < len(sims):
        topk_idx = np.argpartition(sims, -top_k)[-top_k:]
        topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]
    else:
        topk_idx = np.argsort(sims)[::-1]

    candidates = [(nodes[i], float(sims[i]), idxs[i]) for i in topk_idx]

    query_text = str(row['text'] or '')
    sentence_pairs = [(query_text, str(c[0].text or '')) for c in candidates]

    # ── MS-MARCO primary, NLI refinement (asymmetric gate) ──────────────
    # FAKE tree: NLI can lift any candidate with nli > NLI_CONFIDENT_THRESHOLD,
    #   even when msm is negative. The FAKE tree's job is to surface
    #   propaganda templates the query expresses or implies, and NLI is the
    #   right signal for asymmetric paraphrases like "Ukraine is a slave"
    #   ↔ "Ukraine is not independent" that MS-MARCO misses keyword-wise.
    # REAL tree: NLI only lifts when msm > 0 already. This keeps the gate
    #   that prevents NLI from hallucinating entailment on topically-related
    #   neutral news leaves (the earlier real-side regression cause).
    with _reranker_lock:
        msm_raw = get_reranker().predict(sentence_pairs, show_progress_bar=False)
        nli_raw = get_nli_reranker().predict(
            sentence_pairs, bidir=True, show_progress_bar=False,
        )

    msm = np.asarray(msm_raw, dtype=float)
    nli = np.asarray(nli_raw, dtype=float)
    nli_scaled = nli * NLI_SCALE
    if tree_name == "FAKE":
        use_nli_mask = nli > NLI_CONFIDENT_THRESHOLD
    else:
        use_nli_mask = (msm > 0) & (nli > NLI_CONFIDENT_THRESHOLD)
    rerank_scores = np.where(use_nli_mask, np.maximum(nli_scaled, msm), msm)

    rr = rerank_scores
    rr_nan = int(np.isnan(rr).sum())
    msm_nan = int(np.isnan(msm).sum())
    nli_nan = int(np.isnan(nli).sum())
    n_lifted = int(use_nli_mask.sum())

    if rr_nan < len(rr):
        _vprint(
            f"  {tag}rerank scores: n={len(rr)} nan_count={rr_nan} "
            f"min={float(np.nanmin(rr)):.4f} max={float(np.nanmax(rr)):.4f}  "
            f"msm_min={float(np.nanmin(msm)) if msm_nan < len(msm) else float('nan'):.4f} "
            f"msm_max={float(np.nanmax(msm)) if msm_nan < len(msm) else float('nan'):.4f}  "
            f"nli_max={float(np.nanmax(nli)) if nli_nan < len(nli) else float('nan'):.3f} "
            f"n_above_thr={n_lifted}/{len(nli)}",
            flush=True,
        )
    else:
        _vprint(f"  {tag}rerank scores: n={len(rr)} nan_count={rr_nan} (ALL NaN)", flush=True)

    _vprint(f"  {tag}top-5 (cosine, nli, msm, combined, candidate_text):", flush=True)
    for i in range(min(5, len(candidates))):
        node, cos, _ = candidates[i]
        cand_text = str(node.text or '')
        score = float(rr[i])
        nli_i = float(nli[i])
        msm_i = float(msm[i])
        winner = "NLI" if (use_nli_mask[i] and nli_scaled[i] > msm[i]) else "MSM"
        flag = "  <-- NaN" if np.isnan(score) else ""
        empty_flag = "  <-- EMPTY" if not cand_text.strip() else ""
        _vprint(
            f"    [{i}] cos={cos:.4f} nli={nli_i:+.3f} msm={msm_i:+.4f} "
            f"combined={score:+.4f} [{winner}]{flag}{empty_flag} "
            f"text={cand_text[:120]!r}",
            flush=True,
        )

    return candidates, rerank_scores, msm, nli


def get_top_n_scores(row, matches, n=2, tree_name=""):
    """Retrieve top-N nodes by combined cross-encoder + NLI score.

    Returns list of (node, combined_score, msm, nli) tuples, sorted best-first.
    """
    candidates, rerank_scores, msm, nli = _retrieve_candidates(row, matches, tree_name=tree_name)
    sorted_idx = np.argsort(rerank_scores)[::-1][:n]
    return [(candidates[i][0], float(rerank_scores[i]), float(msm[i]), float(nli[i]))
            for i in sorted_idx]


def get_best_score(row, matches, tree_name=""):
    """Retrieve the single best-matching node by cross-encoder score.

    Returns (node, score) tuple. Stable signature for external callers
    (update_model).
    """
    candidates, rerank_scores, _msm, _nli = _retrieve_candidates(row, matches, tree_name=tree_name)
    best_idx = int(np.argmax(rerank_scores))
    return candidates[best_idx][0], float(rerank_scores[best_idx])


# ══════════════════════════════════════════════════════════════════════
# Tree traversal helpers
# ══════════════════════════════════════════════════════════════════════

def get_ancestor_path(node, max_depth=2):
    """Walk up from node, return ancestor texts bottom-up: [parent, grandparent, ...]."""
    path = []
    current = node.parent
    depth = 0
    while current and depth < max_depth:
        path.append(current.text)
        current = current.parent
        depth += 1
    return path


def get_root_path(node):
    """Walk up from node to root, return texts top-down: [root, ..., parent].
    Does NOT include the node itself."""
    path = []
    current = node.parent
    while current:
        path.append(current.text)
        current = current.parent
    path.reverse()
    return path


# ══════════════════════════════════════════════════════════════════════
# V3: Narrative-centered LLM judge (ACTIVE)
# ══════════════════════════════════════════════════════════════════════

_JUDGE_FAKE_TOP_K = 5     # top-N fake leaves to source candidate narratives from
_JUDGE_MAX_NARRATIVES = 5  # cap on numbered narratives shown to the LLM after dedup


def build_judge_context(row, fake_matches, real_matches):
    """Build context for the LLM judge prompt.

    Extracts the immediate-parent narrative from the top fake leaf (used as
    the displayed narrative in the explanation) and builds ``fake_narratives``
    — a numbered block of up to ``_JUDGE_MAX_NARRATIVES`` parent narratives
    drawn from the top-K fake leaves, deduplicated. This block replaces the
    legacy "single narrative + supporting leaves" prompt format so the LLM
    sees multiple propaganda framings to compare against, rather than being
    anchored on one.

    Returns:
        (fake_node, fake_score, fake_msm, fake_nli,
         real_node, real_score, real_msm, real_nli,
         fake_narrative, fake_narratives, real_section)
    """
    _vprint(f"\n=== Classify: {str(row.get('text', ''))[:120]!r} ===", flush=True)
    _vprint(f"--- FAKE tree ---", flush=True)
    fake_top = get_top_n_scores(row, fake_matches, n=_JUDGE_FAKE_TOP_K, tree_name="FAKE")
    fake_node, fake_score, fake_msm, fake_nli = fake_top[0]
    fake_ancestors = get_root_path(fake_node)

    # Narrative = immediate parent of the matched leaf — specific enough to
    # carry the propaganda framing (e.g. "Ukraine lacks independence") while
    # still being a narrative rather than a raw article. Used for the
    # explanation/UI when the LLM returns FAKE.
    fake_narrative = fake_ancestors[0] if fake_ancestors else fake_node.text

    # Numbered block of parent narratives, deduplicated, capped at MAX.
    # Each candidate is the immediate parent of a top-K fake leaf (lifted
    # if the matched node is itself a leaf). Same dedup rule as the previous
    # multi-choice attempt — preserves order of first appearance.
    seen = set()
    narrative_lines = []
    for fnode, _fscore, _fmsm, _fnli in fake_top:
        ancestors = get_root_path(fnode)  # [root, ..., parent]
        if ancestors:
            leaf_narratives = [ancestors[0]]
            if ancestors[-1] != ancestors[0]:
                leaf_narratives.append(ancestors[-1])
        else:
            leaf_narratives = [fnode.text]
        for ntext in leaf_narratives:
            if ntext in seen:
                continue
            seen.add(ntext)
            narrative_lines.append(
                f'Disinformation narrative {len(narrative_lines) + 1}: "{ntext}"'
            )
            if len(narrative_lines) >= _JUDGE_MAX_NARRATIVES:
                break
        if len(narrative_lines) >= _JUDGE_MAX_NARRATIVES:
            break
    fake_narratives = '\n'.join(narrative_lines)

    # Real side: top-2 matches with theme context
    _vprint(f"--- REAL tree ---", flush=True)
    real_top = get_top_n_scores(row, real_matches, n=2, tree_name="REAL")
    real_node, real_score, real_msm, real_nli = real_top[0]
    real_lines = []
    for rnode, rscore, *_ in real_top:
        real_lines.append(f'- "{rnode.text}" (score: {rscore:.2f})')
        r_ancestors = get_ancestor_path(rnode, max_depth=1)
        if r_ancestors:
            real_lines.append(f'  Theme: "{r_ancestors[0]}"')
    real_section = '\n'.join(real_lines)

    return (fake_node, fake_score, fake_msm, fake_nli,
            real_node, real_score, real_msm, real_nli,
            fake_narrative, fake_narratives, real_section)


def get_label_dual_v3(row, fake_matches, real_matches, real_label="unknown"):
    """Classify an article as fake (1) or real (0) using narrative-centered LLM judge.

    Pipeline:
      1. Retrieve top matches from both trees (cosine + cross-encoder + NLI)
      2. Extract the immediate-parent narrative from the top fake leaf
      3. Short-circuit:
         * no_match       : both scores very low → TRUE
         * fake_dominates : fake combined wins by GAP, MSM>0, NLI not a
                            confident contradiction → DISINFORMATION
         * real_dominates : real combined wins by GAP, MSM>0, NLI not a
                            confident contradiction → TRUE
      4. Otherwise ask LLM judge whether the article supports the fake
         narrative — returns FAKE or REAL.

    NLI participates via two mechanisms:
      - Asymmetric lift in `_retrieve_candidates` (FAKE-ungated, REAL-gated).
      - Contradiction-veto on the dominance rules above (`nli > -0.5`).
    The veto routes contradiction cases to the LLM rather than deciding them
    directly — contradicting a fake narrative does not imply the article is
    true (it could be a competing propaganda claim on the same topic).

    Returns (label, narrative_node, fake_score, real_score, explanation_dict).
    The explanation dict carries a short ``reason`` string plus the top
    matched text on either side, suitable for surfacing in the UI.
    """
    (fake_node, fake_score, fake_msm, fake_nli,
     real_node, real_score, real_msm, real_nli,
     fake_narrative, fake_narratives, real_section) = \
        build_judge_context(row, fake_matches, real_matches)

    # Lift leaf nodes to their parent so we return a narrative, not a raw article
    if fake_node and not fake_node.children and fake_node.parent:
        fake_node = fake_node.parent

    fake_match_text = fake_node.text if fake_node is not None else None
    real_match_text = real_node.text if real_node is not None else None

    def explain(reason, mode):
        return {
            "reason": reason,
            "mode": mode,
            "fake_match_text": fake_match_text,
            "fake_score": float(fake_score) if fake_score is not None else None,
            "real_match_text": real_match_text,
            "real_score": float(real_score) if real_score is not None else None,
        }

    def decide(label, node, reason, mode):
        label_text = "DISINFORMATION" if label == 1 else "TRUE"
        narrative = node.text if node is not None else "—"
        # Format the gold label if the caller supplied one (eval path); else
        # leave it as "unknown" (live UI path). Tag a HIT/MISS marker when
        # we can compare.
        try:
            gold = int(real_label)
            gold_text = "DISINFORMATION" if gold == 1 else "TRUE"
            verdict = "HIT" if gold == label else "MISS"
            gold_segment = f"  gold={gold_text}  [{verdict}]"
        except (TypeError, ValueError):
            gold_segment = f"  gold={real_label}"
        _vprint(
            f"[decision] pred={label_text} via {mode}{gold_segment}  "
            f"fake_score={fake_score:.4f}  real_score={real_score:.4f}  "
            f"narrative={narrative[:120]!r}",
            flush=True,
        )
        return label, node, fake_score, real_score, explain(reason, mode)

    #Short-circuit: both scores very low — nothing matches, default to real
    if fake_score < NO_MATCH_THRESHOLD and real_score < NO_MATCH_THRESHOLD:
        return decide(
            0, None,
            "No strong matches in either tree — assumed true.",
            "no_match",
        )

    # Note on contradiction signals: when MSM matches a fake-tree leaf but
    # NLI flags contradiction, we *do not* short-circuit to TRUE directly.
    # Two propaganda statements can contradict each other (different fake
    # framings of the same topic), so "contradicts a fake narrative" does
    # not imply the article is true. Instead, the contradiction-veto on the
    # dominance rules below blocks fake_dominates from firing, and the row
    # is routed to the LLM judge with the article text — which can tell
    # refutation from a competing propaganda claim.

    # Short-circuit: fake clearly dominates. MSM must be positive AND NLI
    # must not be a *confident contradiction* — neutral NLI (mildly +/-) is
    # treated as agnostic so MSM alone can short-circuit.
    if (fake_score > POSITIVE_MATCH_THRESHOLD) and (fake_score - real_score > GAP_THRESHOLD) \
            and (fake_msm > 0) and (fake_nli > -NLI_CONFIDENT_THRESHOLD):
        return decide(
            1, fake_node,
            f"Matches the disinformation narrative \"{fake_match_text}\" (score {fake_score:.2f}).",
            "fake_dominates",
        )

    # Short-circuit: real clearly dominates. Same contradiction-veto on NLI.
    # Pass the real-tree leaf as the card's node so the UI can offer a
    # "Modify element" → spotlight → trash path to delete this specific
    # real-news entry if it turns out to be the wrong target.
    if (real_score > POSITIVE_MATCH_THRESHOLD) and (real_score - fake_score > GAP_THRESHOLD) \
            and (real_msm > 0) and (real_nli > -NLI_CONFIDENT_THRESHOLD):
        return decide(
            0, real_node,
            f"Closely matches real news \"{real_match_text}\" (score {real_score:.2f}).",
            "real_dominates",
        )

    # Ask LLM judge: does the article support any of the candidate narratives?
    verdict = gemma_judge_fake_or_real(row['text'], fake_narratives, real_section)

    if verdict == "FAKE":
        return decide(
            1, fake_node,
            f"LLM judge classified as supporting the narrative \"{fake_narrative}\".",
            "llm_fake",
        )
    return decide(
        0, None,
        "LLM judge classified as not supporting any disinformation narrative.",
        "llm_real",
    )
