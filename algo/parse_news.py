"""
Complete news article evaluation.

Classifies full articles (title + body) by splitting into sentences,
classifying each against dual narrative trees, and applying an adaptive
sliding-window threshold to determine the final article-level label.
"""

# Standard library
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party
import nltk

# Local
from LLM.orchestrator import fetch_embedding
from algo.get_label_dual import get_label_dual_v3
from utils import clean_text

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Narrative consolidation (group by common ancestry)
# ---------------------------------------------------------------------------

def _lca(a, b):
    """Return the Lowest Common Ancestor of two TreeNodes."""
    ancestors_a = set()
    cur = a
    while cur:
        ancestors_a.add(id(cur))
        cur = cur.parent
    cur = b
    while cur:
        if id(cur) in ancestors_a:
            return cur
        cur = cur.parent
    return None


def _node_depth(node):
    """Return the depth of *node* (0 = root)."""
    depth = 0
    cur = node.parent
    while cur:
        depth += 1
        cur = cur.parent
    return depth


def _tree_root(node):
    """Walk up to the root of the tree containing *node*."""
    cur = node
    while cur.parent:
        cur = cur.parent
    return cur


def _consolidate_narratives(matched_nodes):
    """Group matched nodes by common ancestry into consolidated narratives.

    Nodes that share a common ancestor deeper than the tree root are
    iteratively merged under their LCA (most-specific shared ancestor
    first).  Nodes whose only shared ancestor is the root stay as
    separate entries.

    Returns a sorted list of ``{"narrative": str, "count": int}`` dicts.

    Example
    -------
    Sentences A→node1, B→node2 share ancestor node3 (not root):
        → [{"narrative": node3.text, "count": 2}]

    Sentence C→node4 has no shared ancestor with node1/node2 (only root):
        → [{"narrative": node3.text, "count": 2},
           {"narrative": node4.text, "count": 1}]
    """
    if not matched_nodes:
        return []

    root = _tree_root(matched_nodes[0])

    # Pre-count identical node objects
    seen = {}
    for node in matched_nodes:
        nid = id(node)
        if nid in seen:
            seen[nid] = (node, seen[nid][1] + 1)
        else:
            seen[nid] = (node, 1)
    groups = list(seen.values())          # [(node, count), ...]

    # Iteratively merge the pair with the deepest (most specific) LCA
    changed = True
    while changed:
        changed = False
        best_i, best_j, best_anc, best_depth = -1, -1, None, -1

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                anc = _lca(groups[i][0], groups[j][0])
                if anc is None or anc is root:
                    continue                      # different branches — keep separate
                depth = _node_depth(anc)
                # Only merge if both nodes are within 1 hop of the LCA
                if _node_depth(groups[i][0]) - depth > 1 or _node_depth(groups[j][0]) - depth > 1:
                    continue
                if depth > best_depth:
                    best_i, best_j, best_anc, best_depth = i, j, anc, depth

        if best_anc is not None:
            new_count = groups[best_i][1] + groups[best_j][1]
            # Remove higher index first to keep lower index valid
            groups.pop(best_j)
            groups.pop(best_i)
            groups.append((best_anc, new_count))
            changed = True

    return [
        {"narrative": node.text, "count": count}
        for node, count in sorted(groups, key=lambda x: x[1], reverse=True)
    ]


# ---------------------------------------------------------------------------
# Sentence-level classification
# ---------------------------------------------------------------------------

def _classify_sentences(title, sentences, fake_matches, real_matches):
    """Classify title and each sentence against dual narrative trees.

    Returns:
        sent_labels: list[int] — per-sentence labels (0 or 1)
        title_label: int — label for the title
        title_node: TreeNode or None — matched fake narrative node for title
        false_sent: list[tuple] — (text, narrative) pairs flagged as fake
        narrative_counts: dict — narrative text → count
        matched_nodes: list[TreeNode] — all fake-matched nodes (for LCA)
    """
    row = {
        'text': title,
        'embedding': fetch_embedding(clean_text(title)),
    }
    title_label, title_node, _, _, _ = get_label_dual_v3(row, fake_matches, real_matches)

    false_sent = []
    narrative_counts = {}
    matched_nodes = []

    if title_label == 1 and title_node:
        narrative_counts[title_node.text] = narrative_counts.get(title_node.text, 0) + 1
        false_sent.append((title, title_node.text))
        matched_nodes.append(title_node)

    sent_labels = []
    for sent in sentences:
        row = {
            'text': sent,
            'embedding': fetch_embedding(clean_text(sent)),
        }
        pair_label, pair_node, _, _, _ = get_label_dual_v3(row, fake_matches, real_matches)
        sent_labels.append(pair_label)
        if pair_label == 1 and pair_node:
            false_sent.append((sent, pair_node.text))
            narrative_counts[pair_node.text] = narrative_counts.get(pair_node.text, 0) + 1
            matched_nodes.append(pair_node)

    return sent_labels, title_label, title_node, false_sent, narrative_counts, matched_nodes


# ---------------------------------------------------------------------------
# Adaptive threshold
# ---------------------------------------------------------------------------

WINDOW_SIZE = 6

def _compute_adaptive_score(sent_labels):
    """Compute the adaptive threshold and max fake score for an article.

    Short articles (1-3 sentences): threshold=1, score=total fake count.
    Medium articles (4-5 sentences): threshold=2, score=total fake count.
    Long articles (6+ sentences): threshold=3, score=max in sliding window.

    Returns:
        (max_score, threshold)
    """
    n = len(sent_labels)
    total_fake = sum(sent_labels)

    if n <= 3:
        return total_fake, 1
    if n <= 5:
        return total_fake, 2

    # Sliding window for 6+ sentences
    threshold = 3
    w = min(WINDOW_SIZE, n)
    cur_sum = sum(sent_labels[:w])
    max_score = cur_sum
    for i in range(w, n):
        cur_sum += sent_labels[i] - sent_labels[i - w]
        max_score = max(max_score, cur_sum)

    return max_score, threshold


# ---------------------------------------------------------------------------
# Article-level decision
# ---------------------------------------------------------------------------

def _decide_article_label(max_score, threshold, n, title_label, true_label,
                          title, false_sent, narratives):
    """Determine final article label and log the decision.

    Returns:
        int — 0 (real) or 1 (fake)
    """
    # Primary rule: enough fake sentences in a window
    if max_score >= threshold:
        if true_label == 0:
            logger.info("Incorrect fake: %d/%d sentences (threshold=%d), title_label=%d",
                        max_score, n, threshold, title_label)
            logger.info("  Title: %s", title)
            logger.info("  Matched: %s", false_sent)
        return 1

    # Secondary rule: long article with moderate evidence + fake title
    if n > 5 and max_score >= 2 and title_label == 1:
        if true_label == 0:
            logger.info("Incorrect fake via title+window: %d/%d sentences", max_score, n)
            logger.info("  Title: %s", title)
            logger.info("  Matched: %s", false_sent)
        return 1

    # Default: real
    if true_label == 1:
        logger.info("Missed fake: %d/%d sentences (threshold=%d), title_label=%d",
                    max_score, n, threshold, title_label)
        logger.info("  Title: %s", title)
        logger.info("  Matched: %s", false_sent)
    return 0


def _build_narrative_list(matched_nodes):
    """Build the consolidated narrative list from matched nodes.

    Groups nodes that share a common ancestor (deeper than root) under
    their LCA.  Returns a sorted list of {narrative, count} dicts.
    """
    return _consolidate_narratives(matched_nodes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_news_label(title, summary, fake_matches, real_matches, true_label):
    """Classify a complete news article as fake or real.

    Pipeline:
      1. Tokenize into sentences
      2. Classify title + each sentence against dual trees
      3. Compute adaptive sliding-window score
      4. Apply threshold to determine final label

    Returns:
        (label, narratives, marked_sentences) — label is 0/1,
        narratives is list of dicts {narrative, count} consolidated by
        common ancestry (nodes sharing an ancestor are merged under their LCA),
        marked_sentences is list of dicts {sentence, narrative} for fake sentences only
    """
    try:
        sentences = sent_tokenize(summary)
        sent_labels, title_label, title_node, false_sent, narrative_counts, matched_nodes = \
            _classify_sentences(title, sentences, fake_matches, real_matches)

        max_score, threshold = _compute_adaptive_score(sent_labels)
        narratives = _build_narrative_list(matched_nodes)

        final_label = _decide_article_label(
            max_score, threshold, len(sent_labels), title_label,
            true_label, title, false_sent, narratives,
        )

        # Only fake sentences, each with its matched narrative
        marked_sentences = [
            {"sentence": sent, "narrative": nar}
            for sent, nar in false_sent
        ]

        if true_label == 1 and final_label == 1:
            logger.info("Correct fake: %d/%d sentences (threshold=%d), title_label=%d",
                        max_score, len(sent_labels), threshold, title_label)
            logger.info("  Title: %s", title)
            logger.info("  Matched: %s", false_sent)
            logger.info("  Narratives: %s", narratives)

        return final_label, narratives, marked_sentences

    except Exception as e:
        logger.error("Exception for '%s': %s", title, e)
        return 0, [], []


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def worker(i, title, summary, fake_matches, real_matches, true_label):
    """Classify a single article (used by ThreadPoolExecutor)."""
    logger.info("Processing article %d", i)
    label, _, _ = get_news_label(title, summary, fake_matches, real_matches, true_label)
    return i, label

