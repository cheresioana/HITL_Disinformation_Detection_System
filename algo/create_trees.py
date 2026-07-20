"""
Narrative tree construction and management.

Builds hierarchical narrative trees from labeled datasets using iterative
agglomerative clustering and LLM-generated narrative summaries.
"""
import itertools
# Standard library
import glob
import json
import logging
import os
import re
import tempfile
import time
from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor

# Third-party
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_similarity

# Local
from algo.TreeNode import TreeNode
from algo.algo_utils import load_structure_file
from LLM.orchestrator import (
    fetch_embedding,
    get_common_narrative,
    get_common_narrative_truth,
    is_narrative,
)
from config import ACTIVE_THRESHOLD
from constants import CLUSTER_CHUNK_SIZE, COMPRESSION_SIMILARITY_THRESHOLD, CLUSTER_MAX_DIST_NARRATIVE
from utils import clean_text

logger = logging.getLogger(__name__)

# Flag to switch which narrative prompt is used during tree building
_use_truth_prompt = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def set_truth_mode(enabled):
    """Set whether to use truth (factual) prompts for narrative generation."""
    global _use_truth_prompt
    _use_truth_prompt = enabled


def create_tree_list(df):
    """
    Create a list of TreeNode objects from a dataframe with pre-computed embeddings.

    Args:
        df (pandas.DataFrame): Dataframe with 'text' and 'embedding' columns.

    Returns:
        list[TreeNode]: List of leaf-level tree nodes.
    """
    nodes = []
    for idx, row in df.iterrows():
        embedding = row['embedding']
        nodes.append(TreeNode(row['text'], embedding))
        if idx % 100 == 0:
            logger.info("Creating tree nodes: %d processed", idx)
    return nodes


# this should not be deleted
def paralel_process_in_narrative(node, common_narrative):
    """Check if a text is part of a narrative."""
    ent_label1, result = is_narrative(node.text, common_narrative)
    if ent_label1 > 0:
        return node, 1
    return node, 0


# ---------------------------------------------------------------------------
# Clustering and narrative generation
# ---------------------------------------------------------------------------

def process_cluster(node_list):
    """
    Compute common narrative and embedding for a cluster.

    Args:
        node_list (list[TreeNode]): Nodes belonging to the same cluster.

    Returns:
        list[TreeNode]: Nodes with same-narrative members merged as children
        of a new narrative parent node.
    """
    if len(node_list) < 2:
        return node_list

    result = []
    for i in range(0, len(node_list), CLUSTER_CHUNK_SIZE):
        group = node_list[i:i + CLUSTER_CHUNK_SIZE]
        texts = [n.text for n in group]
        common_narrative = (
            get_common_narrative_truth(texts) if _use_truth_prompt
            else get_common_narrative(texts)
        )
        if common_narrative == "no narrative" or common_narrative == "" or common_narrative is None:
            result.extend(group)
            continue

        #This should not be deleted
        if not _use_truth_prompt:
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     ent_results = list(
            #         executor.map(paralel_process_in_narrative, group, itertools.repeat(common_narrative)))
            #
            # not_narrative = [node for node, ent_label in ent_results if ent_label != 1]
            # in_narrative = [node for node, ent_label in ent_results if ent_label == 1]
            not_narrative = []
            in_narrative = group
        else:
            not_narrative = []
            in_narrative = group

        if len(in_narrative) < 2:
            result.extend(group)
            continue

        narrative_embedding = fetch_embedding(common_narrative)
        in_narrative_texts = [node.text for node in in_narrative]
        logger.info("Cluster texts:\n%s", ";  ".join(in_narrative_texts))
        logger.info("Common narrative: %s", common_narrative)
        result.extend(not_narrative)
        result.append(TreeNode(common_narrative, narrative_embedding, children=in_narrative))
    return result


def cluster_new_items(nodes, max_dist=0.5):
    """Cluster a small batch of new leaf nodes into narrative subtrees.

    Lightweight version of run_algo for incremental updates.
    Groups similar unmatched articles and generates narratives for each cluster.

    Args:
        nodes: list of TreeNode (leaf nodes with text + embedding)
        max_dist: cosine distance threshold for clustering (default 0.5)

    Returns:
        list of TreeNode — a mix of narrative parents and standalone leaves
    """
    if len(nodes) < 2:
        return nodes

    embeddings = np.array([n.embedding for n in nodes])
    linkage_matrix = linkage(embeddings, metric='cosine')
    clusters = fcluster(linkage_matrix, t=max_dist, criterion="distance")

    cluster_dict = defaultdict(list)
    for node, cid in zip(nodes, clusters):
        cluster_dict[cid].append(node)

    result = []
    for node_list in cluster_dict.values():
        result.extend(process_cluster(node_list))

    return result


# ---------------------------------------------------------------------------
# Tree compression
# ---------------------------------------------------------------------------

def recursive_compression_tree(node, parent):
    """
    Merge recursively very similar nodes with their parent to reduce tree depth.

    Args:
        node (TreeNode): Current node being inspected.
        parent (TreeNode or None): Parent of the current node.
    """
    if node.children:
        for child in node.children:
            recursive_compression_tree(child, node)
    if node.children and parent:
        sim = cosine_similarity(
            np.array(parent.embedding).reshape(1, -1),
            np.array(node.embedding).reshape(1, -1),
        )[0][0]
        if sim > COMPRESSION_SIMILARITY_THRESHOLD:
            logger.info("Merging nodes: '%s' <- '%s'", parent.text[:80], node.text[:80])
            parent.children.remove(node)
            parent.children.extend(node.children)


def narrative_compression(nodes):
    """Apply recursive compression to all multi-level nodes."""
    for node in nodes:
        if node.level > 1:
            recursive_compression_tree(node, None)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _atomic_write_json(payload, dest_path):
    """Write *payload* to *dest_path* atomically (temp + os.replace).

    Prevents concurrent readers (the live app re-loading trees in another
    thread/process) from seeing half-written buffers on multi-MB JSON
    dumps. Without this, ``json.load`` on the reader side can raise
    ``Expecting ',' delimiter`` mid-file when a writer is partway through
    flushing.
    """
    parent_dir = os.path.dirname(dest_path) or "."
    os.makedirs(parent_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=parent_dir, suffix=".json",
    ) as tmp:
        json.dump(payload, tmp, indent=4)
        tmp_path = tmp.name
    os.replace(tmp_path, dest_path)


def save_state(nodes, file_name, folder=""):
    """
    Save narrative tree structure to JSON files.

    Produces two files per save:
    - ``full_<file_name>.json``     — complete serialization (with embeddings)
    - ``readable_<file_name>.json`` — human-readable version (text only)

    Both writes are atomic — the live app reading via ``load_structure_file``
    in another thread/process always sees either the previous version of
    each file or the new one, never a half-written mix.
    """
    all_nodes = nodes.copy()
    narrative_compression(all_nodes)
    sorted_trees = sorted(all_nodes, key=lambda x: x.level, reverse=True)
    full_payload = [tree.to_dict() for tree in sorted_trees]
    readable_payload = [tree.to_clean_dict() for tree in sorted_trees]
    _atomic_write_json(full_payload,
                       folder + "results/full_" + file_name + ".json")
    _atomic_write_json(readable_payload,
                       folder + "results/readable_" + file_name + ".json")


def latest_checkpoint(folder):
    """Return ``(dist, path)`` for the highest ``full_result_<dist>.json`` in
    ``folder/results/``, or ``None`` if there is no usable checkpoint.

    The 0.0 seed file is ignored — it is not a meaningful resume point.
    """
    pattern = os.path.join(folder, "results", "full_result_*.json")
    best = None
    for path in glob.glob(pattern):
        m = re.search(r"full_result_(.+?)\.json", os.path.basename(path))
        if not m:
            continue
        try:
            dist = float(m.group(1))
        except ValueError:
            continue
        if dist <= 0.0:
            continue
        if best is None or dist > best[0]:
            best = (dist, path)
    return best


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def run_algo(all_nodes, cluster_max_dist=0.1, verbose=True, index=0, folder=""):
    """
    Main algorithm for creating the hierarchical narrative tree.

    Iteratively clusters nodes at increasing distance thresholds, generating
    narrative summaries for each cluster until no further compression is possible.

    Args:
        all_nodes (list[TreeNode]): Initial leaf nodes.
        cluster_max_dist (float): Starting cosine distance threshold.
        verbose (bool): Reserved for future use.
        index (int): Starting iteration index.
        folder (str): Output directory prefix.
    """
    # ── Resume from the latest checkpoint, if one exists ──
    ckpt = latest_checkpoint(folder)
    if ckpt is None:
        logger.info("[%s] No checkpoint found — starting fresh at dist=%.2f",
                    folder, cluster_max_dist)
        save_state(all_nodes, "result_0.0", folder=folder)
    else:
        ckpt_dist, ckpt_path = ckpt
        next_dist = round(ckpt_dist + 0.05, 2)
        if next_dist > CLUSTER_MAX_DIST_NARRATIVE:
            logger.info("[%s] Latest checkpoint dist=%.2f is final — "
                        "tree already complete, skipping.", folder, ckpt_dist)
            return
        logger.info("[%s] Resuming from checkpoint %s — starting at dist=%.2f",
                    folder, ckpt_path, next_dist)
        all_nodes, _ = load_structure_file(ckpt_path)
        cluster_max_dist = next_dist

    stop_criteria = False

    while not stop_criteria:
        start = time.time()
        logger.info("Iteration %d (dist=%.2f, nodes=%d)", index, cluster_max_dist, len(all_nodes))

        embeddings = np.array([node.embedding for node in all_nodes])
        linkage_matrix = linkage(embeddings, metric='cosine')
        clusters = fcluster(linkage_matrix, t=cluster_max_dist, criterion="distance")

        # Group elements from each cluster
        cluster_dict = defaultdict(list)
        for node, cluster_id in zip(all_nodes, clusters):
            cluster_dict[cluster_id].append(node)

        # Process clusters in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            next_iter_nodes = list(executor.map(process_cluster, cluster_dict.values()))
        next_iter_nodes = [item for sublist in next_iter_nodes for item in sublist]

        if len(all_nodes) <= len(next_iter_nodes):
            if cluster_max_dist <= CLUSTER_MAX_DIST_NARRATIVE:
                save_state(next_iter_nodes, "result_" + str(cluster_max_dist), folder=folder)
                cluster_max_dist = round(cluster_max_dist + 0.05, 2)
            else:
                stop_criteria = True

        elapsed = time.time() - start
        logger.info("Iteration %d completed in %.3fs (%d -> %d nodes)",
                     index, elapsed, len(all_nodes), len(next_iter_nodes))

        all_nodes = next_iter_nodes.copy()
        if len(all_nodes) < 2:
            stop_criteria = True
        index = index + 1

    save_state(all_nodes, f"result_{ACTIVE_THRESHOLD}", folder=folder)


# ---------------------------------------------------------------------------
# High-level training entry point
# ---------------------------------------------------------------------------

def _build_tree(sub_df, folder, truth_mode):
    """Build (or resume) a single narrative tree for *sub_df* under *folder*.

    Embeddings are only computed when starting fresh — when a checkpoint
    exists the leaf vectors are reloaded from it inside ``run_algo``, so the
    expensive embedding pass is skipped on resume.
    """
    set_truth_mode(truth_mode)

    if latest_checkpoint(folder) is None:
        logger.info("[%s] Generating embeddings for %d samples...",
                    folder, len(sub_df))
        embeddings = []
        for i, text in enumerate(sub_df['text'].astype(str)):
            if i % 100 == 0:
                logger.info("  Embedding progress: %d/%d", i, len(sub_df))
            embeddings.append(fetch_embedding(clean_text(text)))
        sub_df = sub_df.copy()
        sub_df['embedding'] = embeddings
        nodes = create_tree_list(sub_df)
        run_algo(all_nodes=nodes, folder=folder)
    else:
        # A checkpoint exists: run_algo reloads nodes from it (or skips if the
        # tree is already complete). No embeddings needed.
        run_algo(all_nodes=None, folder=folder)


def train_narrative_tree_from_dataframe(df, folder=""):
    """
    Train dual narrative trees (fake + real) from a labeled dataframe.

    Args:
        df (pandas.DataFrame): Must contain 'text' and 'label' columns
            (label 1 = fake, label 0 = real).
        folder (str): Output directory prefix for saving tree structures.
    """
    df = df[df['text'] != ""].copy()
    df['text'] = df["text"].str.replace('"', '', regex=False)
    df = df.dropna()
    df = df.drop_duplicates(subset='text', keep='first')

    labels = pd.to_numeric(df["label"], errors="coerce")

    logger.info("Total samples for training: %d", len(df))

    df_false = df[labels.eq(1)].copy()
    df_true = df[labels.eq(0)].copy()

    logger.info("Creating tree nodes for false statements %s", df_false.shape)
    logger.info("Creating tree nodes for true statements %s", df_true.shape)

    # Build fake narrative tree (disinformation prompts)
    logger.info("Building fake narrative tree...")
    _build_tree(df_false, folder + "false/", truth_mode=False)

    # Build real narrative tree (factual prompts)
    logger.info("Building truth narrative tree...")
    _build_tree(df_true, folder + "true/", truth_mode=True)

    # Reset to default
    set_truth_mode(False)
